"""Operators: modal frame-by-frame render + pose export, and utilities."""

import os
import re
import math

import bpy
from mathutils import Vector

from . import camera_utils, colmap_export, photara_bridge, indoor_capture


FORMAT_EXT = {
    "PNG": ".png",
    "JPEG": ".jpg",
    "WEBP": ".webp",
    "TIFF": ".tif",
    "OPEN_EXR": ".exr",
}


class PhotaraExportError(Exception):
    pass


_PREVIEW_COLLECTION_TAG = "photara_camera_preview_owner"
_PREVIEW_OBJECT_TAG = "photara_camera_preview"
_PREVIEW_OWNER = "photara_render"
_MAX_PREVIEW_CAMERAS = 10000


def _get_preview_collection(scene, create=False):
    collection = next(
        (item for item in scene.collection.children
         if item.get(_PREVIEW_COLLECTION_TAG) == _PREVIEW_OWNER),
        None,
    )
    if collection is None and create:
        collection = bpy.data.collections.new("Photara Camera Preview")
        collection[_PREVIEW_COLLECTION_TAG] = _PREVIEW_OWNER
        scene.collection.children.link(collection)
    elif collection is not None and not any(
        child == collection for child in scene.collection.children
    ):
        scene.collection.children.link(collection)
    if collection is not None:
        collection.hide_render = True
    return collection


def _clear_preview_objects(collection):
    if collection is None:
        return 0
    preview_objects = [obj for obj in collection.objects if obj.get(_PREVIEW_OBJECT_TAG)]
    for obj in preview_objects:
        data = obj.data
        bpy.data.objects.remove(obj, do_unlink=True)
        if isinstance(data, bpy.types.Mesh) and data.users == 0:
            bpy.data.meshes.remove(data)
        elif isinstance(data, bpy.types.Camera) and data.users == 0 and data.get('photara_preview_optics'):
            bpy.data.cameras.remove(data)
    return len(preview_objects)


def _show_camera_preview(context, poses, source_camera, plan=None):
    if source_camera.get(_PREVIEW_OBJECT_TAG):
        raise PhotaraExportError("请使用场景中的原始相机作为模板，不能使用预览相机")
    collection = _get_preview_collection(context.scene, create=True)
    _clear_preview_objects(collection)
    data = source_camera.data
    if plan is not None:
        data = indoor_capture.optics(source_camera, context.scene.photara)
        data["photara_preview_optics"] = True
    for index, pose in enumerate(poses, 1):
        obj = bpy.data.objects.new("Photara Preview Camera {:04d}".format(index), data)
        obj[_PREVIEW_OBJECT_TAG] = True
        obj.matrix_world = pose
        obj.hide_render = True
        obj.show_in_front = True
        collection.objects.link(obj)
    _reveal_camera_preview(context, collection)
    if plan is not None:
        target = context.scene.photara.box_min_observations
        radius = max(.015, context.scene.photara.box_clearance * .12)
        for name, color, predicate in (
                ("Unseen", (1, .03, .01, 1), lambda n: n == 0),
                ("Insufficient", (1, .4, .01, 1), lambda n: 0 < n < target)):
            vertices, faces = [], []
            for point, observations in zip(plan.samples, plan.observations):
                if not predicate(len(observations)):
                    continue
                offset = len(vertices)
                vertices.extend(tuple(point + Vector(v) * radius) for v in
                                ((1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                 (0, -1, 0), (0, 0, 1), (0, 0, -1)))
                faces.extend(tuple(offset + i for i in face) for face in
                             ((0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
                              (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)))
            if not vertices:
                continue
            mesh = bpy.data.meshes.new("Photara Coverage " + name)
            mesh.from_pydata(vertices, [], faces)
            material = next((mat for mat in bpy.data.materials
                             if mat.get("photara_coverage_color") == name), None)
            if material is None:
                material = bpy.data.materials.new("Photara Coverage " + name)
                material["photara_coverage_color"] = name
                material.diffuse_color = color
            mesh.materials.append(material)
            obj = bpy.data.objects.new("Photara Coverage " + name, mesh)
            obj[_PREVIEW_OBJECT_TAG] = True
            obj.hide_render = True
            obj.show_in_front = True
            obj.color = color
            collection.objects.link(obj)
        context.scene.photara.status = context.scene.photara.box_coverage_status
    else:
        context.scene.photara.status = "相机点位预览已生成：{} 个视点".format(len(poses))


class RenderJob:
    """Prepare / step / finish job used by both the modal and the blocking path."""

    def __init__(self, context):
        self.context = context
        self.props = context.scene.photara
        self.scene = context.scene
        self.camera = None
        self.frames = []
        self.records = []
        self.dataset_dir = ""
        self.images_dir = ""
        self.written = []
        self.width = 0
        self.height = 0
        self._prepared = False
        self._finished = False
        self._index = 0
        self._exported = False
        self._intr_min = None
        self._intr_max = None
        self._originals = {}
        self._original_frame = 0.0
        self._original_camera = None
        self._orbit_camera = None
        self._orbit_target_center = None
        self._orbit_radius = 0.0
        self._generated_poses = []
        self._async_pending_record = None
        self._cancel_requested = False
        self._indoor_plan = None
        self._bounds_visibility = None
        self._render_completed = False
        self._render_cancelled = False
        self._handlers_registered = False

    # ------------------------------------------------------------------ prepare
    def prepare(self):
        props = self.props
        scene = self.scene
        if props.flight_recording:
            raise PhotaraExportError("请先停止自由飞行录制并生成点位")

        camera = props.camera or scene.camera
        if camera is None or camera.type != "CAMERA":
            raise PhotaraExportError("未找到相机：请设置场景相机或在面板中选择一个相机")
        self.camera = camera
        cam_type = camera.data.type
        if cam_type != "PERSP":
            raise PhotaraExportError(
                "相机必须是透视相机（当前：{}），正交/全景相机不适用于 3DGS 重建".format(cam_type)
            )

        if props.capture_mode == "ORBIT":
            target = props.orbit_target
            if target is None:
                raise PhotaraExportError("360° 环绕模式需要选择目标物体")
            if target == camera:
                raise PhotaraExportError("目标物体不能是拍摄相机")
            self._generated_poses = self._build_orbit_poses(target)
            self.frames = list(range(1, len(self._generated_poses) + 1))
        elif props.capture_mode == "BOX_GRID":
            bounds_object = props.box_bounds_object
            if bounds_object is None:
                raise PhotaraExportError("室内 Box 点阵模式需要选择 Room Bounds 边界物体")
            if bounds_object == camera:
                raise PhotaraExportError("Room Bounds 不能是拍摄相机")
            self._generated_poses = self._build_box_grid_poses(bounds_object)
            if props.box_block_incomplete and not self._indoor_plan.report["coverage_target_met"]:
                raise PhotaraExportError("覆盖目标未达到；请先预览红/橙色盲区，调整范围、点阵或 Image Budget")
            self.frames = list(range(1, len(self._generated_poses) + 1))
        else:
            if props.use_scene_frame_range:
                start, end = scene.frame_start, scene.frame_end
            else:
                start, end = props.frame_start, props.frame_end
            if end < start:
                raise PhotaraExportError("帧范围无效：结束帧 {} 小于起始帧 {}".format(end, start))
            self.frames = list(range(int(start), int(end) + 1, max(1, int(props.frame_step))))
        if not self.frames:
            raise PhotaraExportError("帧范围为空")

        if not (props.export_colmap or props.export_transforms_json):
            raise PhotaraExportError("请至少启用 COLMAP 或 transforms.json 导出")
        if scene.render.use_border:
            raise PhotaraExportError("请关闭 Render Region 后导出完整图像")
        render = scene.render
        percentage = render.resolution_percentage / 100.0
        self.width = int(render.resolution_x * percentage)
        self.height = int(render.resolution_y * percentage)
        if self.width < 1 or self.height < 1:
            raise PhotaraExportError("渲染分辨率无效（{}x{}）".format(self.width, self.height))

        self.dataset_dir = os.path.abspath(bpy.path.abspath(props.output_root))
        self.images_dir = os.path.join(self.dataset_dir, "images")
        os.makedirs(self.images_dir, exist_ok=True)
        if self._indoor_plan is not None:
            self._indoor_plan.write(self.dataset_dir)

        if props.overwrite:
            self._cleanup_old_frames()

        # ---- save & apply render settings -------------------------------------
        img = render.image_settings
        self._originals = {
            "filepath": render.filepath,
            "use_file_extension": render.use_file_extension,
            "film_transparent": render.film_transparent,
            "file_format": img.file_format,
            "color_mode": img.color_mode,
            "color_depth": img.color_depth,
            "quality": img.quality,
            "use_motion_blur": getattr(render, "use_motion_blur", None),
        }
        self._original_frame = float(scene.frame_current) + scene.frame_subframe
        self._original_camera = scene.camera
        if props.capture_mode == "ORBIT":
            self._prepare_orbit_camera(camera, props.orbit_target)
        elif props.capture_mode == "BOX_GRID":
            self._create_generated_camera(camera, "Photara Box Grid Camera")
            if props.box_bounds_helper:
                self._bounds_visibility = (props.box_bounds_object, props.box_bounds_object.hide_render)
                props.box_bounds_object.hide_render = True
        if props.capture_mode == "BOX_GRID" and hasattr(render, "use_motion_blur"):
            render.use_motion_blur = False

        fmt = props.image_format
        img.file_format = fmt
        img.color_mode = "RGB"
        img.color_depth = "32" if fmt == "OPEN_EXR" else "8"
        if fmt in ("JPEG", "WEBP"):
            img.quality = int(props.jpeg_quality)
        render.use_file_extension = False
        if props.force_opaque:
            render.film_transparent = False
        if props.capture_mode == "ANIMATION" and props.camera and props.camera != scene.camera:
            scene.camera = props.camera

        self._prepared = True
        self._index = 0
        props.progress = 0.0
        props.status = "准备渲染：{} 帧 → {}".format(len(self.frames), self.dataset_dir)

    def _prepare_orbit_camera(self, source_camera, target):
        """Create a temporary, unanimated camera using the selected camera's optics."""
        self._create_generated_camera(source_camera, "Photara Orbit Camera")

    def _build_orbit_poses(self, target):
        props = self.props
        deps = bpy.context.evaluated_depsgraph_get()
        target_eval = target.evaluated_get(deps)
        corners = [target_eval.matrix_world @ Vector(corner) for corner in target_eval.bound_box]
        if corners:
            center = sum(corners, Vector((0.0, 0.0, 0.0))) / len(corners)
            size = max(
                max(point[i] for point in corners) - min(point[i] for point in corners)
                for i in range(3)
            )
        else:
            center = target_eval.matrix_world.translation.copy()
            size = max(target_eval.dimensions)
        if size <= 1e-6:
            size = 1.0
        radius = size * float(props.orbit_radius_factor)
        levels = int(props.orbit_levels)
        total_shots = int(props.orbit_shots)
        if total_shots < levels:
            raise PhotaraExportError("Total Shots 不能少于 Elevation Rings")
        if props.orbit_coverage == "UPPER_HEMISPHERE":
            elevation_min, elevation_max = 0.0, math.pi * 0.5
        elif props.orbit_coverage == "FULL_SPHERE":
            elevation_min, elevation_max = -math.pi * 0.5, math.pi * 0.5
        else:
            elevation_min = float(props.orbit_elevation_min)
            elevation_max = float(props.orbit_elevation_max)
        if levels > 1 and elevation_max <= elevation_min:
            raise PhotaraExportError("多层球面环绕要求 Max Elevation 大于 Min Elevation")
        if levels == 1:
            if props.orbit_coverage == "UPPER_HEMISPHERE":
                elevations = [math.pi * 0.25]
            else:
                elevations = [min(max(0.0, elevation_min), elevation_max)]
        else:
            elevations = [
                elevation_min + (elevation_max - elevation_min) * ring / (levels - 1)
                for ring in range(levels)
            ]
        # Keep a tiny azimuth radius at the poles so each shot remains a distinct pose.
        pole_limit = math.radians(89.5)
        elevations = [
            max(-pole_limit, min(pole_limit, elevation)) for elevation in elevations
        ]

        base_shots, extra_shots = divmod(total_shots, levels)
        start_angle = float(props.orbit_start_angle)
        poses = []
        for ring, elevation in enumerate(elevations):
            ring_shots = base_shots + (1 if ring < extra_shots else 0)
            stagger = math.pi / ring_shots if ring % 2 else 0.0
            for shot in range(ring_shots):
                azimuth = start_angle + stagger + 2.0 * math.pi * shot / ring_shots
                position = Vector((
                    center.x + radius * math.cos(elevation) * math.cos(azimuth),
                    center.y + radius * math.cos(elevation) * math.sin(azimuth),
                    center.z + radius * math.sin(elevation),
                ))
                matrix = (center - position).to_track_quat("-Z", "Y").to_matrix().to_4x4()
                matrix.translation = position
                poses.append(matrix)
        self._orbit_target_center = center
        self._orbit_radius = radius
        return poses

    def _create_generated_camera(self, source_camera, name):
        camera = source_camera.copy()
        camera.data = source_camera.data.copy()
        camera.data.animation_data_clear()
        if self.props.capture_mode == 'BOX_GRID' and self.props.box_auto_capture:
            camera.data.lens = self.props.box_capture_lens
            camera.data.shift_x = camera.data.shift_y = 0.0
            camera.data.dof.use_dof = False
        camera.name = name
        camera.parent = None
        camera.animation_data_clear()
        for constraint in list(camera.constraints):
            camera.constraints.remove(constraint)
        self.scene.collection.objects.link(camera)
        camera.matrix_world = source_camera.matrix_world.copy()
        self._orbit_camera = camera
        self.camera = camera
        self.scene.camera = camera

    def _build_box_grid_poses(self, bounds_object):
        if self._indoor_plan is None:
            camera = self.props.camera or self.scene.camera
            if camera is None or camera.type != "CAMERA":
                raise PhotaraExportError("请先选择一个透视相机")
            try:
                self._indoor_plan = indoor_capture.build_plan(self.context, bounds_object, camera)
            except indoor_capture.PlanningError as error:
                raise PhotaraExportError(str(error)) from error
        return self._indoor_plan.poses

    def _orbit_matrix(self, shot_index):
        return self._generated_poses[shot_index]

    def _cleanup_old_frames(self):
        """Remove only our own frame_XXXXXXXX files inside the images dir."""
        pattern = re.compile(r"^frame_\d+\.(png|jpg|jpeg|webp|tif|tiff|exr)$", re.IGNORECASE)
        try:
            entries = list(os.scandir(self.images_dir))
        except OSError:
            return
        for entry in entries:
            if entry.is_file(follow_symlinks=False) and pattern.match(entry.name):
                try:
                    os.remove(entry.path)
                except OSError:
                    pass

    # -------------------------------------------------------------------- step
    def step(self):
        """Render one frame. Returns True when everything (incl. exports) is done."""
        if not self._prepared:
            raise PhotaraExportError("任务尚未准备")
        if self._index >= len(self.frames):
            return self._finish_exports()

        record, filepath = self._prepare_current_frame()
        self.scene.render.filepath = filepath
        bpy.ops.render.render(write_still=True)
        self._complete_current_frame(record)

        if self._index >= len(self.frames):
            return self._finish_exports()
        return False

    def _prepare_current_frame(self):
        props = self.props
        scene = self.scene
        frame = self.frames[self._index]
        if props.capture_mode == "ORBIT":
            self.camera.matrix_world = self._orbit_matrix(self._index)
            scene.frame_set(int(self._original_frame), subframe=self._original_frame - int(self._original_frame))
            bpy.context.view_layer.update()
        elif props.capture_mode == "BOX_GRID":
            self.camera.matrix_world = self._generated_poses[self._index]
            scene.frame_set(int(self._original_frame), subframe=self._original_frame - int(self._original_frame))
            bpy.context.view_layer.update()
        else:
            scene.frame_set(frame, subframe=0.0)

        deps = bpy.context.evaluated_depsgraph_get()
        cam_eval = self.camera.evaluated_get(deps)
        rigid_pose = cam_eval.matrix_world.to_quaternion().to_matrix().to_4x4()
        rigid_pose.translation = cam_eval.matrix_world.translation
        matrix = [list(row) for row in rigid_pose]  # copy to plain python
        intr = camera_utils.compute_intrinsics(
            cam_eval.data,
            self.width,
            self.height,
            scene.render.pixel_aspect_x,
            scene.render.pixel_aspect_y,
        )
        fx = intr["fx"]
        self._intr_min = fx if self._intr_min is None else min(self._intr_min, fx)
        self._intr_max = fx if self._intr_max is None else max(self._intr_max, fx)

        filename = "frame_{:08d}{}".format(frame, FORMAT_EXT[props.image_format])
        return (
            {
                "filename": filename,
                "matrix_world": matrix,
                "width": self.width,
                "height": self.height,
                "intrinsics": intr,
                "frame": frame,
            },
            os.path.join(self.images_dir, filename),
        )

    def _complete_current_frame(self, record):
        self.records.append(record)
        self._index += 1
        self.props.progress = self._index / float(len(self.frames))
        self.props.status = "渲染中 {}/{}（frame {}）".format(
            self._index, len(self.frames), record["frame"]
        )

    def step_async(self):
        """Start or poll a UI-friendly render. Returns True when export is complete."""
        if not self._prepared:
            raise PhotaraExportError("任务尚未准备")
        if self._async_pending_record is not None:
            if bpy.app.is_job_running("RENDER"):
                return False
            if self._render_cancelled:
                self.request_cancel()
                self._async_pending_record = None
                return False
            if not self._render_completed:
                return False
            record = self._async_pending_record
            self._async_pending_record = None
            if not os.path.isfile(os.path.join(self.images_dir, record["filename"])):
                raise PhotaraExportError("渲染未写入当前帧文件")
            self._complete_current_frame(record)
            if self._cancel_requested:
                return False
            if self._index >= len(self.frames):
                return self._finish_exports()
            return False

        if self._index >= len(self.frames):
            return self._finish_exports()
        if self._cancel_requested:
            return False

        record, filepath = self._prepare_current_frame()
        self.scene.render.filepath = filepath
        self._async_pending_record = record
        self._render_completed = False
        self._render_cancelled = False
        self._register_handlers()
        result = bpy.ops.render.render("INVOKE_DEFAULT", write_still=True)
        if "CANCELLED" in result:
            self._async_pending_record = None
            raise RuntimeError("Blender 取消了当前图像渲染")
        if "RUNNING_MODAL" not in result and not bpy.app.is_job_running("RENDER"):
            self._render_completed = True
        return False

    def _on_render_complete(self, scene, *args):
        if scene == self.scene:
            self._render_completed = True

    def _on_render_cancel(self, scene, *args):
        if scene == self.scene:
            self._render_cancelled = True

    def _register_handlers(self):
        if not self._handlers_registered:
            bpy.app.handlers.render_complete.append(self._on_render_complete)
            bpy.app.handlers.render_cancel.append(self._on_render_cancel)
            self._handlers_registered = True

    def _remove_handlers(self):
        if self._handlers_registered:
            for handlers, callback in (
                    (bpy.app.handlers.render_complete, self._on_render_complete),
                    (bpy.app.handlers.render_cancel, self._on_render_cancel)):
                if callback in handlers:
                    handlers.remove(callback)
            self._handlers_registered = False

    def request_cancel(self):
        if self._cancel_requested:
            return
        self._cancel_requested = True
        self.props.cancel_requested = True
        self.props.status = "已请求停止采集；当前帧完成后停止（Esc 可中止当前帧）"

    def _finish_exports(self):
        if self._exported:
            return True
        if not self.records:
            raise PhotaraExportError("没有可导出的帧")

        props = self.props
        initial_points = self._indoor_plan.initial_points(self.records) if self._indoor_plan else None
        if self._indoor_plan is not None:
            self._indoor_plan.report["completed_images"] = len(self.records)
            self._indoor_plan.report["capture_completed"] = len(self.records) == len(self.frames)
            self._indoor_plan.write(self.dataset_dir)
        self.written = colmap_export.write_dataset(
            self.dataset_dir,
            self.records,
            self.records[0]["intrinsics"],
            name=props.dataset_name,
            write_colmap=props.export_colmap,
            write_transforms=props.export_transforms_json,
            meta={
                "blender_version": bpy.app.version_string,
                "frame_range": (
                    [self.frames[0], self.frames[-1], 1]
                    if props.capture_mode in ("ORBIT", "BOX_GRID")
                    else [self.frames[0], self.frames[-1], props.frame_step]
                ),
                "image_format": props.image_format,
                "capture_mode": props.capture_mode,
                **({"orbit_target": props.orbit_target.name,
                    "orbit_source_frame": int(self._original_frame),
                    "orbit_levels": props.orbit_levels,
                    "orbit_elevation_range": [
                        float(props.orbit_elevation_min), float(props.orbit_elevation_max)
                    ]}
                   if props.capture_mode == "ORBIT" else {}),
                **({"room_bounds": props.box_bounds_object.name,
                    "box_grid": [props.box_grid_x, props.box_grid_y, props.box_grid_z],
                    "views_per_point": int(props.box_views_per_point),
                    "box_source_frame": int(self._original_frame),
                    "capture_coverage": self._indoor_plan.report}
                   if props.capture_mode == "BOX_GRID" else {}),
            },
            initial_points=initial_points,
        )
        self._exported = True
        msg = "导出完成：{} 帧 → {}".format(len(self.records), self.dataset_dir)

        props.status = msg
        return True

    # ------------------------------------------------------------------ finish
    def finish(self, cancelled=False, deliver=False, use_async=True):
        self._remove_handlers()
        self.props.is_rendering = False
        self.props.cancel_requested = False
        if self._finished or (not self._prepared and self._orbit_camera is None):
            return
        self._finished = True

        props = self.props
        props.is_rendering = False
        props.cancel_requested = False
        if self._bounds_visibility is not None:
            bounds, visibility = self._bounds_visibility
            bounds.hide_render = visibility
            self._bounds_visibility = None
        render = self.scene.render
        img = render.image_settings
        o = self._originals
        try:
            render.filepath = o["filepath"]
            render.use_file_extension = o["use_file_extension"]
            render.film_transparent = o["film_transparent"]
            img.file_format = o["file_format"]
            img.color_mode = o["color_mode"]
            img.color_depth = o["color_depth"]
            img.quality = o["quality"]
            if o["use_motion_blur"] is not None:
                render.use_motion_blur = o["use_motion_blur"]
            self.scene.camera = self._original_camera
            frame = self._original_frame
            self.scene.frame_set(int(frame), subframe=frame - int(frame))
        except Exception:
            pass

        if self._orbit_camera is not None:
            try:
                orbit_data = self._orbit_camera.data
                bpy.data.objects.remove(self._orbit_camera, do_unlink=True)
                if orbit_data.users == 0:
                    bpy.data.cameras.remove(orbit_data)
            except Exception:
                pass
            self._orbit_camera = None

        if cancelled and self.records:
            self._finish_exports()
        if cancelled or not deliver or props.handover_mode == "OFF":
            return

        snapshot = photara_bridge.PropsSnapshot(props)

        if not use_async:
            result = photara_bridge.run_bridge(snapshot, self.dataset_dir)
            if result is not None:
                props.status = result.message
            return

        box = {"result": None}

        def on_done(result):
            box["result"] = result

        photara_bridge.submit_async(snapshot, self.dataset_dir, on_done)

        def poll():
            result = box["result"]
            if result is None:
                return 0.25
            try:
                props.status = result.message
            except Exception:
                pass
            return None

        try:
            bpy.app.timers.register(poll, first_interval=0.25)
        except Exception:
            pass


def _record_poses_blocking(context):
    """Evaluate per-frame poses without rendering (poses-only export)."""
    props = context.scene.photara
    if props.flight_recording:
        raise PhotaraExportError("请先停止自由飞行录制并生成点位")
    scene = context.scene
    camera = props.camera or scene.camera
    if camera is None or camera.type != "CAMERA":
        raise PhotaraExportError("未找到相机：请设置场景相机或在面板中选择一个相机")
    if camera.data.type != "PERSP":
        raise PhotaraExportError("相机必须是透视相机")

    generated_poses = None
    if props.capture_mode == "ORBIT":
        if props.orbit_target is None:
            raise PhotaraExportError("360° 环绕模式需要选择目标物体")
        generator = RenderJob(context)
        generated_poses = generator._build_orbit_poses(props.orbit_target)
        frames = list(range(1, len(generated_poses) + 1))
    elif props.capture_mode == "BOX_GRID":
        if props.box_bounds_object is None:
            raise PhotaraExportError("室内 Box 点阵模式需要选择 Room Bounds 边界物体")
        generator = RenderJob(context)
        generated_poses = generator._build_box_grid_poses(props.box_bounds_object)
        frames = list(range(1, len(generated_poses) + 1))
    else:
        if props.use_scene_frame_range:
            start, end = scene.frame_start, scene.frame_end
        else:
            start, end = props.frame_start, props.frame_end
        frames = list(range(int(start), int(end) + 1, max(1, int(props.frame_step))))
    if not frames:
        raise PhotaraExportError("帧范围为空")

    render = scene.render
    percentage = render.resolution_percentage / 100.0
    width = int(render.resolution_x * percentage)
    height = int(render.resolution_y * percentage)
    dataset_dir = os.path.abspath(bpy.path.abspath(props.output_root))
    images_dir = os.path.join(dataset_dir, "images")

    ext = FORMAT_EXT[props.image_format]
    original_frame = float(scene.frame_current) + scene.frame_subframe
    records = []
    intr = None
    for shot_index, frame in enumerate(frames):
        if generated_poses is None:
            scene.frame_set(frame, subframe=0.0)
        deps = context.evaluated_depsgraph_get()
        cam_eval = camera.evaluated_get(deps)
        if generated_poses is None:
            rigid_pose = cam_eval.matrix_world.to_quaternion().to_matrix().to_4x4()
            rigid_pose.translation = cam_eval.matrix_world.translation
        else:
            rigid_pose = generated_poses[shot_index]
        records.append(
            {
                "filename": "frame_{:08d}{}".format(frame, ext),
                "matrix_world": [list(row) for row in rigid_pose],
                "width": width,
                "height": height,
            }
        )
        current_intrinsics = (generator._indoor_plan.report["intrinsics"]
            if props.capture_mode == "BOX_GRID" else camera_utils.compute_intrinsics(
                cam_eval.data, width, height,
                render.pixel_aspect_x, render.pixel_aspect_y,
            ))
        records[-1]["intrinsics"] = current_intrinsics
        if intr is None:
            intr = current_intrinsics
    if generated_poses is None:
        frame = original_frame
        scene.frame_set(int(frame), subframe=frame - int(frame))

    missing = [
        r["filename"]
        for r in records
        if not os.path.isfile(os.path.join(images_dir, r["filename"]))
    ]
    os.makedirs(images_dir, exist_ok=True)
    initial_points = None
    if props.capture_mode == "BOX_GRID":
        existing_records = [record for record in records if record["filename"] not in missing]
        initial_points = generator._indoor_plan.initial_points(existing_records)
        generator._indoor_plan.report["images_present"] = len(records) - len(missing)
        generator._indoor_plan.report["capture_completed"] = not missing
        generator._indoor_plan.write(dataset_dir)
    colmap_export.write_dataset(
        dataset_dir,
        records,
        intr,
        name=props.dataset_name,
        write_colmap=props.export_colmap,
        write_transforms=props.export_transforms_json,
        initial_points=initial_points,
        meta={
            "blender_version": bpy.app.version_string,
            "frame_range": [frames[0], frames[-1], 1 if generated_poses is not None else props.frame_step],
            "image_format": props.image_format,
            "capture_mode": props.capture_mode,
            **({"orbit_target": props.orbit_target.name,
                "orbit_source_frame": int(original_frame),
                "orbit_levels": props.orbit_levels,
                "orbit_elevation_range": [
                    float(props.orbit_elevation_min), float(props.orbit_elevation_max)
                ]}
               if props.capture_mode == "ORBIT" else {}),
            **({"room_bounds": props.box_bounds_object.name,
                "box_grid": [props.box_grid_x, props.box_grid_y, props.box_grid_z],
                "views_per_point": int(props.box_views_per_point),
                "box_source_frame": int(original_frame),
                "capture_coverage": generator._indoor_plan.report}
               if props.capture_mode == "BOX_GRID" else {}),
        },
    )
    return records, dataset_dir, missing


class PHOTARA_OT_render_and_export(bpy.types.Operator):
    """渲染序列帧、导出每帧相机位姿（COLMAP/transforms.json），完成后交给 Photara"""

    bl_idname = "photara.render_and_export"
    bl_label = "Render & Export for Photara"
    bl_options = {"REGISTER", "UNDO"}

    _timer = None

    @classmethod
    def poll(cls, context):
        return context.scene is not None and not context.scene.photara.is_rendering

    def invoke(self, context, event):
        if context.window is None or context.area is None:
            return self.execute(context)

        self.job = RenderJob(context)
        self._planning = None
        try:
            props = context.scene.photara
            if props.capture_mode == "BOX_GRID":
                camera = props.camera or context.scene.camera
                if props.box_bounds_object is None or camera is None or camera.type != "CAMERA":
                    raise PhotaraExportError("请先选择 Room Bounds 和透视相机")
                self._planner = indoor_capture.IndoorPlanner(context, props.box_bounds_object, camera)
                self._planning = self._planner.steps()
            else:
                self.job.prepare()
        except (PhotaraExportError, indoor_capture.PlanningError, ValueError) as e:
            self.job.finish(cancelled=True)
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except OSError as e:
            self.report({"ERROR"}, "输出目录无法创建：{}".format(e))
            return {"CANCELLED"}

        context.scene.photara.cancel_requested = False
        context.scene.photara.progress = 0.0
        context.scene.photara.is_rendering = True
        self._timer = context.window_manager.event_timer_add(0.1, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def modal(self, context, event):
        if event.type == "ESC":
            self.job.request_cancel()
            if self._planning is not None:
                self._planning.close()
                self._planning = None
            if not bpy.app.is_job_running("RENDER"):
                if self.job._async_pending_record and self.job._render_completed:
                    self.job.step_async()
                self.job.finish(cancelled=True)
                self._remove_timer(context)
                context.scene.photara.status = "已取消"
                self.report({"WARNING"}, "已取消（已渲染文件保留）")
                return {"CANCELLED"}
            return {"PASS_THROUGH"}

        if event.type == "TIMER":
            try:
                if context.scene.photara.cancel_requested and not self.job._cancel_requested:
                    self.job.request_cancel()
                if self._planning is not None and not self.job._cancel_requested:
                    try:
                        context.scene.photara.status = next(self._planning)
                    except StopIteration:
                        self._planning = None
                        self.job._indoor_plan = self._planner.plan
                        self.job.prepare()
                    return {"PASS_THROUGH"}
                if self.job._cancel_requested and not bpy.app.is_job_running("RENDER"):
                    if self._planning is not None:
                        self._planning.close()
                    if self.job._async_pending_record and self.job._render_completed:
                        self.job.step_async()
                    self.job.finish(cancelled=True)
                    self._remove_timer(context)
                    context.scene.photara.status = "已取消"
                    self.report({"WARNING"}, "已取消（已渲染文件保留）")
                    return {"CANCELLED"}
                done = self.job.step_async()
            except (PhotaraExportError, indoor_capture.PlanningError, OSError) as e:
                self.job.finish(cancelled=True)
                self._remove_timer(context)
                self.report({"ERROR"}, str(e))
                return {"CANCELLED"}
            except RuntimeError as e:
                self.job.finish(cancelled=True)
                self._remove_timer(context)
                self.report({"ERROR"}, "渲染失败：{}".format(e))
                return {"CANCELLED"}
            if done:
                self.job.finish(cancelled=False, deliver=True, use_async=True)
                self._remove_timer(context)
                self.report(
                    {"INFO"},
                    "完成：{} 帧 → {}".format(len(self.job.records), self.job.dataset_dir),
                )
                return {"FINISHED"}
        return {"PASS_THROUGH"}

    def execute(self, context):
        """Blocking path: EXEC_DEFAULT from UI scripts or background (blender -b)."""
        job = RenderJob(context)
        try:
            job.prepare()
            while not job.step():
                pass
        except PhotaraExportError as e:
            job.finish(cancelled=True)
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        except (RuntimeError, OSError) as e:
            job.finish(cancelled=True)
            self.report({"ERROR"}, "渲染/导出失败：{}".format(e))
            return {"CANCELLED"}
        job.finish(cancelled=False, deliver=True, use_async=False)
        self.report(
            {"INFO"}, "完成：{} 帧 → {}".format(len(job.records), job.dataset_dir)
        )
        return {"FINISHED"}

    def _remove_timer(self, context):
        if self._timer is not None:
            try:
                context.window_manager.event_timer_remove(self._timer)
            except Exception:
                pass
            self._timer = None


class PHOTARA_OT_cancel_render(bpy.types.Operator):
    """Request cancellation of the active Photara frame sequence."""

    bl_idname = "photara.cancel_render"
    bl_label = "Cancel Render"
    bl_options = {"INTERNAL"}

    def execute(self, context):
        props = context.scene.photara
        if not props.is_rendering:
            self.report({"INFO"}, "当前没有正在运行的 Photara 渲染")
            return {"CANCELLED"}
        props.cancel_requested = True
        props.status = "正在停止采集（渲染中则等待当前帧完成；Esc 可中止当前帧）"
        return {"FINISHED"}


class PHOTARA_OT_preview_camera_positions(bpy.types.Operator):
    """Show generated or animated camera poses in the 3D viewport."""

    bl_idname = "photara.preview_camera_positions"
    bl_label = "Preview Camera Positions"
    bl_options = {"REGISTER", "UNDO"}

    @classmethod
    def poll(cls, context):
        return context.scene is not None and context.scene.photara.capture_mode in {
            "ORBIT", "BOX_GRID", "ANIMATION"
        } and not context.scene.photara.is_rendering and not context.scene.photara.flight_recording

    def invoke(self, context, event):
        if context.scene.photara.capture_mode != "BOX_GRID" or context.window is None:
            return self.execute(context)
        props = context.scene.photara
        self._source = props.camera or context.scene.camera
        try:
            if props.box_bounds_object is None or self._source is None:
                raise indoor_capture.PlanningError("请先选择 Room Bounds 和透视相机")
            self._planner = indoor_capture.IndoorPlanner(context, props.box_bounds_object, self._source)
            self._steps = self._planner.steps()
        except (indoor_capture.PlanningError, ValueError) as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        props.cancel_requested = False
        props.is_rendering = True
        self._timer = context.window_manager.event_timer_add(.05, window=context.window)
        context.window_manager.modal_handler_add(self)
        return {"RUNNING_MODAL"}

    def _end_preview(self, context):
        context.window_manager.event_timer_remove(self._timer)
        context.scene.photara.is_rendering = False
        context.scene.photara.cancel_requested = False

    def modal(self, context, event):
        props = context.scene.photara
        if event.type == "ESC" or props.cancel_requested:
            self._steps.close()
            self._end_preview(context)
            props.status = "已取消点位分析；上次预览保留"
            return {"CANCELLED"}
        if event.type != "TIMER":
            return {"PASS_THROUGH"}
        try:
            props.status = next(self._steps)
        except StopIteration:
            _show_camera_preview(context, self._planner.plan.poses, self._source, self._planner.plan)
            self._end_preview(context)
            return {"FINISHED"}
        except (ValueError, RuntimeError) as error:
            self._end_preview(context)
            props.status = str(error)
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}
        return {"PASS_THROUGH"}

    def execute(self, context):
        props = context.scene.photara
        source_camera = props.camera or context.scene.camera
        if source_camera is None or source_camera.type != "CAMERA":
            self.report({"ERROR"}, "请先选择一个透视相机")
            return {"CANCELLED"}
        if source_camera.data.type != "PERSP":
            self.report({"ERROR"}, "点位预览需要透视相机")
            return {"CANCELLED"}

        generator = RenderJob(context)
        try:
            if props.capture_mode == "ANIMATION":
                scene = context.scene
                start, end = ((scene.frame_start, scene.frame_end) if props.use_scene_frame_range
                              else (props.frame_start, props.frame_end))
                frames = range(start, end + 1, max(1, props.frame_step))
                if not frames or len(frames) > _MAX_PREVIEW_CAMERAS:
                    raise PhotaraExportError("预览帧范围应包含 1 至 10000 个点位")
                original_frame, original_subframe = scene.frame_current, scene.frame_subframe
                poses = []
                try:
                    for frame in frames:
                        scene.frame_set(frame)
                        poses.append(source_camera.evaluated_get(context.evaluated_depsgraph_get()).matrix_world.copy())
                finally:
                    scene.frame_set(original_frame, subframe=original_subframe)
            elif props.capture_mode == "ORBIT":
                if props.orbit_target is None:
                    raise PhotaraExportError("请先选择 Target Object")
                poses = generator._build_orbit_poses(props.orbit_target)
            else:
                if props.box_bounds_object is None:
                    raise PhotaraExportError("请先选择 Room Bounds")
                poses = generator._build_box_grid_poses(props.box_bounds_object)
        except PhotaraExportError as error:
            self.report({"ERROR"}, str(error))
            return {"CANCELLED"}

        if len(poses) > _MAX_PREVIEW_CAMERAS:
            self.report(
                {"ERROR"},
                "预览点位上限为 {} 个；请减少网格数或拍摄张数".format(
                    _MAX_PREVIEW_CAMERAS
                ),
            )
            return {"CANCELLED"}

        _show_camera_preview(context, poses, source_camera, generator._indoor_plan)

        message = (props.box_coverage_status if generator._indoor_plan else
                   "相机点位预览已生成：{} 个视点".format(len(poses)))
        props.status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


def _reveal_camera_preview(context, collection):
    collection.hide_viewport = False
    layer = context.view_layer.layer_collection.children.get(collection.name)
    if layer is not None:
        layer.exclude = False
        layer.hide_viewport = False
    context.view_layer.update()
    space = context.space_data
    if space is not None and space.type == 'VIEW_3D':
        space.overlay.show_overlays = True
        space.overlay.show_extras = True
        space.show_object_viewport_camera = True
        for obj in collection.objects:
            obj.hide_set(False)
            if space.local_view is not None:
                obj.local_view_set(space, True)
        context.area.tag_redraw()


class PHOTARA_OT_focus_camera_preview(bpy.types.Operator):
    bl_idname = 'photara.focus_camera_preview'
    bl_label = 'Frame Camera Positions'
    bl_description = '显示并选中预览相机，将视图定位到所有预览点位'

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == 'VIEW_3D'
                and not context.scene.photara.is_rendering
                and not context.scene.photara.flight_recording)

    def execute(self, context):
        collection = _get_preview_collection(context.scene)
        cameras = [] if collection is None else [o for o in collection.objects
                   if o.type == 'CAMERA' and o.get(_PREVIEW_OBJECT_TAG)]
        if not cameras:
            self.report({'WARNING'}, '没有相机点位预览，请先点击 Preview Camera Positions')
            return {'CANCELLED'}
        region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        if region is None:
            return {'CANCELLED'}
        _reveal_camera_preview(context, collection)
        for obj in context.view_layer.objects:
            obj.select_set(False)
        for obj in cameras:
            obj.select_set(True)
        context.view_layer.objects.active = cameras[0]
        with context.temp_override(region=region):
            bpy.ops.view3d.view_selected('EXEC_DEFAULT', use_all_regions=False)
        return {'FINISHED'}


class PHOTARA_OT_clear_camera_preview(bpy.types.Operator):
    """Remove only the camera objects created by Photara's preview."""

    bl_idname = "photara.clear_camera_preview"
    bl_label = "Clear Preview"
    bl_options = {"REGISTER", "UNDO"}

    def execute(self, context):
        collection = _get_preview_collection(context.scene)
        removed = _clear_preview_objects(collection)
        message = "已清除 {} 个相机预览点位".format(removed)
        context.scene.photara.status = message
        self.report({"INFO"}, message)
        return {"FINISHED"}


class PHOTARA_OT_export_poses(bpy.types.Operator):
    """不渲染，仅按当前设置导出相机位姿文件（图像应已存在）"""

    bl_idname = "photara.export_poses"
    bl_label = "Export Poses Only"
    bl_options = {"REGISTER"}

    @classmethod
    def poll(cls, context):
        return context.scene is not None

    def execute(self, context):
        try:
            records, dataset_dir, missing = _record_poses_blocking(context)
        except PhotaraExportError as e:
            self.report({"ERROR"}, str(e))
            return {"CANCELLED"}
        msg = "已导出 {} 个位姿 → {}".format(len(records), dataset_dir)
        if missing:
            msg += "（警告：{} 张图像缺失，请先渲染）".format(len(missing))
            self.report({"WARNING"}, msg)
        else:
            self.report({"INFO"}, msg)
        context.scene.photara.status = msg
        return {"FINISHED"}


class PHOTARA_OT_open_dataset_folder(bpy.types.Operator):
    """在系统文件管理器中打开数据集目录"""

    bl_idname = "photara.open_dataset_folder"
    bl_label = "Open Dataset Folder"
    bl_options = {"REGISTER"}

    def execute(self, context):
        path = os.path.abspath(bpy.path.abspath(context.scene.photara.output_root))
        if not os.path.isdir(path):
            self.report({"ERROR"}, "目录不存在：{}".format(path))
            return {"CANCELLED"}
        try:
            if os.name == "nt":
                os.startfile(path)  # noqa: S606
            elif os.uname().sysname == "Darwin":
                import subprocess

                subprocess.Popen(["open", path])
            else:
                import subprocess

                subprocess.Popen(["xdg-open", path])
        except OSError as e:
            self.report({"ERROR"}, "打开目录失败：{}".format(e))
            return {"CANCELLED"}
        return {"FINISHED"}


class PHOTARA_OT_test_studio_connection(bpy.types.Operator):
    """测试 Photara Studio Agent Control 连接，并探测本机可执行文件"""

    bl_idname = "photara.test_studio_connection"
    bl_label = "Test Connection"
    bl_options = {"REGISTER"}

    def execute(self, context):
        props = context.scene.photara
        messages = []

        endpoint = photara_bridge.discover_studio(timeout=float(props.studio_timeout))
        if endpoint is not None:
            messages.append(
                "Studio 已连接（{}，session {}）".format(
                    endpoint.get("url"), str(endpoint.get("session_id", "?"))[:8]
                )
            )
        elif photara_bridge.studio_process_running():
            messages.append("Studio 在运行但未开启 Agent Control（--enable-agent-control）")
        else:
            messages.append("未发现运行中的 Studio")

        studio_exe = photara_bridge.find_photara_executable("studio", props.studio_exe)
        cli_exe = photara_bridge.find_photara_executable("cli", props.cli_exe)
        messages.append(
            "Studio: {}".format(studio_exe if studio_exe else "未探测到（可手动指定）")
        )
        messages.append(
            "CLI: {}".format(cli_exe if cli_exe else "未探测到（可手动指定）")
        )

        text = "；".join(messages)
        props.status = text
        self.report({"INFO"}, text)
        return {"FINISHED"}
