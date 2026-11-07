"""Geometry-aware, deterministic indoor capture planning (Blender only).

Coverage is a sampled geometric estimate, not a reconstruction-quality guarantee.
All visibility tests use evaluated mesh geometry and the actual capture intrinsics.
"""

import bisect
import json
import math
import os
import random
from collections import Counter
from dataclasses import dataclass, field

import bpy
import numpy as np
from mathutils import Vector
from mathutils.bvhtree import BVHTree

from . import camera_utils


class PlanningError(ValueError):
    pass


def optics(camera, props):
    """Return a temporary optics copy; the caller owns and removes it."""
    data = camera.data.copy()
    data.animation_data_clear()
    if props.box_auto_capture:
        data.lens = props.box_capture_lens
        data.shift_x = data.shift_y = 0.0
        data.dof.use_dof = False
    return data


def pose_at(position, direction):
    matrix = direction.to_track_quat("-Z", "Y").to_matrix().to_4x4()
    matrix.translation = position
    return matrix


@dataclass
class CapturePlan:
    poses: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    samples: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    report: dict = field(default_factory=dict)
    completed_counts: list = field(default_factory=list)

    def initial_points(self, records):
        """Geometry seeds observed from at least two actually exported positions."""
        allowed = {}
        for index, observations in enumerate(self.observations):
            for position in observations:
                key = tuple(round(float(v), 4) for v in position)
                allowed.setdefault(key, []).append(index)
        seen = [set() for _ in self.samples]
        samples = np.asarray([tuple(point) for point in self.samples], dtype=float)
        intr = self.report["intrinsics"]
        width, height = self.report["resolution"]
        near, far = self.report["clip_range"]
        for frame in records:
            matrix = np.asarray(frame["matrix_world"], dtype=float)
            key = tuple(round(float(v), 4) for v in matrix[:3, 3])
            indices = allowed.get(key, [])
            if not indices:
                continue
            local = (samples[indices] - matrix[:3, 3]) @ matrix[:3, :3]
            depth = -local[:, 2]
            safe = np.maximum(depth, 1e-12)
            u = intr["fx"] * local[:, 0] / safe + intr["cx"]
            v = intr["cy"] - intr["fy"] * local[:, 1] / safe
            mask = ((depth > near) & (depth < far) & (u >= width * .01) & (u <= width * .99) &
                    (v >= height * .01) & (v <= height * .99))
            for index in np.flatnonzero(mask):
                seen[indices[index]].add(key)
        self.completed_counts = [len(positions) for positions in seen]
        target = self.report["required_distinct_positions"]
        self.report["completed_coverage_fraction"] = sum(
            count >= target for count in self.completed_counts) / max(1, len(self.samples))
        self.report["completed_unseen_samples"] = sum(count == 0 for count in self.completed_counts)
        points = []
        for sample, count in zip(self.samples, self.completed_counts):
            if count >= 2:
                points.append({"position": list(sample), "color": [128, 128, 128]})
        self.report["initial_surface_points"] = len(points)
        return points

    def write(self, directory):
        os.makedirs(directory, exist_ok=True)
        with open(os.path.join(directory, "capture_coverage.json"), "w", encoding="utf-8") as stream:
            json.dump({**self.report, "samples": [
                {"position": list(point), "distinct_observations": len(views),
                 **({"completed_observations": self.completed_counts[index]}
                    if self.completed_counts else {})}
                for index, (point, views) in enumerate(zip(self.samples, self.observations))
            ], "camera_positions": [list(point) for point in self.positions]}, stream, indent=2)


class IndoorPlanner:
    def __init__(self, context, bounds, camera):
        self.context = context
        self.scene = context.scene
        self.props = self.scene.photara
        self.bounds = bounds
        self.camera = camera
        self.plan = CapturePlan()
        self.vertices = []
        self.triangles = []
        self.solids = []
        self.sampling_triangles = []
        self.bvh = None
        self.rejected = 0
        self.empty_geometry = False
        self.world = bounds.matrix_world.copy()
        if abs(self.world.determinant()) < 1e-10:
            raise PlanningError("Room Bounds 的缩放不能为零")
        self.inverse = self.world.inverted()
        if bounds.type == "EMPTY":
            size = bounds.empty_display_size
            self.low = Vector((-size,) * 3)
            self.high = Vector((size,) * 3)
        else:
            evaluated = bounds.evaluated_get(context.evaluated_depsgraph_get())
            self.world = evaluated.matrix_world.copy()
            self.inverse = self.world.inverted_safe()
            corners = [Vector(corner) for corner in evaluated.bound_box]
            self.low = Vector(tuple(min(p[i] for p in corners) for i in range(3)))
            self.high = Vector(tuple(max(p[i] for p in corners) for i in range(3)))
        # Convert the world-unit margins to each local axis (supports rotated boxes).
        self.scale = Vector(tuple(self.world.to_3x3().col[i].length for i in range(3)))
        self.margin = Vector(tuple(self.props.box_margin / s for s in self.scale))
        if any(self.high[i] - self.low[i] <= 2 * self.margin[i] for i in range(3)):
            raise PlanningError("Room Bounds 尺寸必须大于两倍边距")
        render = self.scene.render
        self.width = int(render.resolution_x * render.resolution_percentage / 100)
        self.height = int(render.resolution_y * render.resolution_percentage / 100)
        if min(self.width, self.height) < 1:
            raise PlanningError("渲染分辨率无效")
        data = optics(camera, self.props)
        try:
            self.intr = camera_utils.compute_intrinsics(
                data, self.width, self.height, render.pixel_aspect_x, render.pixel_aspect_y)
            self.clip_start = float(data.clip_start)
            self.clip_end = float(data.clip_end)
        finally:
            bpy.data.cameras.remove(data)
        self.epsilon = max(1e-5, min(self.scale) * (self.high - self.low).length * 1e-6)

    def contains(self, point, margin=False):
        local = self.inverse @ point
        m = self.margin if margin else Vector((0, 0, 0))
        return all(self.low[i] + m[i] - self.epsilon <= local[i] <=
                   self.high[i] - m[i] + self.epsilon for i in range(3))

    def _clip_triangle(self, points):
        """Clip a surface triangle to the oriented capture bounds."""
        polygon = [self.inverse @ p for p in points]
        for axis in range(3):
            for edge, sign in ((self.low[axis], 1), (self.high[axis], -1)):
                output = []
                for index, point in enumerate(polygon):
                    previous = polygon[index - 1]
                    a = sign * (previous[axis] - edge)
                    b = sign * (point[axis] - edge)
                    if (a >= 0) != (b >= 0):
                        output.append(previous.lerp(point, a / (a - b)))
                    if b >= 0:
                        output.append(point)
                polygon = output
                if len(polygon) < 3:
                    return
        polygon = [self.world @ p for p in polygon]
        for i in range(1, len(polygon) - 1):
            a, b, c = polygon[0], polygon[i], polygon[i + 1]
            area = (b - a).cross(c - a).length * 0.5
            if area > self.epsilon ** 2:
                self.sampling_triangles.append((a, b, c, area))

    def _geometry_steps(self):
        visibility = {}

        def visit(collection, parent_hidden=False):
            invisible = parent_hidden or collection.hide_render
            for obj in collection.objects:
                visibility[obj.original] = visibility.get(obj.original, False) or not invisible
            for child in collection.children:
                visit(child, invisible)

        visit(self.scene.collection)
        hidden = {obj for obj, visible in visibility.items() if not visible}
        for obj in self.scene.objects:
            if (obj.type in {"MESH", "CURVE", "SURFACE", "FONT", "META"}
                    and not obj.hide_render and obj not in hidden
                    and not (obj == self.bounds and self.props.box_bounds_helper)
                    and not obj.get("photara_camera_preview")
                    and not obj.visible_get(view_layer=self.context.view_layer)):
                raise PlanningError("对象 {} 在渲染中可见、但视口隐藏或被排除；请启用其视口可见性后分析".format(obj.name))
        deps = self.context.evaluated_depsgraph_get()
        bounds_corners = [self.world @ Vector((x, y, z))
                          for x in (self.low.x, self.high.x)
                          for y in (self.low.y, self.high.y)
                          for z in (self.low.z, self.high.z)]
        for index, instance in enumerate(deps.object_instances):
            obj = instance.object
            original = obj.original
            if (obj.type not in {"MESH", "CURVE", "SURFACE", "FONT", "META"}
                    or original.hide_render or original in hidden
                    or original.get("photara_camera_preview")
                    or (original == self.bounds and self.props.box_bounds_helper)):
                continue
            if instance.is_instance and instance.parent and instance.parent.original.hide_render:
                continue
            mesh = obj.to_mesh()
            try:
                if mesh is None or not mesh.polygons:
                    continue
                mesh.calc_loop_triangles()
                vertices = [instance.matrix_world @ vertex.co for vertex in mesh.vertices]
                triangles = [tuple(tri.vertices) for tri in mesh.loop_triangles]
                offset = len(self.vertices)
                self.vertices.extend(vertices)
                self.triangles.extend(tuple(v + offset for v in tri) for tri in triangles)
                for triangle in triangles:
                    self._clip_triangle([vertices[v] for v in triangle])
                # Connected components handle room walls and furniture joined into one mesh.
                parents = list(range(len(vertices)))

                def root(index):
                    while parents[index] != index:
                        parents[index] = parents[parents[index]]
                        index = parents[index]
                    return index

                for a, b, c in triangles:
                    parents[root(b)] = root(a)
                    parents[root(c)] = root(a)
                components = {}
                for triangle in triangles:
                    components.setdefault(root(triangle[0]), []).append(triangle)
                for component in components.values():
                    edges = Counter(tuple(sorted((tri[i], tri[(i + 1) % 3])))
                                    for tri in component for i in range(3))
                    if not edges or any(count != 2 for count in edges.values()):
                        continue
                    tree = BVHTree.FromPolygons(vertices, component, all_triangles=True)
                    points = [vertices[i] for i in {i for tri in component for i in tri}]
                    obj_low = Vector(tuple(min(p[i] for p in points) for i in range(3)))
                    obj_high = Vector(tuple(max(p[i] for p in points) for i in range(3)))
                    encloses_bounds = all(all(obj_low[i] - self.epsilon <= p[i] <=
                                              obj_high[i] + self.epsilon for i in range(3))
                                          for p in bounds_corners)
                    enclosing_shell = encloses_bounds and self._inside(
                        tree, self.world @ ((self.low + self.high) * .5))
                    self.solids.append((tree, obj_low, obj_high, enclosing_shell))
            finally:
                obj.to_mesh_clear()
            if index % 4 == 0:
                yield "分析室内几何：{} 个对象".format(index + 1)
        if self.triangles:
            self.bvh = BVHTree.FromPolygons(self.vertices, self.triangles, all_triangles=True)
        if not self.sampling_triangles:
            raise PlanningError("采集范围内没有可检查的渲染网格；请检查范围、Bounds Is Helper 和对象渲染可见性")

    def _inside(self, tree, point):
        # Majority parity is independent of triangle winding and tolerates edge hits.
        votes = 0
        for raw in ((1, .231, .173), (.119, 1, .337), (.271, .153, 1)):
            direction = Vector(raw).normalized()
            origin = point.copy()
            count = 0
            for _ in range(256):
                hit, _, _, _ = tree.ray_cast(origin, direction)
                if hit is None:
                    break
                count += 1
                origin = hit + direction * self.epsilon
            else:
                return True  # Reject ambiguous, excessively layered geometry.
            votes += count % 2
        return votes >= 2

    def valid_position(self, point):
        if not self.contains(point, margin=True):
            return False
        nearest = self.bvh.find_nearest(point) if self.bvh else None
        if nearest and nearest[0] is not None and nearest[3] < max(
                self.props.box_clearance, self.clip_start * 1.1):
            return False
        for tree, low, high, enclosing in self.solids:
            if enclosing:
                if not self._inside(tree, point):
                    return False
                continue
            if all(low[i] <= point[i] <= high[i] for i in range(3)) and self._inside(tree, point):
                return False
        return True

    def _sample_surfaces(self):
        cumulative = []
        total = 0.0
        for *_, area in self.sampling_triangles:
            total += area
            cumulative.append(total)
        rng = random.Random(74621)
        for index in range(self.props.box_surface_samples):
            selection = (index + rng.random()) * total / self.props.box_surface_samples
            a, b, c, _ = self.sampling_triangles[bisect.bisect_left(cumulative, selection)]
            u = math.sqrt(rng.random())
            v = rng.random()
            self.plan.samples.append(a * (1 - u) + b * (u * (1 - v)) + c * (u * v))
        self.plan.observations = [[] for _ in self.plan.samples]
        self.sample_array = np.array([tuple(p) for p in self.plan.samples], dtype=float)

    def _grid(self, stagger=False):
        values = []
        for axis, count in enumerate((self.props.box_grid_x, self.props.box_grid_y, self.props.box_grid_z)):
            low = self.low[axis] + self.margin[axis]
            high = self.high[axis] - self.margin[axis]
            if axis == 2 and count == 1:
                height = self.low.z + self.props.box_camera_height / self.scale.z
                if height < low or height > high:
                    raise PlanningError("Camera Height 必须位于上下边距范围内")
                values.append([height])
            elif count == 1:
                values.append([(low + high) * .5])
            elif stagger:
                values.append([low + (high - low) * (i + .5) / (count - 1) for i in range(count - 1)])
            else:
                values.append([low + (high - low) * i / (count - 1) for i in range(count)])
        return [self.world @ Vector((x, y, z)) for z in values[2] for y in values[1] for x in values[0]]

    def _views(self, position):
        props = self.props
        views = int(props.box_views_per_point)
        if props.box_auto_capture:
            intr = self.intr
            hfov = 2 * math.atan(min(intr["cx"], self.width - intr["cx"]) / intr["fx"])
            vfov = 2 * math.atan(min(intr["cy"], self.height - intr["cy"]) / intr["fy"])
            if min(hfov, vfov) <= 0:
                raise PlanningError("相机偏移超出画面；请调整透视相机参数")
            views = max(views, math.ceil(2 * math.pi / (hfov * (1 - props.box_overlap))))
            intervals = max(2, math.ceil(math.pi / (vfov * (1 - props.box_overlap))))
            pitches = [-math.pi / 2 + math.pi * i / intervals for i in range(intervals + 1)]
        else:
            pitches = [float(props.box_pitch)]
        poses = []
        for pitch in pitches:
            count = 1 if abs(math.cos(pitch)) < 1e-6 else views
            for view in range(count):
                yaw = float(props.box_start_angle) + 2 * math.pi * view / count
                direction = Vector((math.cos(yaw) * math.cos(pitch),
                                    math.sin(yaw) * math.cos(pitch), math.sin(pitch)))
                poses.append(pose_at(position, direction))
        return poses

    def _in_frame(self, pose):
        matrix = np.asarray(pose, dtype=float)
        local = (self.sample_array - matrix[:3, 3]) @ matrix[:3, :3]
        depth = -local[:, 2]
        intr = self.intr
        safe = np.maximum(depth, 1e-12)
        u = intr["fx"] * local[:, 0] / safe + intr["cx"]
        v = intr["cy"] - intr["fy"] * local[:, 1] / safe
        # Leave a small image-border margin for robust observations.
        return ((depth > self.clip_start) & (depth < self.clip_end) &
                (u >= self.width * .01) & (u <= self.width * .99) &
                (v >= self.height * .01) & (v <= self.height * .99))

    def _visible(self, position, sample):
        delta = sample - position
        distance = delta.length
        if distance <= self.clip_start:
            return False
        hit, _, _, depth = self.bvh.ray_cast(position, delta / distance, distance + self.epsilon * 4)
        return hit is not None and abs(depth - distance) <= self.epsilon * 8

    def _independent(self, index, position):
        sample = self.plan.samples[index]
        direction = (position - sample).normalized()
        for old in self.plan.observations[index]:
            if (old - position).length < self.props.box_clearance:
                return False
            dot = max(-1.0, min(1.0, direction.dot((old - sample).normalized())))
            if math.acos(dot) < self.props.box_min_parallax:
                return False
        return True

    def _add_views(self, position, poses):
        mask = np.zeros(len(self.plan.samples), dtype=bool)
        for pose in poses:
            mask |= self._in_frame(pose)
        for index in np.flatnonzero(mask):
            if (self._independent(index, position) and
                    self._visible(position, self.plan.samples[index])):
                self.plan.observations[index].append(position.copy())
        self.plan.poses.extend(poses)
        if not any((old - position).length < self.epsilon for old in self.plan.positions):
            self.plan.positions.append(position.copy())

    def steps(self):
        """Cooperative planning: each yield lets Blender process redraw/cancel events."""
        yield "分析室内采集范围…"
        yield from self._geometry_steps()
        self._sample_surfaces()
        grid = self._grid()
        base_images = 0
        for index, position in enumerate(grid):
            if self.valid_position(position):
                poses = self._views(position)
                if len(self.plan.poses) + len(poses) > self.props.box_max_images:
                    raise PlanningError("基础全方向采集已超过 Image Budget；请增加预算、缩小点阵或降低 Angular Overlap")
                self._add_views(position, poses)
                base_images += len(poses)
            else:
                self.rejected += 1
            yield "室内基础点位 {}/{}：{} 帧".format(index + 1, len(grid), len(self.plan.poses))
        if not self.plan.poses:
            raise PlanningError("没有有效相机点位：请检查边距、碰撞距离、房间范围与点阵密度")
        target = self.props.box_min_observations
        # Add staggered grid positions and local positions near poorly seen surfaces.
        candidates = self._grid(stagger=True) if self.props.box_auto_capture else []
        offsets = [Vector(v).normalized() for v in
                   ((1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0),
                    (0, 0, 1), (0, 0, -1), (1, 1, .5), (-1, -1, .5))]
        local_distance = max(self.props.box_clearance * 3,
                             min((self.high[i] - self.low[i]) * self.scale[i] for i in range(3)) * .18)
        tested = set()
        added = 0
        if self.props.box_auto_capture:
            for index, sample in enumerate(self.plan.samples):
                if len(self.plan.poses) >= self.props.box_max_images:
                    break
                if len(self.plan.observations[index]) >= target:
                    continue
                # Existing positions may see this surface once aimed toward it.
                near = sorted(self.plan.positions + candidates, key=lambda p: (p - sample).length_squared)[:24]
                near += [sample + offset * local_distance for offset in offsets]
                for position in near:
                    if len(self.plan.observations[index]) >= target or len(self.plan.poses) >= self.props.box_max_images:
                        break
                    key = (index, *(round(float(v), 4) for v in position))
                    if key in tested:
                        continue
                    tested.add(key)
                    if (not self._independent(index, position) or
                            not self.valid_position(position) or not self._visible(position, sample)):
                        continue
                    self._add_views(position, [pose_at(position, sample - position)])
                    added += 1
                if index % 8 == 0:
                    yield "自动补拍：{} 帧；表面检查 {}/{}".format(added, index + 1, len(self.plan.samples))
        counts = [len(views) for views in self.plan.observations]
        covered = sum(count >= target for count in counts)
        unseen = sum(count == 0 for count in counts)
        self.plan.report = {
            "schema_version": 1, "capture_mode": "BOX_GRID",
            "coverage_basis": "sampled_opaque_geometry_planned_poses",
            "surface_samples": len(counts), "surface_area": sum(t[3] for t in self.sampling_triangles),
            "required_distinct_positions": target,
            "minimum_parallax_degrees": math.degrees(self.props.box_min_parallax),
            "covered_samples": covered, "unseen_samples": unseen,
            "insufficient_samples": len(counts) - covered,
            "coverage_fraction": covered / len(counts),
            "coverage_target_met": covered == len(counts),
            "base_images": base_images, "supplemental_images": added,
            "planned_images": len(self.plan.poses), "valid_positions": len(self.plan.positions),
            "rejected_grid_positions": self.rejected,
            "image_budget": self.props.box_max_images,
            "budget_exhausted": len(self.plan.poses) >= self.props.box_max_images,
            "intrinsics": self.intr, "resolution": [self.width, self.height],
            "clip_range": [self.clip_start, self.clip_end],
            "limitations": [
                "Area-weighted surface samples estimate coverage, not a reconstruction guarantee.",
                "Visibility treats evaluated mesh surfaces as opaque; transmission, mirrors and volumes are not simulated.",
                "Closed meshes enclosing the entire capture bounds are treated as room shells.",
                "Point density, surface texture, lighting and training settings also affect 3DGS quality.",
                "Use matching viewport/render modifier levels; animated scenes are captured at one fixed frame.",
                "Geometry initialization colors are neutral gray; training optimizes appearance from rendered images.",
            ],
        }
        self.props.box_coverage_status = "覆盖 {:.1%}；未见 {}；不足 {}；{} 帧".format(
            covered / len(counts), unseen, len(counts) - covered, len(self.plan.poses))
        yield self.props.box_coverage_status


def build_plan(context, bounds, camera):
    planner = IndoorPlanner(context, bounds, camera)
    for status in planner.steps():
        context.scene.photara.status = status
    return planner.plan
