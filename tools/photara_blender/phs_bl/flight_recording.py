"""Record viewport navigation as a reusable, independent camera animation."""

import math
import time

import bpy
from mathutils import Matrix, Quaternion, Vector
from bpy.props import BoolProperty
from bpy.app.handlers import persistent


_session = None
_MAX_POINTS = 10000


def _pose(session):
    if session.get('current_pose') is not None:
        return session['current_pose'].copy()
    space = session['space']
    region = session['region']
    if region.view_perspective == 'CAMERA':
        camera = space.camera or session['scene'].camera
        if camera is None or camera.data.type != 'PERSP':
            raise ValueError('录制需要透视相机或透视视图')
        return camera.evaluated_get(bpy.context.evaluated_depsgraph_get()).matrix_world.copy()
    if region.view_perspective != 'PERSP':
        raise ValueError('请切换到透视视图后录制')
    rotation = region.view_rotation
    position = region.view_location + rotation @ Vector((0, 0, region.view_distance))
    return Matrix.LocRotScale(position, rotation, Vector((1, 1, 1)))


def _sample(session, endpoint=False):
    pose = _pose(session)
    poses = session['poses']
    if len(poses) >= _MAX_POINTS:
        return
    if poses:
        previous = poses[-1]
        distance = (pose.translation - previous.translation).length
        angle = previous.to_quaternion().rotation_difference(pose.to_quaternion()).angle
        angle = min(angle, 2 * math.pi - angle)
        if endpoint:
            if distance < 1e-6 and angle < 1e-6:
                return
        elif distance < session['distance'] and angle < session['angle']:
            return
    poses.append(pose)
    session['scene'].photara.flight_count = len(poses)


def _cleanup():
    global _session
    if bpy.app.timers.is_registered(_tick):
        bpy.app.timers.unregister(_tick)
    if _session:
        controller = _session.get('controller')
        if controller is not None:
            controller._release()
        temp = _session.get('navigation_camera')
        try:
            _session['scene'].photara.flight_recording = False
            if temp is not None:
                space = _session['space']
                if space.camera == temp:
                    space.camera = _session.get('original_camera')
                    space.use_local_camera = _session.get('original_local_camera', False)
                bpy.data.objects.remove(temp, do_unlink=True)
        except (ReferenceError, RuntimeError):
            pass
    _session = None


def _tick():
    try:
        session = _session
        if session is None:
            return None
        if (session.get('window', bpy.context.window).scene != session['scene'] or
                not any(area == session['area'] for area in session['screen'].areas) or
                session['area'].type != 'VIEW_3D' or session['area'].spaces.active != session['space']):
            raise ValueError('录制视图或场景已关闭，录制已取消')
        _sample(session)
        session['area'].tag_redraw()
        if len(session['poses']) >= _MAX_POINTS:
            _finish()
            return None
        return session['interval']
    except Exception as exc:
        try:
            if _session:
                _session['scene'].photara.status = '录制已取消：{}'.format(exc)
        except ReferenceError:
            pass
        shutdown()
        return None


def _finish():
    session = _session
    scene = session['scene']
    props = scene.photara
    camera = None
    data = session['optics']
    old_settings = (props.camera, props.capture_mode, props.use_scene_frame_range,
                    props.frame_start, props.frame_end, props.frame_step)
    try:
        _sample(session, endpoint=True)
        camera = bpy.data.objects.new('Photara Recorded Flight', data)
        scene.collection.objects.link(camera)
        camera.rotation_mode = 'QUATERNION'
        previous = None
        for frame, pose in enumerate(session['poses'], 1):
            camera.location = pose.translation
            rotation = pose.to_quaternion()
            if previous is not None and previous.dot(rotation) < 0:
                rotation.negate()
            camera.rotation_quaternion = rotation
            previous = rotation.copy()
            camera.keyframe_insert(data_path='location', frame=frame)
            camera.keyframe_insert(data_path='rotation_quaternion', frame=frame)
        props.camera = camera
        props.capture_mode = 'ANIMATION'
        props.use_scene_frame_range = False
        props.frame_start = 1
        props.frame_end = len(session['poses'])
        props.frame_step = 1
        camera['photara_recorded_flight'] = True
        camera['photara_point_count'] = len(session['poses'])
        props.status = '已生成自由飞行相机：{} 个点位，可直接渲染导出'.format(len(session['poses']))
        if session.get('navigation_camera') is not None:
            session['space'].camera = camera
    except Exception:
        (props.camera, props.capture_mode, props.use_scene_frame_range,
         props.frame_start, props.frame_end, props.frame_step) = old_settings
        if camera is not None:
            bpy.data.objects.remove(camera, do_unlink=True)
        raise
    finally:
        _cleanup()
        if data.users == 0:
            bpy.data.cameras.remove(data)
    # A display failure must never discard a successfully recorded animation.
    try:
        from .capture_ops import _show_camera_preview
        window = session.get('window')
        if window is not None:
            with bpy.context.temp_override(window=window, area=session['area'], region=session['window_region']):
                _show_camera_preview(bpy.context, session['poses'], camera)
        else:
            _show_camera_preview(bpy.context, session['poses'], camera)
        props.status = '已生成 {} 个录制点位和相机预览'.format(len(session['poses']))
    except Exception as exc:
        props.status = '路径已保存（{} 点）；预览失败，可重新生成：{}'.format(len(session['poses']), exc)


def shutdown():
    data = _session['optics'] if _session else None
    if _session and _session.get('view_backup'):
        try:
            space, region = _session['space'], _session['region']
            backup = _session['view_backup']
            space.camera = _session['original_camera']
            space.use_local_camera = _session.get('original_local_camera', False)
            region.view_rotation, region.view_location, region.view_distance, region.view_perspective = backup
        except (ReferenceError, RuntimeError):
            pass
    _cleanup()
    if data is not None and data.users == 0:
        bpy.data.cameras.remove(data)


@persistent
def before_load(_):
    shutdown()


class PHOTARA_OT_first_person(bpy.types.Operator):
    """Move from the exact recorded pose, without native navigation reinitialization."""
    bl_idname = 'photara.first_person'
    bl_label = 'Photara First Person'
    bl_options = {'INTERNAL', 'BLOCKING', 'GRAB_CURSOR'}

    def invoke(self, context, event):
        if _session is None:
            return {'CANCELLED'}
        self.session = _session
        self.position = _session['poses'][0].translation.copy()
        self.rotation = _session['poses'][0].to_quaternion()
        self.keys = set()
        self.mouse = None
        self.last_time = time.perf_counter()
        self.window_manager = context.window_manager
        self.timer = context.window_manager.event_timer_add(0.02, window=context.window)
        self.session['controller'] = self
        self.session['current_pose'] = self.session['poses'][0].copy()
        self.focused = True
        context.window_manager.modal_handler_add(self)
        return {'RUNNING_MODAL'}

    def _release(self):
        if self.timer is not None:
            try:
                self.window_manager.event_timer_remove(self.timer)
            except (ReferenceError, RuntimeError):
                pass
            self.timer = None
        self.keys.clear()

    def _end(self, context):
        self._release()
        return {'FINISHED'}

    def modal(self, context, event):
        try:
            return self._modal(context, event)
        except Exception as exc:
            self._release()
            shutdown()
            self.report({'ERROR'}, '录制控制已安全退出：{}'.format(exc))
            return {'CANCELLED'}

    def _modal(self, context, event):
        if _session is not self.session:
            return self._end(context)
        if event.type in {'RET', 'NUMPAD_ENTER'} and event.value == 'PRESS':
            self._release()
            _finish()
            return self._end(context)
        if event.type in {'ESC', 'RIGHTMOUSE'} and event.value == 'PRESS':
            shutdown()
            context.scene.photara.status = '已取消录制并恢复起始视图'
            return self._end(context)
        if event.type == 'WINDOW_DEACTIVATE':
            self.keys.clear()
            self.mouse = None
            self.focused = False
        if event.type == 'WINDOW_ACTIVATE':
            self.focused = True
            self.last_time = time.perf_counter()
        if not self.focused:
            return {'RUNNING_MODAL'}
        if event.type in {'W', 'A', 'S', 'D', 'Q', 'E'}:
            if event.value == 'PRESS':
                self.keys.add(event.type)
            elif event.value == 'RELEASE':
                self.keys.discard(event.type)
        if event.type in {'MOUSEMOVE', 'INBETWEEN_MOUSEMOVE'}:
            current = (event.mouse_x, event.mouse_y)
            if self.mouse is not None:
                dx, dy = current[0] - self.mouse[0], current[1] - self.mouse[1]
                sensitivity = context.preferences.inputs.walk_navigation.mouse_speed * 0.002
                self.rotation = (Quaternion((0, 0, 1), -dx * sensitivity)
                                 @ self.rotation @ Quaternion((1, 0, 0), dy * sensitivity))
                self.rotation.normalize()
                self._update()
            self.mouse = current
        # Blender's Python Event exposes the timer event type, not a timer handle.
        # Elapsed time keeps motion independent of other window timers.
        if event.type == 'TIMER':
            now = time.perf_counter()
            elapsed = min(now - self.last_time, 0.1)
            self.last_time = now
            direction = Vector((int('D' in self.keys) - int('A' in self.keys),
                                int('E' in self.keys) - int('Q' in self.keys),
                                int('S' in self.keys) - int('W' in self.keys)))
            if direction.length_squared:
                speed = context.preferences.inputs.walk_navigation.walk_speed
                speed /= max(self.session['scene'].unit_settings.scale_length, 1e-6)
                speed *= 3 if event.shift else 1
                if self.session.get('control', 'FLY') == 'WALK':
                    forward = self.rotation @ Vector((0, 0, -1))
                    forward.z = 0
                    right = self.rotation @ Vector((1, 0, 0))
                    right.z = 0
                    movement = right.normalized() * direction.x - forward.normalized() * direction.z
                    movement.z = direction.y
                else:
                    movement = self.rotation @ direction
                if movement.length_squared:
                    self.position += movement.normalized() * speed * elapsed
                self._update()
        return {'RUNNING_MODAL'}

    def _update(self):
        self.session['current_pose'] = Matrix.LocRotScale(self.position, self.rotation, Vector((1, 1, 1)))
        camera = self.session.get('navigation_camera')
        if camera is not None:
            camera.matrix_world = self.session['current_pose']
            self.session['area'].tag_redraw()
            return
        region = self.session['region']
        region.view_perspective = 'PERSP'
        region.view_distance = 0
        region.view_rotation = self.rotation
        region.view_location = self.position
        region.update()
        self.session['area'].tag_redraw()


class PHOTARA_OT_start_flight(bpy.types.Operator):
    bl_idname = 'photara.start_flight'
    bl_label = 'Start Flight Recording'
    bl_description = '进入所选第一人称/飞行控制并自动记录点位，停止后生成独立相机动画'

    @classmethod
    def poll(cls, context):
        return (context.area is not None and context.area.type == 'VIEW_3D'
                and _session is None and not context.scene.photara.is_rendering)

    def execute(self, context):
        global _session
        props = context.scene.photara
        space = context.space_data
        navigation_region = next((r for r in context.area.regions if r.type == 'WINDOW'), None)
        if props.flight_control != 'MANUAL' and (navigation_region is None or space.region_quadviews):
            self.report({'ERROR'}, '请在普通单视图中启动第一人称录制')
            return {'CANCELLED'}
        session = dict(scene=context.scene, screen=context.screen, area=context.area, window=context.window,
                       window_region=navigation_region, original_camera=space.camera,
                       original_local_camera=space.use_local_camera,
                       control=props.flight_control,
                       space=space, region=space.region_3d, poses=[],
                       distance=props.flight_distance, angle=props.flight_angle,
                       interval=props.flight_interval)
        try:
            region = space.region_3d
            session['view_backup'] = (region.view_rotation.copy(), region.view_location.copy(),
                                      region.view_distance, region.view_perspective)
            _sample(session)
            source = (space.camera or context.scene.camera) if space.region_3d.view_perspective == 'CAMERA' else (props.camera or context.scene.camera)
            data = source.data.copy() if source and source.data.type == 'PERSP' else bpy.data.cameras.new('Photara Flight Optics')
            data.animation_data_clear()
            if space.region_3d.view_perspective != 'CAMERA':
                data.lens = space.lens
            session['optics'] = data
            _session = session
            if props.flight_control != 'MANUAL':
                # Navigate an independent view, so camera lock cannot alter the source animation.
                region = space.region_3d
                if region.view_perspective == 'CAMERA':
                    temp = bpy.data.objects.new('Photara Flight Navigation', data)
                    temp.hide_render = True
                    context.scene.collection.objects.link(temp)
                    temp.matrix_world = session['poses'][0]
                    session['navigation_camera'] = temp
                    space.use_local_camera = True
                    space.camera = temp
                with context.temp_override(region=navigation_region):
                    # Cursor grabbing is handled by the modal operator, never warp here.
                    result = bpy.ops.photara.first_person('INVOKE_DEFAULT')
                    if 'RUNNING_MODAL' not in result:
                        raise ValueError('无法启动导航，请使用 Manual View Navigation')
            props.flight_recording = True
            props.status = '录制中：自由移动视图，完成后点击 Stop & Generate'
            bpy.app.timers.register(_tick, first_interval=session['interval'])
        except Exception as exc:
            shutdown()
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}


class PHOTARA_OT_stop_flight(bpy.types.Operator):
    bl_idname = 'photara.stop_flight'
    bl_label = 'Stop & Generate'
    bl_options = {'REGISTER', 'UNDO'}

    discard: BoolProperty(default=False, options={'HIDDEN'})

    @classmethod
    def poll(cls, context):
        return _session is not None

    def execute(self, context):
        try:
            if self.discard:
                shutdown()
                context.scene.photara.status = '已取消自由飞行录制'
            else:
                _finish()
        except Exception as exc:
            self.report({'ERROR'}, str(exc))
            return {'CANCELLED'}
        return {'FINISHED'}
