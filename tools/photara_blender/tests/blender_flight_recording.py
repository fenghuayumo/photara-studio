"""Run with blender --background --factory-startup --python this_file.py."""
import math
import time
import sys
from pathlib import Path
from unittest.mock import patch
from types import SimpleNamespace

import bpy
from mathutils import Matrix

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import photara_render
from photara_render import flight_recording as flight

photara_render.register()
scene = bpy.context.scene
props = scene.photara
source = scene.camera
source_matrix = source.matrix_world.copy()
data = source.data.copy()
session = dict(scene=scene, poses=[], distance=0.25, angle=math.radians(10), optics=data)
pose = Matrix.Identity(4)
with patch.object(flight, '_pose', side_effect=lambda _: pose.copy()):
    flight._sample(session)
    flight._sample(session)
    assert len(session['poses']) == 1, 'Stationary view duplicated'
    pose.translation.x = 0.1
    flight._sample(session)
    assert len(session['poses']) == 1, 'Distance threshold ignored'
    pose.translation.x = 0.3
    flight._sample(session)
    assert len(session['poses']) == 2
    pose = pose @ Matrix.Rotation(math.radians(15), 4, 'Z')
    flight._sample(session)
    assert len(session['poses']) == 3, 'Rotation-only sample missed'
    pose.translation.x += 0.05
    flight._session = session
    props.flight_recording = True
    flight._finish()

recorded = props.camera
assert recorded != source
assert not props.flight_recording and flight._session is None
assert props.capture_mode == 'ANIMATION'
assert (props.frame_start, props.frame_end, props.frame_step) == (1, 4, 1)
assert not props.use_scene_frame_range
assert source.matrix_world == source_matrix
from photara_render import capture_ops
preview = capture_ops._get_preview_collection(scene)
assert len(preview.objects) == 4
assert preview.hide_render
assert all(obj.show_in_front for obj in preview.objects)
area = next((a for a in bpy.context.screen.areas if a.type == 'VIEW_3D'), None)
if area is not None:
    with bpy.context.temp_override(area=area):
        space = bpy.context.space_data
        space.overlay.show_extras = False
        space.show_object_viewport_camera = False
        preview.hide_viewport = True
        capture_ops._reveal_camera_preview(bpy.context, preview)
        assert space.overlay.show_extras
        assert space.show_object_viewport_camera
        assert not preview.hide_viewport
assert bpy.ops.photara.clear_camera_preview() == {'FINISHED'}
assert len(preview.objects) == 0
saved_frame = scene.frame_current
assert bpy.ops.photara.preview_camera_positions() == {'FINISHED'}
assert scene.frame_current == saved_frame
assert len(preview.objects) == 4
for obj, expected in zip(preview.objects, session['poses']):
    assert (obj.matrix_world.translation - expected.translation).length < 1e-5
for frame, expected in enumerate(session['poses'], 1):
    scene.frame_set(frame)
    actual = recorded.matrix_world
    assert (actual.translation - expected.translation).length < 1e-5
    assert actual.to_quaternion().rotation_difference(expected.to_quaternion()).angle < 1e-4

discarded = source.data.copy()
discarded_name = discarded.name
flight._session = dict(scene=scene, optics=discarded)
props.flight_recording = True
flight.shutdown()
assert not props.flight_recording
assert discarded_name not in bpy.data.cameras
# Sidebar-to-viewport mouse movement must only establish the input baseline.
flight._session = session
controller = SimpleNamespace(session=session, mouse=None, keys=set(),
                             rotation=session['poses'][0].to_quaternion(), focused=True)
updates = []
controller._update = lambda: updates.append(True)
before = controller.rotation.copy()
event = SimpleNamespace(type='MOUSEMOVE', value='NOTHING', mouse_x=900, mouse_y=400)
assert flight.PHOTARA_OT_first_person._modal(controller, bpy.context, event) == {'RUNNING_MODAL'}
assert not updates
assert controller.rotation.rotation_difference(before).angle < 1e-5
event.mouse_x += 10
flight.PHOTARA_OT_first_person._modal(controller, bpy.context, event)
assert updates and controller.rotation.rotation_difference(before).angle > 0.001
controller.position = session['poses'][0].translation.copy()
controller.last_time = time.perf_counter() - 0.02
controller.keys = {'W'}
position_before = controller.position.copy()
# Actual bpy.types.Event has no `timer` attribute.
timer_event = SimpleNamespace(type='TIMER', value='NOTHING', shift=False)
flight.PHOTARA_OT_first_person._modal(controller, bpy.context, timer_event)
assert (controller.position - position_before).length > 0
controller.keys.clear()
position_before = controller.position.copy()
flight.PHOTARA_OT_first_person._modal(controller, bpy.context, timer_event)
assert controller.position == position_before
deactivate = SimpleNamespace(type='WINDOW_DEACTIVATE', value='NOTHING')
controller.keys.add('W')
flight.PHOTARA_OT_first_person._modal(controller, bpy.context, deactivate)
assert not controller.keys and controller.mouse is None and not controller.focused
position_before = controller.position.copy()
flight.PHOTARA_OT_first_person._modal(controller, bpy.context, timer_event)
assert controller.position == position_before
flight._session = None

# View validation must work with Blender RNA collections (not Python `in`).
if area is not None:
    tick_session = dict(scene=scene, window=bpy.context.window, screen=bpy.context.screen,
                        area=area, space=area.spaces.active, poses=[Matrix.Identity(4)],
                        distance=0.25, angle=0.1, interval=0.1, optics=source.data.copy())
    flight._session = tick_session
    props.flight_recording = True
    with patch.object(flight, '_pose', return_value=Matrix.Identity(4)):
        assert flight._tick() == 0.1
        assert props.flight_recording
    flight.shutdown()

# Visualization failure must preserve the generated animation.
failure_session = dict(scene=scene, poses=[Matrix.Identity(4)], distance=0.25,
                       angle=0.1, optics=source.data.copy())
flight._session = failure_session
props.flight_recording = True
with patch.object(flight, '_pose', return_value=Matrix.Identity(4)), patch.object(
        capture_ops, '_show_camera_preview', side_effect=RuntimeError('QA preview failure')):
    flight._finish()
assert props.camera is not None and props.camera.get('photara_point_count') == 1
assert props.camera.animation_data is not None
assert 'QA preview failure' in props.status
assert not props.flight_recording and flight._session is None
photara_render.unregister()
print('Flight recording integration checks passed')
