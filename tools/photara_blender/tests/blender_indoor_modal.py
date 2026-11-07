"""Run Blender with a window: --factory-startup --python this_file.py.

Exercises actual modal planning/render callbacks and sequence cancellation.
The process exits itself; the result is saved to the PHOTARA_MODAL_RESULT path.
"""
import json
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import bpy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import photara_render

photara_render.register()
bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
bpy.ops.mesh.primitive_cube_add(size=6)
room = bpy.context.object
bpy.ops.object.empty_add(type="CUBE")
bounds = bpy.context.object
bounds.empty_display_size = 3
bpy.ops.object.camera_add()
camera = bpy.context.object
scene = bpy.context.scene
scene.camera = camera
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = 32
scene.render.resolution_percentage = 100
props = scene.photara
props.capture_mode = "BOX_GRID"
props.box_bounds_object = bounds
props.box_grid_x = props.box_grid_y = props.box_grid_z = 1
props.box_camera_height = 3
props.box_auto_capture = False
props.box_surface_samples = 100
props.handover_mode = "OFF"
output = tempfile.TemporaryDirectory(prefix="photara_modal_")
props.output_root = output.name
started = time.monotonic()
state = "preview"
result_path = Path(os.environ.get("PHOTARA_MODAL_RESULT", str(Path(tempfile.gettempdir(), "photara_modal_result.json"))))


def invoke(name):
    window = bpy.context.window_manager.windows[0]
    area = next(area for area in window.screen.areas if area.type == "VIEW_3D")
    with bpy.context.temp_override(window=window, area=area):
        assert "RUNNING_MODAL" in getattr(bpy.ops.photara, name)("INVOKE_DEFAULT")


def tick():
    global state
    try:
        assert time.monotonic() - started < 60, (state, props.status)
        if state == "preview":
            invoke("preview_camera_positions")
            state = "wait_preview"
        elif state == "wait_preview" and not props.is_rendering:
            assert "覆盖" in props.box_coverage_status
            invoke("render_and_export")
            state = "wait_render"
        elif state == "wait_render" and not props.is_rendering:
            report = json.loads(Path(output.name, "capture_coverage.json").read_text())
            assert report["capture_completed"] and report["completed_images"] == 4, props.status
            assert scene.camera == camera
            props.box_grid_x = 2
            invoke("render_and_export")
            state = "cancel_after_frame"
        elif state == "cancel_after_frame" and props.progress >= .125:
            bpy.ops.photara.cancel_render()
            state = "wait_cancel"
        elif state == "wait_cancel" and not props.is_rendering:
            report = json.loads(Path(output.name, "capture_coverage.json").read_text())
            assert not report["capture_completed"], report
            assert 0 < report["completed_images"] < report["planned_images"], report
            assert scene.camera == camera
            result_path.write_text(json.dumps({"ok": True, "status": props.status}))
            output.cleanup()
            bpy.ops.wm.quit_blender()
            return None
    except Exception:
        result_path.write_text(json.dumps({"ok": False, "state": state, "error": traceback.format_exc()}))
        bpy.ops.wm.quit_blender()
        return None
    return .05


bpy.app.timers.register(tick, first_interval=.5)
