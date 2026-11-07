"""Run with blender --background --factory-startup --python this_file.py."""
import json
import math
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch

import bpy
import numpy as np
from mathutils import Vector

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import photara_render
from photara_render import indoor_capture, capture_ops

photara_render.register()


def cube(name, position, dimensions):
    bpy.ops.mesh.primitive_cube_add(size=2, location=position)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dimensions
    return obj


bpy.ops.object.select_all(action="SELECT")
bpy.ops.object.delete(use_global=False)
scene = bpy.context.scene
room = cube("Room shell", (0, 0, 1.5), (8, 6, 3))
obstacle = cube("Closed furniture", (0, 0, 1.2), (1.2, 1.2, 1.2))
bounds = cube("Capture bounds helper", (0, 0, 1.5), (8, 6, 3))
bpy.ops.object.camera_add(location=(0, -2, 1.5))
camera = bpy.context.object
camera.data.lens = 50
scene.camera = camera
scene.render.resolution_x = scene.render.resolution_y = 64
scene.render.resolution_percentage = 100
props = scene.photara
props.capture_mode = "BOX_GRID"
props.box_bounds_object = bounds
props.box_surface_samples = 160
props.box_grid_x = props.box_grid_y = props.box_grid_z = 3
props.handover_mode = "OFF"
bpy.context.view_layer.update()

planner = indoor_capture.IndoorPlanner(bpy.context, bounds, camera)
assert not planner.contains(Vector((5, 0, 1.5)))
for status in planner.steps():
    pass
plan = planner.plan
assert plan.report["rejected_grid_positions"] > 0
assert 0 <= plan.report["coverage_fraction"] <= 1
assert plan.report["planned_images"] <= props.box_max_images
assert not planner.valid_position(Vector((0, 0, 1.2))), "Furniture interior accepted"
assert not planner.valid_position(Vector((0, 0, .05))), "Floor clearance ignored"
assert planner.valid_position(Vector((2, 0, 1.2))), "Room interior rejected as solid"
assert all(planner.valid_position(pose.translation) for pose in plan.poses)
assert any((pose.to_3x3() @ Vector((0, 0, -1))).z > .99 for pose in plan.poses)
assert any((pose.to_3x3() @ Vector((0, 0, -1))).z < -.99 for pose in plan.poses)
assert camera.data.lens == 50

# Sparse base coverage must trigger genuine supplemental camera positions.
props.box_grid_x, props.box_grid_y, props.box_grid_z = 2, 1, 1
sparse = indoor_capture.build_plan(bpy.context, bounds, camera)
assert sparse.report["supplemental_images"] > 0, sparse.report
assert sparse.report["valid_positions"] > 2, sparse.report
props.box_grid_x = props.box_grid_y = props.box_grid_z = 3

# Independent coverage audit: no sample can gain multiple observations from a single position.
for sample, observations in zip(plan.samples, plan.observations):
    for i, position in enumerate(observations):
        assert planner._visible(position, sample)
        for other in observations[:i]:
            assert (position - other).length >= props.box_clearance - 1e-5
            assert (position - sample).angle(other - sample) >= props.box_min_parallax - 1e-4

capture_ops._show_camera_preview(bpy.context, plan.poses, camera, plan)
collection = capture_ops._get_preview_collection(scene)
preview_camera = next(obj for obj in collection.objects if obj.type == "CAMERA")
assert preview_camera.data.lens == props.box_capture_lens
assert collection.hide_render and all(obj.hide_render for obj in collection.objects)
capture_ops._clear_preview_objects(collection)
assert not collection.objects

# A restrictive budget fails instead of silently truncating base sampling.
props.box_max_images = 50
try:
    indoor_capture.build_plan(bpy.context, bounds, camera)
except indoor_capture.PlanningError:
    pass
else:
    raise AssertionError("Budget overflow accepted")
props.box_max_images = 4000

# Rotated/scaled bounds retain their local grid and exclude positions outside the box.
bounds.rotation_euler.z = math.radians(20)
bpy.context.view_layer.update()
rotated = indoor_capture.IndoorPlanner(bpy.context, bounds, camera)
assert all(rotated.contains(point, margin=True) for point in rotated._grid())
bounds.rotation_euler.z = 0
bpy.context.view_layer.update()

# Small real image sequence: check image/pose correspondence, helper restoration,
# fixed source frame and no training on cancellation. Workbench needs no GPU setup.
props.box_auto_capture = False
props.box_grid_x, props.box_grid_y, props.box_grid_z = 2, 1, 1
props.box_views_per_point = "4"
props.box_surface_samples = 100
scene.render.engine = "BLENDER_WORKBENCH"
scene.render.resolution_x = scene.render.resolution_y = 32
with tempfile.TemporaryDirectory(prefix="photara_indoor_") as directory:
    props.output_root = directory
    job = capture_ops.RenderJob(bpy.context)
    job.prepare()
    assert bounds.hide_render
    total = len(job.frames)
    while not job.step():
        pass
    job.finish()
    assert not bounds.hide_render and scene.camera == camera
    assert len(list(Path(directory, "images").glob("*.png"))) == total
    transforms = json.loads(Path(directory, "transforms.json").read_text())
    assert len(transforms["frames"]) == total
    report = json.loads(Path(directory, "capture_coverage.json").read_text())
    assert report["capture_completed"] and report["completed_images"] == total
    assert report["initial_surface_points"] > 0
    assert transforms["ply_file_path"] == "initial_points.ply"
    assert Path(directory, "initial_points.ply").exists()
    exported, _, missing = capture_ops._record_poses_blocking(bpy.context)
    assert not missing and len(exported) == total
    np.testing.assert_allclose(exported[0]["matrix_world"], job.records[0]["matrix_world"], atol=1e-6)

    assert exported[0]["intrinsics"] == job.records[0]["intrinsics"]
    job = capture_ops.RenderJob(bpy.context)
    job.prepare()
    job.step()
    with patch.object(capture_ops.photara_bridge, "submit_async") as submit:
        job.finish(cancelled=True, deliver=True)
        submit.assert_not_called()
    partial = json.loads(Path(directory, "capture_coverage.json").read_text())
    assert not partial["capture_completed"] and partial["completed_images"] == 1
    assert not bounds.hide_render and scene.camera == camera

    # Automatic optics must agree between actual rendering and poses-only export.
    props.box_auto_capture = True
    job = capture_ops.RenderJob(bpy.context)
    job.prepare()
    assert job.camera.data.lens == props.box_capture_lens
    job.step()
    job.finish(cancelled=True)
    exported, _, _ = capture_ops._record_poses_blocking(bpy.context)
    assert exported[0]["intrinsics"] == job.records[0]["intrinsics"]
    np.testing.assert_allclose(exported[0]["matrix_world"], job.records[0]["matrix_world"], atol=1e-6)

# Joined geometry must still distinguish room shells from solid furniture.
bpy.ops.object.select_all(action="DESELECT")
room.select_set(True)
obstacle.select_set(True)
bpy.context.view_layer.objects.active = room
bpy.ops.object.join()
bpy.context.view_layer.update()
joined = indoor_capture.IndoorPlanner(bpy.context, bounds, camera)
for _ in joined._geometry_steps():
    pass
assert not joined.valid_position(Vector((0, 0, 1.2))), "Joined furniture interior accepted"
assert joined.valid_position(Vector((2, 0, 1.2))), "Joined room shell rejected"

# Strict coverage must fail before modifying the scene or creating any output images.
props.box_auto_capture = False
props.box_grid_x, props.box_grid_y, props.box_grid_z = 2, 1, 1
props.box_block_incomplete = True
strict_job = capture_ops.RenderJob(bpy.context)
try:
    strict_job.prepare()
except capture_ops.PhotaraExportError:
    pass
else:
    raise AssertionError("Strict incomplete coverage accepted")
strict_job.finish(cancelled=True)
assert not bounds.hide_render and scene.camera == camera
props.box_block_incomplete = False

# Render cancellation handlers must never record a cancelled or absent frame.
job = capture_ops.RenderJob(bpy.context)
job._prepared = True
job._async_pending_record = {"filename": "absent.png"}
job._register_handlers()
job._on_render_cancel(scene)
assert job.step_async() is False and not job.records and job._cancel_requested
job._remove_handlers()
assert job._on_render_cancel not in bpy.app.handlers.render_cancel

print("INDOOR_CAPTURE_OK", json.dumps(plan.report, ensure_ascii=True))
