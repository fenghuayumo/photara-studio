bl_info = {
    "name": "Photara Render & Pose Export",
    "author": "Photara Team",
    "version": (1, 1, 0),
    "blender": (3, 6, 0),
    "location": "View3D > Sidebar (N) > Photara",
    "description": "Render animation to image sequence, export per-frame camera poses "
    "(COLMAP / transforms.json), and hand the dataset to Photara Studio for 3DGS training",
    "doc_url": "",
    "category": "Render",
}

if "bpy" in locals():
    import importlib

    _modules = (
        "props",
        "camera_utils",
        "colmap_export",
        "photara_bridge",
        "indoor_capture",
        "capture_ops",
        "flight_recording",
        "ui",
    )
    for _m in _modules:
        if _m in globals():
            importlib.reload(globals()[_m])
        else:
            globals()[_m] = importlib.import_module('.' + _m, __name__)
    # Keep the compatibility alias used by older scripts.
    from . import capture_ops
    render_ops = capture_ops
else:
    try:
        import bpy  # noqa: F401

        from . import props, camera_utils, colmap_export, photara_bridge, indoor_capture, ui
        from . import capture_ops, flight_recording
        render_ops = capture_ops
    except ImportError:
        # Outside Blender (unit tests): the pure modules still import fine.
        from . import camera_utils, colmap_export, photara_bridge


# Older scripts importing photara_render.render_ops continue to use the current operators.
if globals().get("capture_ops") is not None:
    import sys

    sys.modules[__name__ + ".render_ops"] = capture_ops


def _classes():
    return (
        props.PhotaraProps,
        flight_recording.PHOTARA_OT_first_person,
        flight_recording.PHOTARA_OT_start_flight,
        flight_recording.PHOTARA_OT_stop_flight,
        render_ops.PHOTARA_OT_render_and_export,
        render_ops.PHOTARA_OT_cancel_render,
        render_ops.PHOTARA_OT_preview_camera_positions,
        render_ops.PHOTARA_OT_clear_camera_preview,
        render_ops.PHOTARA_OT_focus_camera_preview,
        render_ops.PHOTARA_OT_export_poses,
        render_ops.PHOTARA_OT_open_dataset_folder,
        render_ops.PHOTARA_OT_test_studio_connection,
        ui.PHOTARA_PT_main,
        ui.PHOTARA_PT_output,
        ui.PHOTARA_PT_frames,
        ui.PHOTARA_PT_photara,
    )


def register():
    if globals().get("props") is None:
        raise RuntimeError("photara_render 只能在 Blender 内启用")
    import bpy

    for cls in _classes():
        bpy.utils.register_class(cls)
    bpy.types.Scene.photara = bpy.props.PointerProperty(type=props.PhotaraProps)
    bpy.app.handlers.load_pre.append(flight_recording.before_load)


def unregister():
    if globals().get("props") is None:
        raise RuntimeError("photara_render 只能在 Blender 内启用")
    import bpy

    flight_recording.shutdown()
    if flight_recording.before_load in bpy.app.handlers.load_pre:
        bpy.app.handlers.load_pre.remove(flight_recording.before_load)
    del bpy.types.Scene.photara
    for cls in reversed(_classes()):
        bpy.utils.unregister_class(cls)
