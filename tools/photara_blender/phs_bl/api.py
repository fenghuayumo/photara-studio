"""Convenience API for scripted / headless use.

Example (inside Blender or ``blender -b scene.blend --python``)::

    import photara_render.api as photara
    photara.render_and_export(
        output_root="D:/data/scene01",
        camera_name="Camera",
        frame_start=1, frame_end=120,
        image_format="PNG",
        handover_mode="CLI",
    )
"""

import bpy


def apply_settings(context=None, **kwargs):
    context = context or bpy.context
    props = context.scene.photara
    for key, value in kwargs.items():
        if key == "camera_name":
            camera = bpy.data.objects.get(value)
            if camera is None or camera.type != "CAMERA":
                raise ValueError("相机不存在：{}".format(value))
            props.camera = camera
        elif hasattr(props, key):
            setattr(props, key, value)
        else:
            raise KeyError("未知设置：{}".format(key))
    if "frame_start" in kwargs or "frame_end" in kwargs:
        props.use_scene_frame_range = False
    return props


def render_and_export(context=None, **kwargs):
    apply_settings(context, **kwargs)
    return bpy.ops.photara.render_and_export("EXEC_DEFAULT")


def export_poses(context=None, **kwargs):
    apply_settings(context, **kwargs)
    return bpy.ops.photara.export_poses("EXEC_DEFAULT")
