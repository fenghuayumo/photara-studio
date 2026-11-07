"""Write the exported dataset: COLMAP sparse model, transforms.json, manifest."""

import json
import os
import time

from . import camera_utils


def _fmt(v):
    return "{:.10f}".format(float(v))


def write_dataset(
    dataset_dir,
    frames,
    intrinsics,
    name="blender_scene",
    write_colmap=True,
    write_transforms=True,
    meta=None,
    initial_points=None,
):
    """Write the dataset folder.

    ``frames``: list of dicts {filename, matrix_world(4x4 nested list), width, height}
    ``intrinsics``: dict {fx, fy, cx, cy} (constant across frames for this addon).
    The COLMAP sparse model is written as text, which COLMAP / 3DGS loaders accept.
    """
    if not frames or not (write_colmap or write_transforms):
        raise ValueError("Enable a camera export format and supply camera frames")
    cameras = {}
    camera_ids = []
    for fr in frames:
        intr = fr.get("intrinsics", intrinsics)
        key = (fr["width"], fr["height"], *(float(intr[k]) for k in ("fx", "fy", "cx", "cy")))
        if key not in cameras:
            cameras[key] = len(cameras) + 1
        camera_ids.append(cameras[key])
    images_dir = os.path.join(dataset_dir, "images")
    os.makedirs(images_dir, exist_ok=True)

    written = []
    if write_colmap:
        sparse_dir = os.path.join(dataset_dir, "sparse", "0")
        os.makedirs(sparse_dir, exist_ok=True)

        w, h = frames[0]["width"], frames[0]["height"]
        with open(os.path.join(sparse_dir, "cameras.txt"), "w", encoding="utf-8") as f:
            f.write(
                "# Camera list with one line of data per camera:\n"
                "#   CAMERA_ID, MODEL, WIDTH, HEIGHT, PARAMS[]\n"
            )
            for (w, h, fx, fy, cx, cy), camera_id in cameras.items():
                f.write("{} PINHOLE {} {} {} {} {} {}\n".format(
                    camera_id, w, h, _fmt(fx), _fmt(fy), _fmt(cx), _fmt(cy)))

        with open(os.path.join(sparse_dir, "images.txt"), "w", encoding="utf-8") as f:
            f.write(
                "# Image list with two lines per image:\n"
                "#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n"
                "#   POINTS2D[] as (X, Y, POINT3D_ID)\n"
            )
            for i, fr in enumerate(frames, start=1):
                q, t = camera_utils.matrix_world_to_colmap(fr["matrix_world"])
                f.write(
                    "{} {} {} {} {} {} {} {} {} {}\n\n".format(
                        i,
                        _fmt(q[0]),
                        _fmt(q[1]),
                        _fmt(q[2]),
                        _fmt(q[3]),
                        _fmt(t[0]),
                        _fmt(t[1]),
                        _fmt(t[2]),
                        camera_ids[i - 1],
                        fr["filename"],
                    )
                )

        with open(os.path.join(sparse_dir, "points3D.txt"), "w", encoding="utf-8") as f:
            f.write("# POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
            for point_id, point in enumerate(initial_points or [], 1):
                f.write("{} {} {} {} {} {} {} 0\n".format(
                    point_id, *(_fmt(v) for v in point["position"]), *point["color"]))
        written.append("sparse/0/*.txt")

    if initial_points:
        with open(os.path.join(dataset_dir, "initial_points.ply"), "w", encoding="ascii") as f:
            f.write("ply\nformat ascii 1.0\nelement vertex {}\n".format(len(initial_points)))
            f.write("property float x\nproperty float y\nproperty float z\n")
            f.write("property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
            for point in initial_points:
                f.write("{} {} {} {} {} {}\n".format(
                    *(_fmt(v) for v in point["position"]), *point["color"]))
        written.append("initial_points.ply")

    if write_transforms:
        w, h = frames[0]["width"], frames[0]["height"]
        data = {
            # Nerfstudio / Blender explicit-intrinsics style, byte-compatible
            # with photara-studio's own save_nerfstudio_transforms output.
            "fl_x": float(intrinsics["fx"]),
            "fl_y": float(intrinsics["fy"]),
            "cx": float(intrinsics["cx"]),
            "cy": float(intrinsics["cy"]),
            "w": int(w),
            "h": int(h),
            "k1": 0.0,
            "k2": 0.0,
            "p1": 0.0,
            "p2": 0.0,
            "camera_model": "OPENCV",
            # camera_angle_* kept for tools that still use the older
            # angle-based Nerfstudio variant.
            "camera_angle_x": camera_utils.fov_x_from_fx(intrinsics["fx"], w),
            "camera_angle_y": camera_utils.fov_x_from_fx(intrinsics["fy"], h),
            "dataset_name": name,
            "frames": [],
        }
        if initial_points:
            data["ply_file_path"] = "initial_points.ply"
        for i, fr in enumerate(frames, start=1):
            data["frames"].append(
                {
                    # transforms.json uses the OpenGL camera convention, which is
                    # identical to Blender's matrix_world (row-major nested list).
                    "file_path": "images/{}".format(fr["filename"]),
                    "transform_matrix": [[float(v) for v in row] for row in fr["matrix_world"]],
                    "colmap_image_id": i,
                    "w": int(fr["width"]), "h": int(fr["height"]),
                    **{dst: float(fr.get("intrinsics", intrinsics)[src])
                       for dst, src in (("fl_x", "fx"), ("fl_y", "fy"), ("cx", "cx"), ("cy", "cy"))},
                }
            )
        with open(os.path.join(dataset_dir, "transforms.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        written.append("transforms.json")

    # Photara manifest: everything the Studio side needs at a glance.
    manifest = {
        "format_version": 1,
        "source": "photara-blender",
        "name": name,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "dataset_path": dataset_dir.replace("\\", "/"),
        "image_count": len(frames),
        "colmap": write_colmap,
        "transforms_json": write_transforms,
        "intrinsics": {k: float(v) for k, v in intrinsics.items()},
        "resolution": [int(frames[0]["width"]), int(frames[0]["height"])],
        "up_axis": "Z",  # Blender world
        "frames": [],
    }
    if meta:
        manifest.update(meta)
    for i, fr in enumerate(frames, start=1):
        q, t = camera_utils.matrix_world_to_colmap(fr["matrix_world"])
        manifest["frames"].append(
            {
                "id": i,
                "file": "images/{}".format(fr["filename"]),
                "colmap_quaternion_wxyz": [float(v) for v in q],
                "colmap_translation": [float(v) for v in t],
            }
        )
    with open(os.path.join(dataset_dir, "photara_dataset.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    written.append("photara_dataset.json")
    return written
