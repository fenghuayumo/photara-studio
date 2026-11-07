"""Pure-numpy camera math shared by the exporter and the unit tests.

Conventions
-----------
Blender / OpenGL camera:  +X right, +Y up, looks along -Z.
COLMAP / OpenCV camera:   +X right, -Y up, looks along +Z.

COLMAP ``images.txt`` stores the world-to-camera transform as
``p_cam = R * p_world + t`` with a quaternion (qw qx qy qz).
"""

import math

import numpy as np

_FLIP = np.diag([1.0, -1.0, -1.0])


def matrix_world_to_colmap(matrix_world):
    """Blender camera-to-world (4x4) -> COLMAP (quat wxyz, translation)."""
    m = np.asarray(matrix_world, dtype=np.float64).reshape(4, 4)
    r_blender = m[:3, :3]  # camera-to-world, Blender convention
    center = m[:3, 3]
    r_w2c = _FLIP @ r_blender.T
    t = -r_w2c @ center
    q = quaternion_from_matrix(r_w2c)
    return q, t


def colmap_to_matrix_world(quat_wxyz, t):
    """COLMAP (qw qx qy qz, t) -> Blender camera-to-world (4x4)."""
    r_w2c = matrix_from_quaternion(np.asarray(quat_wxyz, dtype=np.float64))
    r_blender = r_w2c.T @ _FLIP
    center = -r_w2c.T @ np.asarray(t, dtype=np.float64)
    m = np.eye(4)
    m[:3, :3] = r_blender
    m[:3, 3] = center
    return m


def quaternion_from_matrix(r):
    """Rotation matrix -> quaternion (w, x, y, z), numerically robust (Shepperd)."""
    r = np.asarray(r, dtype=np.float64)
    trace = r[0, 0] + r[1, 1] + r[2, 2]
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (r[2, 1] - r[1, 2]) / s
        y = (r[0, 2] - r[2, 0]) / s
        z = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = math.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        w = (r[2, 1] - r[1, 2]) / s
        x = 0.25 * s
        y = (r[0, 1] + r[1, 0]) / s
        z = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = math.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        w = (r[0, 2] - r[2, 0]) / s
        x = (r[0, 1] + r[1, 0]) / s
        y = 0.25 * s
        z = (r[1, 2] + r[2, 1]) / s
    else:
        s = math.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        w = (r[1, 0] - r[0, 1]) / s
        x = (r[0, 2] + r[2, 0]) / s
        y = (r[1, 2] + r[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / np.linalg.norm(q)


def matrix_from_quaternion(q):
    """Quaternion (w, x, y, z) -> rotation matrix."""
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def compute_intrinsics(camera, width, height, pixel_aspect_x=1.0, pixel_aspect_y=1.0):
    """Compute pixel-space intrinsics from a Blender camera datablock.

    Handles sensor fit (AUTO/HORIZONTAL/VERTICAL), lens shift and pixel aspect.
    Returns a dict with fx, fy, cx, cy.
    """
    if camera.type != "PERSP":
        raise ValueError("仅支持透视相机（PERSP），正交与全景相机不适用于 3DGS 重建")

    fit = camera.sensor_fit
    if fit == "AUTO":
        fit = "HORIZONTAL" if width * pixel_aspect_x >= height * pixel_aspect_y else "VERTICAL"

    sensor = float(camera.sensor_height if camera.sensor_fit == "VERTICAL" else camera.sensor_width)
    aspect = float(pixel_aspect_y) / float(pixel_aspect_x)
    view_factor = width if fit == "HORIZONTAL" else aspect * height
    fx = float(camera.lens) * view_factor / sensor
    fy = fx / aspect
    cx = 0.5 * width - float(camera.shift_x) * view_factor
    cy = 0.5 * height + float(camera.shift_y) * view_factor / aspect
    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy}


def fov_x_from_fx(fx, width):
    return 2.0 * math.atan(0.5 * width / fx)
