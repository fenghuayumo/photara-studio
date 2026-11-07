"""Run with: python tests/test_camera_utils.py (numpy required, no Blender)."""

from pathlib import Path
import json
import math
import os
import shutil
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from photara_render import camera_utils, colmap_export  # noqa: E402
from photara_render.photara_bridge import parse_launch_args  # noqa: E402


def random_rotation(rng):
    axis = rng.normal(size=3)
    axis /= max(1e-12, np.linalg.norm(axis))
    angle = rng.uniform(0.0, 2.0 * math.pi)
    K = np.array(
        [[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]]
    )
    return np.eye(3) + math.sin(angle) * K + (1 - math.cos(angle)) * (K @ K)


class CameraMathTests(unittest.TestCase):
    def test_identity_camera_looks_minus_z(self):
        q, t = camera_utils.matrix_world_to_colmap(np.eye(4))
        self.assertTrue(np.allclose(t, 0.0, atol=1e-12))
        # A point 2m in front of a Blender camera (0,0,-2) must have positive z
        # in COLMAP/OpenCV camera space.
        R = camera_utils.matrix_from_quaternion(q)
        p = R @ np.array([0.0, 0.0, -2.0]) + t
        self.assertGreater(p[2], 0.0)
        self.assertAlmostEqual(p[2], 2.0, places=10)

    def test_roundtrip_random_poses(self):
        rng = np.random.default_rng(42)
        for _ in range(50):
            M = np.eye(4)
            M[:3, :3] = random_rotation(rng)
            M[:3, 3] = rng.uniform(-10, 10, size=3)
            q, t = camera_utils.matrix_world_to_colmap(M)
            M2 = camera_utils.colmap_to_matrix_world(q, t)
            self.assertTrue(np.allclose(M, M2, atol=1e-9))

    def test_quaternion_is_unit_and_valid_rotation(self):
        rng = np.random.default_rng(7)
        q = camera_utils.quaternion_from_matrix(random_rotation(rng))
        self.assertAlmostEqual(np.linalg.norm(q), 1.0, places=12)
        R = camera_utils.matrix_from_quaternion(q)
        self.assertTrue(np.allclose(R @ R.T, np.eye(3), atol=1e-12))
        self.assertAlmostEqual(np.linalg.det(R), 1.0, places=12)

    def test_intrinsics_horizontal_fit(self):
        class FakeCam:
            type = "PERSP"
            lens = 50.0
            sensor_width = 36.0
            sensor_fit = "HORIZONTAL"
            shift_x = 0.0
            shift_y = 0.0

        intr = camera_utils.compute_intrinsics(FakeCam(), 1920, 1080)
        self.assertAlmostEqual(intr["fx"], 50.0 / 36.0 * 1920, places=9)
        self.assertAlmostEqual(intr["fy"], intr["fx"], places=9)
        self.assertAlmostEqual(intr["cx"], 960.0, places=9)
        self.assertAlmostEqual(intr["cy"], 540.0, places=9)

    def test_intrinsics_vertical_fit(self):
        class FakeCam:
            type = "PERSP"
            lens = 35.0
            sensor_width = 36.0
            sensor_height = 24.0
            sensor_fit = "VERTICAL"
            shift_x = 0.0
            shift_y = 0.0

        intr = camera_utils.compute_intrinsics(FakeCam(), 1080, 1920)
        self.assertAlmostEqual(intr["fy"], 35.0 / 24.0 * 1920, places=9)
        self.assertAlmostEqual(intr["fx"], intr["fy"], places=9)

    def test_pixel_aspect_and_shift(self):
        class Cam:
            type = "PERSP"
            lens = 50.0
            sensor_width = 36.0
            sensor_fit = "HORIZONTAL"
            shift_x = 0.1
            shift_y = 0.2
        intr = camera_utils.compute_intrinsics(Cam(), 800, 600, 1, 2)
        self.assertAlmostEqual(intr["fx"], 50 / 36 * 800)
        self.assertAlmostEqual(intr["fy"], intr["fx"] / 2)
        self.assertAlmostEqual(intr["cx"], 320)
        self.assertAlmostEqual(intr["cy"], 380)

    def test_launch_args_keep_windows_paths(self):
        argv = parse_launch_args(
            '--splat-dataset "{dataset}" --output "{dataset}/{name}.sog"',
            {"dataset": r"D:\my data\scene", "name": "demo"},
        )
        self.assertEqual(argv[0], "--splat-dataset")
        self.assertEqual(argv[1], r"D:\my data\scene")
        self.assertEqual(argv[2], "--output")
        self.assertEqual(argv[3], r"D:\my data\scene/demo.sog")


class DatasetWriteTests(unittest.TestCase):
    def _frames(self, n=4):
        frames = []
        for i in range(n):
            angle = 2 * math.pi * i / n
            c, s = math.cos(angle), math.sin(angle)
            M = np.array(
                [
                    [c, 0, s, 5 * c],
                    [0, 1, 0, 2],
                    [-s, 0, c, 5 * s],
                    [0, 0, 0, 1],
                ]
            )
            frames.append(
                {
                    "filename": "frame_{:08d}.png".format(i),
                    "matrix_world": M.tolist(),
                    "width": 800,
                    "height": 600,
                }
            )
        return frames

    def test_write_dataset_layout_and_colmap_format(self):
        tmp = tempfile.mkdtemp(prefix="photara_test_")
        try:
            frames = self._frames()
            intr = {"fx": 1000.0, "fy": 1000.0, "cx": 400.0, "cy": 300.0}
            colmap_export.write_dataset(
                tmp, frames, intr, name="ut_scene",
                write_colmap=True, write_transforms=True,
            )
            images = os.path.join(tmp, "images")
            sparse = os.path.join(tmp, "sparse", "0")
            self.assertTrue(os.path.isdir(images))
            self.assertTrue(os.path.isfile(os.path.join(sparse, "cameras.txt")))
            self.assertTrue(os.path.isfile(os.path.join(sparse, "images.txt")))
            self.assertTrue(os.path.isfile(os.path.join(sparse, "points3D.txt")))

            with open(os.path.join(sparse, "images.txt"), encoding="utf-8") as f:
                lines = [
                    l for l in f.read().splitlines()
                    if l.strip() and not l.startswith("#")
                ]
            self.assertEqual(len(lines), len(frames))
            for line in lines:
                parts = line.split()
                self.assertEqual(len(parts), 10)
                self.assertEqual(parts[8], "1")  # CAMERA_ID
                self.assertTrue(parts[9].endswith(".png"))

            with open(os.path.join(sparse, "cameras.txt"), encoding="utf-8") as f:
                cam_line = f.read().splitlines()[-1]
            # "1 PINHOLE W H fx fy cx cy" == 8 tokens
            self.assertEqual(len(cam_line.split()), 8)
            self.assertEqual(cam_line.split()[1], "PINHOLE")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)

    def test_animated_intrinsics_in_both_formats(self):
        with tempfile.TemporaryDirectory() as tmp:
            frames = self._frames(2)
            intr = {"fx": 1000, "fy": 900, "cx": 400, "cy": 300}
            frames[1]["intrinsics"] = dict(intr, fx=1200, cx=410)
            colmap_export.write_dataset(tmp, frames, intr)
            data = json.loads(Path(tmp, "transforms.json").read_text())
            self.assertEqual(data["frames"][1]["fl_x"], 1200)
            self.assertEqual(data["frames"][1]["cx"], 410)
            lines = Path(tmp, "sparse/0/cameras.txt").read_text().splitlines()
            self.assertEqual(len([line for line in lines if not line.startswith("#")]), 2)
            images = Path(tmp, "sparse/0/images.txt").read_text().splitlines()
            self.assertEqual([line.split()[8] for line in images if line and not line.startswith("#")], ["1", "2"])

    def test_transforms_json_matches_photara_nerfstudio_format(self):
        tmp = tempfile.mkdtemp(prefix="photara_test_")
        try:
            frames = self._frames()
            intr = {"fx": 1000.0, "fy": 1000.0, "cx": 400.0, "cy": 300.0}
            colmap_export.write_dataset(
                tmp, frames, intr, name="ut_scene",
                write_colmap=False, write_transforms=True,
            )
            with open(os.path.join(tmp, "transforms.json"), encoding="utf-8") as f:
                data = json.load(f)
            # photara-studio save_nerfstudio_transforms fields
            self.assertEqual(data["fl_x"], 1000.0)
            self.assertEqual(data["fl_y"], 1000.0)
            self.assertEqual(data["cx"], 400.0)
            self.assertEqual(data["cy"], 300.0)
            self.assertEqual(data["w"], 800)
            self.assertEqual(data["h"], 600)
            self.assertEqual(data["camera_model"], "OPENCV")
            self.assertEqual(data["k1"], 0.0)
            # legacy nerfstudio angle kept for compatibility
            self.assertAlmostEqual(data["camera_angle_x"], 2 * math.atan(400.0 / 1000.0), places=9)
            self.assertEqual(len(data["frames"]), len(frames))
            # transform_matrix is camera-to-world OpenGL == Blender matrix_world
            self.assertTrue(
                np.allclose(
                    np.array(data["frames"][0]["transform_matrix"]),
                    frames[0]["matrix_world"],
                )
            )
            self.assertEqual(data["frames"][0]["file_path"], "images/frame_00000000.png")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
