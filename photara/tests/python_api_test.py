"""Run the Python API contracts against the built extension and real files."""
from pathlib import Path
import importlib.util
import re
import struct
import sys
import tempfile
import unittest

from python_api_fixture import import_photara, write_dataset, write_gaussians, write_mesh

p = import_photara(sys.argv.pop(1))
ROOT = Path(__file__).resolve().parents[2]


class ApiTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="photara-python-api-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def dataset(self):
        sparse, images = write_dataset(self.root)
        request = p.DatasetRequest()
        request.source, request.image_directory = sparse, images
        return p.load_dataset(request)

    def test_strategy_defaults_preserve_identity(self):
        options = p.TrainingOptions()
        options.densification_strategy = p.DensificationStrategy.EMC
        self.assertIs(options.apply_strategy_defaults(), options)
        self.assertIs(p.apply_strategy_defaults(options), options)
        options.apply_strategy_defaults().iterations = 17
        self.assertEqual(options.iterations, 17)
        self.assertAlmostEqual(options.densify_growth_factor, 1.05, places=5)

    def test_sparse_and_dense_metadata_and_options(self):
        scene = self.dataset()
        self.assertEqual(scene.dataset_format, p.DatasetFormat.COLMAP)
        self.assertTrue(scene.resolved_source.exists())
        self.assertEqual(scene.view_count, 3)
        self.assertEqual(scene.dense_point_count, 16)
        self.assertFalse(scene.initial_points_dense)
        self.assertFalse(scene.generated_initial_points)
        options = p.training_options_for_scene(scene)
        self.assertFalse(options.input_is_dense)
        self.assertAlmostEqual(options.means_lr, 2e-5)
        # Sparse initialization must not destroy an already imported cloud.
        p.initialize_scene_from_sparse_points(scene)
        self.assertEqual(scene.dense_point_count, 16)
        self.assertFalse(scene.initial_points_dense)
        cloud = self.root / "cloud.ply"
        scene.save_dense_cloud(cloud)
        scene.load_dense_cloud(cloud)
        self.assertTrue(scene.initial_points_dense)
        self.assertEqual(scene.initial_point_cloud, cloud)
        self.assertTrue(p.training_options_for_scene(scene).input_is_dense)
        p.initialize_scene_from_sparse_points(scene)
        self.assertTrue(scene.initial_points_dense)

    def test_models_on_cpu_and_mesh_roundtrip(self):
        source = self.root / "source.ply"
        write_gaussians(source)
        model = p.load_3dgs(source, p.GaussianFormat.PLY, p.Device.CPU)
        self.assertEqual((model.size, model.sh_degree), (2, 0))
        transform = p.GaussianTransform()
        transform.translation = [1, 2, 3]
        transform.scale = 2
        placed = p.transform_gaussians(model, transform)
        merged = p.merge_gaussians([model, placed])
        self.assertEqual(merged.size, 4)
        p.restrict_sh_degree(merged, 0)
        for name, handle, expected_x in [("original", model, -.2), ("placed", placed, .6)]:
            path = self.root / f"{name}.ply"
            handle.save(path)
            blob = path.read_bytes()
            offset = blob.index(b"end_header\n") + len(b"end_header\n")
            self.assertAlmostEqual(struct.unpack_from("<f", blob, offset)[0], expected_x, places=5)
            self.assertEqual(p.load_3dgs(path, device=p.Device.CPU).size, 2)
        with self.assertRaises((ValueError, RuntimeError)):
            p.merge_gaussians([])
        transform.scale = 0
        with self.assertRaises(RuntimeError):
            p.transform_gaussians(model, transform)
        mesh_path = self.root / "mesh.ply"
        write_mesh(mesh_path)
        mesh = p.load_mesh(mesh_path)
        self.assertEqual((mesh.vertex_count, mesh.face_count), (3, 1))
        mesh.save(self.root / "mesh-copy.ply")
        self.assertEqual(p.load_mesh(self.root / "mesh-copy.ply").face_count, 1)
        scene = self.dataset()
        with self.assertRaises(IndexError):
            p.render_evaluation_png(model, scene, 99, self.root / "bad.png")

    def test_documented_inventory_matches_runtime(self):
        spec = importlib.util.spec_from_file_location("api_reference", ROOT / "scripts/generate_api_reference.py")
        generator = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(generator)
        content = generator.generate()
        self.assertEqual(content, generator.OUTPUT.read_text(encoding="utf-8"))
        for block in content.split("\n## ")[1:]:
            name = block.splitlines()[0]
            if name in {"Enumerations", "Type index"}:
                continue
            self.assertTrue(hasattr(p, name), f"Missing documented class {name}")
            cls = getattr(p, name)
            for field in re.findall(r"^\| `(\w+)` \| read-", block, re.MULTILINE):
                self.assertTrue(hasattr(cls, field), f"{name}.{field}")
        enum_table = content.split("## Enumerations\n", 1)[1].split("\n## ", 1)[0]
        for line in enum_table.splitlines():
            if not line.startswith("| `"):
                continue
            enum_name = line.split("|")[1].strip(" `")
            values = re.findall(r"`(\w+)` →", line)
            self.assertEqual(set(getattr(p, enum_name).__members__), set(values), enum_name)
        hierarchical = content.split("## HierarchicalOptions\n", 1)[1].split("\n## ", 1)[0]
        positioning = content.split("## GlobalPositioningOptions\n", 1)[1].split("\n## ", 1)[0]
        for field in ("global_rotation", "global_positioning"):
            self.assertIn(f"| `{field}` |", hierarchical)
            self.assertNotIn(f"| `{field}` |", positioning)
        self.assertFalse(p.GlobalPositioningOptions().regularize_complete_orbits)
        options = p.TrainingOptions()
        options.checkpoint_path = self.root / "resume.checkpoint"
        options.resume_checkpoint = True
        options.render_method = p.RenderMethod.GAUSSIAN_3DGUT
        options.speedy_pruning = True
        options.rasterization.near_plane = .1
        self.assertTrue(options.resume_checkpoint and options.speedy_pruning)
        self.assertAlmostEqual(options.rasterization.near_plane, .1)
        asfm = p.AsfmOptions()
        asfm.compact = True
        self.assertTrue(asfm.compact)
        self.assertIsNone(p.inspect_training_checkpoint(self.root / "missing.checkpoint"))


if __name__ == "__main__":
    unittest.main()
