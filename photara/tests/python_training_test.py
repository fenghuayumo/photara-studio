"""CUDA integration: callbacks/GIL, early stop, checkpoint resume and evaluation."""
from pathlib import Path
import math
import sys
import tempfile
import threading
import unittest

from python_api_fixture import import_photara, write_dataset

p = import_photara(sys.argv.pop(1))


class TrainingTest(unittest.TestCase):
    def test_training_resume_callbacks_and_evaluation(self):
        with tempfile.TemporaryDirectory(prefix="photara-python-training-") as directory:
            root = Path(directory)
            sparse, images = write_dataset(root)
            request = p.DatasetRequest()
            request.source, request.image_directory = sparse, images
            scene = p.load_dataset(request)
            options = p.training_options_for_scene(scene)
            options.iterations = 4
            options.sh_degree = 0
            options.use_mask = False
            options.preview_interval = 1
            options.log_interval = 1
            options.max_image_dimension = 32
            options.use_depth_normal_loss = False
            options.use_normal_field = False
            options.multi_view_geo_weight = 0
            options.multi_view_ncc_weight = 0
            options.training_device_cache_bytes = 0
            options.adaptive_training_cache = False
            options.checkpoint_path = root / "training.checkpoint"
            options.checkpoint_interval_iterations = 1
            stats, previews = [], []
            signal, advanced = threading.Event(), threading.Event()
            worker = threading.Thread(target=lambda: (signal.wait(30), advanced.set()), daemon=True)
            worker.start()

            def progress(value):
                stats.append(value)
                if value.iteration == 1:
                    signal.set()
                elif value.iteration == 2:
                    self.assertTrue(advanced.is_set(), "training held the GIL between callbacks")
                return value.iteration < 2

            interval = sys.getswitchinterval()
            try:
                # Prevent a Python callback's own scheduling from hiding a
                # missing GIL release around the native training loop.
                sys.setswitchinterval(100)
                model = p.train_3dgs(scene, options, progress, previews.append)
            finally:
                sys.setswitchinterval(interval)
                signal.set()
                worker.join(5)
            self.assertEqual([s.iteration for s in stats], [1, 2])
            self.assertGreater(model.size, 0)
            self.assertTrue(previews)
            self.assertEqual(len(previews[0].rgb), previews[0].width * previews[0].height * 3)
            self.assertIsInstance(previews[0].rgb, bytes)
            info = p.inspect_training_checkpoint(options.checkpoint_path)
            self.assertEqual(info.iteration, 2)
            self.assertTrue(info.model_path.exists())
            self.assertFalse(p.training_checkpoint_in_use(options.checkpoint_path))
            options.resume_checkpoint = True
            resumed = []
            model = p.train_3dgs(scene, options, lambda value: resumed.append(value.iteration))
            self.assertEqual(resumed, [3, 4])
            self.assertEqual(p.inspect_training_checkpoint(options.checkpoint_path).iteration, 4)
            self.assertEqual(stats[0].iteration, 1)  # Retained callback objects own their data.
            metrics = p.render_evaluation_png(model, scene, 0, root / "evaluation.png", options)
            self.assertTrue((root / "evaluation.png").exists())
            self.assertTrue(math.isfinite(metrics.mae))
            model.save(root / "trained.ply")
            self.assertEqual(p.load_3dgs(root / "trained.ply", device=p.Device.CPU).size, model.size)

            class CallbackError(Exception):
                pass

            def fail(value):
                raise CallbackError("callback failure")

            options.resume_checkpoint = False
            options.checkpoint_path = ""
            with self.assertRaisesRegex(CallbackError, "callback failure"):
                p.train_3dgs(scene, options, fail)
            with self.assertRaises(TypeError):
                p.train_3dgs(scene, options, progress=42)

            # The no-options path must resolve sparse initialization itself.
            implicit = p.train_3dgs(scene, progress=lambda value: False)
            self.assertGreater(implicit.size, 0)

            options.popspa_keep_ratio = .75
            options.popspa_refine_iterations = 1
            options.popspa_finetune_iterations = 1
            options.popspa_prune_cycles = 1
            count = model.size
            compacted = p.compact_popspa(model, scene, options)
            self.assertGreater(compacted.size, 0)
            self.assertLessEqual(compacted.size, count)
            self.assertEqual(model.size, count)


if __name__ == "__main__":
    unittest.main()
