from __future__ import annotations

import os
from pathlib import Path
import sys

native_directory = Path(sys.argv[1]).resolve()
handles = []
if hasattr(os, "add_dll_directory"):
    handles.append(os.add_dll_directory(str(native_directory)))
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        handles.append(os.add_dll_directory(str(Path(cuda_path) / "bin")))
sys.path.insert(0, str(native_directory))

import photara as aes

assert aes.__name__ == "photara"
assert hasattr(aes, "__version__")

mvs = aes.MvsOptions()
assert mvs.mesh_tsdf_truncation_voxels == 4.0
assert abs(mvs.grazing_weight_floor - 0.12) < 1e-6
training = aes.TrainingOptions()
assert training.backend == aes.TrainingBackend.CUDA
training.backend = aes.TrainingBackend.VULKAN
assert training.backend == aes.TrainingBackend.VULKAN
assert (
    training.densification_strategy == aes.DensificationStrategy.ADC_IGS
)
training.densification_strategy = aes.DensificationStrategy.ADC_PLUS
assert training.densification_strategy == aes.DensificationStrategy.ADC_PLUS
assert not hasattr(aes.DensificationStrategy, "DEFAULT")
assert hasattr(aes.DensificationStrategy, "DENSE_ADAPTIVE")
assert hasattr(aes.DensificationStrategy, "EMC")

emc = aes.TrainingOptions()
emc.densification_strategy = aes.DensificationStrategy.EMC
emc.apply_strategy_defaults()
assert abs(emc.densify_growth_factor - 1.05) < 1e-6
aes.apply_strategy_defaults(emc)

sfm = aes.SfmOptions()
sfm.frontend.extractor = "sift"
sfm.frontend.matcher = "mutual_ratio"
sfm.frontend.relative.max_epipolar_error_px = 3.0
sfm.global_positioning.backend = aes.PositioningBackend.CPU
sfm.resection.local_ba.maximum_iterations = 7
assert sfm.frontend.extractor == "sift"
assert sfm.frontend.relative.max_epipolar_error_px == 3.0
assert sfm.global_positioning.backend == aes.PositioningBackend.CPU
assert sfm.resection.local_ba.maximum_iterations == 7

extractors = aes.available_extractors()
matchers = aes.available_matchers()
assert "sift" in extractors
assert "mutual_ratio" in matchers
assert "cuda" in aes.compiled_training_backends()
assert "cpu" in aes.available_bundle_backends()
aes.set_bundle_backend(aes.BundleBackend.CPU)
assert aes.bundle_backend() == aes.BundleBackend.CPU
assert hasattr(aes, "TrainingProgress")
assert hasattr(aes, "GaussianFormat")
assert callable(aes.run_sfm)
assert callable(aes.run_sfm_frontend)
assert callable(aes.run_sfm_mapping)
assert callable(aes.estimate_global_rotations)
assert callable(aes.solve_global_positions)
assert callable(aes.bundle_adjust)
assert callable(aes.load_sfm)
assert callable(aes.run_mvs)
assert callable(aes.train_3dgs)
assert callable(aes.extract_tsdf)
