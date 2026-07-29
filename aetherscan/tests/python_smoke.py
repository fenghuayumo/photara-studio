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

import aetherscan_native as aes

mvs = aes.MvsOptions()
assert mvs.mesh_tsdf_truncation_voxels == 4.0
training = aes.TrainingOptions()
training.densification_strategy = aes.DensificationStrategy.ADC_PLUS
assert training.densification_strategy == aes.DensificationStrategy.ADC_PLUS
assert callable(aes.run_sfm)
assert callable(aes.run_mvs)
assert callable(aes.train_3dgs)
assert callable(aes.extract_tsdf)
