"""Guard the training backend boundary without requiring GPU hardware."""
from pathlib import Path
import re

source = Path(__file__).resolve().parents[1] / "src/splat"
errors = []
for path in source.glob("*.cu"):
    # Comments may mention Vulkan parity. Executable CUDA code must never
    # enter the Vulkan runtime or invoke a Vulkan implementation.
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", path.read_text(), flags=re.S)
    if re.search(r"(?:tinytensor|splat_drender)::vulkan::|\b\w+_vulkan\s*\(", text):
        errors.append(f"{path.name}: CUDA translation unit calls Vulkan")
    for include in re.findall(r'#include\s+[<"]([^>"\n]+)', text):
        # This adapter imports exported handles with CUDA APIs; it does not
        # execute Vulkan functions or train with the Vulkan backend.
        if "vulkan" in include and include != "splat/cuda_vulkan_preview.hpp":
            errors.append(f"{path.name}: Vulkan implementation include {include}")

trainer = (source / "trainer.cpp").read_text()
if re.search(r"detail::\w+_(?:vulkan|cuda)\s*\(", trainer):
    errors.append("trainer.cpp: scheduler calls a concrete training backend")
assert not errors, "\n".join(errors)
print("3DGS training backend layering passed")
