"""Keep SfM orchestration independent of concrete GPU solvers."""
from pathlib import Path
import re

root = Path(__file__).resolve().parents[1]
for name in ("bundle.cpp", "global_positioning.cpp"):
    text = (root / "src/sfm" / name).read_text()
    assert not re.search(r"\b(?:Cuda|Vulkan)(?:Bearing)?Optimizer\b|ba::optimize_(?:cuda|vulkan)\s*\(", text), name
    assert not re.search(r'#include\s+"ba/[^"\n]*(?:cuda|vulkan)', text), name
assert "bearing_cuda.hpp" not in (root / "include/ba/bearing_vulkan.hpp").read_text()
for device, other in (("cuda", "vulkan"), ("vulkan", "cuda")):
    text = (root / "src/ba" / f"solver_backend_{device}.cpp").read_text()
    assert other not in text.lower(), f"{device} adapter depends on {other}"
print("SfM solver backend layering passed")
