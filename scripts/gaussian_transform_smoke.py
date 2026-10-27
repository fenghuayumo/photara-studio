"""Exercise Position, Scale and Rotation while a Studio preview is displayed.

Requires an interactive desktop with a Vulkan-capable GPU. The model and
editor state are isolated in a temporary directory.
"""
import argparse
import os
from pathlib import Path
import struct
import subprocess
import tempfile


def write_fixture(path: Path) -> None:
    properties = ["x", "y", "z", "nx", "ny", "nz", "f_dc_0", "f_dc_1", "f_dc_2",
                  "opacity", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    lines = ["ply", "format binary_little_endian 1.0", "element vertex 3"]
    lines.extend(f"property float {name}" for name in properties)
    lines.append("end_header")
    with path.open("wb") as output:
        output.write(("\n".join(lines) + "\n").encode("ascii"))
        for x, y in [(-0.2, -0.2), (0.2, -0.2), (0.0, 0.2)]:
            output.write(struct.pack("<17f", x, y, 2.0, 0.0, 0.0, 0.0,
                                     1.0, 0.5, 0.2, 3.0, -2.0, -2.0, -2.0,
                                     1.0, 0.0, 0.0, 0.0))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--studio", required=True, type=Path)
    parser.add_argument("--model", type=Path, help="Use an existing model instead of the tiny fixture")
    args = parser.parse_args()
    studio = args.studio.resolve(strict=True)
    with tempfile.TemporaryDirectory(prefix="photara_transform_smoke_") as tmp:
        root = Path(tmp)
        model = args.model.resolve(strict=True) if args.model else root / "fixture.ply"
        if not args.model:
            write_fixture(model)
        env = dict(os.environ, APPDATA=str(root), XDG_CONFIG_HOME=str(root),
                   PHOTARA_TRANSFORM_MODEL=str(model))
        result = subprocess.run([str(studio), "--gaussian-transform-smoke"],
                                cwd=studio.parent, env=env, timeout=90,
                                capture_output=True, text=True, errors="replace")
        print(result.stdout, end="")
        if result.stderr:
            print(result.stderr, end="")
        if result.returncode != 0:
            print(f"Studio exited with code {result.returncode}")
            return 1
        if "Gaussian transform preview smoke passed" not in result.stdout:
            print("Studio exited without completing the transform checks")
            return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
