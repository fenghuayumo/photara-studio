"""Compare C++ backend tensor reads against the original PyTorch checkpoint."""
from __future__ import annotations
import argparse
from pathlib import Path
import subprocess
import tempfile
import numpy as np
import torch


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("gguf", type=Path)
    parser.add_argument("executable", type=Path)
    parser.add_argument("--precision", choices=("f32", "f16"), default="f32")
    parser.add_argument("--vulkan", action="store_true")
    args = parser.parse_args()
    weights = torch.load(args.checkpoint, map_location="cpu", weights_only=True)["model"]
    with tempfile.TemporaryDirectory(prefix="photara-moge-reference-") as temporary:
        reference = Path(temporary) / "weights.txt"
        with reference.open("w", encoding="utf-8") as output:
            for name, tensor in weights.items():
                values = tensor.detach().float().numpy().reshape(-1)
                if args.precision == "f16" and name.endswith(".weight") and tensor.ndim >= 2:
                    values = values.astype(np.float16).astype(np.float32)
                # Cover tensor boundaries and interior offsets, not just the
                # first value; sampling is independent of the GGUF serializer.
                for index in sorted({0, 1, len(values) // 3, len(values) // 2, len(values) - 1}):
                    if index < len(values):
                        output.write(f"{name} {index} {float(values[index]):.9g}\n")
        command = [str(args.executable.resolve()), str(args.gguf.resolve()), str(reference)]
        if args.vulkan:
            command.append("--vulkan")
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
