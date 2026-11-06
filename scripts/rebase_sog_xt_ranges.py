"""Losslessly rebase observed code ranges for paper-reference decoder compatibility.

Does not cluster, retrain, change row order, labels, or decoded attributes.
The source archive is never overwritten.
"""
import argparse
import io
import json
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image


def rebase(source, target):
    if source.resolve() == target.resolve():
        raise ValueError("Choose a separate destination archive")
    with zipfile.ZipFile(source) as z:
        files = {name: z.read(name) for name in z.namelist()}
    metadata = json.loads(files["container_meta.yaml"])
    changed = []
    for operation in metadata["ops"]:
        field = operation["input_fields"][0]
        config = operation["transforms"][0]["simple_quantize"]
        wide = field == "means"
        name = field + ("_bytes_0.webp" if wide else ".webp")
        image = np.array(Image.open(io.BytesIO(files[name])))
        channels = 4 if field == "quaternions" else 1 if field == "opacities" else 3
        codes = image[..., :channels].astype(np.uint32)
        upper = None
        if wide:
            upper = np.array(Image.open(io.BytesIO(files["means_bytes_1.webp"])))
            codes += upper[..., :channels].astype(np.uint32) * 256
        low_code, high_code = int(codes.min()), int(codes.max())
        levels = config["levels"]
        if (low_code, high_code) == (0, levels - 1):
            continue
        low = np.asarray(config["min_values"], dtype=np.float64)
        high = np.asarray(config["max_values"], dtype=np.float64)
        rebased_low = (low + (high - low) * low_code / (levels - 1)).astype(np.float32)
        rebased_high = (low + (high - low) * high_code / (levels - 1)).astype(np.float32)
        config["min_values"] = rebased_low.tolist()
        config["max_values"] = rebased_high.tolist()
        config["levels"] = max(2, high_code - low_code + 1)
        if low_code:
            codes -= low_code
            image[..., :channels] = codes & 255
            stream = io.BytesIO()
            Image.fromarray(image).save(stream, format="WEBP", lossless=True, exact=True)
            files[name] = stream.getvalue()
            if wide:
                upper[..., :channels] = codes >> 8
                stream = io.BytesIO()
                Image.fromarray(upper).save(stream, format="WEBP", lossless=True, exact=True)
                files["means_bytes_1.webp"] = stream.getvalue()
        changed.append(field)
    files["container_meta.yaml"] = json.dumps(metadata, separators=(",", ":")).encode()
    with zipfile.ZipFile(target, "w", compression=zipfile.ZIP_STORED) as z:
        for name, contents in files.items():
            z.writestr(name, contents)
    print(json.dumps({"rebased_fields": changed, "output": str(target)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("target", type=Path)
    args = parser.parse_args()
    rebase(args.source, args.target)
