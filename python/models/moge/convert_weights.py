"""Convert an official MoGe-2 ViT-B normal checkpoint to Photara GGUF.

No network access and no model code import. Tensor names and contiguous PyTorch
storage order are preserved; GGUF dimensions are reversed by GGUFWriter.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

import numpy as np
import torch
import gguf

FORMAT_VERSION = 1
MODEL_ID = "Ruicheng/moge-2-vitb-normal"


def checkpoint_summary(checkpoint: dict) -> dict:
    config, weights = checkpoint["model_config"], checkpoint["model"]
    encoder = config["encoder"]
    if encoder["backbone"] != "dinov2_vitb14":
        raise ValueError("Only MoGe-2 dinov2_vitb14 is supported by format version 1")
    if config.get("remap_output") != "exp":
        raise ValueError("Expected the MoGe-2 exp point-map remapping")
    if not all(config.get(head) for head in
               ("points_head", "normal_head", "mask_head", "scale_head")):
        raise ValueError("The point, normal, mask and scale heads are required")
    required_shapes = {
        "encoder.backbone.patch_embed.proj.weight": (768, 3, 14, 14),
        "encoder.backbone.cls_token": (1, 1, 768),
        "encoder.backbone.norm.weight": (768,),
        "encoder.backbone.pos_embed": (1, 1370, 768),
        "points_head.output_blocks.4.weight": (3, 32, 1, 1),
        "normal_head.output_blocks.4.weight": (3, 32, 1, 1),
        "mask_head.output_blocks.4.weight": (1, 32, 1, 1),
        "scale_head.4.weight": (1, 768),
    }
    for block in range(12):
        required_shapes[f"encoder.backbone.blocks.{block}.attn.qkv.weight"] = (2304, 768)
        required_shapes[f"encoder.backbone.blocks.{block}.mlp.fc1.weight"] = (3072, 768)
    for name, shape in required_shapes.items():
        if name not in weights or tuple(weights[name].shape) != shape:
            raise ValueError(f"Missing or malformed tensor: {name}, expected {shape}")
    layers = encoder["intermediate_layers"]
    if layers != [5, 11] or encoder["dim_out"] != 768:
        raise ValueError("Unsupported encoder projections or intermediate layers")
    for name, tensor in weights.items():
        if not isinstance(tensor, torch.Tensor) or not tensor.is_floating_point():
            raise ValueError(f"Expected a floating-point tensor: {name}")
        if not 1 <= tensor.ndim <= 4 or tensor.numel() == 0:
            raise ValueError(f"Unsupported tensor shape: {name}")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Nonfinite checkpoint tensor: {name}")
    return {"config": config, "weights": weights, "embedding_length": 768,
            "block_count": 12, "patch_size": 14, "intermediate_layers": layers}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def convert(source: Path, destination: Path, revision: str, precision: str = "f32") -> dict:
    if not revision.strip():
        raise ValueError("A source checkpoint revision is required")
    if source.resolve() == destination.resolve():
        raise ValueError("Source and destination must differ")
    if destination.exists():
        raise FileExistsError(f"Output already exists: {destination}")
    summary = checkpoint_summary(torch.load(source, map_location="cpu", weights_only=True))
    source_digest = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix=destination.name + ".", suffix=".tmp",
                                     dir=destination.parent)
    os.close(fd)
    writer = gguf.GGUFWriter(temporary, "moge")
    try:
        writer.add_uint32("moge.format_version", FORMAT_VERSION)
        writer.add_string("moge.source.model_id", MODEL_ID)
        writer.add_string("moge.source.revision", revision)
        writer.add_string("moge.source.sha256", source_digest)
        writer.add_string("moge.model_config", json.dumps(summary["config"], sort_keys=True))
        writer.add_string("moge.weight_precision", precision)
        for field in ("embedding_length", "block_count", "patch_size"):
            writer.add_uint32("moge.encoder." + field, summary[field])
        writer.add_array("moge.encoder.intermediate_layers", summary["intermediate_layers"])
        total_bytes = 0
        for name, tensor in summary["weights"].items():
            # Initial correctness path preserves all values in F32. Optional
            # F16 only changes matrix/kernel weights, not affine vectors or
            # positional embeddings. Tensor layout is never silently permuted.
            dtype = np.float16 if precision == "f16" and name.endswith(".weight") \
                and tensor.ndim >= 2 else np.float32
            array = np.ascontiguousarray(tensor.detach().float().numpy(), dtype=dtype)
            if not np.isfinite(array).all():
                raise ValueError(f"Tensor overflows chosen precision: {name}")
            writer.add_tensor(name, array)
            total_bytes += array.nbytes
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file()
        writer.close()
        # Refuse accidental overwrite, including a file created during conversion.
        if destination.exists():
            raise FileExistsError(f"Output appeared during conversion: {destination}")
        os.link(temporary, destination) # Same directory; atomically refuses overwrite.
    finally:
        writer.close()
        Path(temporary).unlink(missing_ok=True)
    return {"model": MODEL_ID, "revision": revision, "source_sha256": source_digest,
            "tensors": len(summary["weights"]), "tensor_bytes": total_bytes,
            "precision": precision, "output": str(destination)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--revision", required=True)
    parser.add_argument("--precision", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()
    print(json.dumps(convert(args.checkpoint, args.output, args.revision, args.precision), indent=2))


if __name__ == "__main__":
    main()
