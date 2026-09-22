#!/usr/bin/env python3
"""
Dump reference intermediate tensors from the SAM3 Python model for C++ verification.
Saves .bin files with raw float32 data and .shape files with tensor dimensions.

Usage:
    python tests/dump_reference.py --checkpoint raw_weights/sam3.pt --image tests/test_truck.jpg --outdir tests/ref/
"""
import argparse
import os
import sys
import struct
import math

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2

# Add SAM3 repo to path
sys.path.insert(0, os.path.expanduser("~/Documents/sam3"))
from sam3.model.vitdet import ViT, compute_axial_cis, window_partition, window_unpartition
from sam3.model_builder import _create_vit_backbone


def save_tensor(path, t):
    """Save a tensor as raw float32 binary + shape file."""
    t = t.detach().cpu().float().contiguous()
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)} bytes={t.numel()*4}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, help="Path to sam3.pt")
    parser.add_argument("--image", required=True, help="Path to test image")
    parser.add_argument("--outdir", default="tests/ref", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"  # Use CPU for reproducibility

    # ── Load model ─────────────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    state_dict = ckpt if isinstance(ckpt, dict) and "model" not in ckpt else ckpt.get("model", ckpt)

    # Extract ViT weights
    vit_prefix = "detector.backbone.visual.trunk."
    vit_state = {k[len(vit_prefix):]: v for k, v in state_dict.items() if k.startswith(vit_prefix)}

    print(f"  Found {len(vit_state)} ViT parameters")

    # ── Create ViT backbone ────────────────────────────────────────────────
    print("Creating ViT backbone...")
    vit = _create_vit_backbone(compile_mode=None)
    vit.use_act_checkpoint = False  # Disable for inference
    missing, unexpected = vit.load_state_dict(vit_state, strict=False)
    if missing:
        print(f"  WARNING: Missing keys: {missing[:5]}...")
    if unexpected:
        print(f"  WARNING: Unexpected keys: {unexpected[:5]}...")
    vit.eval()
    vit.to(device)

    # ── Load and preprocess image ──────────────────────────────────────────
    print(f"Loading image: {args.image}")
    img = Image.open(args.image).convert("RGB")
    print(f"  Original size: {img.size}")

    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    img_tensor = v2.functional.to_image(img)
    img_preprocessed = transform(img_tensor).unsqueeze(0).to(device)
    print(f"  Preprocessed shape: {list(img_preprocessed.shape)}")
    save_tensor(os.path.join(args.outdir, "preprocessed"), img_preprocessed)

    # ── Step-by-step forward pass ──────────────────────────────────────────
    with torch.no_grad():
        # 1. Patch embedding
        x = vit.patch_embed(img_preprocessed)
        print(f"  After patch_embed: {list(x.shape)}")  # [1, 72, 72, 1024]
        save_tensor(os.path.join(args.outdir, "patch_embed"), x)

        # 2. Positional embedding
        h, w = x.shape[1], x.shape[2]
        from sam3.model.vitdet import get_abs_pos
        pos = get_abs_pos(vit.pos_embed, vit.pretrain_use_cls_token, (h, w),
                          retain_cls_token=False, tiling=vit.tile_abs_pos)
        print(f"  Pos embed (tiled): {list(pos.shape)}")  # [1, 72, 72, 1024]
        save_tensor(os.path.join(args.outdir, "pos_embed_tiled"), pos)

        x = x + pos
        save_tensor(os.path.join(args.outdir, "after_pos_embed"), x)

        # 3. ln_pre
        x = vit.ln_pre(x)
        save_tensor(os.path.join(args.outdir, "after_ln_pre"), x)

        # 4. Block 0 output
        x = vit.blocks[0](x)
        save_tensor(os.path.join(args.outdir, "block_0_out"), x)

        # 5. All remaining blocks
        for i in range(1, len(vit.blocks)):
            x = vit.blocks[i](x)

        save_tensor(os.path.join(args.outdir, "vit_final"), x)

        # 6. Permute to [B, C, H, W]
        feats = x.permute(0, 3, 1, 2)
        save_tensor(os.path.join(args.outdir, "vit_output_bchw"), feats)

    # ── RoPE reference ─────────────────────────────────────────────────────
    print("\nDumping RoPE reference values...")

    # Window attention RoPE (24x24, rope_pt_size=24, no interp needed since pt_size == input_size)
    # But wait — for window blocks, input_size is (24, 24), rope_pt_size is from the config
    # Looking at the model_builder: rope_pt_size=None → defaults to (window_size, window_size) = (24, 24)
    # And rope_interp=True, scale_pos = 24/24 = 1.0 → no scaling
    freqs_cis_window = compute_axial_cis(dim=64, end_x=24, end_y=24, theta=10000.0, scale_pos=1.0)
    save_tensor(os.path.join(args.outdir, "rope_window_complex"), freqs_cis_window)
    # Convert to real (cos, sin) pairs: freqs_cis is complex [576, 32]
    freqs_real_window = torch.view_as_real(freqs_cis_window)  # [576, 32, 2]
    save_tensor(os.path.join(args.outdir, "rope_window_real"), freqs_real_window)

    # Global attention RoPE (72x72, rope_pt_size=24, rope_interp=True → scale_pos=24/72=1/3)
    # For global blocks: input_size = (72, 72), rope_pt_size = (24, 24) (from None default)
    # Wait — actually, looking at the code:
    # Block.__init__: input_size=input_size if window_size == 0 else (window_size, window_size)
    # For global blocks, window_size=0, so input_size=(72, 72)
    # Attention.__init__: rope_pt_size from Block, which is (24,24) for rope_pt_size=None
    # scale_pos = rope_pt_size[0] / input_size[0] = 24 / 72 = 1/3
    scale_pos_global = 24.0 / 72.0
    freqs_cis_global = compute_axial_cis(dim=64, end_x=72, end_y=72, theta=10000.0, scale_pos=scale_pos_global)
    freqs_real_global = torch.view_as_real(freqs_cis_global)  # [5184, 32, 2]
    save_tensor(os.path.join(args.outdir, "rope_global_real"), freqs_real_global)

    # ── Neck reference ─────────────────────────────────────────────────────
    print("\nDumping neck reference values...")
    # Load neck weights
    det_neck_prefix = "detector.backbone.visual.convs."
    # We need to instantiate the neck too, but for now just save the ViT output
    # The neck test will be done separately once we verify ViT outputs match

    # ── Sinusoidal PE reference ────────────────────────────────────────────
    print("\nDumping sinusoidal PE reference values...")
    from sam3.model.position_encoding import PositionEmbeddingSine
    pe = PositionEmbeddingSine(num_pos_feats=256, normalize=True, scale=None, temperature=10000)

    for H, W, name in [(288, 288, "pe_288"), (144, 144, "pe_144"), (72, 72, "pe_72"), (36, 36, "pe_36")]:
        dummy = torch.zeros(1, 1, H, W)
        pe_out = pe(dummy)  # [1, 256, H, W]
        save_tensor(os.path.join(args.outdir, name), pe_out)

    print("\nDone! Reference tensors saved to", args.outdir)


if __name__ == "__main__":
    main()
