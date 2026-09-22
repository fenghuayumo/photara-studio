#!/usr/bin/env python3
"""
Phase 3 Numerical Audit — dump ALL intermediate tensors from the ViT backbone,
neck (SimpleFPN), sinusoidal PE, and RoPE for numerical comparison against C++.

Usage:
    uv run python tests/dump_phase3_reference.py \
        --checkpoint raw_weights/sam3.pt \
        --image tests/test_random.jpg \
        --outdir tests/ref_phase3
"""
import argparse
import math
import os
import sys
from functools import partial
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


# ── Helpers ────────────────────────────────────────────────────────────────

def save_tensor(path, t):
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)} "
          f"min={t.min().item():.6f} max={t.max().item():.6f} mean={t.mean().item():.6f}")


# ── Inlined from vitdet.py ─────────────────────────────────────────────────

def init_t_xy(end_x, end_y, scale=1.0, offset=0):
    t = torch.arange(end_x * end_y, dtype=torch.float32)
    t_x = (t % end_x).float()
    t_y = torch.div(t, end_x, rounding_mode="floor").float()
    return t_x * scale + offset, t_y * scale + offset


def compute_axial_cis(dim, end_x, end_y, theta=10000.0, scale_pos=1.0, offset=0):
    freqs_x = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    freqs_y = 1.0 / (theta ** (torch.arange(0, dim, 4)[: (dim // 4)].float() / dim))
    t_x, t_y = init_t_xy(end_x, end_y, scale_pos, offset)
    freqs_x = torch.outer(t_x, freqs_x)
    freqs_y = torch.outer(t_y, freqs_y)
    freqs_cis_x = torch.polar(torch.ones_like(freqs_x), freqs_x)
    freqs_cis_y = torch.polar(torch.ones_like(freqs_y), freqs_y)
    return torch.cat([freqs_cis_x, freqs_cis_y], dim=-1)


def reshape_for_broadcast(freqs_cis, x):
    ndim = x.ndim
    shape = [d if i >= ndim - 2 else 1 for i, d in enumerate(x.shape)]
    return freqs_cis.view(*shape)


def apply_rotary_enc(xq, xk, freqs_cis, repeat_freqs_k=False):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2)) if xk.shape[-2] != 0 else None
    freqs_cis = reshape_for_broadcast(freqs_cis, xq_)
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    if xk_ is None:
        return xq_out.type_as(xq), xk
    if repeat_freqs_k:
        r = xk_.shape[-2] // xq_.shape[-2]
        freqs_cis = freqs_cis.repeat(*([1] * (freqs_cis.ndim - 2)), r, 1)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)


def window_partition(x, window_size):
    B, H, W, C = x.shape
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size
    if pad_h > 0 or pad_w > 0:
        x = F.pad(x, (0, 0, 0, pad_w, 0, pad_h))
    Hp, Wp = H + pad_h, W + pad_w
    x = x.view(B, Hp // window_size, window_size, Wp // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).reshape(-1, window_size, window_size, C)
    return windows, (Hp, Wp)


def window_unpartition(windows, window_size, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // window_size // window_size)
    x = windows.reshape(B, Hp // window_size, Wp // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)
    if Hp > H or Wp > W:
        x = x[:, :H, :W, :]
    return x


def get_abs_pos(abs_pos, has_cls_token, hw, tiling=False):
    h, w = hw
    if has_cls_token:
        abs_pos = abs_pos[:, 1:]
    xy_num = abs_pos.shape[1]
    size = int(math.sqrt(xy_num))
    if size != h or size != w:
        new_abs_pos = abs_pos.reshape(1, size, size, -1).permute(0, 3, 1, 2)
        if tiling:
            new_abs_pos = new_abs_pos.tile(
                [1, 1] + [x // y + 1 for x, y in zip((h, w), new_abs_pos.shape[2:])]
            )[:, :, :h, :w]
        else:
            new_abs_pos = F.interpolate(new_abs_pos, size=(h, w), mode="bicubic", align_corners=False)
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)


# ── Sinusoidal PE (from position_encoding.py) ────────────────────────────

def sinusoidal_pe_2d(H, W, num_pos_feats=256, temperature=10000, scale=2*math.pi):
    """Match PositionEmbeddingSine.forward() exactly."""
    half = num_pos_feats // 2  # 128

    y_embed = torch.arange(1, H + 1, dtype=torch.float32).view(-1, 1).repeat(1, W)
    x_embed = torch.arange(1, W + 1, dtype=torch.float32).view(1, -1).repeat(H, 1)

    # Normalize
    eps = 1e-6
    y_embed = y_embed / (y_embed[-1:, :] + eps) * scale
    x_embed = x_embed / (x_embed[:, -1:] + eps) * scale

    dim_t = torch.arange(half, dtype=torch.float32)
    dim_t = temperature ** (2 * (dim_t // 2) / half)

    pos_x = x_embed[:, :, None] / dim_t  # [H, W, half]
    pos_y = y_embed[:, :, None] / dim_t

    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)

    # [H, W, 256] → [1, 256, H, W]
    pos = torch.cat((pos_y, pos_x), dim=2).permute(2, 0, 1).unsqueeze(0)
    return pos


# ═══════════════════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="tests/ref_phase3")
    parser.add_argument("--blocks", default="0,1,7,15,23,31",
                        help="Comma-separated block indices to dump detailed intermediates")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"
    dump_blocks = set(int(b) for b in args.blocks.split(","))

    # ── Load checkpoint ─────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    vit_prefix = "detector.backbone.vision_backbone.trunk."
    vit_sd = {k[len(vit_prefix):]: v for k, v in ckpt.items() if k.startswith(vit_prefix)}
    print(f"  Found {len(vit_sd)} ViT keys")

    # ── Step 3.4: RoPE ─────────────────────────────────────────────────
    print("\n=== RoPE Frequencies ===")

    # Window RoPE (24x24, scale_pos=1.0)
    window_freqs = compute_axial_cis(dim=64, end_x=24, end_y=24, theta=10000.0, scale_pos=1.0)
    # Store as real: [576, 32, 2] (cos, sin)
    window_real = torch.view_as_real(window_freqs)
    save_tensor(os.path.join(args.outdir, "rope_window_real"), window_real)

    # Global RoPE (72x72, scale_pos=24/72=1/3)
    global_freqs = compute_axial_cis(dim=64, end_x=72, end_y=72, theta=10000.0, scale_pos=24.0/72.0)
    global_real = torch.view_as_real(global_freqs)
    save_tensor(os.path.join(args.outdir, "rope_global_real"), global_real)

    # Also dump the checkpoint's freqs_cis for block 0 (window) and block 7 (global)
    for bi in [0, 7]:
        key = f"blocks.{bi}.attn.freqs_cis"
        if key in vit_sd:
            fc = vit_sd[key]
            fc_real = torch.view_as_real(fc) if fc.is_complex() else fc
            save_tensor(os.path.join(args.outdir, f"ckpt_freqs_cis_block{bi}"), fc_real)

    # ── Step 3.9: Sinusoidal PE ─────────────────────────────────────────
    print("\n=== Sinusoidal PE ===")
    for S, name in [(288, "pe_288"), (144, "pe_144"), (72, "pe_72"), (36, "pe_36")]:
        pe = sinusoidal_pe_2d(S, S, num_pos_feats=256)
        save_tensor(os.path.join(args.outdir, name), pe)

    # ── Load and preprocess image ─────────────────────────────────────
    print(f"\n=== Preprocessing: {args.image} ===")
    img = Image.open(args.image).convert("RGB")
    transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    img_tensor = v2.functional.to_image(img)
    img_preprocessed = transform(img_tensor).unsqueeze(0).to(device)
    save_tensor(os.path.join(args.outdir, "preprocessed"), img_preprocessed)

    # Also save the raw CHW float image before normalization (for debugging preprocessing)
    raw_transform = v2.Compose([
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(1008, 1008)),
        v2.ToDtype(torch.float32, scale=True),
    ])
    img_raw_float = raw_transform(v2.functional.to_image(img)).unsqueeze(0)
    save_tensor(os.path.join(args.outdir, "preprocessed_raw_float"), img_raw_float)

    # ── Manual ViT forward pass ───────────────────────────────────────
    print("\n=== ViT Forward Pass ===")
    with torch.no_grad():
        E = 1024
        NH = 16
        HD = 64
        depth = 32
        MLP_DIM = 4736
        WS = 24
        global_blocks = {7, 15, 23, 31}

        # Step 3.2: Patch embedding
        patch_w = vit_sd["patch_embed.proj.weight"]  # [1024, 3, 14, 14]
        x = F.conv2d(img_preprocessed, patch_w, stride=14)  # [1, 1024, 72, 72]
        x = x.permute(0, 2, 3, 1)  # [1, 72, 72, 1024]
        save_tensor(os.path.join(args.outdir, "patch_embed"), x)

        # Step 3.3: Positional embedding (tiled)
        pos_embed = vit_sd["pos_embed"]  # [1, 577, 1024]
        pos = get_abs_pos(pos_embed, has_cls_token=True, hw=(72, 72), tiling=True)
        save_tensor(os.path.join(args.outdir, "pos_embed_tiled"), pos)
        x = x + pos
        save_tensor(os.path.join(args.outdir, "after_pos_embed"), x)

        # ln_pre
        ln_pre_w = vit_sd["ln_pre.weight"]
        ln_pre_b = vit_sd["ln_pre.bias"]
        x = F.layer_norm(x, [E], ln_pre_w, ln_pre_b, eps=1e-5)
        save_tensor(os.path.join(args.outdir, "after_ln_pre"), x)

        # Step 3.6 & 3.7: Transformer blocks
        for blk_idx in range(depth):
            prefix = f"blocks.{blk_idx}."
            is_global = blk_idx in global_blocks
            ws = 0 if is_global else WS
            dump_this = blk_idx in dump_blocks
            blk_prefix = f"block_{blk_idx}"

            # Pre-norm
            n1_w = vit_sd[prefix + "norm1.weight"]
            n1_b = vit_sd[prefix + "norm1.bias"]
            shortcut = x
            xn = F.layer_norm(x, [E], n1_w, n1_b, eps=1e-5)

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_after_norm1"), xn)

            # Window partition
            if ws > 0:
                H, W = xn.shape[1], xn.shape[2]
                xn, pad_hw = window_partition(xn, ws)
                if dump_this:
                    save_tensor(os.path.join(args.outdir, f"{blk_prefix}_after_winpart"), xn)

            # Attention
            B_cur, Hc, Wc, _ = xn.shape
            L = Hc * Wc
            qkv_w = vit_sd[prefix + "attn.qkv.weight"]
            qkv_b = vit_sd[prefix + "attn.qkv.bias"]
            qkv = F.linear(xn, qkv_w, qkv_b).reshape(B_cur, L, 3, NH, HD)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # [B, NH, L, HD]

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_q_pre_rope"), q)
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_k_pre_rope"), k)

            # RoPE
            freqs_key = prefix + "attn.freqs_cis"
            if freqs_key in vit_sd:
                freqs = vit_sd[freqs_key]
                q, k = apply_rotary_enc(q, k, freqs)

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_q_post_rope"), q)
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_k_post_rope"), k)
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_v"), v)

            # Scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.view(B_cur, NH, Hc, Wc, HD).permute(0, 2, 3, 1, 4).reshape(B_cur, Hc, Wc, E)

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_attn_out"), attn_out)

            # Output projection
            proj_w = vit_sd[prefix + "attn.proj.weight"]
            proj_b = vit_sd[prefix + "attn.proj.bias"]
            attn_out = F.linear(attn_out, proj_w, proj_b)

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_proj_out"), attn_out)

            # Window unpartition
            if ws > 0:
                attn_out = window_unpartition(attn_out, ws, pad_hw, (H, W))

            # Residual
            x = shortcut + attn_out

            if dump_this:
                save_tensor(os.path.join(args.outdir, f"{blk_prefix}_after_attn_residual"), x)

            # FFN
            n2_w = vit_sd[prefix + "norm2.weight"]
            n2_b = vit_sd[prefix + "norm2.bias"]
            shortcut = x
            xn = F.layer_norm(x, [E], n2_w, n2_b, eps=1e-5)

            fc1_w = vit_sd[prefix + "mlp.fc1.weight"]
            fc1_b = vit_sd[prefix + "mlp.fc1.bias"]
            fc2_w = vit_sd[prefix + "mlp.fc2.weight"]
            fc2_b = vit_sd[prefix + "mlp.fc2.bias"]
            h = F.linear(xn, fc1_w, fc1_b)
            h = F.gelu(h)
            h = F.linear(h, fc2_w, fc2_b)

            x = shortcut + h

            # Always save per-block output for selected blocks
            save_tensor(os.path.join(args.outdir, f"block_{blk_idx}_out"), x)

        # Final output
        save_tensor(os.path.join(args.outdir, "vit_final"), x)
        feats = x.permute(0, 3, 1, 2)  # [1, 1024, 72, 72]
        save_tensor(os.path.join(args.outdir, "vit_output_bchw"), feats)

    # ── Neck (SimpleFPN) — detector path ──────────────────────────────
    print("\n=== Neck (Detector) ===")
    neck_prefix = "detector.backbone.vision_backbone.convs."
    neck_sd = {k[len(neck_prefix):]: v for k, v in ckpt.items() if k.startswith(neck_prefix)}

    with torch.no_grad():
        # Scale 0 (4x)
        s0 = F.conv_transpose2d(feats, neck_sd["0.dconv_2x2_0.weight"], neck_sd["0.dconv_2x2_0.bias"], stride=2)
        save_tensor(os.path.join(args.outdir, "neck_det_s0_deconv1"), s0)
        s0 = F.gelu(s0)
        save_tensor(os.path.join(args.outdir, "neck_det_s0_gelu"), s0)
        s0 = F.conv_transpose2d(s0, neck_sd["0.dconv_2x2_1.weight"], neck_sd["0.dconv_2x2_1.bias"], stride=2)
        save_tensor(os.path.join(args.outdir, "neck_det_s0_deconv2"), s0)
        s0 = F.conv2d(s0, neck_sd["0.conv_1x1.weight"], neck_sd["0.conv_1x1.bias"])
        save_tensor(os.path.join(args.outdir, "neck_det_s0_conv1x1"), s0)
        s0 = F.conv2d(s0, neck_sd["0.conv_3x3.weight"], neck_sd["0.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_0"), s0)

        # Scale 1 (2x)
        s1 = F.conv_transpose2d(feats, neck_sd["1.dconv_2x2.weight"], neck_sd["1.dconv_2x2.bias"], stride=2)
        save_tensor(os.path.join(args.outdir, "neck_det_s1_deconv"), s1)
        s1 = F.conv2d(s1, neck_sd["1.conv_1x1.weight"], neck_sd["1.conv_1x1.bias"])
        save_tensor(os.path.join(args.outdir, "neck_det_s1_conv1x1"), s1)
        s1 = F.conv2d(s1, neck_sd["1.conv_3x3.weight"], neck_sd["1.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_1"), s1)

        # Scale 2 (1x)
        s2 = F.conv2d(feats, neck_sd["2.conv_1x1.weight"], neck_sd["2.conv_1x1.bias"])
        save_tensor(os.path.join(args.outdir, "neck_det_s2_conv1x1"), s2)
        s2 = F.conv2d(s2, neck_sd["2.conv_3x3.weight"], neck_sd["2.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_2"), s2)

        # Scale 3 (0.5x)
        s3 = F.max_pool2d(feats, kernel_size=2, stride=2)
        save_tensor(os.path.join(args.outdir, "neck_det_s3_pool"), s3)
        s3 = F.conv2d(s3, neck_sd["3.conv_1x1.weight"], neck_sd["3.conv_1x1.bias"])
        save_tensor(os.path.join(args.outdir, "neck_det_s3_conv1x1"), s3)
        s3 = F.conv2d(s3, neck_sd["3.conv_3x3.weight"], neck_sd["3.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_3"), s3)

    # ── Neck (SimpleFPN) — tracker path ──────────────────────────────
    print("\n=== Neck (Tracker) ===")
    trk_prefix = "detector.backbone.vision_backbone.sam2_convs."
    trk_sd = {k[len(trk_prefix):]: v for k, v in ckpt.items() if k.startswith(trk_prefix)}

    if trk_sd:
        with torch.no_grad():
            s0t = F.conv_transpose2d(feats, trk_sd["0.dconv_2x2_0.weight"], trk_sd["0.dconv_2x2_0.bias"], stride=2)
            s0t = F.gelu(s0t)
            s0t = F.conv_transpose2d(s0t, trk_sd["0.dconv_2x2_1.weight"], trk_sd["0.dconv_2x2_1.bias"], stride=2)
            s0t = F.conv2d(s0t, trk_sd["0.conv_1x1.weight"], trk_sd["0.conv_1x1.bias"])
            s0t = F.conv2d(s0t, trk_sd["0.conv_3x3.weight"], trk_sd["0.conv_3x3.bias"], padding=1)
            save_tensor(os.path.join(args.outdir, "neck_trk_0"), s0t)

            s1t = F.conv_transpose2d(feats, trk_sd["1.dconv_2x2.weight"], trk_sd["1.dconv_2x2.bias"], stride=2)
            s1t = F.conv2d(s1t, trk_sd["1.conv_1x1.weight"], trk_sd["1.conv_1x1.bias"])
            s1t = F.conv2d(s1t, trk_sd["1.conv_3x3.weight"], trk_sd["1.conv_3x3.bias"], padding=1)
            save_tensor(os.path.join(args.outdir, "neck_trk_1"), s1t)

            s2t = F.conv2d(feats, trk_sd["2.conv_1x1.weight"], trk_sd["2.conv_1x1.bias"])
            s2t = F.conv2d(s2t, trk_sd["2.conv_3x3.weight"], trk_sd["2.conv_3x3.bias"], padding=1)
            save_tensor(os.path.join(args.outdir, "neck_trk_2"), s2t)

            s3t = F.max_pool2d(feats, kernel_size=2, stride=2)
            s3t = F.conv2d(s3t, trk_sd["3.conv_1x1.weight"], trk_sd["3.conv_1x1.bias"])
            s3t = F.conv2d(s3t, trk_sd["3.conv_3x3.weight"], trk_sd["3.conv_3x3.bias"], padding=1)
            save_tensor(os.path.join(args.outdir, "neck_trk_3"), s3t)
    else:
        print("  No tracker neck weights found (sam2_convs); skipping.")

    print("\n=== Done! ===")
    print(f"All tensors saved to: {args.outdir}")


if __name__ == "__main__":
    main()
