#!/usr/bin/env python3
"""
Dump ViT backbone intermediate tensors from the SAM3 checkpoint for C++ verification.
Uses inlined functions to avoid heavy SAM3 dependencies.

Usage:
    uv run python tests/dump_vit_reference.py --checkpoint raw_weights/sam3.pt --image tests/test_random.jpg
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


# ── Minimal ViT block implementation ──────────────────────────────────────

try:
    from timm.layers import Mlp
except ImportError:
    class Mlp(nn.Module):
        def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
            super().__init__()
            out_features = out_features or in_features
            hidden_features = hidden_features or in_features
            self.fc1 = nn.Linear(in_features, hidden_features)
            self.act = act_layer()
            self.fc2 = nn.Linear(hidden_features, out_features)
        def forward(self, x):
            return self.fc2(self.act(self.fc1(x)))


def save_tensor(path, t):
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="tests/ref")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"

    # ── Load checkpoint ─────────────────────────────────────────────────
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    # Extract ViT weights
    vit_prefix = "detector.backbone.vision_backbone.trunk."
    vit_sd = {k[len(vit_prefix):]: v for k, v in ckpt.items() if k.startswith(vit_prefix)}
    print(f"  Found {len(vit_sd)} ViT keys")

    # ── Load and preprocess image ─────────────────────────────────────
    print(f"Loading image: {args.image}")
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

    # ── Manual ViT forward pass ───────────────────────────────────────
    with torch.no_grad():
        # Config
        E = 1024
        NH = 16
        HD = 64
        depth = 32
        MLP_DIM = 4736
        WS = 24
        global_blocks = {7, 15, 23, 31}

        # 1. Patch embedding: Conv2d(3, 1024, 14, 14, bias=False)
        patch_w = vit_sd["patch_embed.proj.weight"]  # [1024, 3, 14, 14]
        x = F.conv2d(img_preprocessed, patch_w, stride=14)  # [1, 1024, 72, 72]
        x = x.permute(0, 2, 3, 1)  # [1, 72, 72, 1024]
        save_tensor(os.path.join(args.outdir, "patch_embed"), x)

        # 2. Positional embedding (tiled)
        pos_embed = vit_sd["pos_embed"]  # [1, 577, 1024]
        pos = get_abs_pos(pos_embed, has_cls_token=True, hw=(72, 72), tiling=True)
        save_tensor(os.path.join(args.outdir, "pos_embed_tiled"), pos)
        x = x + pos
        save_tensor(os.path.join(args.outdir, "after_pos_embed"), x)

        # 3. ln_pre
        ln_pre_w = vit_sd["ln_pre.weight"]
        ln_pre_b = vit_sd["ln_pre.bias"]
        x = F.layer_norm(x, [E], ln_pre_w, ln_pre_b, eps=1e-5)
        save_tensor(os.path.join(args.outdir, "after_ln_pre"), x)

        # 4. Transformer blocks
        for blk_idx in range(depth):
            prefix = f"blocks.{blk_idx}."
            is_global = blk_idx in global_blocks
            ws = 0 if is_global else WS

            # Pre-norm
            n1_w = vit_sd[prefix + "norm1.weight"]
            n1_b = vit_sd[prefix + "norm1.bias"]
            shortcut = x
            xn = F.layer_norm(x, [E], n1_w, n1_b, eps=1e-5)

            # Window partition
            if ws > 0:
                H, W = xn.shape[1], xn.shape[2]
                xn, pad_hw = window_partition(xn, ws)

            # Attention
            B_cur, Hc, Wc, _ = xn.shape
            L = Hc * Wc
            qkv_w = vit_sd[prefix + "attn.qkv.weight"]  # [3*E, E]
            qkv_b = vit_sd[prefix + "attn.qkv.bias"]    # [3*E]
            qkv = F.linear(xn, qkv_w, qkv_b).reshape(B_cur, L, 3, NH, HD)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)  # each [B, NH, L, HD]

            # RoPE
            freqs_key = prefix + "attn.freqs_cis"
            if freqs_key in vit_sd:
                freqs = vit_sd[freqs_key]
                q, k = apply_rotary_enc(q, k, freqs)

            # Scaled dot-product attention
            attn_out = F.scaled_dot_product_attention(q, k, v)
            # [B, NH, L, HD] → [B, H, W, E]
            attn_out = attn_out.view(B_cur, NH, Hc, Wc, HD).permute(0, 2, 3, 1, 4).reshape(B_cur, Hc, Wc, E)

            # Output projection
            proj_w = vit_sd[prefix + "attn.proj.weight"]
            proj_b = vit_sd[prefix + "attn.proj.bias"]
            attn_out = F.linear(attn_out, proj_w, proj_b)

            # Window unpartition
            if ws > 0:
                attn_out = window_unpartition(attn_out, ws, pad_hw, (H, W))

            # Residual
            x = shortcut + attn_out

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

            if blk_idx == 0:
                save_tensor(os.path.join(args.outdir, "block_0_out"), x)
            if blk_idx == 7:
                save_tensor(os.path.join(args.outdir, "block_7_out"), x)

        # Final output (x is [1, 72, 72, 1024])
        save_tensor(os.path.join(args.outdir, "vit_final"), x)

        # Permute to [B, C, H, W]
        feats = x.permute(0, 3, 1, 2)  # [1, 1024, 72, 72]
        save_tensor(os.path.join(args.outdir, "vit_output_bchw"), feats)

    # ── Neck (SimpleFPN) ──────────────────────────────────────────────
    print("\nDumping neck outputs...")
    neck_prefix = "detector.backbone.vision_backbone.convs."
    neck_sd = {k[len(neck_prefix):]: v for k, v in ckpt.items() if k.startswith(neck_prefix)}

    with torch.no_grad():
        # Scale 0 (4x): ConvTranspose(1024→512, k=2, s=2) → GELU → ConvTranspose(512→256, k=2, s=2) → Conv1x1 → Conv3x3
        s0 = F.conv_transpose2d(feats, neck_sd["0.dconv_2x2_0.weight"], neck_sd["0.dconv_2x2_0.bias"], stride=2)
        s0 = F.gelu(s0)
        s0 = F.conv_transpose2d(s0, neck_sd["0.dconv_2x2_1.weight"], neck_sd["0.dconv_2x2_1.bias"], stride=2)
        s0 = F.conv2d(s0, neck_sd["0.conv_1x1.weight"], neck_sd["0.conv_1x1.bias"])
        s0 = F.conv2d(s0, neck_sd["0.conv_3x3.weight"], neck_sd["0.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_0"), s0)

        # Scale 1 (2x)
        s1 = F.conv_transpose2d(feats, neck_sd["1.dconv_2x2.weight"], neck_sd["1.dconv_2x2.bias"], stride=2)
        s1 = F.conv2d(s1, neck_sd["1.conv_1x1.weight"], neck_sd["1.conv_1x1.bias"])
        s1 = F.conv2d(s1, neck_sd["1.conv_3x3.weight"], neck_sd["1.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_1"), s1)

        # Scale 2 (1x)
        s2 = F.conv2d(feats, neck_sd["2.conv_1x1.weight"], neck_sd["2.conv_1x1.bias"])
        s2 = F.conv2d(s2, neck_sd["2.conv_3x3.weight"], neck_sd["2.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_2"), s2)

        # Scale 3 (0.5x)
        s3 = F.max_pool2d(feats, kernel_size=2, stride=2)
        s3 = F.conv2d(s3, neck_sd["3.conv_1x1.weight"], neck_sd["3.conv_1x1.bias"])
        s3 = F.conv2d(s3, neck_sd["3.conv_3x3.weight"], neck_sd["3.conv_3x3.bias"], padding=1)
        save_tensor(os.path.join(args.outdir, "neck_det_3"), s3)

    print("\nDone!")


if __name__ == "__main__":
    main()
