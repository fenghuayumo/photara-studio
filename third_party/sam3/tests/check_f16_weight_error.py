#!/usr/bin/env python3
"""Check f16 quantization error on ViT weights, focusing on dimension 679.

Also: run the same ViT forward pass in Python with f16 weights to see if
PyTorch shows the same amplification. This tells us if the issue is inherent
to f16 or a C++ bug.
"""
import numpy as np
import torch
import os

# Load checkpoint
print("Loading checkpoint...")
ckpt = torch.load("raw_weights/sam3.pt", map_location="cpu", weights_only=False)
if "model" in ckpt:
    ckpt = ckpt["model"]

vit_prefix = "detector.backbone.vision_backbone.trunk."

# Check f16 quantization error on specific block weights
print("\n═══ F16 Quantization Error on ViT Weights ═══\n")
print(f"{'Weight':50s} {'MaxRelErr':>12} {'MaxAbsErr':>12} {'ErrAt679':>12} {'MaxVal':>12}")
print("-" * 100)

for blk in [0, 7, 14, 15, 16, 31]:
    for suffix in ["attn.qkv.weight", "attn.qkv.bias", "attn.proj.weight",
                   "mlp.fc1.weight", "mlp.fc2.weight"]:
        key = f"{vit_prefix}blocks.{blk}.{suffix}"
        if key not in ckpt:
            continue
        w = ckpt[key].float()
        w_f16 = w.half().float()
        err = (w - w_f16).abs()
        max_abs = err.max().item()
        max_rel = (err / (w.abs() + 1e-12)).max().item()
        max_val = w.abs().max().item()

        # Check dimension 679 specifically
        if w.ndim == 2 and w.shape[1] > 679:
            err_679 = err[:, 679].max().item()
        elif w.ndim == 2 and w.shape[0] > 679:
            err_679 = err[679, :].max().item()
        elif w.ndim == 1 and w.shape[0] > 679:
            err_679 = err[679].item()
        else:
            err_679 = 0.0

        name = f"blk{blk}.{suffix}"
        print(f"  {name:48s} {max_rel:>12.6e} {max_abs:>12.6e} {err_679:>12.6e} {max_val:>12.4f}")

# ═══ Compare: Python f32 vs Python f16 ViT forward ═══
print("\n═══ Python f32 vs f16 ViT Forward Pass ═══")
print("(This tells us if the error is inherent to f16 or a C++ bug)\n")

# Load the preprocessed image
ref_dir = "tests/ref_phase3"
img_data = np.fromfile(f"{ref_dir}/preprocessed.bin", dtype=np.float32)
img = torch.from_numpy(img_data).reshape(1, 3, 1008, 1008)

# Extract ViT weights
vit_sd = {k[len(vit_prefix):]: v for k, v in ckpt.items() if k.startswith(vit_prefix)}

# Run ViT forward pass with f32 and f16 weights, compare outputs
# We'll just check the QKV output at block 15 and the block output

from torchvision.transforms import v2
import torch.nn.functional as F
import math

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
        return new_abs_pos.permute(0, 2, 3, 1)
    return abs_pos.reshape(1, h, w, -1)

def apply_rotary_enc(xq, xk, freqs_cis):
    xq_ = torch.view_as_complex(xq.float().reshape(*xq.shape[:-1], -1, 2))
    xk_ = torch.view_as_complex(xk.float().reshape(*xk.shape[:-1], -1, 2))
    freqs_cis = freqs_cis.view(*[d if i >= xq_.ndim - 2 else 1 for i, d in enumerate(xq_.shape)])
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(3)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(3)
    return xq_out.type_as(xq), xk_out.type_as(xk)

def window_partition(x, ws):
    B, H, W, C = x.shape
    x = x.view(B, H // ws, ws, W // ws, ws, C)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(-1, ws, ws, C), (H, W)

def window_unpartition(windows, ws, pad_hw, hw):
    Hp, Wp = pad_hw
    H, W = hw
    B = windows.shape[0] // (Hp * Wp // ws // ws)
    x = windows.reshape(B, Hp // ws, Wp // ws, ws, ws, -1)
    return x.permute(0, 1, 3, 2, 4, 5).reshape(B, Hp, Wp, -1)[:, :H, :W, :]

def run_vit(vit_sd, img, use_f16_weights=False):
    """Run ViT forward pass. If use_f16_weights, quantize all 2D weights to f16."""
    E, NH, HD, WS = 1024, 16, 64, 24
    global_blocks = {7, 15, 23, 31}

    # Optionally quantize weights to f16
    sd = {}
    for k, v in vit_sd.items():
        if use_f16_weights and v.ndim >= 2:
            sd[k] = v.half().float()  # f32 → f16 → f32 roundtrip
        else:
            sd[k] = v.float()

    with torch.no_grad():
        patch_w = sd["patch_embed.proj.weight"]
        x = F.conv2d(img, patch_w, stride=14).permute(0, 2, 3, 1)
        pos = get_abs_pos(sd["pos_embed"], True, (72, 72), tiling=True)
        x = x + pos
        x = F.layer_norm(x, [E], sd["ln_pre.weight"], sd["ln_pre.bias"])

        block_outs = {}
        for blk_idx in range(32):
            prefix = f"blocks.{blk_idx}."
            is_global = blk_idx in global_blocks
            ws = 0 if is_global else WS
            shortcut = x
            xn = F.layer_norm(x, [E], sd[prefix + "norm1.weight"], sd[prefix + "norm1.bias"])
            if ws > 0:
                H, W = xn.shape[1], xn.shape[2]
                xn, pad_hw = window_partition(xn, ws)
            B_cur, Hc, Wc, _ = xn.shape
            L = Hc * Wc
            qkv = F.linear(xn, sd[prefix + "attn.qkv.weight"], sd[prefix + "attn.qkv.bias"])
            qkv = qkv.reshape(B_cur, L, 3, NH, HD)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            freqs_key = prefix + "attn.freqs_cis"
            if freqs_key in sd:
                q, k = apply_rotary_enc(q, k, sd[freqs_key])
            attn_out = F.scaled_dot_product_attention(q, k, v)
            attn_out = attn_out.view(B_cur, NH, Hc, Wc, HD).permute(0, 2, 3, 1, 4).reshape(B_cur, Hc, Wc, E)
            attn_out = F.linear(attn_out, sd[prefix + "attn.proj.weight"], sd[prefix + "attn.proj.bias"])
            if ws > 0:
                attn_out = window_unpartition(attn_out, ws, pad_hw, (H, W))
            x = shortcut + attn_out
            shortcut = x
            xn = F.layer_norm(x, [E], sd[prefix + "norm2.weight"], sd[prefix + "norm2.bias"])
            h = F.gelu(F.linear(xn, sd[prefix + "mlp.fc1.weight"], sd[prefix + "mlp.fc1.bias"]))
            h = F.linear(h, sd[prefix + "mlp.fc2.weight"], sd[prefix + "mlp.fc2.bias"])
            x = shortcut + h
            block_outs[blk_idx] = x.clone()

    return block_outs

print("Running f32 forward pass...")
outs_f32 = run_vit(vit_sd, img, use_f16_weights=False)
print("Running f16-weights forward pass...")
outs_f16 = run_vit(vit_sd, img, use_f16_weights=True)

print(f"\n{'Block':>6} {'MaxErr':>12} {'MeanErr':>12} {'Cosine':>14}")
print("-" * 50)
for blk in sorted(outs_f32.keys()):
    a = outs_f32[blk].flatten().numpy()
    b = outs_f16[blk].flatten().numpy()
    diff = np.abs(a - b)
    cos = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    print(f"{blk:>6} {diff.max():>12.6e} {diff.mean():>12.6e} {cos:>14.10f}")

# Check the worst element at dimension 679
print(f"\n═══ Worst Element (dim 679) in Python f16 ═══")
for blk in [14, 15, 16, 23, 30]:
    a = outs_f32[blk][0].numpy()  # [72, 72, 1024]
    b = outs_f16[blk][0].numpy()
    diff = np.abs(a - b)
    worst = np.unravel_index(np.argmax(diff), diff.shape)
    print(f"  Block {blk:2d}: worst at (h={worst[0]:2d}, w={worst[1]:2d}, e={worst[2]:4d})  "
          f"f32={a[worst]:>10.4f}  f16={b[worst]:>10.4f}  diff={diff[worst]:>8.4f}")
