#!/usr/bin/env python3
"""
Dump geometry encoder reference tensors for numerical comparison against sam3.cpp.

Tests the SequenceGeometryEncoder with exemplar box prompts.
Dumps intermediate tensors at each stage:
  1. Input box coordinates (CxCyWH normalized)
  2. Direct box projection output
  3. ROI align pooled features
  4. Box positional encoding
  5. Label embedding
  6. Combined box embedding
  7. CLS token
  8. Final projection output
  9. Per-layer transformer outputs
  10. Final output

Also tests the dummy prompt case (no exemplars, just CLS token).

Usage:
  cd ~/Documents/sam3
  uv run python /Users/pierre-antoine/Documents/sam3.cpp/tests/dump_geom_encoder_reference.py \
      --checkpoint raw_weights/sam3.pt \
      --prephase-ref /Users/pierre-antoine/Documents/sam3.cpp/tests/ref_phase3 \
      --outdir /Users/pierre-antoine/Documents/sam3.cpp/tests/ref_geom
"""

import argparse
import math
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
import torchvision

# ── Tensor save utilities ───────────────────────────────────────────────────

def save_raw(path: str, arr: np.ndarray, shape) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(arr.astype(np.float32, copy=False).tobytes())
    with open(path + ".shape", "w", encoding="utf-8") as f:
        f.write(",".join(str(int(d)) for d in shape))


def save_ggml_bnd(path: str, x_bnd: torch.Tensor) -> None:
    """Save [B, N, D] as ggml [D, N, B]."""
    assert x_bnd.ndim == 3
    x = x_bnd.detach().cpu().float().contiguous()
    b, n, d = x.shape
    save_raw(path, x.numpy(), (d, n, b))


def save_ggml_sbd(path: str, x_sbd: torch.Tensor) -> None:
    """Save [S, B, D] (seq-first) as ggml [D, S, B]."""
    assert x_sbd.ndim == 3
    x = x_sbd.detach().cpu().float().contiguous()
    s, b, d = x.shape
    # Permute to [B, S, D] then save as ggml [D, S, B]
    x_bsd = x.permute(1, 0, 2).contiguous()
    save_raw(path, x_bsd.numpy(), (d, s, b))


def save_ggml_nchw(path: str, x_nchw: torch.Tensor) -> None:
    """Save [N, C, H, W] as ggml [C, W, H, N]."""
    assert x_nchw.ndim == 4
    x = x_nchw.detach().cpu().float().contiguous()
    n, c, h, w = x.shape
    x_nhwc = x.permute(0, 2, 3, 1).contiguous()
    save_raw(path, x_nhwc.numpy(), (c, w, h, n))


def save_ggml_1d(path: str, x: torch.Tensor) -> None:
    """Save a 1D tensor."""
    x = x.detach().cpu().float().contiguous()
    save_raw(path, x.numpy(), tuple(x.shape))


def load_tensor(path: str) -> torch.Tensor:
    with open(path + ".shape", "r") as f:
        shape = [int(x) for x in f.read().strip().split(",") if x]
    data = np.fromfile(path + ".bin", dtype=np.float32).reshape(shape)
    return torch.from_numpy(data)


# ── Sinusoidal position encoding (matches PositionEmbeddingSine) ──────────

def sine_encode_xy(x: torch.Tensor, y: torch.Tensor, num_pos_feats: int = 128,
                    temperature: int = 10000, scale: float = 2 * math.pi):
    """Encode (x, y) with sinusoidal PE. x, y: [N] normalized to [0,1]."""
    x_embed = x * scale
    y_embed = y * scale

    dim_t = torch.arange(num_pos_feats, dtype=torch.float32, device=x.device)
    dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

    pos_x = x_embed[:, None] / dim_t  # [N, 128]
    pos_y = y_embed[:, None] / dim_t  # [N, 128]
    pos_x = torch.stack((pos_x[:, 0::2].sin(), pos_x[:, 1::2].cos()), dim=2).flatten(1)  # [N, 128]
    pos_y = torch.stack((pos_y[:, 0::2].sin(), pos_y[:, 1::2].cos()), dim=2).flatten(1)  # [N, 128]
    return pos_x, pos_y


def sine_encode_boxes(cx, cy, w, h, num_pos_feats=128, temperature=10000):
    """Encode box (cx, cy, w, h) with sinusoidal PE. Returns [N, 258]."""
    pos_x, pos_y = sine_encode_xy(cx, cy, num_pos_feats, temperature)
    pos = torch.cat((pos_y, pos_x, h[:, None], w[:, None]), dim=1)  # [N, 258]
    return pos


# ── Box coordinate conversion ──────────────────────────────────────────────

def box_cxcywh_to_xyxy(x):
    """Convert CxCyWH to XYXY format."""
    cx, cy, w, h = x.unbind(-1)
    x0 = cx - 0.5 * w
    y0 = cy - 0.5 * h
    x1 = cx + 0.5 * w
    y1 = cy + 0.5 * h
    return torch.stack([x0, y0, x1, y1], dim=-1)


# ── ROI Align implementation ────────────────────────────────────────────────

def do_roi_align(img_feats_nchw, boxes_cxcywh, roi_size=7):
    """
    ROI Align from image features using box coordinates in CxCyWH [0,1] format.

    Args:
        img_feats_nchw: [B, C, H, W] backbone feature map
        boxes_cxcywh: [N_boxes, B, 4] in normalized CxCyWH format
        roi_size: output spatial size

    Returns:
        roi_pooled: [N_boxes*B, C, roi_size, roi_size]
    """
    n_boxes, bs, _ = boxes_cxcywh.shape
    H, W = img_feats_nchw.shape[-2:]

    boxes_xyxy = box_cxcywh_to_xyxy(boxes_cxcywh)
    scale = torch.tensor([W, H, W, H], dtype=boxes_xyxy.dtype, device=boxes_xyxy.device)
    scale = scale.view(1, 1, 4)
    boxes_xyxy = boxes_xyxy * scale

    # roi_align expects list of [N_boxes, 4] per batch element
    rois_list = boxes_xyxy.float().transpose(0, 1).unbind(0)
    sampled = torchvision.ops.roi_align(img_feats_nchw, rois_list, roi_size)
    return sampled  # [B*N_boxes, C, roi_size, roi_size]


# ── Main dump logic ───────────────────────────────────────────────────────

def dump_geometry_encoder(
    ckpt: Dict[str, torch.Tensor],
    img_feats_hwc: torch.Tensor,       # [H*W, B, C] seq-first image features
    img_feats_nchw: torch.Tensor,      # [B, C, H, W] image features for ROI align
    img_pe_hwc: torch.Tensor,          # [H*W, B, C] image positional encoding
    boxes_cxcywh: torch.Tensor,        # [N_boxes, B, 4] normalized CxCyWH [0,1]
    box_labels: torch.Tensor,          # [N_boxes, B] long (0=pos, 1=neg)
    box_mask: torch.Tensor,            # [B, N_boxes] bool (True=padded)
    outdir: str,
    case_name: str,
):
    D = 256
    roi_size = 7
    case_dir = os.path.join(outdir, case_name)
    os.makedirs(case_dir, exist_ok=True)

    n_boxes, bs = boxes_cxcywh.shape[:2]
    print(f"\n=== Geometry Encoder: {case_name} ===")
    print(f"  boxes: {n_boxes}, batch: {bs}, img_feats: {img_feats_hwc.shape}")

    # Save input coordinates
    save_raw(os.path.join(case_dir, "input_boxes_cxcywh"),
             boxes_cxcywh.detach().cpu().float().numpy(),
             boxes_cxcywh.shape)
    save_raw(os.path.join(case_dir, "input_box_labels"),
             box_labels.detach().cpu().float().numpy(),
             box_labels.shape)

    # ── 1. Direct box projection: Linear(4, 256) ───────────────────────
    box_proj_w = ckpt["detector.geometry_encoder.boxes_direct_project.weight"].float()
    box_proj_b = ckpt["detector.geometry_encoder.boxes_direct_project.bias"].float()
    boxes_direct = F.linear(boxes_cxcywh, box_proj_w, box_proj_b)  # [N, B, D]
    save_ggml_sbd(os.path.join(case_dir, "boxes_direct_proj"), boxes_direct)
    print(f"  boxes_direct_proj: {boxes_direct.shape} "
          f"mean={boxes_direct.mean():.6f} std={boxes_direct.std():.6f}")

    # ── 2. ROI Align pooled features ────────────────────────────────────
    # img_pre_norm on image features before pooling
    img_pre_norm_w = ckpt["detector.geometry_encoder.img_pre_norm.weight"].float()
    img_pre_norm_b = ckpt["detector.geometry_encoder.img_pre_norm.bias"].float()
    # Apply LayerNorm to seq-first features for pooling (operate on the NCHW version)
    img_normed_hwc = F.layer_norm(img_feats_hwc, [D], img_pre_norm_w, img_pre_norm_b)
    H, W = img_feats_nchw.shape[-2:]
    img_normed_nchw = img_normed_hwc.permute(1, 2, 0).reshape(bs, D, H, W)

    if n_boxes > 0:
        roi_pooled = do_roi_align(img_normed_nchw, boxes_cxcywh, roi_size)
        # roi_pooled: [B*N_boxes, D, roi_size, roi_size]

        # boxes_pool_project is Conv2d(D, D, roi_size)
        pool_proj_w = ckpt["detector.geometry_encoder.boxes_pool_project.weight"].float()
        pool_proj_b = ckpt["detector.geometry_encoder.boxes_pool_project.bias"].float()
        pool_proj = F.conv2d(roi_pooled, pool_proj_w, pool_proj_b)  # [B*N, D, 1, 1]
        pool_proj = pool_proj.view(bs, n_boxes, D).transpose(0, 1)  # [N, B, D]
        save_ggml_sbd(os.path.join(case_dir, "boxes_pool_proj"), pool_proj)
        print(f"  boxes_pool_proj: {pool_proj.shape} "
              f"mean={pool_proj.mean():.6f} std={pool_proj.std():.6f}")
    else:
        pool_proj = torch.zeros(0, bs, D)

    # ── 3. Box positional encoding: sinusoidal PE + Linear(258, 256) ───
    if n_boxes > 0:
        cx = boxes_cxcywh[:, :, 0].flatten()
        cy = boxes_cxcywh[:, :, 1].flatten()
        w = boxes_cxcywh[:, :, 2].flatten()
        h = boxes_cxcywh[:, :, 3].flatten()
        pos_enc = sine_encode_boxes(cx, cy, w, h)  # [N*B, 258]
        pos_enc = pos_enc.view(n_boxes, bs, 258)  # [N, B, 258]
        save_ggml_sbd(os.path.join(case_dir, "boxes_pos_enc_raw"), pos_enc)

        pos_proj_w = ckpt["detector.geometry_encoder.boxes_pos_enc_project.weight"].float()
        pos_proj_b = ckpt["detector.geometry_encoder.boxes_pos_enc_project.bias"].float()
        pos_proj = F.linear(pos_enc, pos_proj_w, pos_proj_b)  # [N, B, D]
        save_ggml_sbd(os.path.join(case_dir, "boxes_pos_proj"), pos_proj)
        print(f"  boxes_pos_proj: {pos_proj.shape} "
              f"mean={pos_proj.mean():.6f} std={pos_proj.std():.6f}")
    else:
        pos_proj = torch.zeros(0, bs, D)

    # ── 4. Label embedding ───────────────────────────────────────────────
    label_embed_w = ckpt["detector.geometry_encoder.label_embed.weight"].float()  # [2, D]
    if n_boxes > 0:
        type_embed = F.embedding(box_labels.long(), label_embed_w)  # [N, B, D]
        save_ggml_sbd(os.path.join(case_dir, "label_embed"), type_embed)
        print(f"  label_embed: {type_embed.shape}")
    else:
        type_embed = torch.zeros(0, bs, D)

    # ── 5. Combined box embedding ────────────────────────────────────────
    if n_boxes > 0:
        boxes_embed = boxes_direct + pool_proj + pos_proj + type_embed  # [N, B, D]
    else:
        boxes_embed = torch.zeros(0, bs, D)
    save_ggml_sbd(os.path.join(case_dir, "boxes_combined_embed"), boxes_embed)
    print(f"  boxes_combined: {boxes_embed.shape}")

    # ── 6. CLS token ────────────────────────────────────────────────────
    cls_embed_w = ckpt["detector.geometry_encoder.cls_embed.weight"].float()  # [1, D]
    cls = cls_embed_w.view(1, 1, D).repeat(1, bs, 1)  # [1, B, D]
    cls_mask = torch.zeros(bs, 1, dtype=torch.bool)

    # Concatenate: [boxes, CLS]
    # Inline concat_padded_sequences for simplicity (no mask padding needed here)
    if n_boxes > 0:
        final_embeds = torch.cat([boxes_embed, cls], dim=0)  # [N+1, B, D]
        final_mask = torch.cat([box_mask, cls_mask], dim=1)   # [B, N+1]
    else:
        final_embeds = cls  # [1, B, D]
        final_mask = cls_mask  # [B, 1]
    save_ggml_sbd(os.path.join(case_dir, "pre_final_proj"), final_embeds)
    print(f"  pre_final_proj: {final_embeds.shape}")

    # ── 7. Final projection + norm ──────────────────────────────────────
    final_proj_w = ckpt["detector.geometry_encoder.final_proj.weight"].float()
    final_proj_b = ckpt["detector.geometry_encoder.final_proj.bias"].float()
    norm_w = ckpt["detector.geometry_encoder.norm.weight"].float()
    norm_b = ckpt["detector.geometry_encoder.norm.bias"].float()

    final_embeds = F.layer_norm(
        F.linear(final_embeds, final_proj_w, final_proj_b),
        [D], norm_w, norm_b
    )
    save_ggml_sbd(os.path.join(case_dir, "post_final_proj"), final_embeds)
    print(f"  post_final_proj: {final_embeds.shape}")

    # ── 8. Transformer layers (3 layers) ─────────────────────────────────
    for layer_idx in range(3):
        lp = f"detector.geometry_encoder.encode.{layer_idx}."

        # Self-attention (pre-norm, pos_enc_at_attn=False)
        shortcut = final_embeds
        xn = F.layer_norm(final_embeds, [D],
                          ckpt[lp + "norm1.weight"].float(),
                          ckpt[lp + "norm1.bias"].float())

        # Self-attention with fused in_proj
        sa_w = ckpt[lp + "self_attn.in_proj_weight"].float()  # [3D, D]
        sa_b = ckpt[lp + "self_attn.in_proj_bias"].float()    # [3D]
        sa_ow = ckpt[lp + "self_attn.out_proj.weight"].float()
        sa_ob = ckpt[lp + "self_attn.out_proj.bias"].float()

        # pos_enc_at_attn=False, so Q=K=V=xn (no pos added)
        # key_padding_mask = final_mask
        S, B, _ = xn.shape
        qkv = F.linear(xn, sa_w, sa_b)  # [S, B, 3D]
        q, k, v = qkv.chunk(3, dim=-1)
        n_heads = 8
        hd = D // n_heads

        q = q.reshape(S, B, n_heads, hd).permute(1, 2, 0, 3)
        k = k.reshape(S, B, n_heads, hd).permute(1, 2, 0, 3)
        v = v.reshape(S, B, n_heads, hd).permute(1, 2, 0, 3)

        # Apply key_padding_mask
        attn_mask = None
        if final_mask.any():
            # final_mask: [B, S] True=padded -> attn_mask: [B, 1, 1, S] with -inf
            attn_mask = final_mask.unsqueeze(1).unsqueeze(2).float() * (-1e9)

        sa_out = F.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
        sa_out = sa_out.permute(2, 0, 1, 3).reshape(S, B, D)
        sa_out = F.linear(sa_out, sa_ow, sa_ob)
        final_embeds = shortcut + sa_out

        save_ggml_sbd(os.path.join(case_dir, f"layer{layer_idx}_after_sa"), final_embeds)

        # Cross-attention (pre-norm, pos_enc_at_cross_attn_keys=True)
        shortcut = final_embeds
        xn = F.layer_norm(final_embeds, [D],
                          ckpt[lp + "norm2.weight"].float(),
                          ckpt[lp + "norm2.bias"].float())

        ca_w = ckpt[lp + "cross_attn_image.in_proj_weight"].float()  # [3D, D]
        ca_b = ckpt[lp + "cross_attn_image.in_proj_bias"].float()
        ca_ow = ckpt[lp + "cross_attn_image.out_proj.weight"].float()
        ca_ob = ckpt[lp + "cross_attn_image.out_proj.bias"].float()

        # Q from xn (no pos), K from img_feats + img_pe, V from img_feats
        q_w, k_w, v_w = ca_w.chunk(3, dim=0)
        q_b, k_b, v_b = ca_b.chunk(3, dim=0)

        # pos_enc_at_cross_attn_queries=False, so Q = proj(xn)
        # pos_enc_at_cross_attn_keys=True, so K = proj(img_feats + img_pe)
        q_proj = F.linear(xn, q_w, q_b)  # [S_q, B, D]
        k_input = img_feats_hwc + img_pe_hwc  # [S_kv, B, D]
        k_proj = F.linear(k_input, k_w, k_b)  # [S_kv, B, D]
        v_proj = F.linear(img_feats_hwc, v_w, v_b)  # [S_kv, B, D]

        S_q = q_proj.shape[0]
        S_kv = k_proj.shape[0]
        q_proj = q_proj.reshape(S_q, B, n_heads, hd).permute(1, 2, 0, 3)
        k_proj = k_proj.reshape(S_kv, B, n_heads, hd).permute(1, 2, 0, 3)
        v_proj = v_proj.reshape(S_kv, B, n_heads, hd).permute(1, 2, 0, 3)

        ca_out = F.scaled_dot_product_attention(q_proj, k_proj, v_proj)
        ca_out = ca_out.permute(2, 0, 1, 3).reshape(S_q, B, D)
        ca_out = F.linear(ca_out, ca_ow, ca_ob)
        final_embeds = shortcut + ca_out

        save_ggml_sbd(os.path.join(case_dir, f"layer{layer_idx}_after_ca"), final_embeds)

        # FFN (pre-norm, ReLU)
        shortcut = final_embeds
        xn = F.layer_norm(final_embeds, [D],
                          ckpt[lp + "norm3.weight"].float(),
                          ckpt[lp + "norm3.bias"].float())
        ffn = F.linear(xn, ckpt[lp + "linear1.weight"].float(),
                       ckpt[lp + "linear1.bias"].float())
        ffn = F.relu(ffn)
        ffn = F.linear(ffn, ckpt[lp + "linear2.weight"].float(),
                       ckpt[lp + "linear2.bias"].float())
        final_embeds = shortcut + ffn

        save_ggml_sbd(os.path.join(case_dir, f"layer{layer_idx}_after_ffn"), final_embeds)
        print(f"  layer{layer_idx}: shape={final_embeds.shape} "
              f"mean={final_embeds.mean():.6f} std={final_embeds.std():.6f}")

    # ── 9. Final encode norm ────────────────────────────────────────────
    enc_norm_w = ckpt["detector.geometry_encoder.encode_norm.weight"].float()
    enc_norm_b = ckpt["detector.geometry_encoder.encode_norm.bias"].float()
    final_embeds = F.layer_norm(final_embeds, [D], enc_norm_w, enc_norm_b)
    save_ggml_sbd(os.path.join(case_dir, "geom_output"), final_embeds)
    print(f"  geom_output: {final_embeds.shape} "
          f"mean={final_embeds.mean():.6f} std={final_embeds.std():.6f}")

    # Also save the mask
    save_raw(os.path.join(case_dir, "geom_mask"),
             final_mask.cpu().float().numpy(), final_mask.shape)

    return final_embeds, final_mask


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prephase-ref", required=True,
                        help="Phase 3 reference dir with neck features")
    parser.add_argument("--outdir", required=True)
    args = parser.parse_args()

    D = 256
    H = 72

    # Load checkpoint
    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=True)

    # Load backbone features from Phase 3 reference (saved in NCHW format)
    print("Loading Phase 3 reference features...")
    neck_det_2 = load_tensor(os.path.join(args.prephase_ref, "neck_det_2"))  # [1, 256, 72, 72]
    print(f"  neck_det_2: {neck_det_2.shape}")

    # Image features in sequence-first format [H*W, B, C]
    img_feats_hwc = neck_det_2.flatten(2).permute(2, 0, 1)  # [H*W, B, C]

    # Image positional encoding [H*W, B, C] — sinusoidal PE computed on the fly
    def compute_sine_pe(h, w, num_pos_feats=128, temperature=10000, scale=2*math.pi):
        """Sinusoidal PE matching SAM3's PositionEmbeddingSine."""
        not_mask = torch.ones(1, h, w, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        eps = 1e-6
        y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * scale
        x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [1, D, H, W]
        return pos

    img_pe_nchw = compute_sine_pe(H, H, num_pos_feats=D // 2)  # [1, 256, 72, 72]
    img_pe_hwc = img_pe_nchw.flatten(2).permute(2, 0, 1)  # [H*W, 1, C]
    print(f"  img_pe: {img_pe_hwc.shape}")

    # ══════════════════════════════════════════════════════════════════════
    # Test Case 1: Dummy prompt (no exemplars, just CLS)
    # ══════════════════════════════════════════════════════════════════════
    dummy_boxes = torch.zeros(0, 1, 4)
    dummy_labels = torch.zeros(0, 1, dtype=torch.long)
    dummy_mask = torch.zeros(1, 0, dtype=torch.bool)

    dump_geometry_encoder(
        ckpt, img_feats_hwc, neck_det_2, img_pe_hwc,
        dummy_boxes, dummy_labels, dummy_mask,
        args.outdir, "dummy_prompt"
    )

    # ══════════════════════════════════════════════════════════════════════
    # Test Case 2: Single positive exemplar box
    # Box at center of image: cx=0.5, cy=0.5, w=0.3, h=0.3
    # ══════════════════════════════════════════════════════════════════════
    single_box = torch.tensor([[[0.5, 0.5, 0.3, 0.3]]])  # [1, 1, 4]
    single_label = torch.tensor([[0]], dtype=torch.long)  # [1, 1] positive=0
    single_mask = torch.tensor([[False]])  # [1, 1]

    dump_geometry_encoder(
        ckpt, img_feats_hwc, neck_det_2, img_pe_hwc,
        single_box, single_label, single_mask,
        args.outdir, "single_box"
    )

    # ══════════════════════════════════════════════════════════════════════
    # Test Case 3: Two exemplar boxes (one positive, one negative)
    # ══════════════════════════════════════════════════════════════════════
    two_boxes = torch.tensor([
        [[0.3, 0.4, 0.2, 0.25]],   # box 1: positive
        [[0.7, 0.6, 0.15, 0.2]],   # box 2: negative
    ])  # [2, 1, 4]
    two_labels = torch.tensor([[0], [1]], dtype=torch.long)  # [2, 1]
    two_mask = torch.tensor([[False, False]])  # [1, 2]

    dump_geometry_encoder(
        ckpt, img_feats_hwc, neck_det_2, img_pe_hwc,
        two_boxes, two_labels, two_mask,
        args.outdir, "two_boxes"
    )

    print(f"\nAll reference tensors saved to {args.outdir}/")


if __name__ == "__main__":
    main()
