#!/usr/bin/env python3
"""
Dump Phase 5 (Detector/PCS) reference tensors from SAM3 checkpoint for C++ verification.
Covers: Fusion Encoder, DETR Decoder, DotProductScoring, Segmentation Head.
Self-contained — builds forward pass from raw weights, no triton dependency.

Usage:
    uv run python tests/dump_phase5_reference.py --checkpoint raw_weights/sam3.pt --image tests/test_random.jpg --outdir tests/ref_phase5/
"""
import argparse
import math
import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms import v2


def save_tensor(path, t):
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)}")


def save_tensor_i32(path, t):
    t = t.detach().cpu().to(torch.int32).contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)} dtype=int32")


# ── Inlined helpers ─────────────────────────────────────────────────────

def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor, num_feats=256):
    """Generate sinusoidal positional embedding for 2D/4D reference points."""
    assert num_feats % 2 == 0
    num_feats = num_feats // 2
    scale = 2 * math.pi
    dim_t = torch.arange(num_feats, dtype=torch.float32, device=pos_tensor.device)
    dim_t = 10000 ** (2 * (torch.div(dim_t, 2, rounding_mode="floor")) / num_feats)
    x_embed = pos_tensor[:, :, 0] * scale
    y_embed = pos_tensor[:, :, 1] * scale
    pos_x = x_embed[:, :, None] / dim_t
    pos_y = y_embed[:, :, None] / dim_t
    pos_x = torch.stack((pos_x[:, :, 0::2].sin(), pos_x[:, :, 1::2].cos()), dim=3).flatten(2)
    pos_y = torch.stack((pos_y[:, :, 0::2].sin(), pos_y[:, :, 1::2].cos()), dim=3).flatten(2)
    if pos_tensor.size(-1) == 2:
        pos = torch.cat((pos_y, pos_x), dim=2)
    elif pos_tensor.size(-1) == 4:
        w_embed = pos_tensor[:, :, 2] * scale
        pos_w = w_embed[:, :, None] / dim_t
        pos_w = torch.stack((pos_w[:, :, 0::2].sin(), pos_w[:, :, 1::2].cos()), dim=3).flatten(2)
        h_embed = pos_tensor[:, :, 3] * scale
        pos_h = h_embed[:, :, None] / dim_t
        pos_h = torch.stack((pos_h[:, :, 0::2].sin(), pos_h[:, :, 1::2].cos()), dim=3).flatten(2)
        pos = torch.cat((pos_y, pos_x, pos_w, pos_h), dim=2)
    else:
        raise ValueError(f"Unknown pos_tensor shape(-1):{pos_tensor.size(-1)}")
    return pos


def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def mlp_forward(x, sd, prefix, num_layers, activation="relu"):
    """Forward through MLP layers from state dict."""
    for i in range(num_layers):
        w = sd[f"{prefix}.layers.{i}.weight"]
        b = sd[f"{prefix}.layers.{i}.bias"]
        x = F.linear(x, w, b)
        if i < num_layers - 1:
            x = F.relu(x)
    return x


def multihead_attention_forward(query, key, value, w_q, b_q, w_k, b_k, w_v, b_v,
                                  w_out, b_out, num_heads, attn_mask=None,
                                  key_padding_mask=None):
    """Manual multi-head attention forward."""
    D = query.shape[-1]
    HD = D // num_heads

    Q = F.linear(query, w_q, b_q)
    K = F.linear(key, w_k, b_k)
    V = F.linear(value, w_v, b_v)

    # [BS, N, D] -> [BS, N, NH, HD] -> [BS, NH, N, HD]
    BS = Q.shape[0]
    N_q = Q.shape[1]
    N_kv = K.shape[1]

    Q = Q.reshape(BS, N_q, num_heads, HD).permute(0, 2, 1, 3)
    K = K.reshape(BS, N_kv, num_heads, HD).permute(0, 2, 1, 3)
    V = V.reshape(BS, N_kv, num_heads, HD).permute(0, 2, 1, 3)

    combined_mask = attn_mask
    if key_padding_mask is not None:
        pad_bias = torch.zeros(
            BS, 1, 1, N_kv, dtype=Q.dtype, device=Q.device
        )
        pad_bias = pad_bias.masked_fill(
            key_padding_mask[:, None, None, :], float("-inf")
        )
        combined_mask = pad_bias if combined_mask is None else combined_mask + pad_bias

    # Attention
    attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=combined_mask)

    # [BS, NH, N_q, HD] -> [BS, N_q, D]
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(BS, N_q, D)
    out = F.linear(attn_output, w_out, b_out)
    return out


def fused_mha_forward(query, key, value, in_proj_w, in_proj_b, out_w, out_b,
                       num_heads, attn_mask=None, key_padding_mask=None):
    """Multi-head attention with fused in_proj weights."""
    D = query.shape[-1]
    w_q, w_k, w_v = in_proj_w[:D], in_proj_w[D:2*D], in_proj_w[2*D:]
    b_q, b_k, b_v = in_proj_b[:D], in_proj_b[D:2*D], in_proj_b[2*D:]
    return multihead_attention_forward(query, key, value, w_q, b_q, w_k, b_k, w_v, b_v,
                                        out_w, out_b, num_heads, attn_mask, key_padding_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--outdir", default="tests/ref_phase5")
    parser.add_argument("--text", default="yellow school bus")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"
    D = 256
    NH = 8
    HD = D // NH
    NQ = 200
    T = 32  # text context length
    H = 72  # image feature spatial dim (1008 / 14)

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    # ── Load required weights from specific modules ─────────────────────

    # We need the image features & text features as inputs.
    # To get them, we first need to run the ViT backbone + neck + text encoder.
    # These were already verified in prior phases, so we load cached reference tensors.
    # But we need fresh tensors to match exact C++ inputs.

    # Let's first generate fresh image features using dump_vit_reference approach
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

    # ── ViT forward (reuse from dump_vit_reference.py) ─────────────────
    # ... (this is too long to inline, we'll load pre-computed neck features)
    # Instead, we'll load from the reference directory if available
    ref_dir = "tests/ref"
    neck_det_path = os.path.join(ref_dir, "neck_det_2")
    if os.path.exists(neck_det_path + ".bin"):
        print("Loading pre-computed neck features from tests/ref/")
        # Load neck features
        def load_tensor(path):
            shape = list(map(int, open(path + ".shape").read().strip().split(",")))
            data = np.fromfile(path + ".bin", dtype=np.float32).reshape(shape)
            return torch.tensor(data)

        # Neck det features: [1, 256, H, H] for scale 2 (72x72)
        neck_det_2 = load_tensor(os.path.join(ref_dir, "neck_det_2"))  # [1, 256, 72, 72]
        neck_det_1 = load_tensor(os.path.join(ref_dir, "neck_det_1"))  # [1, 256, 144, 144]
        neck_det_0 = load_tensor(os.path.join(ref_dir, "neck_det_0"))  # [1, 256, 288, 288]
    else:
        print("ERROR: Pre-computed neck features not found. Run dump_vit_reference.py first.")
        sys.exit(1)

    # ── Text encoder forward ────────────────────────────────────────────
    # For simplicity, we need text features. The text encoder was verified in phase 4.
    # We can compute them here with the same approach.
    # For now, load from existing ref or compute inline.

    # Actually, let's compute text features inline since we need them.
    # The text encoder is: embedding -> transformer -> projection

    # Tokenize
    from tokenizers import Tokenizer
    tokenizer_path = os.path.join(os.path.dirname(args.checkpoint), "tokenizer.json")
    if os.path.exists(tokenizer_path):
        tokenizer = Tokenizer.from_file(tokenizer_path)
    else:
        tokenizer = Tokenizer.from_file("raw_weights/tokenizer.json")

    encoded = tokenizer.encode(args.text)
    # tokenizer already includes SOT/EOT
    token_ids = list(encoded.ids)
    while len(token_ids) < T:
        token_ids.append(0)  # pad
    token_ids = token_ids[:T]

    tokens_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)  # [1, T]
    print(f"  Token IDs: {token_ids[:10]}...")
    save_tensor_i32(os.path.join(args.outdir, "token_ids"), tokens_tensor)
    with open(os.path.join(args.outdir, "prompt.txt"), "w") as f:
        f.write(args.text + "\n")

    # Text encoder forward
    text_prefix = "detector.backbone.language_backbone.encoder."
    text_sd = {k[len(text_prefix):]: v for k, v in ckpt.items() if k.startswith(text_prefix)}
    text_width = 1024
    text_heads = 16
    text_layers = 24

    with torch.no_grad():
        # Token + positional embedding
        tok_emb = text_sd["token_embedding.weight"]  # [49408, 1024]
        pos_emb = text_sd["positional_embedding"]  # [32, 1024]
        x = tok_emb[tokens_tensor] + pos_emb[:T]  # [1, T, 1024]
        x = x.permute(1, 0, 2)  # [T, 1, 1024]

        # Causal mask
        causal_mask = torch.full((T, T), float("-inf"), device=device)
        causal_mask = torch.triu(causal_mask, diagonal=1)

        # Transformer blocks
        for i in range(text_layers):
            p = f"transformer.resblocks.{i}."
            # Pre-norm SA
            xn = F.layer_norm(x, [text_width], text_sd[p + "ln_1.weight"], text_sd[p + "ln_1.bias"])
            # Fused in_proj
            qkv_w = text_sd[p + "attn.in_proj_weight"]
            qkv_b = text_sd[p + "attn.in_proj_bias"]
            out_w = text_sd[p + "attn.out_proj.weight"]
            out_b = text_sd[p + "attn.out_proj.bias"]

            # Manual MHA (T, BS, D) -> seq-first
            xn_bf = xn.permute(1, 0, 2)  # [1, T, 1024]
            sa_out = fused_mha_forward(xn_bf, xn_bf, xn_bf, qkv_w, qkv_b, out_w, out_b,
                                        text_heads, attn_mask=causal_mask)
            x = x + sa_out.permute(1, 0, 2)

            # Pre-norm FFN
            xn = F.layer_norm(x, [text_width], text_sd[p + "ln_2.weight"], text_sd[p + "ln_2.bias"])
            fc1_w = text_sd[p + "mlp.c_fc.weight"]
            fc1_b = text_sd[p + "mlp.c_fc.bias"]
            fc2_w = text_sd[p + "mlp.c_proj.weight"]
            fc2_b = text_sd[p + "mlp.c_proj.bias"]
            h = F.linear(xn, fc1_w, fc1_b)
            h = F.gelu(h, approximate="tanh")
            h = F.linear(h, fc2_w, fc2_b)
            x = x + h

        # Final LN
        x = F.layer_norm(x, [text_width], text_sd["ln_final.weight"], text_sd["ln_final.bias"])
        # x: [T, 1, 1024]

        # Project to 256-dim via resizer (Linear(1024→256))
        resizer_prefix = "detector.backbone.language_backbone.resizer."
        resizer_sd = {k[len(resizer_prefix):]: v for k, v in ckpt.items() if k.startswith(resizer_prefix)}
        txt_projected = F.linear(x, resizer_sd["weight"], resizer_sd.get("bias"))
        # txt_projected: [T, 1, 256]

    save_tensor(os.path.join(args.outdir, "text_features"), txt_projected)
    print(f"  Text features: {list(txt_projected.shape)}")

    # Positional encoding for image features (sinusoidal)
    # Compute sinusoidal PE manually
    def compute_sine_pe(h, w, num_pos_feats=256, temperature=10000, normalize=True, scale=2*math.pi):
        """Sinusoidal PE matching SAM3's PositionEmbeddingSine."""
        not_mask = torch.ones(1, h, w, dtype=torch.float32)
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if normalize:
            eps = 1e-6
            y_embed = (y_embed - 0.5) / (y_embed[:, -1:, :] + eps) * scale
            x_embed = (x_embed - 0.5) / (x_embed[:, :, -1:] + eps) * scale

        dim_t = torch.arange(num_pos_feats, dtype=torch.float32)
        dim_t = temperature ** (2 * (dim_t // 2) / num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t  # [1, H, W, D/2]
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # [1, D, H, W]
        return pos

    img_pe_72 = compute_sine_pe(H, H, num_pos_feats=D//2)  # [1, 256, 72, 72]
    save_tensor(os.path.join(args.outdir, "img_pe_72"), img_pe_72)

    # ═══════════════════════════════════════════════════════════════════════
    #  Step 5.1: Fusion Encoder (6 layers)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n=== Fusion Encoder ===")

    fenc_prefix = "detector.transformer.encoder."
    fenc_sd = {k[len("detector.transformer."):]: v for k, v in ckpt.items()
               if k.startswith("detector.transformer.encoder.")}

    with torch.no_grad():
        # Image features: [1, 256, 72, 72] -> flatten to [1, 5184, 256] (batch-first)
        img_feat_flat = neck_det_2.flatten(2).permute(0, 2, 1)  # [1, 5184, 256]
        img_pe_flat = img_pe_72.flatten(2).permute(0, 2, 1)  # [1, 5184, 256]

        # Text features: [T, 1, 256] -> [1, T, 256] for batch-first
        prompt_bf = txt_projected.permute(1, 0, 2)  # [1, T, 256]

        # Text mask: True for padding (0 tokens), False for valid
        txt_mask = torch.tensor([[tid == 0 for tid in token_ids]], dtype=torch.bool)  # [1, T]
        valid_scale = float(T) / max(1, sum(tid != 0 for tid in token_ids))
        text_valid_mask = torch.tensor(
            [[[valid_scale if tid != 0 else 0.0] for tid in token_ids]],
            dtype=torch.float32,
        ).permute(1, 2, 0)  # [T, 1, 1]

        save_tensor(os.path.join(args.outdir, "fenc_img_input"), img_feat_flat)
        save_tensor(os.path.join(args.outdir, "fenc_pos_embed"), img_pe_flat)
        save_tensor(os.path.join(args.outdir, "fenc_prompt"), prompt_bf)
        save_tensor(os.path.join(args.outdir, "text_valid_mask"), text_valid_mask)

        # Forward through 6 fusion encoder layers
        output = img_feat_flat  # [1, HW, D] batch-first

        for layer_idx in range(6):
            lp = f"encoder.layers.{layer_idx}."

            # 1. Self-attention (pre-norm, pos_enc_at_attn=True)
            shortcut = output
            xn = F.layer_norm(output, [D],
                              fenc_sd[lp + "norm1.weight"], fenc_sd[lp + "norm1.bias"])
            q = k = xn + img_pe_flat  # add pos to Q and K

            # SA weights
            sa_qkv_w = fenc_sd[lp + "self_attn.in_proj_weight"]
            sa_qkv_b = fenc_sd[lp + "self_attn.in_proj_bias"]
            sa_out_w = fenc_sd[lp + "self_attn.out_proj.weight"]
            sa_out_b = fenc_sd[lp + "self_attn.out_proj.bias"]

            sa_out = fused_mha_forward(q, k, xn, sa_qkv_w, sa_qkv_b, sa_out_w, sa_out_b, NH)
            output = shortcut + sa_out

            # 2. Cross-attention (pre-norm, no pos at CA)
            shortcut = output
            xn = F.layer_norm(output, [D],
                              fenc_sd[lp + "norm2.weight"], fenc_sd[lp + "norm2.bias"])
            # cross_attn_image: Q from image, K/V from text
            ca_qkv_w = fenc_sd[lp + "cross_attn_image.in_proj_weight"]
            ca_qkv_b = fenc_sd[lp + "cross_attn_image.in_proj_bias"]
            ca_out_w = fenc_sd[lp + "cross_attn_image.out_proj.weight"]
            ca_out_b = fenc_sd[lp + "cross_attn_image.out_proj.bias"]

            # Split in_proj into separate Q/K/V for cross-attention
            w_q_ca = ca_qkv_w[:D]
            w_k_ca = ca_qkv_w[D:2*D]
            w_v_ca = ca_qkv_w[2*D:]
            b_q_ca = ca_qkv_b[:D]
            b_k_ca = ca_qkv_b[D:2*D]
            b_v_ca = ca_qkv_b[2*D:]

            ca_out = multihead_attention_forward(
                xn, prompt_bf, prompt_bf,
                w_q_ca, b_q_ca, w_k_ca, b_k_ca, w_v_ca, b_v_ca,
                ca_out_w, ca_out_b, NH,
                key_padding_mask=txt_mask)

            output = shortcut + ca_out

            # 3. FFN (pre-norm, relu activation)
            shortcut = output
            xn = F.layer_norm(output, [D],
                              fenc_sd[lp + "norm3.weight"], fenc_sd[lp + "norm3.bias"])
            h = F.linear(xn, fenc_sd[lp + "linear1.weight"], fenc_sd[lp + "linear1.bias"])
            h = F.relu(h)
            h = F.linear(h, fenc_sd[lp + "linear2.weight"], fenc_sd[lp + "linear2.bias"])
            output = shortcut + h

            save_tensor(os.path.join(args.outdir, f"fenc_layer{layer_idx}_out"), output)

        # Transpose to seq-first for downstream: [HW, 1, D]
        fenc_output_sf = output.permute(1, 0, 2)  # [5184, 1, D] seq-first
        save_tensor(os.path.join(args.outdir, "fenc_output"), fenc_output_sf)
        save_tensor(os.path.join(args.outdir, "fenc_output_bf"), output)  # batch-first for decoder
        print(f"  fenc_output shape: {list(fenc_output_sf.shape)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Step 5.2: DETR Decoder (6 layers)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n=== DETR Decoder ===")

    ddec_prefix = "detector.transformer.decoder."
    ddec_sd = {k[len("detector.transformer."):]: v for k, v in ckpt.items()
               if k.startswith("detector.transformer.decoder.")}

    with torch.no_grad():
        # Query embeddings and reference points
        query_embed = ddec_sd["decoder.query_embed.weight"]  # [NQ, D]
        ref_pts_raw = ddec_sd["decoder.reference_points.weight"]  # [NQ, 4]
        reference_boxes = ref_pts_raw.sigmoid().unsqueeze(1)  # [NQ, 1, 4]

        save_tensor(os.path.join(args.outdir, "ddec_query_embed"), query_embed)
        save_tensor(os.path.join(args.outdir, "ddec_ref_pts_raw"), ref_pts_raw)
        save_tensor(os.path.join(args.outdir, "ddec_ref_boxes_init"), reference_boxes)

        # Presence token
        presence_token_w = ddec_sd["decoder.presence_token.weight"]  # [1, D]
        presence_out = presence_token_w[None].expand(1, 1, -1)  # [1, 1, D]
        save_tensor(os.path.join(args.outdir, "ddec_presence_token"), presence_out)

        # Encoder output is [HW, BS, D] (seq-first) for the decoder
        enc_memory = fenc_output_sf  # [5184, 1, D]
        enc_pos = img_pe_flat.permute(1, 0, 2)  # [5184, 1, D] seq-first

        # valid_ratios: [1, 1, 2] = ones (no masking)
        valid_ratios = torch.ones(1, 1, 2, device=device)
        spatial_shapes = torch.tensor([[H, H]], dtype=torch.long)

        # Init output (tgt)
        output = query_embed.unsqueeze(1).repeat(1, 1, 1)  # [NQ, 1, D]

        # Text features for cross-attention: seq-first [T, 1, D]
        prompt_sf = txt_projected  # already [T, 1, D]

        # ── RPB coordinates ─────────────────────────────────────────
        coords_h = torch.arange(0, H, dtype=torch.float32) / H
        coords_w = torch.arange(0, H, dtype=torch.float32) / H

        for layer_idx in range(6):
            lp = f"decoder.layers.{layer_idx}."

            # Reference points input
            reference_points_input = reference_boxes[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], D)
            # query_sine_embed: [NQ, 1, 512]

            # Conditional query pos: ref_point_head MLP (512→256→256)
            query_pos = mlp_forward(query_sine_embed, ddec_sd,
                                     "decoder.ref_point_head", 2)  # [NQ, 1, D]

            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_query_pos_0"), query_pos)
                save_tensor(os.path.join(args.outdir, "ddec_query_sine_0"), query_sine_embed)

            # ── Box RPB ─────────────────────────────────────────────
            boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).permute(1, 0, 2)  # [1, NQ, 4]
            bs_d, nq_d, _ = boxes_xyxy.shape

            deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
            deltas_y = deltas_y.view(bs_d, nq_d, -1, 2)  # [1, NQ, H, 2]
            deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
            deltas_x = deltas_x.view(bs_d, nq_d, -1, 2)  # [1, NQ, W, 2]

            # Log transform
            deltas_x_log = deltas_x * 8
            deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8)
            deltas_y_log = deltas_y * 8
            deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8)
            deltas_x = deltas_x_log
            deltas_y = deltas_y_log

            # RPB MLP: [1, NQ, W, 2] → [1, NQ, W, NH]
            rpb_x = mlp_forward(deltas_x, ddec_sd, "decoder.boxRPB_embed_x", 2)
            rpb_y = mlp_forward(deltas_y, ddec_sd, "decoder.boxRPB_embed_y", 2)

            # Outer sum: [1, NQ, H, W, NH]
            B_rpb = rpb_y.unsqueeze(3) + rpb_x.unsqueeze(2)
            B_rpb = B_rpb.flatten(2, 3)  # [1, NQ, H*W, NH]
            B_rpb = B_rpb.permute(0, 3, 1, 2).contiguous()  # [1, NH, NQ, H*W]

            # For presence token: prepend zeros
            pres_rpb = torch.zeros(1, NH, 1, H*H, device=device)
            memory_mask = torch.cat([pres_rpb, B_rpb], dim=2)  # [1, NH, NQ+1, H*W]
            memory_mask = memory_mask.flatten(0, 1)  # [NH, NQ+1, H*W]

            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_rpb_mask_0"), memory_mask)

            # ── Self-attention ──────────────────────────────────────
            # Prepend presence token
            tgt_with_pres = torch.cat([presence_out, output], dim=0)  # [NQ+1, 1, D]
            qpos_with_pres = torch.cat([torch.zeros_like(presence_out), query_pos], dim=0)

            q_sa = k_sa = tgt_with_pres + qpos_with_pres

            sa_w = ddec_sd[lp + "self_attn.in_proj_weight"]
            sa_b = ddec_sd[lp + "self_attn.in_proj_bias"]
            sa_ow = ddec_sd[lp + "self_attn.out_proj.weight"]
            sa_ob = ddec_sd[lp + "self_attn.out_proj.bias"]

            # Convert to batch-first for MHA
            q_sa_bf = q_sa.permute(1, 0, 2)  # [1, NQ+1, D]
            k_sa_bf = k_sa.permute(1, 0, 2)
            v_sa_bf = tgt_with_pres.permute(1, 0, 2)

            sa_out = fused_mha_forward(q_sa_bf, k_sa_bf, v_sa_bf, sa_w, sa_b, sa_ow, sa_ob, NH)
            tgt_with_pres = tgt_with_pres + sa_out.permute(1, 0, 2)
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm2.weight"], ddec_sd[lp + "norm2.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_after_sa"), tgt_with_pres)

            # ── Text cross-attention ─────────────────────────────────
            tgt_q_ca_text = tgt_with_pres + qpos_with_pres
            ca_text_w = ddec_sd[lp + "ca_text.in_proj_weight"]
            ca_text_b = ddec_sd[lp + "ca_text.in_proj_bias"]
            ca_text_ow = ddec_sd[lp + "ca_text.out_proj.weight"]
            ca_text_ob = ddec_sd[lp + "ca_text.out_proj.bias"]

            # Q from queries, K/V from text
            w_q_ct = ca_text_w[:D]
            w_k_ct = ca_text_w[D:2*D]
            w_v_ct = ca_text_w[2*D:]
            b_q_ct = ca_text_b[:D]
            b_k_ct = ca_text_b[D:2*D]
            b_v_ct = ca_text_b[2*D:]

            tq_bf = tgt_q_ca_text.permute(1, 0, 2)  # [1, NQ+1, D]
            p_bf = prompt_sf.permute(1, 0, 2)  # [1, T, D]
            ct_out = multihead_attention_forward(tq_bf, p_bf, p_bf,
                                                  w_q_ct, b_q_ct, w_k_ct, b_k_ct, w_v_ct, b_v_ct,
                                                  ca_text_ow, ca_text_ob, NH,
                                                  key_padding_mask=txt_mask)
            tgt_with_pres = tgt_with_pres + ct_out.permute(1, 0, 2)
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "catext_norm.weight"],
                                          ddec_sd[lp + "catext_norm.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_after_text_ca"), tgt_with_pres)

            # ── Image cross-attention with RPB ──────────────────────
            tgt_q_ca_img = tgt_with_pres + qpos_with_pres
            mem_k = enc_memory + enc_pos  # [5184, 1, D]

            ca_w = ddec_sd[lp + "cross_attn.in_proj_weight"]
            ca_b = ddec_sd[lp + "cross_attn.in_proj_bias"]
            ca_ow = ddec_sd[lp + "cross_attn.out_proj.weight"]
            ca_ob = ddec_sd[lp + "cross_attn.out_proj.bias"]

            w_q_ci = ca_w[:D]
            w_k_ci = ca_w[D:2*D]
            w_v_ci = ca_w[2*D:]
            b_q_ci = ca_b[:D]
            b_k_ci = ca_b[D:2*D]
            b_v_ci = ca_b[2*D:]

            # Batch first
            tqi_bf = tgt_q_ca_img.permute(1, 0, 2)  # [1, NQ+1, D]
            mk_bf = mem_k.permute(1, 0, 2)  # [1, 5184, D]
            mv_bf = enc_memory.permute(1, 0, 2)  # [1, 5184, D]

            # Need to handle RPB mask as attn_mask
            # memory_mask: [NH, NQ+1, H*W] — for PyTorch SDPA, need [BS*NH, NQ+1, H*W]
            # For BS=1, just use as is
            ci_out = multihead_attention_forward(tqi_bf, mk_bf, mv_bf,
                                                  w_q_ci, b_q_ci, w_k_ci, b_k_ci, w_v_ci, b_v_ci,
                                                  ca_ow, ca_ob, NH,
                                                  attn_mask=memory_mask)
            tgt_with_pres = tgt_with_pres + ci_out.permute(1, 0, 2)
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm1.weight"], ddec_sd[lp + "norm1.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_after_img_ca"), tgt_with_pres)

            # ── FFN ─────────────────────────────────────────────────
            ffn = F.linear(tgt_with_pres, ddec_sd[lp + "linear1.weight"], ddec_sd[lp + "linear1.bias"])
            ffn = F.relu(ffn)
            ffn = F.linear(ffn, ddec_sd[lp + "linear2.weight"], ddec_sd[lp + "linear2.bias"])
            tgt_with_pres = tgt_with_pres + ffn
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm3.weight"], ddec_sd[lp + "norm3.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_full_out"), tgt_with_pres)

            # Split presence and queries
            presence_out = tgt_with_pres[:1]  # [1, 1, D]
            output = tgt_with_pres[1:]  # [NQ, 1, D]

            # Box refinement
            reference_before_sigmoid = inverse_sigmoid(reference_boxes)
            normed_output = F.layer_norm(output, [D],
                                          ddec_sd["decoder.norm.weight"], ddec_sd["decoder.norm.bias"])
            delta_unsig = mlp_forward(normed_output, ddec_sd, "decoder.bbox_embed", 3)
            outputs_unsig = delta_unsig + reference_before_sigmoid
            new_reference_points = outputs_unsig.sigmoid()
            reference_boxes = new_reference_points.detach()

            save_tensor(os.path.join(args.outdir, f"ddec_layer{layer_idx}_out"), output)
            save_tensor(os.path.join(args.outdir, f"ddec_layer{layer_idx}_refboxes"), reference_boxes)
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, f"ddec_layer{layer_idx}_presence"), presence_out)

        # Final normalization
        normed_output = F.layer_norm(output, [D],
                                      ddec_sd["decoder.norm.weight"], ddec_sd["decoder.norm.bias"])
        save_tensor(os.path.join(args.outdir, "ddec_normed_output"), normed_output)
        save_tensor(os.path.join(args.outdir, "ddec_pred_boxes"), reference_boxes)

        # Presence head
        pres_normed = F.layer_norm(presence_out, [D],
                                    ddec_sd["decoder.presence_token_out_norm.weight"],
                                    ddec_sd["decoder.presence_token_out_norm.bias"])
        pres_logit = mlp_forward(pres_normed, ddec_sd, "decoder.presence_token_head", 3)
        save_tensor(os.path.join(args.outdir, "ddec_presence_logit"), pres_logit)
        print(f"  Presence logit: {pres_logit.item():.4f}")

    # ═══════════════════════════════════════════════════════════════════════
    #  Step 5.2b: DotProductScoring
    # ═══════════════════════════════════════════════════════════════════════
    print("\n=== DotProductScoring ===")

    scoring_prefix = "detector.dot_prod_scoring."
    scoring_sd = {k[len(scoring_prefix):]: v for k, v in ckpt.items()
                  if k.startswith(scoring_prefix)}

    with torch.no_grad():
        # prompt_mlp: residual MLP + LayerNorm
        prompt_for_scoring = prompt_sf.clone()  # [T, 1, D]
        prompt_mlp_out = prompt_for_scoring
        orig = prompt_mlp_out.clone()
        prompt_mlp_out = F.linear(prompt_mlp_out, scoring_sd["prompt_mlp.layers.0.weight"],
                                   scoring_sd["prompt_mlp.layers.0.bias"])
        prompt_mlp_out = F.relu(prompt_mlp_out)
        prompt_mlp_out = F.linear(prompt_mlp_out, scoring_sd["prompt_mlp.layers.1.weight"],
                                   scoring_sd["prompt_mlp.layers.1.bias"])
        prompt_mlp_out = prompt_mlp_out + orig  # residual
        prompt_mlp_out = F.layer_norm(prompt_mlp_out, [D],
                                       scoring_sd["prompt_mlp.out_norm.weight"],
                                       scoring_sd["prompt_mlp.out_norm.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_prompt_mlp_out"), prompt_mlp_out)

        # Mean pool text
        # prompt_mask: True for padding, False for valid
        is_valid = (~txt_mask).float().permute(1, 0)[..., None]  # [T, 1, 1]
        num_valid = torch.clamp(is_valid.sum(dim=0), min=1.0)  # [1, 1]
        pooled = (prompt_mlp_out * is_valid).sum(dim=0) / num_valid  # [1, D]
        save_tensor(os.path.join(args.outdir, "scoring_pooled"), pooled)

        # Project
        proj_pooled = F.linear(pooled, scoring_sd["prompt_proj.weight"],
                                scoring_sd["prompt_proj.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_proj_pooled"), proj_pooled)

        # Project hs
        hs = normed_output.permute(1, 0, 2).unsqueeze(0)  # [1, 1, NQ, D]
        proj_hs = F.linear(hs, scoring_sd["hs_proj.weight"], scoring_sd["hs_proj.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_proj_hs"), proj_hs)

        # Dot product scores
        scores = torch.matmul(proj_hs, proj_pooled.unsqueeze(-1))  # [1, 1, NQ, 1]
        scores = scores * (1.0 / math.sqrt(D))
        scores = scores.clamp(-12.0, 12.0)
        save_tensor(os.path.join(args.outdir, "scoring_class_scores"), scores)
        print(f"  scores shape: {list(scores.shape)}, range: [{scores.min():.4f}, {scores.max():.4f}]")

    # ═══════════════════════════════════════════════════════════════════════
    #  Step 5.3: Segmentation Head
    # ═══════════════════════════════════════════════════════════════════════
    print("\n=== Segmentation Head ===")

    seg_prefix = "detector.segmentation_head."
    seg_sd = {k[len(seg_prefix):]: v for k, v in ckpt.items()
              if k.startswith(seg_prefix)}

    with torch.no_grad():
        # Cross-attend encoder hidden states to prompt
        enc_hs = fenc_output_sf.clone()  # [5184, 1, D]
        tgt2 = F.layer_norm(enc_hs, [D], seg_sd["cross_attn_norm.weight"], seg_sd["cross_attn_norm.bias"])

        ca_w = seg_sd["cross_attend_prompt.in_proj_weight"]
        ca_b = seg_sd["cross_attend_prompt.in_proj_bias"]
        ca_ow = seg_sd["cross_attend_prompt.out_proj.weight"]
        ca_ob = seg_sd["cross_attend_prompt.out_proj.bias"]

        w_q = ca_w[:D]; w_k = ca_w[D:2*D]; w_v = ca_w[2*D:]
        b_q = ca_b[:D]; b_k = ca_b[D:2*D]; b_v = ca_b[2*D:]

        tgt2_bf = tgt2.permute(1, 0, 2)
        p_bf = prompt_sf.permute(1, 0, 2)
        ca_out = multihead_attention_forward(tgt2_bf, p_bf, p_bf,
                                              w_q, b_q, w_k, b_k, w_v, b_v,
                                              ca_ow, ca_ob, NH,
                                              key_padding_mask=txt_mask)
        enc_hs = enc_hs + ca_out.permute(1, 0, 2)
        save_tensor(os.path.join(args.outdir, "seg_enc_after_ca"), enc_hs)

        # Replace lowest-res FPN feat with encoder visual output
        enc_visual = enc_hs.permute(1, 2, 0).reshape(1, D, H, H)
        save_tensor(os.path.join(args.outdir, "seg_enc_visual"), enc_visual)

        # Pixel decoder
        modified_feats = [neck_det_0.clone(), neck_det_1.clone(), enc_visual]

        prev_fpn = modified_feats[-1]  # [1, D, 72, 72]
        for pd_idx, bb_feat in enumerate(modified_feats[:-1][::-1]):
            curr_fpn = bb_feat
            prev_fpn = curr_fpn + F.interpolate(prev_fpn, size=curr_fpn.shape[-2:], mode="nearest")
            prev_fpn = F.conv2d(prev_fpn, seg_sd[f"pixel_decoder.conv_layers.{pd_idx}.weight"],
                                 seg_sd[f"pixel_decoder.conv_layers.{pd_idx}.bias"], padding=1)
            prev_fpn = F.group_norm(prev_fpn, 8,
                                     seg_sd[f"pixel_decoder.norms.{pd_idx}.weight"],
                                     seg_sd[f"pixel_decoder.norms.{pd_idx}.bias"])
            prev_fpn = F.relu(prev_fpn)
            save_tensor(os.path.join(args.outdir, f"seg_pixel_dec_stage{pd_idx}"), prev_fpn)

        pixel_embed = prev_fpn  # [1, D, 288, 288]
        save_tensor(os.path.join(args.outdir, "seg_pixel_decoder_out"), pixel_embed)

        # Instance seg head
        instance_embed = F.conv2d(pixel_embed, seg_sd["instance_seg_head.weight"],
                                   seg_sd["instance_seg_head.bias"])
        save_tensor(os.path.join(args.outdir, "seg_instance_embed"), instance_embed)

        # Mask predictor
        obj_queries = normed_output.permute(1, 0, 2)  # [1, NQ, D]
        mask_embed = obj_queries
        for j in range(3):
            w = seg_sd[f"mask_predictor.mask_embed.layers.{j}.weight"]
            b = seg_sd[f"mask_predictor.mask_embed.layers.{j}.bias"]
            mask_embed = F.linear(mask_embed, w, b)
            if j < 2:
                mask_embed = F.relu(mask_embed)
        save_tensor(os.path.join(args.outdir, "seg_mask_embed"), mask_embed)

        # Einsum: bqc,bchw -> bqhw
        mask_preds = torch.einsum("bqc,bchw->bqhw", mask_embed, instance_embed)
        save_tensor(os.path.join(args.outdir, "seg_mask_logits"), mask_preds)
        print(f"  mask_preds shape: {list(mask_preds.shape)}")

    # ═══════════════════════════════════════════════════════════════════════
    #  End-to-end summary
    # ═══════════════════════════════════════════════════════════════════════
    print("\n=== End-to-end Summary ===")
    with torch.no_grad():
        class_logits = scores[-1, :, :, 0]  # [1, NQ]
        pres_sig = pres_logit.squeeze(-1).squeeze(0).sigmoid()  # scalar
        probs = class_logits.sigmoid() * pres_sig
        top_k = min(10, NQ)
        top_probs, top_idx = probs[0].topk(top_k)
        print(f"  Top {top_k} detections:")
        for i in range(top_k):
            qi = top_idx[i].item()
            box = reference_boxes[qi, 0]
            print(f"    q{qi}: score={top_probs[i]:.4f}, box=({box[0]:.3f},{box[1]:.3f},{box[2]:.3f},{box[3]:.3f})")

    print(f"\nDone! Reference tensors saved to {args.outdir}")


if __name__ == "__main__":
    main()
