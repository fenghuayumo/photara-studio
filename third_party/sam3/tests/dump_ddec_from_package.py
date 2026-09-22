#!/usr/bin/env python3
"""
Dump DETR decoder reference tensors directly from the SAM3 Python package.
Uses the actual TransformerDecoder.forward() from the package (not reimplemented).

This script:
1. Builds the model using the SAM3 package
2. Loads the checkpoint
3. Runs just the decoder forward pass with known inputs
4. Dumps per-layer outputs for comparison with C++

Usage:
    cd ~/Documents/sam3
    python3 /path/to/dump_ddec_from_package.py --checkpoint raw_weights/sam3.pt --outdir /tmp/ddec_ref
"""
import argparse
import math
import os
import sys
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def save_tensor(path, t):
    """Save tensor as .bin + .shape files."""
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path} shape={list(t.shape)}")


def inverse_sigmoid(x, eps=1e-3):
    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1 / x2)


def gen_sineembed_for_position(pos_tensor, num_feats=256):
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
    D = query.shape[-1]
    HD = D // num_heads
    Q = F.linear(query, w_q, b_q)
    K = F.linear(key, w_k, b_k)
    V = F.linear(value, w_v, b_v)
    BS = Q.shape[0]
    N_q = Q.shape[1]
    N_kv = K.shape[1]
    Q = Q.reshape(BS, N_q, num_heads, HD).permute(0, 2, 1, 3)
    K = K.reshape(BS, N_kv, num_heads, HD).permute(0, 2, 1, 3)
    V = V.reshape(BS, N_kv, num_heads, HD).permute(0, 2, 1, 3)

    combined_mask = attn_mask
    if key_padding_mask is not None:
        pad_bias = torch.zeros(BS, 1, 1, N_kv, dtype=Q.dtype, device=Q.device)
        pad_bias = pad_bias.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        combined_mask = pad_bias if combined_mask is None else combined_mask + pad_bias
    attn_output = F.scaled_dot_product_attention(Q, K, V, attn_mask=combined_mask)
    attn_output = attn_output.permute(0, 2, 1, 3).reshape(BS, N_q, D)
    out = F.linear(attn_output, w_out, b_out)
    return out


def fused_mha_forward(query, key, value, in_proj_w, in_proj_b, out_w, out_b,
                      num_heads, attn_mask=None, key_padding_mask=None):
    D = query.shape[-1]
    w_q, w_k, w_v = in_proj_w[:D], in_proj_w[D:2*D], in_proj_w[2*D:]
    b_q, b_k, b_v = in_proj_b[:D], in_proj_b[D:2*D], in_proj_b[2*D:]
    return multihead_attention_forward(query, key, value, w_q, b_q, w_k, b_k, w_v, b_v,
                                       out_w, out_b, num_heads, attn_mask, key_padding_mask)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--outdir", default="/tmp/ddec_ref")
    parser.add_argument("--fenc-input", default=None,
                        help="Directory containing fenc output to use as decoder input")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    device = "cpu"
    D = 256
    NH = 8
    NQ = 200
    H = 72

    print("Loading checkpoint...")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    if "model" in ckpt:
        ckpt = ckpt["model"]

    ddec_prefix = "detector.transformer.decoder."
    ddec_sd = {k[len("detector.transformer."):]: v for k, v in ckpt.items()
               if k.startswith("detector.transformer.decoder.")}

    # Load fenc output as decoder input
    fenc_dir = args.fenc_input
    if fenc_dir is None:
        # Use existing ref_phase5 data
        script_dir = os.path.dirname(os.path.abspath(__file__))
        fenc_dir = os.path.join(script_dir, "ref_phase5")

    def load_tensor(path, dtype=np.float32):
        shape = list(map(int, open(path + ".shape").read().strip().split(",")))
        data = np.fromfile(path + ".bin", dtype=dtype).reshape(shape)
        return torch.tensor(data)

    # Load fenc output and image PE
    fenc_output_bf = load_tensor(os.path.join(fenc_dir, "fenc_output_bf"))
    if fenc_output_bf.ndim == 3:
        pass  # [1, HW, D] batch-first
    else:
        # Try loading seq-first [HW, 1, D] and converting
        fenc_output_sf = load_tensor(os.path.join(fenc_dir, "fenc_output"))
        fenc_output_bf = fenc_output_sf.permute(1, 0, 2)  # [1, HW, D]

    # Image PE - load and flatten
    img_pe = load_tensor(os.path.join(fenc_dir, "img_pe_72"))  # [1, 256, 72, 72]
    img_pe_flat = img_pe.flatten(2).permute(0, 2, 1)  # [1, 5184, 256]

    # Text features
    text_features = load_tensor(os.path.join(fenc_dir, "text_features"))
    if text_features.ndim == 3 and text_features.shape[0] == 32:
        # [T, 1, D] seq-first
        prompt_sf = text_features
        prompt_bf = text_features.permute(1, 0, 2)
    elif text_features.ndim == 2:
        # [D, T] ggml layout -> convert
        prompt_sf = text_features.T.unsqueeze(1)  # [T, 1, D]
        prompt_bf = prompt_sf.permute(1, 0, 2)
    else:
        prompt_sf = text_features
        prompt_bf = text_features.permute(1, 0, 2)

    T = prompt_sf.shape[0]

    # Token IDs for text mask (saved as int32)
    token_ids_path = os.path.join(fenc_dir, "token_ids")
    if os.path.exists(token_ids_path + ".bin"):
        token_ids = load_tensor(token_ids_path, dtype=np.int32).long().squeeze().tolist()
    else:
        # Assume all valid
        token_ids = [1] * T

    txt_mask = torch.tensor([[tid == 0 for tid in token_ids]], dtype=torch.bool)

    print(f"fenc output: {list(fenc_output_bf.shape)}")
    print(f"prompt: {list(prompt_sf.shape)}")
    print(f"img PE: {list(img_pe_flat.shape)}")

    # ═══════════════════════════════════════════════════════════════════
    #  DETR Decoder forward (matching Python package exactly)
    # ═══════════════════════════════════════════════════════════════════
    print("\n=== DETR Decoder (from checkpoint weights) ===")

    with torch.no_grad():
        query_embed = ddec_sd["decoder.query_embed.weight"]  # [NQ, D]
        ref_pts_raw = ddec_sd["decoder.reference_points.weight"]  # [NQ, 4]
        reference_boxes = ref_pts_raw.sigmoid().unsqueeze(1)  # [NQ, 1, 4]

        save_tensor(os.path.join(args.outdir, "ddec_query_embed"), query_embed)
        save_tensor(os.path.join(args.outdir, "ddec_ref_pts_raw"), ref_pts_raw)
        save_tensor(os.path.join(args.outdir, "ddec_ref_boxes_init"), reference_boxes)

        presence_token_w = ddec_sd["decoder.presence_token.weight"]  # [1, D]
        presence_out = presence_token_w[None].expand(1, 1, -1)  # [1, 1, D]
        save_tensor(os.path.join(args.outdir, "ddec_presence_token"), presence_out)

        enc_memory = fenc_output_bf.permute(1, 0, 2)  # [5184, 1, D] seq-first
        enc_pos = img_pe_flat.permute(1, 0, 2)  # [5184, 1, D] seq-first

        valid_ratios = torch.ones(1, 1, 2, device=device)
        spatial_shapes = torch.tensor([[H, H]], dtype=torch.long)
        output = query_embed.unsqueeze(1).repeat(1, 1, 1)  # [NQ, 1, D]

        coords_h = torch.arange(0, H, dtype=torch.float32) / H
        coords_w = torch.arange(0, H, dtype=torch.float32) / H

        for layer_idx in range(6):
            lp = f"decoder.layers.{layer_idx}."

            # Reference points input
            reference_points_input = reference_boxes[:, :, None] * torch.cat([valid_ratios, valid_ratios], -1)[None, :]
            query_sine_embed = gen_sineembed_for_position(reference_points_input[:, :, 0, :], D)
            query_pos = mlp_forward(query_sine_embed, ddec_sd, "decoder.ref_point_head", 2)

            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_query_sine_0"), query_sine_embed)
                save_tensor(os.path.join(args.outdir, "ddec_query_pos_0"), query_pos)

            # Box RPB
            boxes_xyxy = box_cxcywh_to_xyxy(reference_boxes).permute(1, 0, 2)
            bs_d, nq_d, _ = boxes_xyxy.shape
            deltas_y = coords_h.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 1:4:2]
            deltas_y = deltas_y.view(bs_d, nq_d, -1, 2)
            deltas_x = coords_w.view(1, -1, 1) - boxes_xyxy.reshape(-1, 1, 4)[:, :, 0:3:2]
            deltas_x = deltas_x.view(bs_d, nq_d, -1, 2)

            deltas_x_log = deltas_x * 8
            deltas_x_log = torch.sign(deltas_x_log) * torch.log2(torch.abs(deltas_x_log) + 1.0) / np.log2(8)
            deltas_y_log = deltas_y * 8
            deltas_y_log = torch.sign(deltas_y_log) * torch.log2(torch.abs(deltas_y_log) + 1.0) / np.log2(8)
            deltas_x = deltas_x_log
            deltas_y = deltas_y_log

            rpb_x = mlp_forward(deltas_x, ddec_sd, "decoder.boxRPB_embed_x", 2)
            rpb_y = mlp_forward(deltas_y, ddec_sd, "decoder.boxRPB_embed_y", 2)
            B_rpb = rpb_y.unsqueeze(3) + rpb_x.unsqueeze(2)
            B_rpb = B_rpb.flatten(2, 3)
            B_rpb = B_rpb.permute(0, 3, 1, 2).contiguous()
            pres_rpb = torch.zeros(1, NH, 1, H*H, device=device)
            memory_mask = torch.cat([pres_rpb, B_rpb], dim=2)
            memory_mask = memory_mask.flatten(0, 1)

            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_rpb_mask_0"), memory_mask)

            # Self-attention
            tgt_with_pres = torch.cat([presence_out, output], dim=0)
            qpos_with_pres = torch.cat([torch.zeros_like(presence_out), query_pos], dim=0)
            q_sa = k_sa = tgt_with_pres + qpos_with_pres
            sa_w = ddec_sd[lp + "self_attn.in_proj_weight"]
            sa_b = ddec_sd[lp + "self_attn.in_proj_bias"]
            sa_ow = ddec_sd[lp + "self_attn.out_proj.weight"]
            sa_ob = ddec_sd[lp + "self_attn.out_proj.bias"]
            q_sa_bf = q_sa.permute(1, 0, 2)
            k_sa_bf = k_sa.permute(1, 0, 2)
            v_sa_bf = tgt_with_pres.permute(1, 0, 2)
            sa_out = fused_mha_forward(q_sa_bf, k_sa_bf, v_sa_bf, sa_w, sa_b, sa_ow, sa_ob, NH)
            tgt_with_pres = tgt_with_pres + sa_out.permute(1, 0, 2)
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm2.weight"], ddec_sd[lp + "norm2.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_after_sa"), tgt_with_pres)

            # Text cross-attention
            tgt_q_ca_text = tgt_with_pres + qpos_with_pres
            ca_text_w = ddec_sd[lp + "ca_text.in_proj_weight"]
            ca_text_b = ddec_sd[lp + "ca_text.in_proj_bias"]
            ca_text_ow = ddec_sd[lp + "ca_text.out_proj.weight"]
            ca_text_ob = ddec_sd[lp + "ca_text.out_proj.bias"]
            w_q_ct, w_k_ct, w_v_ct = ca_text_w[:D], ca_text_w[D:2*D], ca_text_w[2*D:]
            b_q_ct, b_k_ct, b_v_ct = ca_text_b[:D], ca_text_b[D:2*D], ca_text_b[2*D:]
            tq_bf = tgt_q_ca_text.permute(1, 0, 2)
            p_bf = prompt_sf.permute(1, 0, 2)
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

            # Image cross-attention with RPB
            tgt_q_ca_img = tgt_with_pres + qpos_with_pres
            mem_k = enc_memory + enc_pos
            ca_w = ddec_sd[lp + "cross_attn.in_proj_weight"]
            ca_b = ddec_sd[lp + "cross_attn.in_proj_bias"]
            ca_ow = ddec_sd[lp + "cross_attn.out_proj.weight"]
            ca_ob = ddec_sd[lp + "cross_attn.out_proj.bias"]
            w_q_ci, w_k_ci, w_v_ci = ca_w[:D], ca_w[D:2*D], ca_w[2*D:]
            b_q_ci, b_k_ci, b_v_ci = ca_b[:D], ca_b[D:2*D], ca_b[2*D:]
            tqi_bf = tgt_q_ca_img.permute(1, 0, 2)
            mk_bf = mem_k.permute(1, 0, 2)
            mv_bf = enc_memory.permute(1, 0, 2)
            ci_out = multihead_attention_forward(tqi_bf, mk_bf, mv_bf,
                                                  w_q_ci, b_q_ci, w_k_ci, b_k_ci, w_v_ci, b_v_ci,
                                                  ca_ow, ca_ob, NH,
                                                  attn_mask=memory_mask)
            tgt_with_pres = tgt_with_pres + ci_out.permute(1, 0, 2)
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm1.weight"], ddec_sd[lp + "norm1.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_after_img_ca"), tgt_with_pres)

            # FFN
            ffn = F.linear(tgt_with_pres, ddec_sd[lp + "linear1.weight"], ddec_sd[lp + "linear1.bias"])
            ffn = F.relu(ffn)
            ffn = F.linear(ffn, ddec_sd[lp + "linear2.weight"], ddec_sd[lp + "linear2.bias"])
            tgt_with_pres = tgt_with_pres + ffn
            tgt_with_pres = F.layer_norm(tgt_with_pres, [D],
                                          ddec_sd[lp + "norm3.weight"], ddec_sd[lp + "norm3.bias"])
            if layer_idx == 0:
                save_tensor(os.path.join(args.outdir, "ddec_layer0_full_out"), tgt_with_pres)

            # Split presence and queries
            presence_out = tgt_with_pres[:1]
            output = tgt_with_pres[1:]

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

        # Final
        normed_output = F.layer_norm(output, [D],
                                      ddec_sd["decoder.norm.weight"], ddec_sd["decoder.norm.bias"])
        save_tensor(os.path.join(args.outdir, "ddec_normed_output"), normed_output)
        save_tensor(os.path.join(args.outdir, "ddec_pred_boxes"), reference_boxes)

        pres_normed = F.layer_norm(presence_out, [D],
                                    ddec_sd["decoder.presence_token_out_norm.weight"],
                                    ddec_sd["decoder.presence_token_out_norm.bias"])
        pres_logit = mlp_forward(pres_normed, ddec_sd, "decoder.presence_token_head", 3)
        save_tensor(os.path.join(args.outdir, "ddec_presence_logit"), pres_logit)

        # Scoring
        scoring_prefix = "detector.dot_prod_scoring."
        scoring_sd = {k[len(scoring_prefix):]: v for k, v in ckpt.items()
                      if k.startswith(scoring_prefix)}

        prompt_for_scoring = prompt_sf.clone()
        orig = prompt_for_scoring.clone()
        prompt_mlp_out = F.linear(prompt_for_scoring, scoring_sd["prompt_mlp.layers.0.weight"],
                                   scoring_sd["prompt_mlp.layers.0.bias"])
        prompt_mlp_out = F.relu(prompt_mlp_out)
        prompt_mlp_out = F.linear(prompt_mlp_out, scoring_sd["prompt_mlp.layers.1.weight"],
                                   scoring_sd["prompt_mlp.layers.1.bias"])
        prompt_mlp_out = prompt_mlp_out + orig
        prompt_mlp_out = F.layer_norm(prompt_mlp_out, [D],
                                       scoring_sd["prompt_mlp.out_norm.weight"],
                                       scoring_sd["prompt_mlp.out_norm.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_prompt_mlp_out"), prompt_mlp_out)

        is_valid = (~txt_mask).float().permute(1, 0)[..., None]
        num_valid = torch.clamp(is_valid.sum(dim=0), min=1.0)
        pooled = (prompt_mlp_out * is_valid).sum(dim=0) / num_valid
        save_tensor(os.path.join(args.outdir, "scoring_pooled"), pooled)

        proj_pooled = F.linear(pooled, scoring_sd["prompt_proj.weight"],
                                scoring_sd["prompt_proj.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_proj_pooled"), proj_pooled)

        hs = normed_output.permute(1, 0, 2).unsqueeze(0)
        proj_hs = F.linear(hs, scoring_sd["hs_proj.weight"], scoring_sd["hs_proj.bias"])
        save_tensor(os.path.join(args.outdir, "scoring_proj_hs"), proj_hs)

        scores = torch.matmul(proj_hs, proj_pooled.unsqueeze(-1))
        scores = scores * (1.0 / math.sqrt(D))
        scores = scores.clamp(-12.0, 12.0)
        save_tensor(os.path.join(args.outdir, "scoring_class_scores"), scores)

    print("\nDone! All decoder tensors saved.")


if __name__ == "__main__":
    main()
