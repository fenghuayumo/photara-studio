#!/usr/bin/env python3
"""
Dump fusion encoder inputs and per-layer outputs from the ACTUAL SAM3 Python package.

This script monkey-patches TransformerEncoder.forward to intercept the exact tensors
flowing through the fusion encoder during real inference. This ensures the reference
tensors are bit-identical to what the real model produces — no reimplementation.

Usage:
    cd ~/Documents/sam3
    uv run python /Users/pierre-antoine/Documents/sam3.cpp/tests/dump_fenc_from_package.py \
        [--image /path/to/image.jpg] [--text "yellow school bus"] \
        [--outdir /Users/pierre-antoine/Documents/sam3.cpp/tests/ref_fenc_pkg]
"""
import argparse
import os
import sys
import types

# ── Mock triton for macOS (must happen before any torch/sam3 import) ──
def _install_triton_mock():
    """Install a mock triton package so SAM3 and torch._dynamo can import."""
    if "triton" in sys.modules:
        return  # Already real or already mocked

    triton = types.ModuleType("triton")
    triton.__path__ = []  # Make it a package
    triton.jit = lambda fn=None, **kw: fn if fn else (lambda f: f)
    triton.autotune = lambda **kw: lambda f: f
    triton.heuristics = lambda **kw: lambda f: f
    triton.Config = type("Config", (), {"__init__": lambda self, **kw: None})

    lang = types.ModuleType("triton.language")
    lang.__path__ = []
    lang.constexpr = int
    lang.dtype = type  # torch._dynamo.utils needs this
    triton.language = lang

    core = types.ModuleType("triton.language.core")
    core.dtype = type
    lang.core = core

    backends = types.ModuleType("triton.backends")
    backends.__path__ = []
    compiler = types.ModuleType("triton.backends.compiler")
    backends.compiler = compiler

    runtime = types.ModuleType("triton.runtime")
    runtime.__path__ = []

    triton_compiler = types.ModuleType("triton.compiler")
    triton_compiler.__path__ = []
    triton_compiler_compiler = types.ModuleType("triton.compiler.compiler")
    triton_compiler.compiler = triton_compiler_compiler

    # Wire submodule attributes (so triton.backends.compiler works)
    triton.backends = backends
    triton.runtime = runtime
    triton.compiler = triton_compiler
    lang.core = core
    backends.compiler = compiler
    triton_compiler.compiler = triton_compiler_compiler

    for name, mod in [
        ("triton", triton),
        ("triton.language", lang),
        ("triton.language.core", core),
        ("triton.backends", backends),
        ("triton.backends.compiler", compiler),
        ("triton.runtime", runtime),
        ("triton.compiler", triton_compiler),
        ("triton.compiler.compiler", triton_compiler_compiler),
    ]:
        sys.modules[name] = mod

_install_triton_mock()

import numpy as np
import torch

# Force CPU — SAM3 model builder uses device="cuda" in some places.
# Monkey-patch torch.zeros/torch.ones/torch.empty to replace device="cuda" with "cpu".
_orig_zeros = torch.zeros
_orig_ones = torch.ones
_orig_empty = torch.empty

def _cpu_override(orig_fn):
    def wrapper(*args, **kwargs):
        dev = kwargs.get("device")
        if isinstance(dev, str) and "cuda" in dev:
            kwargs["device"] = "cpu"
        elif isinstance(dev, torch.device) and dev.type == "cuda":
            kwargs["device"] = "cpu"
        return orig_fn(*args, **kwargs)
    return wrapper

_orig_arange = torch.arange
_orig_linspace = torch.linspace
_orig_full = torch.full
_orig_randn = torch.randn
_orig_rand = torch.rand
_orig_tensor = torch.tensor

# Also neuter pin_memory — it tries to use MPS on macOS and causes device mismatch
_orig_pin_memory = torch.Tensor.pin_memory
torch.Tensor.pin_memory = lambda self, *a, **kw: self

torch.zeros = _cpu_override(_orig_zeros)
torch.ones = _cpu_override(_orig_ones)
torch.empty = _cpu_override(_orig_empty)
torch.arange = _cpu_override(_orig_arange)
torch.linspace = _cpu_override(_orig_linspace)
torch.full = _cpu_override(_orig_full)
torch.randn = _cpu_override(_orig_randn)
torch.rand = _cpu_override(_orig_rand)
torch.tensor = _cpu_override(_orig_tensor)

# ---------------------------------------------------------------------------
# Tensor save helpers (matching existing test infrastructure format)
# ---------------------------------------------------------------------------

def save_tensor(path, t):
    """Save tensor as .bin + .shape (PyTorch native layout)."""
    t = t.detach().cpu().float().contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path}  shape={list(t.shape)}  "
          f"min={t.min().item():.4f} max={t.max().item():.4f}")


def save_tensor_i32(path, t):
    t = t.detach().cpu().to(torch.int32).contiguous()
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path + ".bin", "wb") as f:
        f.write(t.numpy().tobytes())
    with open(path + ".shape", "w") as f:
        f.write(",".join(str(d) for d in t.shape))
    print(f"  saved {path}  shape={list(t.shape)} dtype=int32")


# ---------------------------------------------------------------------------
# Global capture dict — filled by the monkey-patched encoder forward
# ---------------------------------------------------------------------------

captured = {}


def install_encoder_patch():
    """
    Monkey-patch TransformerEncoder.forward (the base class used by
    TransformerEncoderFusion) to dump inputs and per-layer outputs.

    TransformerEncoderFusion.forward() does:
      1. Reshape src from seq-first to NCHW
      2. (Optional pooled text — disabled with add_pooled_text_to_img_feat=False)
      3. Call super().forward(src_NCHW, ..., prompt=prompt.transpose(0,1), ...)

    super().forward() (TransformerEncoder.forward) does:
      1. _prepare_multilevel_features → src_flatten [B, HW, D], lvl_pos_embed_flatten [B, HW, D]
      2. Loop through 6 encoder layers

    We patch TransformerEncoder.forward to capture the flattened tensors.
    """
    from sam3.model.encoder import TransformerEncoder
    from sam3.model.act_ckpt_utils import activation_ckpt_wrapper

    original_forward = TransformerEncoder.forward

    def patched_forward(self, src, src_key_padding_masks=None, pos=None,
                        prompt=None, prompt_key_padding_mask=None,
                        encoder_extra_kwargs=None):
        # --- _prepare_multilevel_features (same as original) ---
        (
            src_flatten,
            key_padding_masks_flatten,
            lvl_pos_embed_flatten,
            level_start_index,
            valid_ratios,
            spatial_shapes,
        ) = self._prepare_multilevel_features(src, src_key_padding_masks, pos)

        reference_points = self.get_reference_points(
            spatial_shapes, valid_ratios, device=src_flatten.device
        )

        # ═══ DUMP INPUTS ═══
        # src_flatten: [B, HW, D] = [1, 5184, 256] batch-first image features
        captured["fenc_input_tgt"] = src_flatten.detach().clone()
        # lvl_pos_embed_flatten: [B, HW, D] = [1, 5184, 256] positional encoding
        captured["fenc_input_pos"] = lvl_pos_embed_flatten.detach().clone()
        # prompt: [B, T, D] = [1, 32, 256] batch-first text tokens
        if prompt is not None:
            captured["fenc_input_prompt"] = prompt.detach().clone()
        # prompt_key_padding_mask: [B, T] = [1, 32] boolean (True=padding)
        if prompt_key_padding_mask is not None:
            captured["fenc_input_prompt_mask"] = prompt_key_padding_mask.detach().clone()

        # --- Layer loop (same as original) ---
        output = src_flatten
        for layer_idx, layer in enumerate(self.layers):
            layer_kwargs = {}

            assert hasattr(layer, 'forward_pre') or hasattr(layer, 'forward_post')
            layer_kwargs["memory"] = prompt
            layer_kwargs["memory_key_padding_mask"] = prompt_key_padding_mask
            layer_kwargs["query_pos"] = lvl_pos_embed_flatten
            layer_kwargs["tgt"] = output
            layer_kwargs["tgt_key_padding_mask"] = key_padding_masks_flatten

            if encoder_extra_kwargs is not None:
                layer_kwargs.update(encoder_extra_kwargs)
            output = activation_ckpt_wrapper(layer)(
                **layer_kwargs,
                act_ckpt_enable=self.training and self.use_act_checkpoint,
            )

            # ═══ DUMP PER-LAYER OUTPUT ═══
            captured[f"fenc_layer{layer_idx}_out"] = output.detach().clone()

        # --- Return (same as original) ---
        return (
            output.transpose(0, 1),
            (
                key_padding_masks_flatten.transpose(0, 1)
                if key_padding_masks_flatten is not None
                else None
            ),
            lvl_pos_embed_flatten.transpose(0, 1),
            level_start_index,
            spatial_shapes,
            valid_ratios,
        )

    TransformerEncoder.forward = patched_forward
    print("[PATCH] TransformerEncoder.forward patched for tensor dumping")


def install_text_encoder_hook(model):
    """
    Hook into the text encoder output to capture text features and token info.
    We intercept backbone.forward_text by wrapping it.
    """
    original_forward_text = model.backbone.forward_text

    def patched_forward_text(text_list, **kwargs):
        result = original_forward_text(text_list, **kwargs)
        # language_features: [T, num_prompts, D] = [32, 1, 256]
        if "language_features" in result:
            captured["text_features_raw"] = result["language_features"].detach().clone()
        if "language_mask" in result:
            captured["text_mask_raw"] = result["language_mask"].detach().clone()
        return result

    model.backbone.forward_text = patched_forward_text
    print("[PATCH] backbone.forward_text wrapped for text feature capture")


def install_prompt_hook(model):
    """
    Hook into _encode_prompt to capture the assembled prompt (text + geo + visual).
    """
    original_encode_prompt = model._encode_prompt

    def patched_encode_prompt(*args, **kwargs):
        prompt, prompt_mask, backbone_out = original_encode_prompt(*args, **kwargs)
        # prompt: [seq, batch, dim] = [T+geo, 1, 256] seq-first
        captured["prompt_assembled"] = prompt.detach().clone()
        captured["prompt_mask_assembled"] = prompt_mask.detach().clone()
        return prompt, prompt_mask, backbone_out

    model._encode_prompt = patched_encode_prompt
    print("[PATCH] _encode_prompt wrapped for prompt capture")


def install_backbone_hook(model):
    """
    Capture backbone FPN features after set_image.
    """
    original_forward_image = model.backbone.forward_image

    def patched_forward_image(image):
        result = original_forward_image(image)
        if "backbone_fpn" in result:
            for i, feat in enumerate(result["backbone_fpn"]):
                captured[f"backbone_fpn_{i}"] = feat.detach().clone()
        if "vision_pos_enc" in result:
            for i, pe in enumerate(result["vision_pos_enc"]):
                captured[f"vision_pos_enc_{i}"] = pe.detach().clone()
        return result

    model.backbone.forward_image = patched_forward_image
    print("[PATCH] backbone.forward_image wrapped for FPN capture")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Dump fenc tensors from SAM3 package")
    parser.add_argument("--image", default=None,
                        help="Test image path (default: tests/test_random.jpg)")
    parser.add_argument("--text", default="yellow school bus",
                        help="Text prompt")
    parser.add_argument("--outdir",
                        default="/Users/pierre-antoine/Documents/sam3.cpp/tests/ref_fenc_pkg",
                        help="Output directory for dumped tensors")
    args = parser.parse_args()

    if args.image is None:
        args.image = "/Users/pierre-antoine/Documents/sam3.cpp/tests/test_random.jpg"

    os.makedirs(args.outdir, exist_ok=True)

    # ── Load model ──
    print("Loading SAM3 model...")
    sys.path.insert(0, os.path.expanduser("~/Documents/sam3"))

    import sam3
    from sam3 import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    sam3_pkg_dir = os.path.dirname(sam3.__file__)  # ~/Documents/sam3/sam3/
    torch.set_default_dtype(torch.float32)

    bpe_path = os.path.join(sam3_pkg_dir, "assets", "bpe_simple_vocab_16e6.txt.gz")
    if not os.path.exists(bpe_path):
        # Try parent dir
        bpe_path = os.path.join(sam3_pkg_dir, "..", "assets", "bpe_simple_vocab_16e6.txt.gz")
    print(f"  BPE path: {bpe_path} (exists={os.path.exists(bpe_path)})")

    ckpt_path = "/Users/pierre-antoine/Documents/sam3.cpp/raw_weights/sam3.pt"
    print(f"  Checkpoint: {ckpt_path} (exists={os.path.exists(ckpt_path)})")

    model = build_sam3_image_model(
        bpe_path=bpe_path,
        checkpoint_path=ckpt_path,
        device="cpu",
        load_from_HF=False,
    )
    model = model.float()
    model.eval()

    # ── Install all patches ──
    install_encoder_patch()
    install_text_encoder_hook(model)
    install_prompt_hook(model)
    install_backbone_hook(model)

    # ── Load and preprocess image ──
    from PIL import Image
    print(f"Loading image: {args.image}")
    image = Image.open(args.image).convert("RGB")

    processor = Sam3Processor(model, confidence_threshold=0.5, device="cpu")

    # ── Step 1: Encode image (runs backbone) ──
    print("\n=== Step 1: Encoding image ===")
    state = processor.set_image(image)

    # Save backbone features
    for key in sorted(captured.keys()):
        if key.startswith("backbone_fpn_") or key.startswith("vision_pos_enc_"):
            save_tensor(os.path.join(args.outdir, key), captured[key])

    # ── Step 2: Run text prompt (runs text encoder + fusion encoder + decoder) ──
    print(f"\n=== Step 2: Running text prompt '{args.text}' ===")
    state = processor.set_text_prompt(prompt=args.text, state=state)

    # ── Save all captured tensors ──
    print(f"\n=== Saving tensors to {args.outdir} ===")

    # Save prompt text
    with open(os.path.join(args.outdir, "prompt.txt"), "w") as f:
        f.write(args.text + "\n")

    # -- Fusion encoder inputs --
    print("\n-- Fusion encoder inputs --")
    # fenc_input_tgt: [1, 5184, 256] batch-first image features
    # Memory layout: 5184 blocks of 256 floats = same as ggml [256, 5184, 1]
    if "fenc_input_tgt" in captured:
        save_tensor(os.path.join(args.outdir, "fenc_input_tgt"), captured["fenc_input_tgt"])

    # fenc_input_pos: [1, 5184, 256] batch-first positional encoding
    if "fenc_input_pos" in captured:
        save_tensor(os.path.join(args.outdir, "fenc_input_pos"), captured["fenc_input_pos"])

    # fenc_input_prompt: [1, T, 256] batch-first prompt tokens
    if "fenc_input_prompt" in captured:
        save_tensor(os.path.join(args.outdir, "fenc_input_prompt"), captured["fenc_input_prompt"])

    # fenc_input_prompt_mask: [1, T] boolean (True=padding)
    if "fenc_input_prompt_mask" in captured:
        mask = captured["fenc_input_prompt_mask"]
        save_tensor(os.path.join(args.outdir, "fenc_input_prompt_mask"),
                    mask.float())
        # Also save as attention bias (0.0 for valid, -1e9 for padding)
        # This is what the C++ uses directly
        attn_bias = torch.where(mask, torch.tensor(-1e9), torch.tensor(0.0))
        save_tensor(os.path.join(args.outdir, "fenc_attn_bias"), attn_bias)

    # -- Also save in format compatible with existing phase 5 test --
    print("\n-- Phase 5 compatible format --")
    if "fenc_input_tgt" in captured:
        # [1, 5184, 256] → same binary layout as [32, 1, 256] seq-first
        save_tensor(os.path.join(args.outdir, "fenc_img_input"), captured["fenc_input_tgt"])
    if "fenc_input_pos" in captured:
        save_tensor(os.path.join(args.outdir, "fenc_pos_embed"), captured["fenc_input_pos"])

    # Text features in seq-first format [T, B, D] for compatibility
    if "prompt_assembled" in captured:
        # prompt_assembled is [T, B, D] seq-first
        save_tensor(os.path.join(args.outdir, "text_features"), captured["prompt_assembled"])

    # -- Fusion encoder per-layer outputs --
    print("\n-- Fusion encoder per-layer outputs --")
    for i in range(6):
        key = f"fenc_layer{i}_out"
        if key in captured:
            save_tensor(os.path.join(args.outdir, key), captured[key])

    # -- Text encoder raw output --
    print("\n-- Text encoder output --")
    if "text_features_raw" in captured:
        save_tensor(os.path.join(args.outdir, "text_features_raw"), captured["text_features_raw"])
    if "text_mask_raw" in captured:
        save_tensor(os.path.join(args.outdir, "text_mask_raw"), captured["text_mask_raw"].float())

    # -- Also save neck features for potential full pipeline test --
    print("\n-- Neck detector features --")
    # The detector uses backbone_fpn levels. Save the last 3 for the seg head.
    for i in range(4):
        key = f"backbone_fpn_{i}"
        if key in captured:
            t = captured[key]
            # Save as neck_det_{i} in NCHW format
            save_tensor(os.path.join(args.outdir, f"neck_det_{i}"), t)

    # -- Summary --
    print(f"\n=== Summary ===")
    print(f"Total tensors captured: {len(captured)}")
    if "fenc_input_tgt" in captured:
        print(f"  fenc image features: {list(captured['fenc_input_tgt'].shape)}")
    if "fenc_input_prompt" in captured:
        t = captured["fenc_input_prompt"]
        print(f"  fenc prompt tokens:  {list(t.shape)}")
    if "fenc_input_prompt_mask" in captured:
        mask = captured["fenc_input_prompt_mask"]
        n_valid = (~mask).sum().item()
        n_total = mask.numel()
        print(f"  prompt mask: {n_valid}/{n_total} valid tokens")
    if "fenc_layer5_out" in captured:
        print(f"  fenc output:         {list(captured['fenc_layer5_out'].shape)}")

    # Check for detection results
    if "boxes" in state and len(state.get("boxes", [])) > 0:
        print(f"\n  Detections: {len(state['boxes'])} objects found")
    else:
        print(f"\n  Detections: none (may need different prompt/image)")

    print(f"\nAll tensors saved to {args.outdir}/")
    print("Next: run the C++ test_debug_fenc against these references.")


if __name__ == "__main__":
    main()
