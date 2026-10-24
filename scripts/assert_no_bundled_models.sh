#!/usr/bin/env bash
set -euo pipefail
stage="${1:?Usage: assert_no_bundled_models.sh STAGE}"
forbidden="$(find "$stage" \( -type d \( -iname models -o -iname model-agreements \) \) -o \( \( -type f -o -type l \) \( -iname '*.bin' -o -iname '*.bin.*' -o -iname '*.ggml' -o -iname '*.ggml.*' -o -iname '*.gguf' -o -iname '*.gguf.*' -o -iname '*.safetensors' -o -iname '*.safetensors.*' -o -iname '*.onnx' -o -iname '*.onnx.*' -o -iname '*.pt' -o -iname '*.pt.*' -o -iname '*.pth' -o -iname '*.pth.*' -o -iname '*.ckpt' -o -iname '*.ckpt.*' -o -iname '*.tflite' -o -iname '*.tflite.*' -o -iname '*.agreement' -o -iname '*.agreement.*' -o -iname '*.verified' -o -iname '*.verified.*' -o -iname '*license-accepted*' -o -iname '*licence-accepted*' -o -iname 'pytorch_model*.bin' -o -iname 'grounding-dino-vocab.txt' \) \))"
if [[ -n "$forbidden" ]]; then
    printf 'Release package contains model weights, caches or user agreements:\n%s\n' "$forbidden" >&2
    exit 1
fi
printf '%s\n' 'Release guard passed: no model weights or user agreement records.'
