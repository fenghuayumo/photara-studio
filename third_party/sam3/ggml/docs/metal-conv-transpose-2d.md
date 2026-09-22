# ggml `CONV_TRANSPOSE_2D` semantics

Source of truth: [`ggml_compute_forward_conv_transpose_2d()`](../src/ggml-cpu/ops.cpp).

This note documents the exact contract the Metal kernel mirrors.

## Tensor layouts

- `src0` weights: `(KW, KH, Cout, Cin)`, contiguous, `F16` in the CPU implementation used by SAM.
- `src1` input: `(IW, IH, Cin, N)`, contiguous, `F32`.
- `dst` output: `(OW, OH, Cout, N)`, `F32`.

For `ggml_conv_transpose_2d_p0()` the output size is:

- `OW = (IW - 1) * stride + KW`
- `OH = (IH - 1) * stride + KH`

The `p0` variant used here has:

- zero padding only
- one shared spatial stride `s0`
- no dilation
- no output padding
- no bias term

## CPU computation

The CPU implementation permutes weights and inputs into temporary buffers, zeros `dst`,
then performs a scatter-add over output channel, input spatial position, and kernel tap:

```text
for out_c in [0, Cout):
  for in_y in [0, IH):
    for in_x in [0, IW):
      for kh in [0, KH):
        for kw in [0, KW):
          dst[in_x * stride + kw, in_y * stride + kh, out_c] +=
            dot(input[in_x, in_y, :, 0], weight[kw, kh, out_c, :, 0])
```

The Metal reference kernel uses the equivalent gather form for one output element:

```text
dst[ow, oh, out_c] =
  sum over kh, kw, in_c where
    ow = in_x * stride + kw
    oh = in_y * stride + kh
  of weight[kw, kh, out_c, in_c] * input[in_x, in_y, in_c]
```

## Numeric behavior

- input source type: `F32`, but the CPU implementation first packs it to `F16`
- weight type: `F16` or `F32` on Metal, `F16` in the CPU path used here
- multiply inputs: `F16 x F16` in the reference CPU path
- accumulation type: `float`
- output type: `F32`

## Scope

The current CPU implementation only uses batch `N = 1` in practice for this path, and
the Metal implementation matches that contract explicitly.
