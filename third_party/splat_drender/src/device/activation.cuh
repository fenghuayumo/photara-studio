#pragma once
#include "splat_drender/api.h"
#include <cuda_runtime.h>

namespace splat_drender {
struct ActivatedRow {
    float3 scale;
    float3 unfiltered_scale;
    float4 rotation;
    float opacity;
};
// Match Photara's activation chain, including the Mip3D determinant correction.
__device__ inline ActivatedRow activate_row(RawActivation raw, int i) {
    ActivatedRow a;
    float* s = &a.scale.x;
    float ratio = 1.f;
    const float f2 = raw.filter_3d ? raw.filter_3d[i] * raw.filter_3d[i] : 0.f;
#pragma unroll
    for (int axis = 0; axis < 3; ++axis) {
        const float v = expf(raw.log_scales[3 * i + axis]);
        (&a.unfiltered_scale.x)[axis] = v;
        s[axis] = sqrtf(v * v + f2);
        ratio *= v / s[axis];
    }
    const float w = raw.quaternions[4 * i], x = raw.quaternions[4 * i + 1];
    const float y = raw.quaternions[4 * i + 2], z = raw.quaternions[4 * i + 3];
    const float inv = rsqrtf(fmaxf(w * w + x * x + y * y + z * z, 1e-20f));
    a.rotation = make_float4(w * inv, x * inv, y * inv, z * inv);
    a.opacity = (1.f / (1.f + expf(-raw.opacity_logits[i]))) * ratio;
    return a;
}
} // namespace splat_drender
