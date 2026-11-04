#include "splat_gut_sh_math.hlsli"
float3 evaluate_gut_sh_unclamped(uint degree, float3 pos, float3 eye,
                                 StructuredBuffer<float> coefficients, uint offset,
                                 out float clamped0, out float clamped1, out float clamped2) {
    float3 d = gut_unit_direction(pos - eye);
    GutShBasis basis = gut_sh_basis(int(degree), d.x, d.y, d.z);
    float3 color = float3(gut_mul(basis.value[0], coefficients[offset]),
                          gut_mul(basis.value[0], coefficients[offset + 1u]),
                          gut_mul(basis.value[0], coefficients[offset + 2u]));
    if (degree > 0u) {
        [unroll] for (uint channel = 0u; channel < 3u; ++channel) {
            float pair = gut_fma(basis.value[1], coefficients[offset + 3u + channel],
                                 gut_mul(basis.value[2], coefficients[offset + 6u + channel]));
            float first =
                gut_add(pair, gut_mul(basis.value[3], coefficients[offset + 9u + channel]));
            color[channel] = gut_add(color[channel], first);
        }
    }
    uint active = (degree + 1u) * (degree + 1u);
    [unroll] for (uint k = 4u; k < 16u; ++k) if (k < active) {
        color.x = gut_fma(basis.value[k], coefficients[offset + 3u * k], color.x);
        color.y = gut_fma(basis.value[k], coefficients[offset + 3u * k + 1u], color.y);
        color.z = gut_fma(basis.value[k], coefficients[offset + 3u * k + 2u], color.z);
    }
    color = float3(gut_add(color.x, .5f), gut_add(color.y, .5f), gut_add(color.z, .5f));
    clamped0 = color.x < 0.0f ? 1.0f : 0.0f;
    clamped1 = color.y < 0.0f ? 1.0f : 0.0f;
    clamped2 = color.z < 0.0f ? 1.0f : 0.0f;
    return color;
}
