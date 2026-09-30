#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessian;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> factors;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dim = pc.p0;
    uint stride = pc.p1;
    uint step = max(pc.groups, 1u) * 256u;
    for (uint block = group_id.x * 256u + group_thread; block < pc.n; block += step) {
        double lower[64];
        [unroll] for (uint i = 0u; i < 64u; ++i) lower[i] = 0.0;
        bool valid = true;
        for (uint row = 0u; row < dim && valid; ++row) {
            for (uint column = 0u; column <= row; ++column) {
                double value = hessian[block * dim * dim + row * dim + column];
                for (uint k = 0u; k < column; ++k)
                    value -= lower[row * stride + k] * lower[column * stride + k];
                if (row == column) {
                    if (!(value > 1e-24) || !isfinite_d(value)) {
                        valid = false;
                        break;
                    }
                    lower[row * stride + column] = sqrt_d(value);
                } else {
                    lower[row * stride + column] = value / lower[column * stride + column];
                }
            }
        }
        if (!valid) lower[0] = 0.0;
        for (uint i = 0u; i < stride * stride; ++i)
            factors[block * stride * stride + i] = lower[i];
    }
}
