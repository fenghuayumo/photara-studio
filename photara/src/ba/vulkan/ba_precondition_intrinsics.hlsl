#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessian;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> factors;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> residual;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint dof = pc.dof;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint group = group_id.x * 256u + group_thread; group < pc.n; group += stride) {
        uint rhs_base = pc.camera_values + group * dof;
        bool valid = factors[group * 64u] > 0.0;
        if (!valid) {
            for (uint row = 0u; row < dof; ++row) {
                output[rhs_base + row] =
                    residual[rhs_base + row] /
                    fmax_d(hessian[group * dof * dof + row * dof + row], 1e-12);
            }
            continue;
        }
        double temporary[8];
        double solution[8];
        [unroll] for (uint row = 0u; row < 8u; ++row) {
            temporary[row] = 0.0;
            solution[row] = 0.0;
        }
        for (uint row = 0u; row < dof; ++row) {
            double value = residual[rhs_base + row];
            for (uint k = 0u; k < row; ++k)
                value -= factors[group * 64u + row * 8u + k] * temporary[k];
            temporary[row] = value / factors[group * 64u + row * 8u + row];
        }
        for (uint remaining = dof; remaining > 0u; --remaining) {
            uint row = remaining - 1u;
            double value = temporary[row];
            for (uint k = row + 1u; k < dof; ++k)
                value -= factors[group * 64u + k * 8u + row] * solution[k];
            solution[row] = value / factors[group * 64u + row * 8u + row];
        }
        for (uint row = 0u; row < dof; ++row) output[rhs_base + row] = solution[row];
    }
}
