#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> hessian;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> factors;
[[vk::binding(2, 0)]] RWStructuredBuffer<double> residual;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    bool fix_first = (pc.flags & kFixPose) != 0u;
    bool opt_rot = (pc.flags & kOptRot) != 0u;
    bool opt_trans = (pc.flags & kOptTrans) != 0u;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        if (fix_first && camera == 0u) {
            [unroll] for (uint row = 0u; row < 6u; ++row) output[camera * 6u + row] = 0.0;
            continue;
        }
        bool valid = factors[camera * 36u] > 0.0;
        double solution[6];
        if (!valid) {
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                solution[row] = residual[camera * 6u + row] /
                                fmax_d(hessian[camera * 36u + row * 6u + row], 1e-12);
            }
        } else {
            double temporary[6];
            [unroll] for (uint row = 0u; row < 6u; ++row) temporary[row] = 0.0;
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                double value = residual[camera * 6u + row];
                for (uint k = 0u; k < row; ++k)
                    value -= factors[camera * 36u + row * 6u + k] * temporary[k];
                temporary[row] = value / factors[camera * 36u + row * 6u + row];
            }
            [unroll] for (uint row = 0u; row < 6u; ++row) solution[row] = 0.0;
            for (int row = 5; row >= 0; --row) {
                double value = temporary[row];
                for (int k = row + 1; k < 6; ++k)
                    value -= factors[camera * 36u + uint(k) * 6u + uint(row)] * solution[k];
                solution[row] = value / factors[camera * 36u + uint(row) * 6u + uint(row)];
            }
        }
        [unroll] for (uint row = 0u; row < 6u; ++row) {
            bool active = row < 3u ? opt_rot : opt_trans;
            output[camera * 6u + row] = active ? solution[row] : 0.0;
        }
    }
}
