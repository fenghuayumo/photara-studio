#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(1, 0)]] RWByteAddressBuffer indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer track_ids;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> camera_rhs;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> reduced;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> cross;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> output;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    bool fix_first = (pc.flags & kFixPose) != 0u;
    bool opt_rot = (pc.flags & kOptRot) != 0u;
    bool opt_trans = (pc.flags & kOptTrans) != 0u;
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint camera = group_id.x * 256u + group_thread; camera < pc.n; camera += stride) {
        double result[6];
        [unroll] for (uint row = 0u; row < 6u; ++row) result[row] = camera_rhs[camera * 6u + row];
        if (fix_first && camera == 0u) {
            [unroll] for (uint row = 0u; row < 6u; ++row) result[row] = 0.0;
        } else {
            uint begin = load_uint(offsets, camera);
            uint end = load_uint(offsets, camera + 1u);
            for (uint cursor = begin; cursor < end; ++cursor) {
                uint observation = load_uint(indices, cursor);
                uint track = load_uint(track_ids, observation);
                [unroll] for (uint row = 0u; row < 6u; ++row) {
                    result[row] -= cross[observation * 18u + row * 3u] * reduced[track * 3u] +
                                   cross[observation * 18u + row * 3u + 1u] * reduced[track * 3u + 1u] +
                                   cross[observation * 18u + row * 3u + 2u] * reduced[track * 3u + 2u];
                }
            }
        }
        [unroll] for (uint row = 0u; row < 6u; ++row) {
            bool active = row < 3u ? opt_rot : opt_trans;
            output[camera * 6u + row] = active ? result[row] : 0.0;
        }
    }
}
