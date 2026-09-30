#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(2, 0)]] RWByteAddressBuffer indices;
[[vk::binding(3, 0)]] RWStructuredBuffer<double> track_h;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> track_b;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> cross;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> track_intr;

[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double red[32];

[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u);
    uint dof = pc.dof;
    for (uint track = group_id.x; track < pc.n; track += stride) {
        double local_h[9];
        double local_b[3];
        [unroll] for (uint z = 0u; z < 9u; ++z) local_h[z] = 0.0;
        [unroll] for (uint z = 0u; z < 3u; ++z) local_b[z] = 0.0;
        uint begin = load_uint(offsets, track);
        uint end = load_uint(offsets, track + 1u);
        for (uint cursor = begin + lane; cursor < end; cursor += 32u) {
            uint observation = load_uint(indices, cursor);
            if (!lin_valid(lin, observation)) {
                [unroll] for (uint i = 0u; i < 18u; ++i) cross[observation * 18u + i] = 0.0;
                if (dof > 0u) {
                    for (uint i = 0u; i < dof * 3u; ++i)
                        track_intr[observation * dof * 3u + i] = 0.0;
                }
                continue;
            }
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                double jc0 = load_lin(lin, observation, 2u + row);
                double jc1 = load_lin(lin, observation, 8u + row);
                [unroll] for (uint column = 0u; column < 3u; ++column) {
                    cross[observation * 18u + row * 3u + column] =
                        jc0 * load_lin(lin, observation, 14u + column) +
                        jc1 * load_lin(lin, observation, 17u + column);
                }
            }
            [unroll] for (uint row = 0u; row < 3u; ++row) {
                double jp0 = load_lin(lin, observation, 14u + row);
                double jp1 = load_lin(lin, observation, 17u + row);
                if (dof > 0u) {
                    for (uint param = 0u; param < dof; ++param) {
                        track_intr[observation * dof * 3u + row * dof + param] =
                            jp0 * load_lin(lin, observation, 20u + param) +
                            jp1 * load_lin(lin, observation, 28u + param);
                    }
                }
                local_b[row] -= jp0 * load_lin(lin, observation, 0u) +
                                jp1 * load_lin(lin, observation, 1u);
                [unroll] for (uint column = 0u; column < 3u; ++column) {
                    local_h[row * 3u + column] +=
                        jp0 * load_lin(lin, observation, 14u + column) +
                        jp1 * load_lin(lin, observation, 17u + column);
                }
            }
        }
        [unroll] for (uint element = 0u; element < 9u; ++element) {
            red[lane] = local_h[element];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 16u) red[lane] += red[lane + 16u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 8u) red[lane] += red[lane + 8u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 4u) red[lane] += red[lane + 4u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 2u) red[lane] += red[lane + 2u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 1u) red[lane] += red[lane + 1u];
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u) track_h[track * 9u + element] = red[0];
        }
        [unroll] for (uint element = 0u; element < 3u; ++element) {
            red[lane] = local_b[element];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 16u) red[lane] += red[lane + 16u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 8u) red[lane] += red[lane + 8u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 4u) red[lane] += red[lane + 4u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 2u) red[lane] += red[lane + 2u];
            GroupMemoryBarrierWithGroupSync();
            if (lane < 1u) red[lane] += red[lane + 1u];
            GroupMemoryBarrierWithGroupSync();
            if (lane == 0u) track_b[track * 3u + element] = red[0];
        }
    }
}
