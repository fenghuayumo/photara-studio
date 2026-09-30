#include "ba_common.hlsli"
#include "ba_atomic.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(2, 0)]] RWByteAddressBuffer indices;
[[vk::binding(3, 0)]] RWByteAddressBuffer pose_group;
[[vk::binding(4, 0)]] RWStructuredBuffer<double> camera_h;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> camera_b;
[[vk::binding(6, 0)]] RWStructuredBuffer<double> pose_intr;
[[vk::binding(7, 0)]] RWStructuredBuffer<double> intrinsic_h;
[[vk::binding(8, 0)]] RWStructuredBuffer<double> intrinsic_b;

[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double red[32];

[numthreads(32, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u);
    uint dof = pc.dof;
    for (uint camera = group_id.x; camera < pc.n; camera += stride) {
        double local_h[36];
        double local_b[6];
        double local_ci[48];
        double local_ii[64];
        double local_bi[8];
        [unroll] for (uint z = 0u; z < 36u; ++z) local_h[z] = 0.0;
        [unroll] for (uint z = 0u; z < 6u; ++z) local_b[z] = 0.0;
        [unroll] for (uint z = 0u; z < 48u; ++z) local_ci[z] = 0.0;
        [unroll] for (uint z = 0u; z < 64u; ++z) local_ii[z] = 0.0;
        [unroll] for (uint z = 0u; z < 8u; ++z) local_bi[z] = 0.0;
        uint begin = load_uint(offsets, camera);
        uint end = load_uint(offsets, camera + 1u);
        for (uint cursor = begin + lane; cursor < end; cursor += 32u) {
            uint observation = load_uint(indices, cursor);
            if (!lin_valid(lin, observation)) continue;
            [unroll] for (uint row = 0u; row < 6u; ++row) {
                double jc0 = load_lin(lin, observation, 2u + row);
                double jc1 = load_lin(lin, observation, 8u + row);
                local_b[row] -= jc0 * load_lin(lin, observation, 0u) +
                                jc1 * load_lin(lin, observation, 1u);
                [unroll] for (uint column = 0u; column < 6u; ++column) {
                    local_h[row * 6u + column] +=
                        jc0 * load_lin(lin, observation, 2u + column) +
                        jc1 * load_lin(lin, observation, 8u + column);
                }
            }
            if (dof > 0u) {
                for (uint param = 0u; param < dof; ++param) {
                    double ji0 = load_lin(lin, observation, 20u + param);
                    double ji1 = load_lin(lin, observation, 28u + param);
                    local_bi[param] -= ji0 * load_lin(lin, observation, 0u) +
                                       ji1 * load_lin(lin, observation, 1u);
                    [unroll] for (uint row = 0u; row < 6u; ++row) {
                        local_ci[row * 8u + param] +=
                            load_lin(lin, observation, 2u + row) * ji0 +
                            load_lin(lin, observation, 8u + row) * ji1;
                    }
                    for (uint other = 0u; other < dof; ++other) {
                        local_ii[param * 8u + other] +=
                            ji0 * load_lin(lin, observation, 20u + other) +
                            ji1 * load_lin(lin, observation, 28u + other);
                    }
                }
            }
        }
        [unroll] for (uint element = 0u; element < 36u; ++element) {
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
            if (lane == 0u) camera_h[camera * 36u + element] = red[0];
        }
        [unroll] for (uint element = 0u; element < 6u; ++element) {
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
            if (lane == 0u) camera_b[camera * 6u + element] = red[0];
        }
        if (dof == 0u) continue;
        for (uint row = 0u; row < 6u; ++row) {
            for (uint param = 0u; param < dof; ++param) {
                red[lane] = local_ci[row * 8u + param];
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
                if (lane == 0u)
                    pose_intr[camera * 6u * dof + row * dof + param] = red[0];
            }
        }
        uint group = load_uint(pose_group, camera);
        for (uint param = 0u; param < dof; ++param) {
            red[lane] = local_bi[param];
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
            if (lane == 0u)
                atomic_add_f64(intrinsic_b[group * dof + param], 1u, 0u, red[0]);
            for (uint other = 0u; other < dof; ++other) {
                red[lane] = local_ii[param * 8u + other];
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
                if (lane == 0u)
                    atomic_add_f64(
                        intrinsic_h[group * dof * dof + param * dof + other],
                        1u, 0u, red[0]);
            }
        }
    }
}
