#include "common.hlsli"

// Six exact finite order statistics, without sorting or N-sized scratch.
struct PushConstants {
    uint count;
    uint src_offset;
    uint pass;
    uint mode; // 0 histogram, 1 resolve digit, 2 write min.xyz/max.xyz
    float percentile;
};
[[vk::binding(0, 0)]] RWByteAddressBuffer src;
[[vk::binding(1, 0)]] RWByteAddressBuffer bins;
[[vk::binding(2, 0)]] RWByteAddressBuffer state;
[[vk::binding(3, 0)]] RWByteAddressBuffer result;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;
groupshared uint local_bins[256];

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    const uint target = group.y;
    const uint state_offset = target * 16u; // prefix, remaining rank, finite count, reserved
    const uint bin_offset = (pc.pass == 0u ? target % 3u : target) * 1024u;
    if (pc.mode == 2u) {
        if (lane >= 6u) return;
        bool empty = false;
        [unroll] for (uint t = 0u; t < 6u; ++t)
            empty = empty || state.Load(t * 16u + 8u) == 0u;
        const uint key = state.Load(lane * 16u);
        const float value = asfloat((key & 0x80000000u) != 0u ? key & 0x7fffffffu : ~key);
        result.Store(lane * 4u, asuint(empty ? (lane < 3u ? -1.0f : 1.0f) : value));
        return;
    }
    if (pc.mode == 1u) {
        if (lane != 0u) return;
        uint rank = state.Load(state_offset + 4u);
        if (pc.pass == 0u) {
            uint total = 0u;
            for (uint b = 0u; b < 256u; ++b) total += bins.Load(bin_offset + b * 4u);
            state.Store(state_offset + 8u, total);
            // Match the existing float32 percentile-to-rank rule exactly.
            precise float fraction = (target < 3u ? 1.0f - pc.percentile : 1.0f + pc.percentile) * 0.5f;
            rank = total == 0u ? 0u : min(uint(fraction * float(total)), total - 1u) + 1u;
        }
        if (rank == 0u) return;
        uint below = 0u;
        for (uint digit = 0u; digit < 256u; ++digit) {
            const uint n = bins.Load(bin_offset + digit * 4u);
            if (below + n >= rank) {
                state.Store(state_offset, (state.Load(state_offset) << 8u) | digit);
                state.Store(state_offset + 4u, rank - below);
                break;
            }
            below += n;
        }
        return;
    }
    local_bins[lane] = 0u;
    GroupMemoryBarrierWithGroupSync();
    const uint index = group.x * TT_GROUP_SIZE + lane;
    if (index < pc.count) {
        const float value = load_f32(src, pc.src_offset + (index * 3u + target % 3u) * 4u);
        if (isfinite(value)) {
            const uint bits = asuint(value);
            const uint key = (bits & 0x80000000u) != 0u ? ~bits : bits | 0x80000000u;
            if (pc.pass == 0u || (key >> (32u - pc.pass * 8u)) == state.Load(state_offset)) {
                const uint digit = (key >> (24u - pc.pass * 8u)) & 255u;
                InterlockedAdd(local_bins[digit], 1u);
            }
        }
    }
    GroupMemoryBarrierWithGroupSync();
    if (local_bins[lane] != 0u)
        bins.InterlockedAdd(bin_offset + lane * 4u, local_bins[lane]);
}
