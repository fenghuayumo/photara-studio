#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer tracks;
[[vk::binding(1, 0)]] RWStructuredBuffer<double> step;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint track = group_id.x * 256u + group_thread; track < pc.n; track += stride) {
        uint byte = track * kPointBytes;
        store_at(tracks, byte, load_at(tracks, byte) + step[track * 3u]);
        store_at(tracks, byte + 8u, load_at(tracks, byte + 8u) + step[track * 3u + 1u]);
        store_at(tracks, byte + 16u, load_at(tracks, byte + 16u) + step[track * 3u + 2u]);
    }
}
