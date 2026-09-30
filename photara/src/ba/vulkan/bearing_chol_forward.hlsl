#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer system;
[[vk::binding(1, 0)]] RWByteAddressBuffer unknowns;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double part[256];
groupshared double reduced;

[numthreads(256, 1, 1)]
void main(uint group_thread : SV_GroupIndex) {
    uint n = pc.n;
    for (uint row = 0u; row < n; ++row) {
        double value = 0.0;
        for (uint col = group_thread; col < row; col += 256u) {
            value += load_double(system, row + col * n) * load_double(unknowns, col);
        }
        part[group_thread] = value;
        GroupMemoryBarrierWithGroupSync();
        for (uint span = 128u; span > 0u; span >>= 1u) {
            if (group_thread < span) part[group_thread] += part[group_thread + span];
            GroupMemoryBarrierWithGroupSync();
        }
        if (group_thread == 0u) {
            reduced = (load_double(unknowns, row) - part[0]) / load_double(system, row + row * n);
        }
        GroupMemoryBarrierWithGroupSync();
        store_double(unknowns, row, reduced);
        GroupMemoryBarrierWithGroupSync();
    }
}
