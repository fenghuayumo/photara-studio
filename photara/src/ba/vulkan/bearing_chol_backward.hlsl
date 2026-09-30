#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer system;
[[vk::binding(1, 0)]] RWByteAddressBuffer unknowns;
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared double part[256];
groupshared double reduced;

[numthreads(256, 1, 1)]
void main(uint group_thread : SV_GroupIndex) {
    uint n = pc.n;
    for (int row = int(n) - 1; row >= 0; --row) {
        double value = 0.0;
        for (uint col = uint(row) + 1u + group_thread; col < n; col += 256u) {
            value += load_double(system, col + uint(row) * n) * load_double(unknowns, col);
        }
        part[group_thread] = value;
        GroupMemoryBarrierWithGroupSync();
        for (uint span = 128u; span > 0u; span >>= 1u) {
            if (group_thread < span) part[group_thread] += part[group_thread + span];
            GroupMemoryBarrierWithGroupSync();
        }
        if (group_thread == 0u) {
            reduced = (load_double(unknowns, uint(row)) - part[0]) / load_double(system, uint(row) + uint(row) * n);
        }
        GroupMemoryBarrierWithGroupSync();
        store_double(unknowns, uint(row), reduced);
        GroupMemoryBarrierWithGroupSync();
    }
}
