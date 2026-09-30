#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer system;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint n = pc.n;
    uint column = pc.p0;
    uint start = column + 1u;
    uint extent = n - start;
    uint total = extent * extent;
    uint step = max(pc.groups, 1u) * 256u;
    for (uint item = id.x; item < total; item += step) {
        uint col = start + item / extent;
        uint row = start + item % extent;
        if (row < col) continue;
        uint index = row + col * n;
        double value = load_double(system, index);
        value -= load_double(system, row + column * n) * load_double(system, col + column * n);
        store_double(system, index, value);
    }
}
