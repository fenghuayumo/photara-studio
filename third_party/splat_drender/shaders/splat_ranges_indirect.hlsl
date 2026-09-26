#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> key_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> key_hi;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> ranges;
[[vk::binding(3, 0)]] StructuredBuffer<uint> control;

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint count = control[0];
    uint index = dispatch_id.x;
    if (control[2] != 0u || index >= count) return;
    uint tile = key_lo[index];
    if (index == 0u) ranges[tile * 2u] = 0u;
    else {
        uint previous = key_lo[index - 1u];
        if (tile != previous) {
            ranges[previous * 2u + 1u] = index;
            ranges[tile * 2u] = index;
        }
    }
    if (index == count - 1u) ranges[tile * 2u + 1u] = count;
}
