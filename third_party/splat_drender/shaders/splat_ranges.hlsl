#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> key_lo;
[[vk::binding(1, 0)]] StructuredBuffer<uint> key_hi;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> ranges;

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint count = pc.u0;
    uint index = dispatch_id.x;
    if (index >= count) return;
    uint packed = pc.u1;
    uint tile = packed != 0 ? key_hi[index] : key_lo[index];
    if (index == 0) {
        ranges[tile * 2] = 0;
    } else {
        uint previous = packed != 0 ? key_hi[index - 1] : key_lo[index - 1];
        if (tile != previous) {
            ranges[previous * 2 + 1] = index;
            ranges[tile * 2] = index;
        }
    }
    if (index == count - 1) ranges[tile * 2 + 1] = count;
}
