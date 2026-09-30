#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer cameras;
[[vk::binding(1, 0)]] RWByteAddressBuffer step;
[[vk::binding(2, 0)]] RWByteAddressBuffer candidate;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint stride = max(pc.groups, 1u) * 256u;
    for (uint i = id.x; i < pc.n; i += stride) {
        store_double(candidate, i, load_double(cameras, i) + load_double(step, i));
    }
}
