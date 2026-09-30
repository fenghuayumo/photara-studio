#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer system;
[[vk::binding(1, 0)]] RWByteAddressBuffer info;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(1, 1, 1)]
void main() {
    uint n = pc.n;
    uint column = pc.p0;
    uint diagonal = column + column * n;
    double pivot = load_double(system, diagonal);
    if (!(pivot > 0.0) || !isfinite_d(pivot)) {
        if (info.Load(0u) == 0u) info.Store(0u, column + 1u);
        return;
    }
    pivot = sqrt_d(pivot);
    store_double(system, diagonal, pivot);
    for (uint row = column + 1u; row < n; ++row) {
        uint index = row + column * n;
        store_double(system, index, load_double(system, index) / pivot);
    }
}
