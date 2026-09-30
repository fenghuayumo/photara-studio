#include "bearing_common.hlsli"

[[vk::binding(0, 0)]] RWByteAddressBuffer lin;
[[vk::binding(1, 0)]] RWByteAddressBuffer observations;
[[vk::binding(2, 0)]] RWByteAddressBuffer eliminated;
[[vk::binding(3, 0)]] RWByteAddressBuffer offsets;
[[vk::binding(4, 0)]] RWByteAddressBuffer indices;
[[vk::binding(5, 0)]] RWStructuredBuffer<double> system;

[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_capability(6034)]]
[[vk::ext_instruction(6035)]]
double atomic_add_f64(
    [[vk::ext_reference]] double destination, uint scope, uint semantics, double value);

[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint width = pc.p0 * 3u;
    uint stride = max(pc.groups, 1u);
    for (uint p = group_id.x; p < pc.n; p += stride) {
        uint begin = load_uint(offsets, p);
        uint count = load_uint(offsets, p + 1u) - begin;
        if (count == 0u) continue;
        // 64-bit bound: one track can be seen often enough that count*count*9
        // does not fit in 32 bits. A zero count never reaches this division.
        uint64_t total = uint64_t(count) * uint64_t(count) * 9ull;
        for (uint64_t v = group_thread; v < total; v += 256ull) {
            uint a = load_uint(indices, begin + uint(v / (uint64_t(count) * 9ull)));
            uint b = load_uint(indices, begin + uint((v / 9ull) % uint64_t(count)));
            uint row = uint((v % 9ull) / 3ull);
            uint col = uint(v % 3ull);
            double value = 0.0;
            [unroll] for (uint k = 0u; k < 3u; ++k) {
                value += load_double(eliminated, a * 9u + row * 3u + k) *
                         load_double(lin, b * 12u + k * 3u + col);
            }
            uint camera_a = observations.Load(a * 32u);
            uint camera_b = observations.Load(b * 32u);
            uint index = (camera_a * 3u + row) * width + camera_b * 3u + col;
            atomic_add_f64(system[index], 1u, 0u, -value);
        }
    }
}
