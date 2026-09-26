StructuredBuffer<uint> wg_counts : register(t10);
RWStructuredBuffer<uint> wg_offsets : register(u11);
RWStructuredBuffer<uint> totals : register(u13);
RWStructuredBuffer<uint> bin_start : register(u14);

struct PushConstants {
    uint count;
    uint groups;
    uint shift;
    uint parity;
};
[[vk::push_constant]]
PushConstants push;

groupshared uint totals_shared[256];

[numthreads(256, 1, 1)]
void main(uint group_thread : SV_GroupIndex) {
    uint digit = group_thread;
    uint sum = 0u;
    for (uint group = 0u; group < push.groups; ++group) {
        uint value = wg_counts[group * 256u + digit];
        wg_offsets[group * 256u + digit] = sum;
        sum += value;
    }
    totals[digit] = sum;
    totals_shared[digit] = sum;
    GroupMemoryBarrierWithGroupSync();
    if (digit != 0u) return;
    uint acc = 0u;
    for (int bin = 0; bin < 256; ++bin) {
        bin_start[(uint)bin] = acc;
        acc += totals_shared[bin];
    }
}
