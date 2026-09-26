StructuredBuffer<uint> keys0 : register(t6);
StructuredBuffer<uint> keys1 : register(t7);
RWStructuredBuffer<uint> wg_counts : register(u10);
RWStructuredBuffer<uint> ranks : register(u12);

struct PushConstants {
    uint count;
    uint groups;
    uint shift;
    uint parity;
};
[[vk::push_constant]]
PushConstants push;

groupshared uint digits[256];

[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint group = group_id.x;
    uint index = group * 256u + group_thread;
    uint digit = 0u;
    if (index < push.count) {
        uint key = push.parity == 0u ? keys0[index] : keys1[index];
        digit = (key >> push.shift) & 255u;
    }
    digits[group_thread] = digit;
    GroupMemoryBarrierWithGroupSync();
    if (group_thread != 0u) return;

    uint hist[256];
    uint cursor[256];
    for (int bin = 0; bin < 256; ++bin) hist[bin] = 0u;
    uint base = group * 256u;
    uint limit = base >= push.count ? 0u : min(256u, push.count - base);
    for (uint item = 0u; item < limit; ++item) hist[digits[item]]++;

    // Rank inside this workgroup's digit. The global bin start is added
    // when scattering, so this must not include counts of lower digits.
    for (int bin = 0; bin < 256; ++bin) {
        wg_counts[group * 256u + (uint)bin] = hist[bin];
        cursor[bin] = 0u;
    }
    for (uint item = 0u; item < limit; ++item) {
        uint bin = digits[item];
        ranks[base + item] = cursor[bin];
        cursor[bin]++;
    }
}
