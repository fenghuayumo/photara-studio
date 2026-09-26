RWStructuredBuffer<uint> keys0 : register(u6);
RWStructuredBuffer<uint> keys1 : register(u7);
RWStructuredBuffer<uint> vals0 : register(u8);
RWStructuredBuffer<uint> vals1 : register(u9);
StructuredBuffer<uint> wg_offsets : register(t11);
StructuredBuffer<uint> ranks : register(t12);
StructuredBuffer<uint> bin_start : register(t14);

struct PushConstants {
    uint count;
    uint groups;
    uint shift;
    uint parity;
};
[[vk::push_constant]]
PushConstants push;

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint index = tid.x;
    if (index >= push.count) return;
    uint group = index >> 8u;
    uint key = push.parity == 0u ? keys0[index] : keys1[index];
    uint value = push.parity == 0u ? vals0[index] : vals1[index];
    uint digit = (key >> push.shift) & 255u;
    uint dest = bin_start[digit] + wg_offsets[group * 256u + digit] + ranks[index];
    if (push.parity == 0u) {
        keys1[dest] = key;
        vals1[dest] = value;
    } else {
        keys0[dest] = key;
        vals0[dest] = value;
    }
}
