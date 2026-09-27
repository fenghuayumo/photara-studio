#include "common.hlsli"

struct PushConstants
{
    uint count;
    uint padded;
    uint merge_size;
    uint compare_distance;
    uint descending;
    uint first_pass;
    uint src_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer src_values;
[[vk::binding(1, 0)]] RWByteAddressBuffer src_indices;
[[vk::binding(2, 0)]] RWByteAddressBuffer dst_values;
[[vk::binding(3, 0)]] RWByteAddressBuffer dst_indices;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct Item
{
    float value;
    uint index;
};

Item read_item(uint i)
{
    Item item;
    if (i >= pc.count && pc.first_pass != 0)
    {
        item.value = 0.0f;
        item.index = 0xffffffffu;
    }
    else
    {
        item.value = asfloat(src_values.Load(pc.src_offset + i * 4u));
        item.index = pc.first_pass != 0 ? i : src_indices.Load(i * 8u);
    }
    return item;
}

bool less_item(Item a, Item b)
{
    if (a.index == 0xffffffffu) return false;
    if (b.index == 0xffffffffu) return true;
    if (a.value == b.value) return a.index < b.index;
    return pc.descending != 0 ? a.value > b.value : a.value < b.value;
}

void write_item(uint i, Item item)
{
    dst_values.Store(i * 4u, asuint(item.value));
    dst_indices.Store2(i * 8u, uint2(item.index, 0u));
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint pair = dtid.x;
    if (pair >= pc.padded / 2u) return;
    const uint distance = pc.compare_distance;
    const uint lo = (pair / distance) * (distance * 2u) + pair % distance;
    const uint hi = lo + distance;
    const Item a = read_item(lo);
    const Item b = read_item(hi);
    const bool ascending = (lo & pc.merge_size) == 0u;
    const bool a_first = less_item(a, b);
    if (ascending == a_first)
    {
        write_item(lo, a);
        write_item(hi, b);
    }
    else
    {
        write_item(lo, b);
        write_item(hi, a);
    }
}
