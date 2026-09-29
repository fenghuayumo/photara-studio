#include "sift_common.hlsli"

// ListGen: walk one histogram pyramid level down, converting a coarse seed
// position into the child that owns its rank.
struct PushConstants
{
    uint list_length;
    uint hist_width;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer list;
[[vk::binding(1, 0)]] RWByteAddressBuffer hist;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.list_length) return;

    int4 pos = load_i4(list, index * 16u);
    const int4 temp = load_i4(hist, uint(pos.y * int(pc.hist_width) + pos.x) * 16u);
    const int sum1 = temp.x + temp.y;
    const int sum2 = sum1 + temp.z;
    pos.x <<= 2;
    if (pos.z >= sum2)
    {
        pos.x += 3;
        pos.z -= sum2;
    }
    else if (pos.z >= sum1)
    {
        pos.x += 2;
        pos.z -= sum1;
    }
    else if (pos.z >= temp.x)
    {
        pos.x += 1;
        pos.z -= temp.x;
    }
    store_i4(list, index * 16u, pos);
}
