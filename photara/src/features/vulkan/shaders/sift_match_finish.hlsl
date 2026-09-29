#include "sift_common.hlsli"

// Translation of the CUDA `finish_matches` kernel: merge the per-tile top-two
// reductions, apply the angular distance threshold and ratio test, and check
// mutuality. Accepted matches store the train index, everything else -1.
struct PushConstants
{
    uint query_count;
    uint train_count;
    uint row_chunks;
    uint col_chunks;
    float ratio;
    float mutual;
    uint pad0;
    uint pad1;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer row_partials;
[[vk::binding(1, 0)]] RWByteAddressBuffer col_partials;
[[vk::binding(2, 0)]] RWByteAddressBuffer output;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

int3 load_partial(RWByteAddressBuffer buffer, uint index)
{
    const uint3 words = buffer.Load3(index * 12u);
    return int3(asint(words.x), asint(words.y), asint(words.z));
}

void insert(int value, int index, inout int3 best)
{
    if (value > best.x)
    {
        best.z = best.x;
        best.x = value;
        best.y = index;
    }
    else
    {
        best.z = max(best.z, value);
    }
}

int accepted_match(int3 best, float ratio)
{
    const float d = acos(min(float(best.x) * (1.0f / 262144.0f), 1.0f));
    const float second = acos(min(float(best.z) * (1.0f / 262144.0f), 1.0f));
    return d < 0.7f && d < ratio * second ? best.y : -1;
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint i = dtid.x;
    if (i >= pc.query_count) return;

    int3 best = int3(0, -1, 0);
    for (uint j = 0u; j < pc.row_chunks; ++j)
    {
        const int3 candidate = load_partial(row_partials, j * pc.query_count + i);
        insert(candidate.x, candidate.y, best);
        best.z = max(best.z, candidate.z);
    }

    int match = accepted_match(best, pc.ratio);
    if (pc.mutual > 0.5f && match >= 0)
    {
        int3 reverse = int3(0, -1, 0);
        for (uint j = 0u; j < pc.col_chunks; ++j)
        {
            const int3 candidate = load_partial(col_partials, j * pc.train_count + uint(match));
            insert(candidate.x, candidate.y, reverse);
            reverse.z = max(reverse.z, candidate.z);
        }
        if (accepted_match(reverse, pc.ratio) != int(i)) match = -1;
    }
    output.Store(i * 4u, asuint(match));
}
