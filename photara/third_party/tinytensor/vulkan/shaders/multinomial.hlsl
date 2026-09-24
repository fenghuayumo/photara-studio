#include "common.hlsli"

struct PushConstants
{
    uint op;
    uint n_weights;
    uint n_samples;
    uint seed;
    uint weights_offset;
    uint cdf_offset;
    uint keys_offset;
    uint out_offset;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer weights;
[[vk::binding(1, 0)]] RWByteAddressBuffer aux;
[[vk::binding(2, 0)]] RWByteAddressBuffer out_indices;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

static const uint kOpSampleReplacement = 0;
static const uint kOpGumbelKeys = 1;
static const uint kOpArgmaxMasked = 2;

uint pcg_hash(uint value)
{
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}

float rand01(uint seed, uint idx)
{
    const uint hashed = pcg_hash(seed ^ pcg_hash(idx + 1u));
    return float(hashed >> 8u) * (1.0f / 16777216.0f);
}

uint lower_bound_cdf(float u, uint n)
{
    uint lo = 0u;
    uint hi = n;
    while (lo < hi)
    {
        const uint mid = lo + (hi - lo) / 2u;
        const float cdf = asfloat(aux.Load(pc.cdf_offset + mid * 4u));
        if (cdf < u)
        {
            lo = mid + 1u;
        }
        else
        {
            hi = mid;
        }
    }
    return min(lo, n - 1u);
}

groupshared float shared_max[TT_GROUP_SIZE];
groupshared uint shared_idx[TT_GROUP_SIZE];

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID, uint gidx : SV_GroupIndex)
{
    if (pc.op == kOpSampleReplacement)
    {
        const uint sample = dtid.x;
        if (sample >= pc.n_samples)
        {
            return;
        }
        const float total = asfloat(aux.Load(pc.cdf_offset + (pc.n_weights - 1u) * 4u));
        float u = rand01(pc.seed, sample) * total;
        if (u <= 0.0f)
        {
            u = total * 1e-6f;
        }
        store_i64_from_i32(out_indices, pc.out_offset + sample * 8u, int(lower_bound_cdf(u, pc.n_weights)));
        return;
    }

    if (pc.op == kOpGumbelKeys)
    {
        const uint index = dtid.x;
        if (index >= pc.n_weights)
        {
            return;
        }
        const float w = asfloat(weights.Load(pc.weights_offset + index * 4u));
        if (w <= 0.0f)
        {
            aux.Store(pc.keys_offset + index * 4u, asuint(-3.4e38f));
            return;
        }
        float u = rand01(pc.seed, index);
        u = clamp(u, 1e-10f, 1.0f - 1e-10f);
        const float gumbel = -log(-log(u));
        aux.Store(pc.keys_offset + index * 4u, asuint(log(w) + gumbel));
        return;
    }

    // kOpArgmaxMasked: reduction within this dispatch. Host loops over samples
    // and launches one workgroup-striped reduction; we write the global argmax
    // of keys into out[sample] using atomics-free two-pass via shared memory
    // of a single group when n_weights <= 256. For larger n, each group writes
    // a partial into aux then a second dispatch (n_samples packed in seed high).
    const uint index = dtid.x;
    float best = -3.4e38f;
    uint best_i = 0u;
    for (uint i = index; i < pc.n_weights; i += TT_GROUP_SIZE)
    {
        const float key = asfloat(aux.Load(pc.keys_offset + i * 4u));
        if (key > best)
        {
            best = key;
            best_i = i;
        }
    }
    shared_max[gidx] = best;
    shared_idx[gidx] = best_i;
    GroupMemoryBarrierWithGroupSync();

    for (uint stride = TT_GROUP_SIZE / 2u; stride > 0u; stride >>= 1)
    {
        if (gidx < stride)
        {
            if (shared_max[gidx + stride] > shared_max[gidx])
            {
                shared_max[gidx] = shared_max[gidx + stride];
                shared_idx[gidx] = shared_idx[gidx + stride];
            }
        }
        GroupMemoryBarrierWithGroupSync();
    }

    if (gidx == 0)
    {
        const uint sample = pc.n_samples; // reused as the current sample slot
        store_i64_from_i32(out_indices, pc.out_offset + sample * 8u, int(shared_idx[0]));
        aux.Store(pc.keys_offset + shared_idx[0] * 4u, asuint(-3.4e38f));
    }
}
