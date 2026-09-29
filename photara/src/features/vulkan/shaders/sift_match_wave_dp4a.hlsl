#include "sift_common.hlsli"

// Low-shared-memory matcher for 32-lane subgroups. Each wave owns one 32x32
// query/train tile: lanes are train columns, and query descriptor words are
// broadcast across the wave. Sixteen waves process 512 query rows per group.
// This keeps several groups resident per SM instead of spending ~45 KiB of
// shared memory on a materialized dot matrix.
struct PushConstants
{
    uint query_count;
    uint train_count;
    uint mutual;
    uint pad0;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer query;
[[vk::binding(1, 0)]] RWByteAddressBuffer train;
[[vk::binding(2, 0)]] RWByteAddressBuffer row_partials;
[[vk::binding(3, 0)]] RWByteAddressBuffer col_partials;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

groupshared uint train_tile[32][32];

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

void store_partial(RWByteAddressBuffer buffer, uint index, int3 value)
{
    buffer.Store3(index * 12u, uint3(asuint(value.x), asuint(value.y), asuint(value.z)));
}

[numthreads(512, 1, 1)]
void main(uint3 gtid : SV_GroupThreadID, uint3 gid : SV_GroupID)
{
    const uint tid = gtid.x;
    const uint lane = WaveGetLaneIndex();
    const uint wave = tid >> 5u;
    const uint train_base = gid.x * 32u;
    const uint query_base = gid.y * 512u + wave * 32u;

    for (uint e = tid; e < 1024u; e += 512u)
    {
        const uint row = e >> 5u;
        const uint word = e & 31u;
        train_tile[row][word] = train_base + row < pc.train_count
            ? train.Load(((train_base + row) * 32u + word) * 4u)
            : 0u;
    }
    GroupMemoryBarrierWithGroupSync();

    const bool valid_train = train_base + lane < pc.train_count;
    // One lane owns one train descriptor. Keep its 128 bytes in registers for
    // all 32 query rows handled by the wave instead of rereading shared memory
    // for every dot product.
    uint train_words[32];
    [unroll]
    for (uint word = 0u; word < 32u; ++word)
        train_words[word] = train_tile[lane][word];
    int3 column_best = int3(0, -1, 0);
    [unroll]
    for (uint row = 0u; row < 32u; ++row)
    {
        const uint query_index = query_base + row;
        const bool valid_query = query_index < pc.query_count;
        const uint query_word = valid_query
            ? query.Load((query_index * 32u + lane) * 4u)
            : 0u;
        uint dot = 0u;
        [unroll]
        for (uint word = 0u; word < 32u; ++word)
            dot = dot4add_u8packed(WaveReadLaneAt(query_word, word),
                                   train_words[word], dot);

        if (valid_query && valid_train)
            insert(int(dot), int(query_index), column_best);

        const uint candidate = valid_train ? dot : 0u;
        // The score needs only 19 bits. Pack the reverse lane in its low five
        // bits so one max reduction returns both the highest score and, for
        // ties, the lowest train index.
        const uint packed_candidate = valid_train && candidate != 0u
            ? (candidate << 5u) | (31u - lane)
            : 0u;
        const uint packed_best = WaveActiveMax(packed_candidate);
        const uint best_value = packed_best >> 5u;
        const uint best_lane = packed_best == 0u
            ? 0xffffffffu
            : 31u - (packed_best & 31u);
        const uint second_value = WaveActiveMax(
            valid_train && lane != best_lane ? candidate : 0u);
        if (lane == 0u && valid_query)
        {
            const int best_index = best_lane == 0xffffffffu
                ? -1
                : int(train_base + best_lane);
            store_partial(row_partials,
                          gid.x * pc.query_count + query_index,
                          int3(int(best_value), best_index, int(second_value)));
        }
    }

    if (pc.mutual != 0u && valid_train)
        store_partial(col_partials,
                      (query_base / 32u) * pc.train_count + train_base + lane,
                      column_best);
}
