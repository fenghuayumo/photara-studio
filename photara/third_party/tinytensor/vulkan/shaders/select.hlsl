#include "common.hlsli"

// Exact order-statistic selection over Float32 rows.
//
// The sort key is 64 bits: the row's transform of its float in the high half,
// its row index in the low half. Folding the index in is what makes the
// selection reproduce a sort exactly rather than merely approximately, the tie
// rule included: two rows with the same value are ordered by row index, which
// is what the bitonic comparator this replaces does, and the key is then
// unique so the emit pass never has to break a tie at the cut.
//
// The caller resolves the requested rank one 16-bit digit at a time. Each
// "histogram" dispatch counts the rows whose remaining key matches the prefix
// already resolved; the host walks the 65536 bins from the low end to find the
// bin that holds the rank, folds that bin into the prefix, and calls again.
// Four passes therefore pin down the exact key of the rank-th row, and the
// "emit" dispatch writes the row indices at or below that key.
//
// This replaces the full bitonic sort IGS selection used to run, five to six
// times per refinement. Sorting a million rows costs about two hundred
// dispatches and two hundred barriers to produce at most twelve thousand
// indices; two histogram passes cost six dispatches and no barriers between
// rows, and the result is exact rather than distribution dependent.
//
// The key is the row's float with its sign bit folded in: IEEE-754 floats
// compare like sign-magnitude integers once the negatives are complemented.
// Ties (two rows with the same key at the cut) are impossible for the
// continuous scores this is used on; when they do occur the emit order inside
// the tied group is unspecified, which is harmless because every caller treats
// the result as a set.

struct PushConstants
{
    uint count;        // rows to scan
    uint src_offset;   // Float32 values
    uint mask_offset;  // one Bool per row; a zero excludes the row
    uint has_mask;
    uint bins_offset;  // 65536 u32 bins, or the emit counter at offset 0
    uint dst_offset;   // Int32 indices, emit mode only
    uint prefix_hi;    // resolved high half of the key, right aligned
    uint prefix_lo;    // resolved low half of the key, right aligned
    uint prefix_bits;  // 0, 16, 32 or 48
    uint limit;        // emit: at most this many indices
    uint descending;   // 1 selects the largest values instead of the smallest
    uint mode;         // 0 histogram, 1 emit
    uint threshold_hi; // emit: the key resolved by the histogram passes
    uint threshold_lo;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer values_buf;
[[vk::binding(1, 0)]] RWByteAddressBuffer mask_buf;
[[vk::binding(2, 0)]] RWByteAddressBuffer bins_buf;
[[vk::binding(3, 0)]] RWByteAddressBuffer out_buf;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

// Strictly increasing in `value` for every non-NaN float.
uint order_key(float value)
{
    const uint bits = asuint(value);
    return (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
}

// Ascending selection compares the key directly; descending inverts it so both
// directions pick "the smallest keys first" inside the histogram.
uint select_key(float value)
{
    const uint key = order_key(value);
    return pc.descending != 0u ? ~key : key;
}

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.count)
    {
        return;
    }
    if (pc.has_mask != 0u && load_u8(mask_buf, pc.mask_offset + index) == 0u)
    {
        return;
    }
    const uint key_hi = select_key(load_f32(values_buf, pc.src_offset + index * 4u));
    const uint key_lo = index;

    if (pc.mode == 0u)
    {
        uint bin;
        if (pc.prefix_bits == 0u)
        {
            bin = key_hi >> 16u;
        }
        else if (pc.prefix_bits == 16u)
        {
            if ((key_hi >> 16u) != pc.prefix_hi)
            {
                return;
            }
            bin = key_hi & 0xFFFFu;
        }
        else if (pc.prefix_bits == 32u)
        {
            if (key_hi != pc.prefix_hi)
            {
                return;
            }
            bin = key_lo >> 16u;
        }
        else
        {
            if (key_hi != pc.prefix_hi || (key_lo >> 16u) != pc.prefix_lo)
            {
                return;
            }
            bin = key_lo & 0xFFFFu;
        }
        bins_buf.InterlockedAdd(pc.bins_offset + bin * 4u, 1u);
        return;
    }

    // Emit: every masked row at or below the resolved key, capped at `limit`.
    // The counter lives in the first bin, which the caller clears.
    const bool past = key_hi > pc.threshold_hi ||
        (key_hi == pc.threshold_hi && key_lo > pc.threshold_lo);
    if (past)
    {
        return;
    }
    uint slot = 0u;
    bins_buf.InterlockedAdd(pc.bins_offset, 1u, slot);
    if (slot < pc.limit)
    {
        out_buf.Store(pc.dst_offset + slot * 4u, index);
    }
}
