#include "common.hlsli"

// Exact float/index radix selection. The device resolves each 16-bit digit;
// histogram bins never cross to the host. Ties select the lowest row indices.

struct PushConstants
{
    uint count;        // rows to scan
    uint src_offset;   // Float32 values
    uint mask_offset;  // one Bool per row; a zero excludes the row
    uint has_mask;
    uint bins_offset;  // 65536 u32 histogram bins
    uint dst_offset;   // Int32 indices, emit mode only
    uint prefix_bits;  // 0, 16, 32 or 48
    uint limit;        // emit: at most this many indices
    uint descending;   // 1 selects the largest values instead of the smallest
    uint mode;         // 0 histogram, 1 emit, 2 resolve digit
};

[[vk::binding(0, 0)]] RWByteAddressBuffer values_buf;
[[vk::binding(1, 0)]] RWByteAddressBuffer mask_buf;
[[vk::binding(2, 0)]] RWByteAddressBuffer bins_buf;
[[vk::binding(3, 0)]] RWByteAddressBuffer out_buf;
// key_hi, key_lo, remaining rank, qualifying count, emitted count
[[vk::binding(4, 0)]] RWByteAddressBuffer state_buf;
groupshared uint block_counts[TT_GROUP_SIZE];
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
void main(uint3 dtid : SV_DispatchThreadID, uint lane : SV_GroupIndex)
{
    if (pc.mode == 2u) {
        uint subtotal = 0u;
        for (uint b = 0u; b < 256u; ++b)
            subtotal += bins_buf.Load(pc.bins_offset + (lane * 256u + b) * 4u);
        block_counts[lane] = subtotal;
        GroupMemoryBarrierWithGroupSync();
        if (lane != 0u) return;
        uint rank = state_buf.Load(8u);
        if (pc.prefix_bits == 0u) {
            uint total = 0u;
            for (uint b = 0u; b < 256u; ++b) total += block_counts[b];
            state_buf.Store(12u, total);
            rank = min(pc.limit, total);
        }
        if (rank == 0u) return;
        uint below = 0u, block = 0u;
        for (; block < 256u; ++block) {
            if (below + block_counts[block] >= rank) break;
            below += block_counts[block];
        }
        for (uint b = 0u; b < 256u; ++b) {
            const uint digit = block * 256u + b;
            const uint n = bins_buf.Load(pc.bins_offset + digit * 4u);
            if (below + n >= rank) {
                const uint offset = pc.prefix_bits < 32u ? 0u : 4u;
                state_buf.Store(offset, (state_buf.Load(offset) << 16u) | digit);
                state_buf.Store(8u, rank - below);
                break;
            }
            below += n;
        }
        return;
    }
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
            if ((key_hi >> 16u) != state_buf.Load(0u))
            {
                return;
            }
            bin = key_hi & 0xFFFFu;
        }
        else if (pc.prefix_bits == 32u)
        {
            if (key_hi != state_buf.Load(0u))
            {
                return;
            }
            bin = key_lo >> 16u;
        }
        else
        {
            if (key_hi != state_buf.Load(0u) || (key_lo >> 16u) != state_buf.Load(4u))
            {
                return;
            }
            bin = key_lo & 0xFFFFu;
        }
        bins_buf.InterlockedAdd(pc.bins_offset + bin * 4u, 1u);
        return;
    }

    // Emit: every masked row at or below the resolved key, capped at `limit`.
    // The emitted count lives beside the resolved threshold in device state.
    if (state_buf.Load(12u) == 0u) return;
    const bool past = key_hi > state_buf.Load(0u) ||
        (key_hi == state_buf.Load(0u) && key_lo > state_buf.Load(4u));
    if (past)
    {
        return;
    }
    uint slot = 0u;
    state_buf.InterlockedAdd(16u, 1u, slot);
    if (slot < pc.limit)
    {
        out_buf.Store(pc.dst_offset + slot * 4u, index);
    }
}
