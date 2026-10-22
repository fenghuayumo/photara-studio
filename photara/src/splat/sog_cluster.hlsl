[[vk::binding(0,0)]] StructuredBuffer<float> points;
[[vk::binding(1,0)]] RWStructuredBuffer<float> centers;
[[vk::binding(2,0)]] RWStructuredBuffer<uint> labels;
[[vk::binding(3,0)]] RWStructuredBuffer<uint> counts;
[[vk::binding(4,0)]] RWStructuredBuffer<uint> offsets;
[[vk::binding(5,0)]] RWStructuredBuffer<uint> cursors;
[[vk::binding(6,0)]] RWStructuredBuffer<uint> indices;
[[vk::binding(7,0)]] StructuredBuffer<uint> parentOffsets;
[[vk::binding(8,0)]] StructuredBuffer<uint> parentIndices;
[[vk::binding(9,0)]] RWStructuredBuffer<uint> blockOffsets;
struct Push { uint mode, count, dims, clusters, branches, iteration; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
groupshared uint scan[64];
uint hash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}
uint sample(uint cluster, uint iteration) {
    uint parent = cluster / pc.branches;
    uint begin = parentOffsets[parent], end = parentOffsets[parent + 1];
    return end > begin ? parentIndices[begin + hash(cluster ^ (iteration * 0x9e3779b9u)) % (end - begin)] : 0;
}

[numthreads(64,1,1)]
void main(uint3 tid:SV_DispatchThreadID, uint3 gid:SV_GroupID, uint lane:SV_GroupIndex) {
    uint i = tid.x + tid.y * 65535u * 64u;
    uint group = gid.x + gid.y * 65535u;
    if (pc.mode == 0) { // Initial root membership.
        if (i < pc.count) { labels[i] = 0; indices[i] = i; }
        if (i == 0) { offsets[0] = 0; offsets[1] = pc.count; }
    } else if (pc.mode == 1) { // Seed each child from its actual parent members.
        if (i < pc.clusters * pc.dims)
            centers[i] = points[sample(i / pc.dims, 1) * pc.dims + i % pc.dims];
    } else if (pc.mode == 2) { // Only compare sibling centers, never N x K distances.
        if (i >= pc.count) return;
        uint parent = pc.iteration == 0 ? labels[i] : labels[i] / pc.branches;
        float value[45];
        for (uint d = 0; d < pc.dims; ++d) value[d] = points[i * pc.dims + d];
        float best = 3.402823466e+38f;
        uint selected = parent * pc.branches;
        for (uint child = 0; child < pc.branches; ++child) {
            uint c = parent * pc.branches + child;
            float distance = 0;
            for (uint d = 0; d < pc.dims; ++d) {
                float delta = value[d] - centers[c * pc.dims + d];
                distance += delta * delta;
            }
            if (distance < best) { best = distance; selected = c; }
        }
        labels[i] = selected;
    } else if (pc.mode == 3) {
        if (i < pc.clusters) { counts[i] = 0; cursors[i] = 0; }
    } else if (pc.mode == 4) {
        if (i < pc.count) InterlockedAdd(counts[labels[i]], 1);
    } else if (pc.mode == 5) { // Exclusive block scan. All lanes reach barriers.
        scan[lane] = i < pc.clusters ? counts[i] : 0;
        GroupMemoryBarrierWithGroupSync();
        for (uint step = 1; step < 64; step <<= 1) {
            uint v = lane >= step ? scan[lane - step] : 0;
            GroupMemoryBarrierWithGroupSync();
            scan[lane] += v;
            GroupMemoryBarrierWithGroupSync();
        }
        if (i < pc.clusters) offsets[i] = scan[lane] - counts[i];
        if (lane == 63) blockOffsets[group] = scan[63];
    } else if (pc.mode == 6) { // Small block-total scan into a disjoint region.
        uint blocks = (pc.clusters + 63) / 64;
        if (i < blocks) {
            uint sum = 0;
            for (uint b = 0; b < i; ++b) sum += blockOffsets[b];
            blockOffsets[blocks + i] = sum;
        }
    } else if (pc.mode == 7) {
        if (i < pc.clusters) offsets[i] += blockOffsets[(pc.clusters + 63) / 64 + i / 64];
        if (i == 0) offsets[pc.clusters] = pc.count;
    } else if (pc.mode == 8) { // Counting-sort scatter; only integer atomics.
        if (i < pc.count) {
            uint label = labels[i], slot;
            InterlockedAdd(cursors[label], 1, slot);
            indices[offsets[label] + slot] = i;
        }
    } else if (pc.mode == 9) { // One workgroup per center, lanes cover SH channels.
        uint c = group;
        if (c >= pc.clusters || lane >= pc.dims) return;
        uint begin = offsets[c], end = offsets[c + 1];
        float mean = 0;
        if (end > begin) {
            // Scale before accumulation to avoid overflow in large clusters.
            float inv = 1.0f / float(end - begin);
            for (uint at = begin; at < end; ++at)
                mean += points[indices[at] * pc.dims + lane] * inv;
        } else {
            mean = points[sample(c, pc.iteration + 2) * pc.dims + lane];
        }
        centers[c * pc.dims + lane] = mean;
    }
}
