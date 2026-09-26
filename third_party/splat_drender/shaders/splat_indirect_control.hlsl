#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> control;
// Per-slot host-visible mirror of the counts. The shader writes it directly:
// a vkCmdCopyBuffer readback would need a transfer barrier that costs more
// than the whole sort stage. The host reads a slot only after its fence.
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> mirrors;

void write_dispatch(uint word, uint groups) {
    control[word] = groups;
    control[word + 1u] = 1u;
    control[word + 2u] = 1u;
}

[numthreads(1, 1, 1)]
void main() {
    uint gaussian_count = pc.u0;
    uint capacity = pc.u1;
    uint mirror_word = pc.u2;
    uint instances = gauss_u[5u * gaussian_count - 1u];
    uint visible = gauss_u[6u * gaussian_count - 1u];
    uint overflow = instances > capacity ? 1u : 0u;
    control[0] = instances;
    control[1] = visible;
    control[2] = overflow;
    control[3] = capacity;
    mirrors[mirror_word] = instances;
    mirrors[mirror_word + 1u] = visible;
    mirrors[mirror_word + 2u] = overflow;

    uint enabled = overflow == 0u ? 1u : 0u;
    write_dispatch(4u, enabled * ((visible + 255u) / 256u));
    write_dispatch(7u, enabled * ((visible + 255u) / 256u));
    write_dispatch(10u, enabled * ((visible + 1023u) / 1024u));
    write_dispatch(13u, enabled * ((instances + 1023u) / 1024u));
    write_dispatch(16u, enabled * ((instances + 255u) / 256u));
}
