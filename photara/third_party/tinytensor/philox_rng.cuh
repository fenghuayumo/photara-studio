#pragma once

// Minimal Philox4x32-10 counter-based RNG for device code. It replaces the
// cuRAND dependency (curand64_10.dll, ~69 MB) while staying in the same
// generation family the code used before (CURAND_RNG_PSEUDO_PHILOX4_32_10,
// the Philox4x32-10 variant popularized by Random123 / PyTorch).
//
// Streams map to the familiar curand_init(seed, subsequence, offset) layout:
// the 64-bit seed forms the Philox key and (subsequence, offset) address the
// 128-bit counter, so every thread owns an independent, reproducible stream.

#include <cstdint>

namespace tinytensor {

struct PhiloxState {
    unsigned long long seed;
    std::uint32_t counter[4];
    std::uint32_t output[4];
    int used;
};

__device__ __forceinline__ void philox_round4(std::uint32_t* c,
                                              std::uint32_t k0,
                                              std::uint32_t k1) {
    const std::uint32_t m0 = 0xD2511F53u, m1 = 0xCD9E8D57u;
    const unsigned long long p0 =
        static_cast<unsigned long long>(c[0]) * m0;
    const unsigned long long p1 =
        static_cast<unsigned long long>(c[2]) * m1;
    const std::uint32_t lo0 = static_cast<std::uint32_t>(p0);
    const std::uint32_t hi0 = static_cast<std::uint32_t>(p0 >> 32);
    const std::uint32_t lo1 = static_cast<std::uint32_t>(p1);
    const std::uint32_t hi1 = static_cast<std::uint32_t>(p1 >> 32);
    c[0] = hi1 ^ c[1] ^ k0;
    c[1] = lo0;
    c[2] = hi0 ^ c[3] ^ k1;
    c[3] = lo1;
}

// Generates the next 128-bit block for the current counter and advances it.
__device__ __forceinline__ void philox_next4(PhiloxState* s) {
    std::uint32_t k0 = static_cast<std::uint32_t>(s->seed);
    std::uint32_t k1 = static_cast<std::uint32_t>(s->seed >> 32);
    std::uint32_t c[4] = {s->counter[0], s->counter[1],
                          s->counter[2], s->counter[3]};
#pragma unroll
    for (int r = 0; r < 10; ++r) {
        if (r > 0) {
            k0 += 0x9E3779B9u;
            k1 += 0xBB67AE85u;
        }
        philox_round4(c, k0, k1);
    }
    for (int i = 0; i < 4; ++i) s->output[i] = c[i];
    if (++s->counter[0] == 0u) ++s->counter[1];
    s->used = 0;
}

__device__ __forceinline__ void philox_init(unsigned long long seed,
                                            unsigned long long subsequence,
                                            unsigned long long offset,
                                            PhiloxState* s) {
    s->seed = seed;
    s->counter[0] = static_cast<std::uint32_t>(offset);
    s->counter[1] = static_cast<std::uint32_t>(offset >> 32);
    s->counter[2] = static_cast<std::uint32_t>(subsequence);
    s->counter[3] = static_cast<std::uint32_t>(subsequence >> 32);
    s->used = 4;  // force generation on the first draw
}

__device__ __forceinline__ std::uint32_t philox_next_u32(PhiloxState* s) {
    if (s->used >= 4) philox_next4(s);
    return s->output[s->used++];
}

// Uniform in (0, 1], matching curand_uniform semantics.
__device__ __forceinline__ float philox_uniform(PhiloxState* s) {
    return 1.0f - static_cast<float>(philox_next_u32(s)) *
                      2.3283064365386963e-10f;
}

// Standard normal via Box-Muller; consumes two uniforms per draw.
__device__ __forceinline__ float philox_normal(PhiloxState* s) {
    float u1 = philox_uniform(s);
    while (!(u1 > 0.0f)) u1 = philox_uniform(s);
    const float u2 = philox_uniform(s);
    return sqrtf(-2.0f * logf(u1)) *
           sinf(6.283185307179586f * u2);
}

}  // namespace tinytensor
