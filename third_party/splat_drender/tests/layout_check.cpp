// Host-side audit: bytes() must cover from_pool()'s actual cursor extent.
#include "splat_drender/buffers.h"

#include <cstdio>
#include <cstring>
#include <vector>

using namespace splat_drender::ws;

template <typename State, typename... Args>
void audit(const char* name, std::size_t claimed, Args... args) {
    std::vector<char> buffer(claimed, 0);
    // Poison beyond the claimed size to detect overruns relative to bytes().
    State s = State::from_pool(buffer.data(), args...);
    (void)s;
    // Recompute the cursor extent by re-running from_pool on a second cursor.
    // from_pool advances by align_up per take; emulate by differencing two
    // probes with distinct sizes is complex, so instead check that every
    // pointer lies within [base, base+claimed).
    const char* base = buffer.data();
    const char* end = base + claimed;
    int bad = 0;
    const char* fields[32];
    int n = 0;
    // Generic: walk the struct's pointer members via a second from_pool on a
    // max-sized buffer and compare offsets.
    std::vector<char> big(1 << 20, 0);
    State s2 = State::from_pool(big.data(), args...);
    const char* p2 = reinterpret_cast<const char*>(&s2);
    for (std::size_t off = 0; off + sizeof(void*) <= sizeof(State);
         off += sizeof(void*)) {
        const void* v;
        std::memcpy(&v, p2 + off, sizeof(v));
        fields[n++] = static_cast<const char*>(v);
    }
    // Deduplicate nulls and check bounds.
    for (int i = 0; i < n; ++i) {
        if (!fields[i]) continue;
        if (fields[i] < base || fields[i] >= end) {
            printf("%s: field %d at %td outside [%td, %td)\n", name, i,
                   fields[i] - base, base - base, end - base);
            ++bad;
        }
    }
    if (!bad) printf("%s: OK (%zu bytes)\n", name, claimed);
}

int main() {
    audit<GaussianState>("GaussianState n=1",
                         GaussianState::bytes(1, 4096), 1, 4096);
    audit<GaussianState>("GaussianState n=1000",
                         GaussianState::bytes(1000, 4096), 1000, 4096);
    audit<GradState>("GradState n=1", GradState::bytes(1), 1);
    audit<GradState>("GradState n=1000", GradState::bytes(1000), 1000);
    audit<InstanceState>("InstanceState v=1 i=1",
                         InstanceState::bytes(1, 1, 4096), 1, 1, 4096);
    audit<InstanceState>("InstanceState v=999 i=1000",
                         InstanceState::bytes(999, 1000, 4096), 999, 1000, 4096);
    audit<PixelState>("PixelState p=1 b=0", PixelState::bytes(1, 0, false),
                      1, 0, false);
    audit<PixelState>("PixelState p=100000 b=4096 geom",
                      PixelState::bytes(100000, 4096, true),
                      100000, 4096, true);
    audit<PixelState>("PixelState p=100000 b=4096 rgb",
                      PixelState::bytes(100000, 4096, false),
                      100000, 4096, false);
    audit<TileState>("TileState t=1 b=0", TileState::bytes(1, 0), 1, 0);
    audit<TileState>("TileState t=1000 b=4096",
                     TileState::bytes(1000, 4096), 1000, 4096);
    audit<PointState>("PointState p=1 t=1", PointState::bytes(1, 1, 4096),
                      1, 1, 4096);
    audit<PointState>("PointState p=1000 t=1000",
                      PointState::bytes(1000, 1000, 4096), 1000, 1000, 4096);
    return 0;
}
