#pragma once

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace aetherscan::parallel {

inline unsigned hardware_threads() noexcept {
    const unsigned count = std::thread::hardware_concurrency();
    return count == 0 ? 1U : count;
}

inline unsigned resolve_thread_count(unsigned requested) noexcept {
    const unsigned available = hardware_threads();
    if (requested == 0) return available;
    return std::min(requested, available);
}

inline std::atomic<int>& parallel_region_depth() noexcept {
    static std::atomic<int> depth{0};
    return depth;
}

inline bool in_parallel_region() noexcept {
    return parallel_region_depth().load(std::memory_order_relaxed) > 0;
}

// True when OpenMP/SIMD inner loops should run; false under an outer task pool.
inline bool allow_inner_parallelism() noexcept { return !in_parallel_region(); }

class ScopedParallelRegion {
public:
    explicit ScopedParallelRegion(const bool active) : active_(active) {
        if (active_) parallel_region_depth().fetch_add(1, std::memory_order_relaxed);
    }
    ~ScopedParallelRegion() {
        if (active_) parallel_region_depth().fetch_sub(1, std::memory_order_relaxed);
    }
    ScopedParallelRegion(const ScopedParallelRegion&) = delete;
    ScopedParallelRegion& operator=(const ScopedParallelRegion&) = delete;

private:
    bool active_;
};

// Dynamic work-stealing style parallel-for over [0, count). Each worker claims the
// next index with an atomic counter so uneven tasks (SIFT, RANSAC) stay balanced.
template <class Function>
void parallel_for(const std::size_t count, unsigned thread_count, Function&& function) {
    if (count == 0) return;
    thread_count = resolve_thread_count(thread_count);
    thread_count = static_cast<unsigned>(std::min<std::size_t>(thread_count, count));
    if (thread_count <= 1) {
        for (std::size_t index = 0; index < count; ++index) function(index);
        return;
    }

    const ScopedParallelRegion region(true);
    std::atomic<std::size_t> next{0};
    std::vector<std::thread> workers;
    workers.reserve(thread_count);
    std::exception_ptr error;
    std::mutex error_mutex;

    const auto worker = [&]() {
        try {
            for (;;) {
                const std::size_t index = next.fetch_add(1, std::memory_order_relaxed);
                if (index >= count) break;
                function(index);
            }
        } catch (...) {
            std::lock_guard lock(error_mutex);
            if (!error) error = std::current_exception();
            next.store(count, std::memory_order_relaxed);
        }
    };

    for (unsigned thread = 1; thread < thread_count; ++thread) workers.emplace_back(worker);
    worker();
    for (auto& thread : workers) thread.join();
    if (error) std::rethrow_exception(error);
}

}  // namespace aetherscan::parallel
