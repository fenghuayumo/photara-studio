#pragma once

#include <cuda_runtime.h>
#include <cstddef>
#include <mutex>
#include <atomic>
#include <unordered_map>
#include <vector>

namespace tinytensor {

// Full pinned memory allocator with caching
class PinnedMemoryAllocator {
public:
    struct Block {
        void* ptr = nullptr;
        size_t size = 0;
        cudaStream_t last_stream = nullptr;
        cudaEvent_t ready_event = nullptr;

        Block() = default;
        Block(void* p, size_t s, cudaStream_t stream);
        ~Block();
        Block(Block&& other) noexcept;
        Block& operator=(Block&& other) noexcept;
        Block(const Block&) = delete;
        Block& operator=(const Block&) = delete;
    };

    struct Stats {
        size_t allocated_bytes = 0;
        size_t cached_bytes = 0;
        size_t num_allocs = 0;
        size_t num_deallocs = 0;
        size_t cache_hits = 0;
        size_t cache_misses = 0;
    };

    static PinnedMemoryAllocator& instance();

    void* allocate(size_t bytes);
    void deallocate(void* ptr, cudaStream_t stream = nullptr);
    void empty_cache();

    Stats get_stats() const;
    void reset_stats();
    void prewarm();

    void set_enabled(bool enabled) { enabled_ = enabled; }
    bool is_enabled() const { return enabled_; }

private:
    PinnedMemoryAllocator();
    ~PinnedMemoryAllocator();
    PinnedMemoryAllocator(const PinnedMemoryAllocator&) = delete;
    PinnedMemoryAllocator& operator=(const PinnedMemoryAllocator&) = delete;

    void shutdown();
    static size_t round_size(size_t bytes);

    mutable std::mutex mutex_;
    std::atomic<bool> shutdown_{false};
    std::atomic<bool> enabled_{true};

    std::unordered_map<size_t, std::vector<Block>> cache_;
    std::unordered_map<void*, size_t> allocated_blocks_;
    Stats stats_;
};

} // namespace tinytensor
