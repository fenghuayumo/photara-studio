#pragma once

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <deque>
#include <exception>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <stop_token>
#include <thread>
#include <type_traits>
#include <utility>
#include <vector>

#if defined(AETHERSCAN_HAS_OPENMP)
#include <omp.h>
#endif

namespace aetherscan::parallel {

inline unsigned hardware_threads() noexcept {
    const unsigned count = std::thread::hardware_concurrency();
    return count == 0 ? 1U : count;
}

inline unsigned& parallel_region_depth() noexcept {
    thread_local unsigned depth = 0;
    return depth;
}

inline bool in_parallel_region() noexcept {
    return parallel_region_depth() > 0;
}

inline unsigned resolve_thread_count(unsigned requested) noexcept {
    if (in_parallel_region()) return 1;
    const unsigned available = hardware_threads();
    if (requested == 0) return available;
    return std::min(requested, available);
}

// True when OpenMP/SIMD inner loops should run; false under an outer task pool.
inline bool allow_inner_parallelism() noexcept { return !in_parallel_region(); }

class ScopedParallelRegion {
public:
    explicit ScopedParallelRegion(const bool active) : active_(active) {
        if (active_) ++parallel_region_depth();
    }
    ~ScopedParallelRegion() {
        if (active_) --parallel_region_depth();
    }
    ScopedParallelRegion(const ScopedParallelRegion&) = delete;
    ScopedParallelRegion& operator=(const ScopedParallelRegion&) = delete;

private:
    bool active_;
};

class ScopedOpenMpThreads {
public:
    explicit ScopedOpenMpThreads(const unsigned thread_count) {
#if defined(AETHERSCAN_HAS_OPENMP)
        previous_ = omp_get_max_threads();
        omp_set_num_threads(static_cast<int>(std::max(1U, thread_count)));
#else
        (void)thread_count;
#endif
    }
    ~ScopedOpenMpThreads() {
#if defined(AETHERSCAN_HAS_OPENMP)
        omp_set_num_threads(previous_);
#endif
    }
    ScopedOpenMpThreads(const ScopedOpenMpThreads&) = delete;
    ScopedOpenMpThreads& operator=(const ScopedOpenMpThreads&) = delete;

private:
#if defined(AETHERSCAN_HAS_OPENMP)
    int previous_{1};
#endif
};

class ThreadPool {
public:
    explicit ThreadPool(const unsigned worker_count) {
        workers_.reserve(worker_count);
        for (unsigned worker = 0; worker < worker_count; ++worker) {
            workers_.emplace_back([this](const std::stop_token stop) {
                for (;;) {
                    std::function<void()> task;
                    {
                        std::unique_lock lock(mutex_);
                        condition_.wait(
                            lock, stop,
                            [this] { return !tasks_.empty(); });
                        if (stop.stop_requested() && tasks_.empty()) return;
                        if (tasks_.empty()) continue;
                        task = std::move(tasks_.front());
                        tasks_.pop_front();
                    }
                    const ScopedParallelRegion task_region(true);
                    task();
                }
            });
        }
    }

    ~ThreadPool() {
        for (auto& worker : workers_) worker.request_stop();
        condition_.notify_all();
    }

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    [[nodiscard]] unsigned worker_count() const noexcept {
        return static_cast<unsigned>(workers_.size());
    }

    template <class Function>
    std::future<void> submit(Function&& function) {
        auto task = std::make_shared<std::packaged_task<void()>>(
            std::forward<Function>(function));
        std::future<void> result = task->get_future();
        // Zero-worker pools (single-core hosts) must run inline; otherwise
        // callers that wait on the returned future deadlock forever.
        if (workers_.empty()) {
            (*task)();
            return result;
        }
        {
            std::lock_guard lock(mutex_);
            tasks_.emplace_back([task = std::move(task)] { (*task)(); });
        }
        condition_.notify_one();
        return result;
    }

private:
    std::mutex mutex_;
    std::condition_variable_any condition_;
    std::deque<std::function<void()>> tasks_;
    std::vector<std::jthread> workers_;
};

// Wait for every future even if one throws, then rethrow the first error.
inline void wait_all(std::vector<std::future<void>>& futures) {
    std::exception_ptr error;
    for (std::future<void>& future : futures) {
        if (!future.valid()) continue;
        try {
            future.get();
        } catch (...) {
            if (!error) error = std::current_exception();
        }
    }
    futures.clear();
    if (error) std::rethrow_exception(error);
}

// Owns outstanding pool futures and always drains them on destruction so
// stack-capturing tasks cannot outlive their coordinator frame (UAF).
class FutureGroup {
public:
    FutureGroup() = default;
    FutureGroup(const FutureGroup&) = delete;
    FutureGroup& operator=(const FutureGroup&) = delete;

    ~FutureGroup() {
        for (std::future<void>& future : futures_) {
            if (!future.valid()) continue;
            try {
                future.get();
            } catch (...) {
                // Destructor must not throw; the owning scope already failed.
            }
        }
    }

    template <class Function>
    void submit(ThreadPool& pool, Function&& function) {
        futures_.push_back(pool.submit(std::forward<Function>(function)));
    }

    void wait() { wait_all(futures_); }

    void reserve(const std::size_t count) { futures_.reserve(count); }

private:
    std::vector<std::future<void>> futures_;
};

inline ThreadPool& global_thread_pool() {
    static ThreadPool pool(hardware_threads() > 1 ? hardware_threads() - 1 : 0);
    return pool;
}

// Choose a grain large enough to amortize atomic claims while keeping enough
// chunks for load balancing on irregular work.
inline std::size_t parallel_for_grain(
    const std::size_t count, const unsigned thread_count) noexcept {
    if (count <= thread_count) return 1;
    constexpr std::size_t k_min_grain = 32;
    const std::size_t target_chunks =
        static_cast<std::size_t>(std::max(1U, thread_count)) * 8U;
    return std::max(k_min_grain, (count + target_chunks - 1) / target_chunks);
}

// Dynamic chunked parallel-for over [0, count). Workers claim ranges with an
// atomic counter so small items avoid one atomic op per iteration.
template <class Function>
void parallel_for(const std::size_t count, unsigned thread_count, Function&& function) {
    const auto invoke = [&](const std::size_t index, const unsigned worker_id) {
        if constexpr (std::is_invocable_v<Function&, std::size_t, unsigned>)
            function(index, worker_id);
        else {
            (void)worker_id;
            function(index);
        }
    };
    if (count == 0) return;
    thread_count = resolve_thread_count(thread_count);
    thread_count = std::min(
        thread_count, global_thread_pool().worker_count() + 1);
    thread_count = static_cast<unsigned>(std::min<std::size_t>(thread_count, count));
    if (thread_count <= 1) {
        for (std::size_t index = 0; index < count; ++index) invoke(index, 0);
        return;
    }

    const ScopedParallelRegion region(true);
    const std::size_t grain = parallel_for_grain(count, thread_count);
    std::atomic<std::size_t> next{0};
    std::vector<std::future<void>> workers;
    workers.reserve(thread_count - 1);
    std::exception_ptr error;
    std::mutex error_mutex;

    const auto worker = [&](const unsigned worker_id) {
        const ScopedParallelRegion worker_region(true);
        try {
            for (;;) {
                const std::size_t begin =
                    next.fetch_add(grain, std::memory_order_relaxed);
                if (begin >= count) break;
                const std::size_t end = std::min(begin + grain, count);
                for (std::size_t index = begin; index < end; ++index)
                    invoke(index, worker_id);
            }
        } catch (...) {
            std::lock_guard lock(error_mutex);
            if (!error) error = std::current_exception();
            next.store(count, std::memory_order_relaxed);
        }
    };

    for (unsigned thread = 1; thread < thread_count; ++thread)
        workers.emplace_back(
            global_thread_pool().submit(
                [&, thread] { worker(thread); }));
    worker(0);
    for (auto& thread : workers) thread.get();
    if (error) std::rethrow_exception(error);
}

}  // namespace aetherscan::parallel
