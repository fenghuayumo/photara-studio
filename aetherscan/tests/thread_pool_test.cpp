#include "parallel/thread_pool.hpp"

#include <atomic>
#include <cstddef>
#include <iostream>
#include <stdexcept>
#include <vector>

int main() {
    using aetherscan::parallel::parallel_for;
    const unsigned threads =
        std::min(8U, aetherscan::parallel::hardware_threads());
    std::atomic<std::size_t> sum{0};
    std::vector<std::atomic_flag> active(threads);
    for (auto& flag : active) flag.clear();

    parallel_for(
        10000, threads,
        [&](const std::size_t index, const unsigned worker_id) {
            if (worker_id >= active.size() ||
                active[worker_id].test_and_set(std::memory_order_acquire))
                throw std::runtime_error("worker id used concurrently");
            sum.fetch_add(index, std::memory_order_relaxed);
            active[worker_id].clear(std::memory_order_release);
        });
    if (sum.load() != 9999ULL * 10000ULL / 2ULL) {
        std::cerr << "parallel sum mismatch\n";
        return 1;
    }

    // Chunked scheduling must still cover every index exactly once.
    std::vector<std::atomic<unsigned>> hits(1000);
    for (auto& hit : hits) hit.store(0, std::memory_order_relaxed);
    parallel_for(hits.size(), threads, [&](const std::size_t index) {
        hits[index].fetch_add(1, std::memory_order_relaxed);
    });
    for (const auto& hit : hits) {
        if (hit.load(std::memory_order_relaxed) != 1U) {
            std::cerr << "chunked parallel_for missed or duplicated work\n";
            return 5;
        }
    }
    if (aetherscan::parallel::parallel_for_grain(10000, threads) < 32) {
        std::cerr << "unexpected parallel_for grain\n";
        return 6;
    }

    std::atomic<unsigned> nested_workers{0};
    parallel_for(threads, threads, [&](const std::size_t) {
        parallel_for(
            4, threads,
            [&](const std::size_t, const unsigned worker_id) {
                if (worker_id != 0)
                    nested_workers.fetch_add(1, std::memory_order_relaxed);
            });
    });
    if (nested_workers.load() != 0) {
        std::cerr << "nested parallelism exceeded thread budget\n";
        return 2;
    }

    auto& pool = aetherscan::parallel::global_thread_pool();
    std::atomic<unsigned> direct_nested_count{0};
    std::vector<std::future<void>> direct_tasks;
    direct_tasks.reserve(pool.worker_count());
    for (unsigned task = 0; task < pool.worker_count(); ++task) {
        direct_tasks.push_back(pool.submit([&] {
            parallel_for(8, threads, [&](const std::size_t) {
                direct_nested_count.fetch_add(1, std::memory_order_relaxed);
            });
        }));
    }
    for (auto& task : direct_tasks) task.get();
    if (direct_nested_count.load() != pool.worker_count() * 8U) {
        std::cerr << "direct nested pool work did not complete\n";
        return 3;
    }

    bool propagated = false;
    try {
        parallel_for(100, threads, [](const std::size_t index) {
            if (index == 17) throw std::runtime_error("expected");
        });
    } catch (const std::runtime_error&) {
        propagated = true;
    }
    if (!propagated) {
        std::cerr << "worker exception was not propagated\n";
        return 4;
    }

    // Zero-worker pools must run submit() inline instead of deadlocking.
    {
        aetherscan::parallel::ThreadPool empty_pool(0);
        std::atomic<unsigned> ran{0};
        empty_pool.submit([&] {
            ran.fetch_add(1, std::memory_order_relaxed);
        }).get();
        if (ran.load(std::memory_order_relaxed) != 1U) {
            std::cerr << "empty pool did not run task inline\n";
            return 7;
        }
    }

    // FutureGroup must drain outstanding work even when wait() sees an error.
    {
        aetherscan::parallel::FutureGroup group;
        std::atomic<unsigned> completed{0};
        group.submit(pool, [&] {
            completed.fetch_add(1, std::memory_order_relaxed);
        });
        group.submit(pool, [] { throw std::runtime_error("group failure"); });
        group.submit(pool, [&] {
            completed.fetch_add(1, std::memory_order_relaxed);
        });
        bool group_error = false;
        try {
            group.wait();
        } catch (const std::runtime_error&) {
            group_error = true;
        }
        if (!group_error) {
            std::cerr << "FutureGroup did not propagate worker error\n";
            return 8;
        }
        if (completed.load(std::memory_order_relaxed) != 2U) {
            std::cerr << "FutureGroup did not drain sibling tasks\n";
            return 9;
        }
    }
    return 0;
}
