#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"
#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

// Manual verification tool (not a ctest entry): measures what one tensor op
// costs on the tinytensor Vulkan backend, split into the per-op fixed cost and
// the per-element cost. Both matter: the runtime submits and waits per op, so
// the fixed part is a function of the op count, not of the tensor size.
//
// Usage: photara_tinytensor_vulkan_bench [REPEATS]

namespace {

using tinytensor::DataType;
using tinytensor::Device;
using tinytensor::Tensor;

double median(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

template <typename Body>
double median_ms(Body&& body, const int repeats) {
    volatile float sink = body();
    (void)sink;
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        sink = body();
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    return median(samples);
}

// A dependent chain of `ops` elementwise steps: every step reads the previous
// result, so nothing can be reordered or collapsed. The chain ends with a full
// readback, which is also what forces the queued work to run.
double chain_ms(const std::size_t count, const int ops, const int repeats, const Device device) {
    auto a = Tensor::from_vector(std::vector<float>(count, 0.5F), {count}, device);
    auto b = Tensor::from_vector(std::vector<float>(count, 1.25F), {count}, device);
    const auto body = [&] {
        Tensor c = a;
        for (int i = 0; i < ops; ++i) c = c.add(b);
        return c.to_vector()[0];
    };
    return median_ms(body, repeats);
}

// Same chain, but flushed explicitly instead of read back, so the measured
// time is the GPU work and the submission, not the device-to-host copy.
double chain_flush_ms(const std::size_t count, const int ops, const int repeats) {
    auto a = Tensor::from_vector(std::vector<float>(count, 0.5F), {count}, Device::Vulkan);
    auto b = Tensor::from_vector(std::vector<float>(count, 1.25F), {count}, Device::Vulkan);
    auto& context = tinytensor::vulkan::runtime::Context::get();
    const auto body = [&] {
        Tensor c = a;
        for (int i = 0; i < ops; ++i) c = c.add(b);
        // No readback: the flush is what runs the chain, so the time is the GPU
        // work plus one submission.
        context.flush();
        return static_cast<float>(c.numel());
    };
    return median_ms(body, repeats);
}

// Readback on its own: creating a tensor and pulling it back to the host.
double readback_ms(const std::size_t count, const int repeats) {
    auto tensor = Tensor::from_vector(std::vector<float>(count, 1.0F), {count}, Device::Vulkan);
    return median_ms([&] { return tensor.to_vector()[0]; }, repeats);
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!tinytensor::vulkan::available())
            throw std::runtime_error("the tinytensor Vulkan backend is not available");
        const int repeats = argc > 1 ? std::stoi(argv[1]) : 15;
        const auto& info = tinytensor::vulkan::device_info();
        std::printf("device=%s api=%u subgroup=%u\n", info.name.c_str(), info.api_version,
                    info.subgroup_size);

        // Which elementwise ops actually run on the Vulkan backend. A failure
        // here is a coverage gap, not a timing result, so probe before timing.
        std::printf("\n=== op coverage on Device::Vulkan ===\n");
        {
            // Before anything else: is a plain upload/readback round trip
            // correct, and does it stay correct as the staging grows?
            for (const std::size_t count : {std::size_t{1}, std::size_t{4}, std::size_t{7},
                                            std::size_t{64}, std::size_t{4096}, std::size_t{7},
                                            std::size_t{4096}}) {
                auto t = Tensor::from_vector(std::vector<float>(count, 1.25F), {count},
                                             Device::Vulkan);
                std::printf("upload/readback count=%llu -> %g (want 1.25)\n",
                            static_cast<unsigned long long>(count), t.to_vector()[0]);
            }
        }
        const auto probe = [](const char* name, const auto& op) {
            try {
                Tensor result = op();
                const bool vulkan = result.device() == Device::Vulkan;
                const auto values = result.to_vector();
                std::printf("%-12s ok (device=%s first=%g)\n", name,
                            vulkan ? "vulkan" : "non-vulkan", values.empty() ? 0.0 : values[0]);
                return vulkan;
            } catch (const std::exception& error) {
                std::printf("%-12s FAILED: %s\n", name, error.what());
                return false;
            }
        };
        auto probe_a = Tensor::from_vector(std::vector<float>(256, 0.5F), {std::size_t{256}},
                                           Device::Vulkan);
        auto probe_b = Tensor::from_vector(std::vector<float>(256, 1.25F), {std::size_t{256}},
                                           Device::Vulkan);
        const int supported =
            probe("add", [&] { return probe_a.add(probe_b); }) +
            probe("sub", [&] { return probe_a.sub(probe_b); }) +
            probe("mul", [&] { return probe_a.mul(probe_b); }) +
            probe("div", [&] { return probe_a.div(probe_b); }) +
            probe("relu", [&] { return probe_a.relu(); }) +
            probe("exp", [&] { return probe_a.exp(); }) +
            probe("sigmoid", [&] { return probe_a.sigmoid(); }) +
            probe("sqrt", [&] { return probe_a.sqrt(); }) +
            probe("tanh", [&] { return probe_a.tanh(); }) +
            probe("neg", [&] { return probe_a.neg(); }) +
            probe("sum", [&] { return probe_a.sum(); });
        std::printf("%d of 11 probed ops stay on the Vulkan device\n", supported);

        // 1. Fixed cost per op: 1024 floats is far below any bandwidth limit.
        std::printf("\n=== tiny tensors (1024 floats), chained adds ===\n");
        std::printf("%8s %12s %14s %12s %14s %8s\n", "ops", "vulkan_ms", "vulkan_us/op",
                    "cuda_ms", "cuda_us/op", "ratio");
        for (const int ops : {1, 10, 50, 200}) {
            const double vulkan = chain_ms(1024, ops, repeats, Device::Vulkan);
            const double cuda = chain_ms(1024, ops, repeats, Device::CUDA);
            std::printf("%8d %12.4f %14.4f %12.4f %14.4f %7.1fx\n", ops, vulkan,
                        vulkan / ops * 1000.0, cuda, cuda / ops * 1000.0, vulkan / cuda);
        }

        // 2. Per-element cost: same op count, growing tensors. Each elementwise
        // op reads two arrays and writes one, so 12 bytes per element.
        constexpr int k_sweep_ops = 20;
        std::printf("\n=== size sweep (chained adds, one flush, no readback) ===\n");
        std::printf("%12s %11s %11s %11s %11s %12s %11s\n", "elements", "1_op_ms", "20_op_ms",
                    "us/op", "GB/s", "readback_ms", "cuda_ms");
        for (const std::size_t count : {std::size_t{1} << 12, std::size_t{1} << 16,
                                        std::size_t{1} << 20, std::size_t{1} << 22}) {
            const double one_op = chain_flush_ms(count, 1, repeats);
            const double many_ops = chain_flush_ms(count, k_sweep_ops, repeats);
            const double readback = readback_ms(count, repeats);
            const double cuda = chain_ms(count, k_sweep_ops, repeats, Device::CUDA);
            const double bytes = static_cast<double>(k_sweep_ops) * static_cast<double>(count) * 12.0;
            std::printf("%12llu %11.4f %11.4f %11.4f %11.2f %12.4f %11.4f\n",
                        static_cast<unsigned long long>(count), one_op, many_ops,
                        many_ops / k_sweep_ops * 1000.0, bytes / (many_ops * 1.0e6), readback, cuda);
        }

        // 3. Correctness under the new batching and pooling: chains of varying
        // length, shapes that change between rounds, and readbacks interleaved
        // with the chain (which is what forces a flush mid-sequence). Every
        // result is checked against the same chain on the CPU.
        std::printf("\n=== batching/pooling stress against a CPU reference ===\n");
        std::size_t rounds_checked = 0;
        bool stress_ok = true;
        for (const std::size_t count : {std::size_t{1}, std::size_t{7}, std::size_t{4096},
                                        std::size_t{100000}, std::size_t{4096}, std::size_t{65536}}) {
            const int length = static_cast<int>(1 + (rounds_checked * 7) % 11);
            const float seed = 0.25F + static_cast<float>(rounds_checked) * 0.5F;
            const std::vector<float> host_a(count, seed);
            const std::vector<float> host_b(count, -0.5F);
            auto a = Tensor::from_vector(host_a, {count}, Device::Vulkan);
            auto b = Tensor::from_vector(host_b, {count}, Device::Vulkan);
            auto cpu_a = Tensor::from_vector(host_a, {count}, Device::CPU);
            auto cpu_b = Tensor::from_vector(host_b, {count}, Device::CPU);

            Tensor value = a;
            Tensor reference = cpu_a;
            for (int step = 0; step < length; ++step) {
                value = step % 3 == 2 ? value.sub(b) : value.add(b);
                reference = step % 3 == 2 ? reference.sub(cpu_b) : reference.add(cpu_b);
                if (step % 4 == 1) {
                    // Readback in the middle of the chain: flush, then continue.
                    const float mid = value.to_vector()[0];
                    const float expected_mid = reference.to_vector()[0];
                    if (std::abs(mid - expected_mid) > 1e-3F) {
                        std::printf("  MISMATCH mid-chain count=%llu step=%d got=%g want=%g\n",
                                    static_cast<unsigned long long>(count), step, mid, expected_mid);
                        stress_ok = false;
                    }
                }
            }
            const auto got = value.to_vector();
            const auto want = reference.to_vector();
            for (std::size_t i = 0; i < got.size(); i += (count > 64 ? count / 64 : 1)) {
                if (std::abs(got[i] - want[i]) > 1e-3F) {
                    std::printf("  MISMATCH count=%llu index=%llu got=%g want=%g\n",
                                static_cast<unsigned long long>(count),
                                static_cast<unsigned long long>(i), got[i], want[i]);
                    stress_ok = false;
                    break;
                }
            }
            ++rounds_checked;
        }
        std::printf("%zu rounds checked against the CPU reference: %s\n", rounds_checked,
                    stress_ok ? "all match" : "FAILED");

        // 4. Focused repro of the failing pattern: a readback in the middle of
        // a chain, which is what splits it into two batches.
        for (const int mid_readback : {0, 1}) {
            const std::size_t count = 4096;
            auto a = Tensor::from_vector(std::vector<float>(count, 1.25F), {count}, Device::Vulkan);
            auto b = Tensor::from_vector(std::vector<float>(count, -0.5F), {count}, Device::Vulkan);
            float want = 1.25F;
            Tensor value = a;
            std::printf("repro mid_readback=%d: a[0]=%g b[0]=%g\n", mid_readback,
                        value.to_vector()[0], b.to_vector()[0]);
            for (int step = 0; step < 4; ++step) {
                if (step % 3 == 2) {
                    value = value.sub(b);
                    want += 0.5F;
                } else {
                    value = value.add(b);
                    want -= 0.5F;
                }
                if (mid_readback == 1 && step == 1) {
                    std::printf("   after %d ops: got %g want %g\n", step + 1, value.to_vector()[0],
                                want);
                }
            }
            std::printf("   final: got %g want %g\n", value.to_vector()[0], want);
        }
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
