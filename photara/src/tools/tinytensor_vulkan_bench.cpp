#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"
#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
#include <numeric>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime_api.h>

// Manual verification tool (not a ctest entry): measures what one tensor op
// costs on the tinytensor Vulkan backend, split into the per-op fixed cost and
// the per-element cost. Both matter: the runtime submits and waits per op, so
// the fixed part is a function of the op count, not of the tensor size.
//
// Usage: photara_tinytensor_vulkan_bench [REPEATS | --million [REPEATS] |
//                                        --densify [COUNT] | --ops [REPEATS]]

namespace {

using tinytensor::DataType;
using tinytensor::Device;
using tinytensor::ScatterMode;
using tinytensor::Tensor;

double median(std::vector<double>& samples) {
    if (samples.empty()) return 0.0;
    std::sort(samples.begin(), samples.end());
    return samples[samples.size() / 2];
}

void sync_device(const Device device) {
    if (device == Device::Vulkan) {
        tinytensor::vulkan::synchronize();
        return;
    }
    if (cudaDeviceSynchronize() != cudaSuccess)
        throw std::runtime_error("cudaDeviceSynchronize failed");
}

template <typename Body>
double time_call(Body&& body, const int repeats) {
    body();
    std::vector<double> samples;
    samples.reserve(static_cast<std::size_t>(repeats));
    for (int i = 0; i < repeats; ++i) {
        const auto start = std::chrono::steady_clock::now();
        body();
        const auto finish = std::chrono::steady_clock::now();
        samples.push_back(std::chrono::duration<double, std::milli>(finish - start).count());
    }
    return median(samples);
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
double chain_sync_ms(const std::size_t count, const int ops, const int repeats,
                     const Device device) {
    auto a = Tensor::from_vector(std::vector<float>(count, 0.5F), {count}, device);
    auto b = Tensor::from_vector(std::vector<float>(count, 1.25F), {count}, device);
    const auto body = [&] {
        Tensor c = a;
        for (int i = 0; i < ops; ++i) c = c.add(b);
        // Force both backends to finish without charging either one for a
        // device-to-host copy.  CUDA launches are asynchronous, while Vulkan
        // records the whole chain and submits it here.
        if (device == Device::Vulkan) {
            tinytensor::vulkan::runtime::Context::get().flush();
        } else {
            if (cudaDeviceSynchronize() != cudaSuccess)
                throw std::runtime_error("cudaDeviceSynchronize failed");
        }
        return static_cast<float>(c.numel());
    };
    return median_ms(body, repeats);
}

// A tensor-level Adam expression representative of an unfused optimizer.  The
// production splat optimizer should eventually be one fused Vulkan shader, but
// this exposes runtime overhead and Float32 elementwise throughput while that
// kernel is being built.
double adam_expression_ms(const std::size_t count, const int repeats, const Device device) {
    auto parameter = Tensor::from_vector(std::vector<float>(count, 0.25F), {count}, device);
    auto gradient = Tensor::from_vector(std::vector<float>(count, 0.01F), {count}, device);
    auto first = Tensor::from_vector(std::vector<float>(count, 0.0F), {count}, device);
    auto second = Tensor::from_vector(std::vector<float>(count, 0.0F), {count}, device);
    const auto body = [&] {
        Tensor next_first = first.mul(0.9F).add(gradient.mul(0.1F));
        Tensor next_second = second.mul(0.999F).add(gradient.square().mul(0.001F));
        Tensor update = next_first.div(next_second.sqrt().add(1e-8F)).mul(1e-3F);
        Tensor next_parameter = parameter.sub(update);
        if (device == Device::Vulkan) {
            tinytensor::vulkan::synchronize();
        } else if (cudaDeviceSynchronize() != cudaSuccess) {
            throw std::runtime_error("cudaDeviceSynchronize failed");
        }
        return static_cast<float>(next_parameter.numel());
    };
    return median_ms(body, repeats);
}

double adam_fused_vulkan_ms(const std::size_t count, const int repeats) {
    auto parameter = Tensor::from_vector(std::vector<float>(count, 0.25F), {count}, Device::Vulkan);
    auto gradient = Tensor::from_vector(std::vector<float>(count, 0.01F), {count}, Device::Vulkan);
    auto first = Tensor::from_vector(std::vector<float>(count, 0.0F), {count}, Device::Vulkan);
    auto second = Tensor::from_vector(std::vector<float>(count, 0.0F), {count}, Device::Vulkan);
    tinytensor::vulkan::AdamStepOptions options;
    options.correction1 = 0.1F;
    options.correction2 = 0.001F;
    return median_ms([&] {
        tinytensor::vulkan::adam_step(parameter, gradient, first, second, options);
        tinytensor::vulkan::synchronize();
        return static_cast<float>(parameter.numel());
    }, repeats);
}

double full_reduce_ms(const std::size_t count, const int repeats, const Device device,
                      const bool mean) {
    auto input = Tensor::from_vector(std::vector<float>(count, 1.0F), {count}, device);
    const auto body = [&] {
        Tensor result = mean ? input.mean() : input.sum();
        return result.to_vector()[0];
    };
    return median_ms(body, repeats);
}

// Readback on its own: creating a tensor and pulling it back to the host.
double readback_ms(const std::size_t count, const int repeats) {
    auto tensor = Tensor::from_vector(std::vector<float>(count, 1.0F), {count}, Device::Vulkan);
    return median_ms([&] { return tensor.to_vector()[0]; }, repeats);
}

bool million_correctness(const std::size_t count) {
    std::vector<float> host_a(count), host_b(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_a[i] = 0.25F + static_cast<float>(i % 1009) * 0.001F;
        host_b[i] = 0.5F + static_cast<float>(i % 127) * 0.002F;
    }
    auto vk_a = Tensor::from_vector(host_a, {count}, Device::Vulkan);
    auto vk_b = Tensor::from_vector(host_b, {count}, Device::Vulkan);
    auto cu_a = Tensor::from_vector(host_a, {count}, Device::CUDA);
    auto cu_b = Tensor::from_vector(host_b, {count}, Device::CUDA);

    const char* profile_env = std::getenv("TINYTENSOR_VULKAN_PROFILE_OPS");
    const bool profile = profile_env != nullptr && std::strtoul(profile_env, nullptr, 10) != 0;
    const auto dispatches = [](const char* shader) {
        std::uint64_t total = 0;
        for (const auto& entry : tinytensor::vulkan::runtime::Context::get().op_profile())
            if (std::string(entry.name) == shader) total += entry.calls;
        return total;
    };
    const auto compare = [&](const char* name, const auto& vk_op, const auto& cu_op,
                             const float tolerance, const char* shader,
                             const std::uint64_t min_dispatches) {
        const auto before = profile ? dispatches(shader) : 0;
        const Tensor vk = vk_op();
        const Tensor cu = cu_op();
        if (vk.device() != Device::Vulkan || cu.device() != Device::CUDA)
            throw std::runtime_error(std::string(name) + " result device mismatch");
        const auto actual = vk.to_vector();
        const auto expected = cu.to_vector();
        const auto vk_dispatches = profile ? dispatches(shader) - before : 0;
        if (actual.size() != expected.size())
            throw std::runtime_error(std::string(name) + " output size mismatch");
        float max_abs = 0.0F, max_rel = 0.0F;
        std::size_t bad = 0;
        for (std::size_t i = 0; i < actual.size(); ++i) {
            const float abs_error = std::abs(actual[i] - expected[i]);
            const float rel_error = abs_error / std::max(1.0F, std::abs(expected[i]));
            if (!std::isfinite(actual[i]) || !std::isfinite(expected[i]) ||
                abs_error > tolerance * std::max(1.0F, std::abs(expected[i]))) ++bad;
            max_abs = std::max(max_abs, abs_error);
            max_rel = std::max(max_rel, rel_error);
        }
        std::printf("%-12s count=%llu max_abs=%.9g max_rel=%.9g mismatches=%llu",
                    name, static_cast<unsigned long long>(actual.size()), max_abs, max_rel,
                    static_cast<unsigned long long>(bad));
        if (profile) std::printf(" vk_%s_dispatches=%llu", shader,
                                 static_cast<unsigned long long>(vk_dispatches));
        std::printf("\n");
        return bad == 0 && (!profile || vk_dispatches >= min_dispatches);
    };

    bool ok = compare("roundtrip", [&] { return vk_a; }, [&] { return cu_a; }, 0.0F,
                      "elementwise", 0);
    ok = compare("add", [&] { return vk_a.add(vk_b); }, [&] { return cu_a.add(cu_b); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("mul", [&] { return vk_a.mul(vk_b); }, [&] { return cu_a.mul(cu_b); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("sub", [&] { return vk_a.sub(vk_b); }, [&] { return cu_a.sub(cu_b); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("div", [&] { return vk_a.div(vk_b); }, [&] { return cu_a.div(cu_b); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("relu", [&] { return vk_a.sub(vk_b).relu(); },
                 [&] { return cu_a.sub(cu_b).relu(); },
                 1e-5F, "elementwise", 2) && ok;
    ok = compare("sqrt", [&] { return vk_a.sqrt(); }, [&] { return cu_a.sqrt(); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("exp", [&] { return vk_a.exp(); }, [&] { return cu_a.exp(); },
                 2e-5F, "elementwise", 1) && ok;
    ok = compare("sigmoid", [&] { return vk_a.sigmoid(); }, [&] { return cu_a.sigmoid(); },
                 2e-5F, "elementwise", 1) && ok;
    ok = compare("tanh", [&] { return vk_a.tanh(); }, [&] { return cu_a.tanh(); },
                 2e-5F, "elementwise", 1) && ok;
    ok = compare("neg", [&] { return vk_a.neg(); }, [&] { return cu_a.neg(); },
                 1e-5F, "elementwise", 1) && ok;
    ok = compare("20-add chain", [&] {
        Tensor value = vk_a;
        for (int i = 0; i < 20; ++i) value = value.add(vk_b);
        return value;
    }, [&] {
        Tensor value = cu_a;
        for (int i = 0; i < 20; ++i) value = value.add(cu_b);
        return value;
    }, 2e-5F, "elementwise", 20) && ok;
    ok = compare("scalar chain", [&] {
        Tensor value = vk_a;
        for (int i = 0; i < 20; ++i) value = value.add(1.0F).mul(0.5F);
        return value;
    }, [&] {
        Tensor value = cu_a;
        for (int i = 0; i < 20; ++i) value = value.add(1.0F).mul(0.5F);
        return value;
    }, 2e-5F, "elementwise", 40) && ok;
    ok = compare("sum", [&] { return vk_a.sum(); }, [&] { return cu_a.sum(); },
                 1e-4F, "reduce_all_f32", 1) && ok;
    ok = compare("mean", [&] { return vk_a.mean(); }, [&] { return cu_a.mean(); },
                 1e-4F, "reduce_all_f32", 1) && ok;
    return ok;
}

void million_benchmark(const int repeats) {
    constexpr std::size_t count = 1U << 20;
    std::printf("=== CUDA/Vulkan comparison: %llu elements, %d repeats, median wall ms ===\n",
                static_cast<unsigned long long>(count), repeats);
    if (!million_correctness(count))
        throw std::runtime_error("million-element CUDA/Vulkan correctness comparison failed");
    std::printf("\n%-20s %12s %12s %10s\n", "operation", "vulkan_ms", "cuda_ms", "vk/cuda");
    const auto row = [](const char* name, const double vk, const double cu) {
        std::printf("%-20s %12.4f %12.4f %9.2fx\n", name, vk, cu, vk / cu);
    };
    row("1 add, sync", chain_sync_ms(count, 1, repeats, Device::Vulkan),
        chain_sync_ms(count, 1, repeats, Device::CUDA));
    row("20 adds, sync", chain_sync_ms(count, 20, repeats, Device::Vulkan),
        chain_sync_ms(count, 20, repeats, Device::CUDA));
    row("sum + readback", full_reduce_ms(count, repeats, Device::Vulkan, false),
        full_reduce_ms(count, repeats, Device::CUDA, false));
    row("mean + readback", full_reduce_ms(count, repeats, Device::Vulkan, true),
        full_reduce_ms(count, repeats, Device::CUDA, true));
    row("Adam expression", adam_expression_ms(count, repeats, Device::Vulkan),
        adam_expression_ms(count, repeats, Device::CUDA));
    std::printf("%-20s %12.4f %12s %10s\n", "Vulkan fused Adam",
                adam_fused_vulkan_ms(count, repeats), "n/a", "n/a");
}

// The ADC-IGS refinement is built from a handful of tensor primitives that
// all have a host-visible result: masks are counted, indices are compacted,
// selections are sorted. This isolates each one on both backends at the sizes
// the glass scene actually reaches during refinement.
void densify_benchmark(const std::size_t count, const int repeats) {
    std::printf("=== densification primitives: %llu elements, %d repeats, median wall ms ===\n",
                static_cast<unsigned long long>(count), repeats);
    std::printf("%-28s %12s %12s %10s\n", "operation", "vulkan_ms", "cuda_ms", "vk/cuda");
    const auto row = [](const char* name, const double vk, const double cu) {
        if (cu > 0.0)
            std::printf("%-28s %12.4f %12.4f %9.2fx\n", name, vk, cu, vk / cu);
        else
            std::printf("%-28s %12.4f %12s %10s\n", name, vk, "n/a", "n/a");
    };

    std::vector<float> host_values(count);
    std::vector<bool> host_mask(count);
    for (std::size_t i = 0; i < count; ++i) {
        host_values[i] = static_cast<float>((i * 2654435761u) % 1000u) / 1000.0F;
        host_mask[i] = (i % 3u) == 0u ? 1 : 0;
    }

    const auto make_bool = [&](const Device device) {
        return Tensor::from_vector(host_mask, {count}, device);
    };
    const auto make_values = [&](const Device device) {
        return Tensor::from_vector(host_values, {count}, device);
    };

    row("count_nonzero(bool)",
        median_ms([&] { return static_cast<float>(make_bool(Device::Vulkan).count_nonzero()); }, 1),
        median_ms([&] { return static_cast<float>(make_bool(Device::CUDA).count_nonzero()); }, 1));

    {
        // Broadcast probe: [N,3] - [1,3] must stay row-wise on Vulkan. A
        // regression here would silently corrupt device-side prune masks.
        const std::size_t rows = count / 3;
        std::vector<float> host_rows(3 * rows);
        for (std::size_t i = 0; i < 3 * rows; ++i) host_rows[i] = host_values[i];
        auto values = Tensor::from_vector(
            host_rows, {rows, 3}, Device::Vulkan);
        auto center = Tensor::from_vector(
            std::vector<float>{0.25F, 0.5F, 0.75F}, {1, 3}, Device::Vulkan);
        const auto shifted = values.sub(center).abs();
        const auto host = shifted.to_vector();
        double worst = 0.0;
        for (std::size_t i = 0; i < 3 * rows; ++i) {
            const double expected = std::abs(host_values[i] -
                (i % 3 == 0 ? 0.25F : i % 3 == 1 ? 0.5F : 0.75F));
            worst = std::max(worst, std::abs(static_cast<double>(host[i]) - expected));
        }
        std::printf("broadcast [N,3]-[1,3] max_abs_error=%g\n", worst);
        auto finite_rows = values.isfinite().all(1);
        std::printf("all(dim=1) finite_rows=%lld/%llu\n",
            static_cast<long long>(finite_rows.count_nonzero()),
            static_cast<unsigned long long>(rows));
    }

    {
        // Replay of the device-side prune_masks composition at the glass
        // scene's initial size. This isolates any pathological op from the
        // full trainer.
        const std::size_t rows = count;
        std::vector<float> means(3 * rows), scales(3 * rows),
            rotations(4 * rows), logits(rows), sh(48 * rows);
        for (std::size_t i = 0; i < sh.size(); ++i) sh[i] = 0.5F;
        for (std::size_t i = 0; i < logits.size(); ++i) logits[i] = 0.25F;
        auto model_means = Tensor::from_vector(means, {rows, 3}, Device::Vulkan);
        auto model_scales = Tensor::from_vector(scales, {rows, 3}, Device::Vulkan);
        auto model_rotations = Tensor::from_vector(rotations, {rows, 4}, Device::Vulkan);
        auto model_logits = Tensor::from_vector(logits, {rows}, Device::Vulkan);
        auto model_sh = Tensor::from_vector(sh, {rows, 48}, Device::Vulkan);
        auto keep = Tensor::zeros_bool({rows}, Device::Vulkan);
        auto hard = Tensor::zeros_bool({rows}, Device::Vulkan);
        auto opacities = Tensor::empty({rows}, Device::Vulkan);
        const auto run = [&] {
            tinytensor::vulkan::prune_masks(
                model_means, model_scales, model_rotations, model_logits,
                model_sh, keep, hard, opacities, 0.01F, 1000.F, 0.F, 0.F, 0.F);
            return keep;
        };
        auto result = run();
        std::printf("prune replay hard=%lld keep=%lld\n",
            static_cast<long long>(result.logical_not().count_nonzero()),
            static_cast<long long>(result.count_nonzero()));
        std::fflush(stdout);
        {
            // Poison representative rows and compare against the reference
            // sweep the fused shader replaced.
            auto host_means = model_means.to_vector();
            auto host_scales = model_scales.to_vector();
            auto host_rotations = model_rotations.to_vector();
            auto host_logits = model_logits.to_vector();
            auto host_sh = model_sh.to_vector();
            const auto poison = [](std::vector<float>& values,
                                   const std::size_t row, const float value) {
                for (std::size_t axis = 0; axis < 3; ++axis)
                    values[3 * row + axis] = value;
            };
            poison(host_means, 1, std::numeric_limits<float>::quiet_NaN());
            host_sh[48 * 2 + 7] = std::numeric_limits<float>::quiet_NaN();
            poison(host_means, 3, 5000.F);
            host_scales[3 * 4 + 1] = 12.F;
            host_logits[5] = -12.F;
            host_rotations[4 * 6 + 2] = std::numeric_limits<float>::quiet_NaN();
            model_means = Tensor::from_vector(host_means, model_means.shape(), Device::Vulkan);
            model_scales = Tensor::from_vector(host_scales, model_scales.shape(), Device::Vulkan);
            model_rotations = Tensor::from_vector(host_rotations, model_rotations.shape(), Device::Vulkan);
            model_logits = Tensor::from_vector(host_logits, model_logits.shape(), Device::Vulkan);
            model_sh = Tensor::from_vector(host_sh, model_sh.shape(), Device::Vulkan);
            run();
            const auto keep_host = keep.to_vector_bool();
            const auto hard_host = hard.to_vector_bool();
            const auto opacity_host = opacities.to_vector();
            std::size_t mismatches = 0;
            for (std::size_t row = 0; row < rows; ++row) {
                const float opacity = 1.F / (1.F + std::exp(-host_logits[row]));
                bool bad = !std::isfinite(host_logits[row]);
                float maximum_scale = 0.F;
                bool outside = false;
                for (std::size_t axis = 0; axis < 3; ++axis) {
                    const float mean = host_means[3 * row + axis];
                    const float scale = host_scales[3 * row + axis];
                    bad = bad || !std::isfinite(mean) || !std::isfinite(scale);
                    if (std::isfinite(scale))
                        maximum_scale = std::max(maximum_scale, std::exp(scale));
                    outside = outside || std::abs(mean) > 1000.F;
                }
                for (std::size_t component = 0; component < 4; ++component)
                    bad = bad || !std::isfinite(host_rotations[4 * row + component]);
                for (std::size_t component = 0; component < 48; ++component)
                    bad = bad || !std::isfinite(host_sh[48 * row + component]);
                const bool hard_row = bad || outside || maximum_scale > 1000.F;
                const bool keep_row = !hard_row && opacity >= 0.01F;
                if (keep_host[row] != keep_row ||
                    hard_host[row] != hard_row ||
                    std::abs(opacity_host[row] - opacity) > 1e-6F)
                    ++mismatches;
            }
            std::printf("prune parity poisoned_mismatches=%llu (rows 1-6 expected hard)\n",
                static_cast<unsigned long long>(mismatches));
        }
        std::printf("prune replay ms=%.4f\n",
            median_ms([&] { run(); return 1.0F; }, 3));
    }

    {
        auto vk_mask = make_bool(Device::Vulkan);
        auto cu_mask = make_bool(Device::CUDA);
        row("nonzero",
            median_ms([&] { return static_cast<float>(vk_mask.nonzero().numel()); }, repeats),
            median_ms([&] { return static_cast<float>(cu_mask.nonzero().numel()); }, repeats));
    }

    {
        auto vk_values = make_values(Device::Vulkan);
        auto cu_values = make_values(Device::CUDA);
        row("sort descending",
            median_ms([&] { return static_cast<float>(vk_values.sort(0, true).second.numel()); },
                      repeats),
            median_ms([&] { return static_cast<float>(cu_values.sort(0, true).second.numel()); },
                      repeats));
    }

    {
        auto vk_values = make_values(Device::Vulkan);
        auto cu_values = make_values(Device::CUDA);
        row("max + readback",
            median_ms([&] { return vk_values.max().to_vector().front(); }, repeats),
            median_ms([&] { return cu_values.max().to_vector().front(); }, repeats));
    }

    {
        auto vk_values = make_values(Device::Vulkan);
        auto cu_values = make_values(Device::CUDA);
        const std::vector<int> index_host(64);
        std::vector<int> indices(64);
        for (std::size_t i = 0; i < indices.size(); ++i)
            indices[i] = static_cast<int>((i * 7919u) % count);
        auto vk_indices = Tensor::from_vector(indices, {indices.size()}, Device::Vulkan);
        auto cu_indices = Tensor::from_vector(indices, {indices.size()}, Device::CUDA);
        row("index_select(64)",
            median_ms([&] { return static_cast<float>(
                vk_values.index_select(0, vk_indices).numel()); }, repeats),
            median_ms([&] { return static_cast<float>(
                cu_values.index_select(0, cu_indices).numel()); }, repeats));
    }

    // The two forms of the Gumbel top-k used by IGS selection: the device
    // form the Vulkan backend now takes, and the host partial_sort the CUDA
    // backend has always taken.
    const std::size_t requested = 4096;
    {
        auto weights = make_values(Device::Vulkan).clamp_min(1e-7F);
        row("gumbel device sort",
            median_ms([&] {
                const auto eligible = weights.isfinite().logical_and(weights.gt(0.F));
                const auto uniform = Tensor::rand(weights.shape(), weights.device())
                    .clamp_min(1e-7F).clamp_max(1.F - 1e-7F);
                auto scores = weights.log().sub(uniform.log().mul(-1.F).log());
                scores.masked_fill_(eligible.logical_not(),
                                    -std::numeric_limits<float>::infinity());
                return static_cast<float>(scores.sort(0, true).second
                    .slice(0, 0, requested).to(DataType::Int32).numel());
            }, repeats), -1.0);
    }
    {
        auto weights = make_values(Device::CUDA).clamp_min(1e-7F);
        row("gumbel host partial_sort",
            median_ms([&] {
                const auto values = weights.to_vector();
                std::vector<std::pair<float, int>> scores;
                scores.reserve(values.size());
                for (std::size_t i = 0; i < values.size(); ++i) {
                    const float u = 1e-7F + (1.F - 2e-7F) *
                        static_cast<float>((i * 48271u) % 100003u) / 100003.0F;
                    scores.emplace_back(
                        std::log(values[i]) - std::log(-std::log(u)), static_cast<int>(i));
                }
                std::partial_sort(scores.begin(), scores.begin() +
                    static_cast<std::ptrdiff_t>(requested), scores.end(), std::greater<>());
                std::vector<int> picked(requested);
                for (std::size_t i = 0; i < requested; ++i) picked[i] = scores[i].second;
                return static_cast<float>(Tensor::from_vector(
                    picked, {requested}, Device::CUDA).numel());
            }, repeats), -1.0);
    }
}

constexpr std::size_t k_ops_count = 1000000;
constexpr std::size_t k_ops_rows = 1000;
constexpr std::size_t k_ops_cols = 1000;
constexpr std::size_t k_ops_prod_rows = 100000;
constexpr std::size_t k_ops_prod_cols = 10;
constexpr std::size_t k_ops_mm = 1024;
constexpr std::size_t k_ops_small = 4096;
constexpr std::size_t k_ops_fill = 64;
constexpr double k_transcendental_atol = 1e-5;
constexpr double k_transcendental_rtol = 2e-5;

enum class OpKind { Float, Bool, Index, Count, SortAsc, SortDesc, SortRows };

struct Diff {
    bool pass = false;
    std::size_t bad = 0;
    double max_abs = 0.0;
    double max_rel = 0.0;
    std::string detail;
};

struct Side {
    Device device = Device::CPU;
    Tensor a, b, u, special, divisor;
    Tensor grid, prod, mm_a, mm_b, row, col, keys;
    Tensor mask, mask2, mask_all, mask_grid;
    Tensor perm, row_perm, idx_small, col_idx, gather_idx;
    Tensor scatter_dst, scatter_add, row_dst, row_add, fill_dst, mask_dst;
};

struct Spec {
    const char* group = "";
    const char* name = "";
    OpKind kind = OpKind::Float;
    double atol = 0.0;
    double rtol = 0.0;
    std::function<Tensor(const Side&)> eval;
    std::function<std::size_t(const Side&)> count;
    std::function<void(Side&)> bench;
};

Tensor prepare(const Tensor& tensor) {
    Tensor prepared = tensor.contiguous();
    sync_device(prepared.device());
    return prepared;
}

std::string device_problem(const Tensor& vulkan, const Tensor& cuda) {
    if (!vulkan.is_valid() || !cuda.is_valid()) return "invalid result";
    if (vulkan.device() != Device::Vulkan) return "Vulkan op left Device::Vulkan";
    if (cuda.device() != Device::CUDA) return "CUDA op left Device::CUDA";
    if (vulkan.dtype() != cuda.dtype()) return "dtype mismatch";
    if (vulkan.shape() != cuda.shape())
        return "shape " + vulkan.shape().str() + " vs " + cuda.shape().str();
    return {};
}

bool near_float(const float got, const float expected, const double atol, const double rtol) {
    if (std::isnan(got) && std::isnan(expected)) return true;
    if (std::isinf(got) && std::isinf(expected) && std::signbit(got) == std::signbit(expected))
        return true;
    if (!std::isfinite(got) || !std::isfinite(expected)) return false;
    const double absolute = std::abs(static_cast<double>(got) - static_cast<double>(expected));
    return absolute <= atol + rtol * std::abs(static_cast<double>(expected));
}

void note_first(Diff& diff, const std::size_t index, const std::string& text) {
    if (!diff.detail.empty()) return;
    diff.detail = "first[" + std::to_string(index) + "] " + text;
}

Diff compare_float(const Tensor& vulkan, const Tensor& cuda, const double atol, const double rtol) {
    Diff diff;
    diff.detail = device_problem(vulkan, cuda);
    if (!diff.detail.empty()) return diff;
    const auto got = prepare(vulkan).to_vector();
    const auto expected = prepare(cuda).to_vector();
    if (got.size() != expected.size() || got.empty()) {
        diff.detail = "downloaded " + std::to_string(got.size()) + " vs " +
                      std::to_string(expected.size());
        diff.bad = std::max(got.size(), expected.size());
        return diff;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double absolute = std::abs(static_cast<double>(got[i]) - static_cast<double>(expected[i]));
        const double relative = absolute / std::max(1.0, std::abs(static_cast<double>(expected[i])));
        if (std::isfinite(absolute)) diff.max_abs = std::max(diff.max_abs, absolute);
        if (std::isfinite(relative)) diff.max_rel = std::max(diff.max_rel, relative);
        if (!near_float(got[i], expected[i], atol, rtol)) {
            ++diff.bad;
            char text[160];
            std::snprintf(text, sizeof(text), "vk=%.9g cu=%.9g", got[i], expected[i]);
            note_first(diff, i, text);
        }
    }
    diff.pass = diff.bad == 0;
    if (diff.pass) diff.detail.clear();
    return diff;
}

Diff compare_bool(const Tensor& vulkan, const Tensor& cuda) {
    Diff diff;
    diff.detail = device_problem(vulkan, cuda);
    if (!diff.detail.empty()) return diff;
    const auto got = prepare(vulkan).to_vector_bool();
    const auto expected = prepare(cuda).to_vector_bool();
    if (got.size() != expected.size() || got.empty()) {
        diff.detail = "downloaded " + std::to_string(got.size()) + " vs " +
                      std::to_string(expected.size());
        diff.bad = std::max(got.size(), expected.size());
        return diff;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        if (static_cast<bool>(got[i]) == static_cast<bool>(expected[i])) continue;
        ++diff.bad;
        diff.max_abs = 1.0;
        note_first(diff, i, std::string("vk=") + (got[i] ? "1" : "0") +
                                 " cu=" + (expected[i] ? "1" : "0"));
    }
    diff.pass = diff.bad == 0;
    if (diff.pass) diff.detail.clear();
    return diff;
}

Diff compare_index(const Tensor& vulkan, const Tensor& cuda) {
    Diff diff;
    diff.detail = device_problem(vulkan, cuda);
    if (!diff.detail.empty()) return diff;
    const auto got = prepare(vulkan).to_vector_int64();
    const auto expected = prepare(cuda).to_vector_int64();
    if (got.size() != expected.size() || got.empty()) {
        diff.detail = "downloaded " + std::to_string(got.size()) + " vs " +
                      std::to_string(expected.size());
        diff.bad = std::max(got.size(), expected.size());
        return diff;
    }
    for (std::size_t i = 0; i < got.size(); ++i) {
        const double absolute = std::abs(static_cast<double>(got[i] - expected[i]));
        diff.max_abs = std::max(diff.max_abs, absolute);
        if (got[i] == expected[i]) continue;
        ++diff.bad;
        note_first(diff, i, "vk=" + std::to_string(got[i]) + " cu=" + std::to_string(expected[i]));
    }
    diff.pass = diff.bad == 0;
    if (diff.pass) diff.detail.clear();
    return diff;
}

bool sort_self_ok(const std::vector<float>& values, const std::vector<std::int64_t>& order,
                  const std::vector<float>& keys, const bool descending, std::string& why) {
    if (values.size() != keys.size() || order.size() != keys.size()) {
        why = "size";
        return false;
    }
    std::vector<unsigned char> seen(keys.size(), 0);
    for (std::size_t i = 0; i < keys.size(); ++i) {
        const std::int64_t index = order[i];
        if (index < 0 || static_cast<std::size_t>(index) >= keys.size() ||
            seen[static_cast<std::size_t>(index)] != 0) {
            why = "index";
            return false;
        }
        seen[static_cast<std::size_t>(index)] = 1;
        if (values[i] != keys[static_cast<std::size_t>(index)]) {
            why = "value";
            return false;
        }
        if (i > 0) {
            const bool ordered = descending ? values[i - 1] >= values[i] : values[i - 1] <= values[i];
            if (!ordered) {
                why = "order";
                return false;
            }
        }
    }
    return true;
}

std::vector<int> permute_indices(const std::size_t count, std::uint32_t rng) {
    std::vector<int> values(count);
    std::iota(values.begin(), values.end(), 0);
    for (std::size_t i = count; i > 1; --i) {
        rng = rng * 1664525u + 1013904223u;
        std::swap(values[i - 1], values[rng % static_cast<std::uint32_t>(i)]);
    }
    return values;
}

void print_op_row(const char* name, const char* status, const Diff& diff,
                  const double vulkan_ms, const double cuda_ms) {
    char ratio[16] = "n/a";
    if (vulkan_ms >= 0.0 && cuda_ms > 1e-9)
        std::snprintf(ratio, sizeof(ratio), "%.2fx", vulkan_ms / cuda_ms);
    else if (vulkan_ms < 0.0 || cuda_ms < 0.0)
        std::snprintf(ratio, sizeof(ratio), "err");
    std::printf("%-34s %-4s %8llu %12.4g %12.4g %10.4f %10.4f %8s\n", name, status,
                static_cast<unsigned long long>(diff.bad), diff.max_abs, diff.max_rel,
                std::max(0.0, vulkan_ms), std::max(0.0, cuda_ms), ratio);
    if (std::string(status) != "OK" && !diff.detail.empty())
        std::printf("    %s\n", diff.detail.c_str());
    std::fflush(stdout);
}

Diff check_sort(const Side& vulkan, const Side& cuda, const std::vector<float>& keys,
                const OpKind kind) {
    const bool rows = kind == OpKind::SortRows;
    const bool descending = kind == OpKind::SortDesc;
    const auto vk_sorted = rows ? vulkan.grid.sort(1, false) : vulkan.keys.sort(0, descending);
    const auto cu_sorted = rows ? cuda.grid.sort(1, false) : cuda.keys.sort(0, descending);
    Diff values = compare_float(vk_sorted.first, cu_sorted.first, 0.0, 0.0);
    Diff indices = compare_index(vk_sorted.second, cu_sorted.second);
    Diff merged;
    merged.max_abs = values.max_abs;
    merged.max_rel = values.max_rel;
    merged.bad = values.bad + indices.bad;
    merged.pass = values.pass && indices.pass;
    if (!values.pass) merged.detail = "values " + values.detail;
    if (!indices.pass) {
        if (!merged.detail.empty()) merged.detail += "; ";
        merged.detail += "indices " + indices.detail;
    }
    if (!rows && values.detail.find("left") == std::string::npos &&
        values.detail.find("invalid") == std::string::npos) {
        std::string why;
        const bool self_ok = sort_self_ok(prepare(vk_sorted.first).to_vector(),
                                           prepare(vk_sorted.second).to_vector_int64(), keys,
                                           descending, why);
        if (!self_ok) {
            merged.pass = false;
            ++merged.bad;
            if (!merged.detail.empty()) merged.detail += "; ";
            merged.detail += "vulkan self-check " + why;
        }
    }
    return merged;
}

int ops_sweep(const int repeats) {
    const auto started = std::chrono::steady_clock::now();
    const auto& info = tinytensor::vulkan::device_info();
    cudaDeviceProp prop{};
    const char* cuda_name = "unknown";
    if (cudaGetDeviceProperties(&prop, 0) == cudaSuccess) cuda_name = prop.name;
    std::printf("=== tinytensor ops: %llu float32 elements, %d repeats ===\n",
                static_cast<unsigned long long>(k_ops_count), repeats);
    std::printf("vulkan=%s cuda=%s\n", info.name.c_str(), cuda_name);
    std::printf("time is the median wall ms of the call plus device sync, without copying\n");
    std::printf("the full result back. CUDA lazy unary ops are materialized before the sync.\n");
    std::printf("correctness compares Vulkan with CUDA. max_abs/max_rel are the observed gap.\n");
    std::printf("sort dim1 is the host fallback; sort asc/desc are the 1d device kernels.\n");
    std::fflush(stdout);

    std::vector<float> host_a(k_ops_count), host_b(k_ops_count), host_u(k_ops_count);
    for (std::size_t i = 0; i < k_ops_count; ++i) {
        host_a[i] = -1.5F + static_cast<float>(i % 1000) * 0.003F;
        host_b[i] = 0.35F + static_cast<float>(i % 251) * 0.01F;
        host_u[i] = -0.99F + static_cast<float>(i % 199) * 0.01F;
    }
    std::vector<float> host_divisor(k_ops_count);
    for (std::size_t i = 0; i < k_ops_count; ++i)
        host_divisor[i] = std::abs(host_u[i]) + 0.3F;
    std::vector<float> host_special = host_b;
    host_special[0] = std::numeric_limits<float>::quiet_NaN();
    host_special[1] = std::numeric_limits<float>::infinity();
    host_special[2] = -std::numeric_limits<float>::infinity();
    host_special[10] = std::numeric_limits<float>::quiet_NaN();

    std::vector<float> host_grid(k_ops_rows * k_ops_cols);
    std::vector<bool> host_mask_grid(k_ops_rows * k_ops_cols);
    for (std::size_t row = 0; row < k_ops_rows; ++row) {
        for (std::size_t col = 0; col < k_ops_cols; ++col) {
            const std::size_t index = row * k_ops_cols + col;
            host_grid[index] = static_cast<float>((col + row * 17) % 1000) +
                               static_cast<float>(row) * 1000.F;
            host_mask_grid[index] = ((row + col) % 3) != 0;
        }
    }
    std::vector<float> host_prod(k_ops_prod_rows * k_ops_prod_cols);
    for (std::size_t i = 0; i < host_prod.size(); ++i)
        host_prod[i] = 0.92F + static_cast<float>(i % 5) * 0.03F;
    std::vector<float> host_mm_a(k_ops_mm * k_ops_mm), host_mm_b(k_ops_mm * k_ops_mm);
    for (std::size_t i = 0; i < host_mm_a.size(); ++i) {
        host_mm_a[i] = (static_cast<float>(i % 32) - 16.F) * 0.01F;
        host_mm_b[i] = (static_cast<float>((i * 3) % 32) - 16.F) * 0.01F;
    }
    std::vector<float> host_row(k_ops_cols), host_col(k_ops_rows);
    for (std::size_t i = 0; i < k_ops_cols; ++i)
        host_row[i] = 0.02F * static_cast<float>(i % 11);
    for (std::size_t i = 0; i < k_ops_rows; ++i)
        host_col[i] = 0.1F * static_cast<float>(i % 7);

    const std::vector<int> host_perm = permute_indices(k_ops_count, 0x12345678u);
    const std::vector<int> host_row_perm = permute_indices(k_ops_rows, 0x89abcdefu);
    const std::vector<int> host_small(host_perm.begin(), host_perm.begin() +
                                                             static_cast<std::ptrdiff_t>(k_ops_small));
    const std::vector<int> host_fill(host_row_perm.begin(),
                                     host_row_perm.begin() + static_cast<std::ptrdiff_t>(k_ops_fill));
    std::vector<int> host_gather(k_ops_rows * k_ops_cols);
    for (std::size_t row = 0; row < k_ops_rows; ++row) {
        for (std::size_t col = 0; col < k_ops_cols; ++col) {
            host_gather[row * k_ops_cols + col] =
                static_cast<int>((col * 17 + row) % k_ops_cols);
        }
    }
    std::vector<float> host_keys(k_ops_count);
    for (std::size_t i = 0; i < k_ops_count; ++i)
        host_keys[i] = static_cast<float>(host_perm[i]);
    std::vector<bool> host_mask(k_ops_count), host_mask2(k_ops_count), host_mask_all(k_ops_count, true);
    for (std::size_t i = 0; i < k_ops_count; ++i) {
        host_mask[i] = (i % 3) == 0;
        host_mask2[i] = (i % 5) == 0;
    }

    std::printf("uploading tensors...\n");
    std::fflush(stdout);
    const auto upload = [&](const Device device) {
        Side side;
        side.device = device;
        side.a = Tensor::from_vector(host_a, {k_ops_count}, device);
        side.b = Tensor::from_vector(host_b, {k_ops_count}, device);
        side.u = Tensor::from_vector(host_u, {k_ops_count}, device);
        side.special = Tensor::from_vector(host_special, {k_ops_count}, device);
        side.divisor = Tensor::from_vector(host_divisor, {k_ops_count}, device);
        side.grid = Tensor::from_vector(host_grid, {k_ops_rows, k_ops_cols}, device);
        side.prod = Tensor::from_vector(host_prod, {k_ops_prod_rows, k_ops_prod_cols}, device);
        side.mm_a = Tensor::from_vector(host_mm_a, {k_ops_mm, k_ops_mm}, device);
        side.mm_b = Tensor::from_vector(host_mm_b, {k_ops_mm, k_ops_mm}, device);
        side.row = Tensor::from_vector(host_row, {k_ops_cols}, device);
        side.col = Tensor::from_vector(host_col, {k_ops_rows, std::size_t{1}}, device);
        side.keys = Tensor::from_vector(host_keys, {k_ops_count}, device);
        side.mask = Tensor::from_vector(host_mask, {k_ops_count}, device);
        side.mask2 = Tensor::from_vector(host_mask2, {k_ops_count}, device);
        side.mask_all = Tensor::from_vector(host_mask_all, {k_ops_count}, device);
        side.mask_grid = Tensor::from_vector(host_mask_grid, {k_ops_rows, k_ops_cols}, device);
        side.perm = Tensor::from_vector(host_perm, {k_ops_count}, device);
        side.row_perm = Tensor::from_vector(host_row_perm, {k_ops_rows}, device);
        side.idx_small = Tensor::from_vector(host_small, {k_ops_small}, device);
        side.col_idx = Tensor::from_vector(host_fill, {k_ops_fill}, device);
        side.gather_idx = Tensor::from_vector(host_gather, {k_ops_rows, k_ops_cols}, device);
        side.scatter_dst = Tensor::zeros({k_ops_count}, device);
        side.scatter_add = Tensor::ones({k_ops_count}, device);
        side.row_dst = Tensor::zeros({k_ops_rows, k_ops_cols}, device);
        side.row_add = Tensor::ones({k_ops_rows, k_ops_cols}, device);
        side.fill_dst = Tensor::zeros({k_ops_rows, k_ops_cols}, device);
        side.mask_dst = side.a.clone();
        return side;
    };
    Side vk = upload(Device::Vulkan);
    Side cu = upload(Device::CUDA);
    std::printf("upload done\n");
    std::fflush(stdout);

    std::vector<Spec> specs;
    const auto add = [&](const char* group, const char* name, const OpKind kind, const double atol,
                         const double rtol, std::function<Tensor(const Side&)> eval,
                         std::function<void(Side&)> bench = {}) {
        Spec spec;
        spec.group = group;
        spec.name = name;
        spec.kind = kind;
        spec.atol = atol;
        spec.rtol = rtol;
        spec.eval = std::move(eval);
        spec.bench = std::move(bench);
        specs.push_back(std::move(spec));
    };
    const auto add_count = [&](const char* name, std::function<std::size_t(const Side&)> count) {
        Spec spec;
        spec.group = "reduce";
        spec.name = name;
        spec.kind = OpKind::Count;
        spec.count = std::move(count);
        specs.push_back(std::move(spec));
    };
    const auto add_sort = [&](const char* name, const OpKind kind) {
        Spec spec;
        spec.group = "sort";
        spec.name = name;
        spec.kind = kind;
        specs.push_back(std::move(spec));
    };

    const double trans_atol = k_transcendental_atol;
    const double trans_rtol = k_transcendental_rtol;
    add("unary", "neg", OpKind::Float, 0, 0, [](const Side& s) { return s.a.neg(); });
    add("unary", "abs", OpKind::Float, 0, 0, [](const Side& s) { return s.a.abs(); });
    add("unary", "sign", OpKind::Float, 0, 0, [](const Side& s) { return s.a.sign(); });
    add("unary", "square", OpKind::Float, 0, 0, [](const Side& s) { return s.a.square(); });
    add("unary", "floor", OpKind::Float, 0, 0, [](const Side& s) { return s.a.floor(); });
    add("unary", "ceil", OpKind::Float, 0, 0, [](const Side& s) { return s.a.ceil(); });
    add("unary", "round", OpKind::Float, 0, 0, [](const Side& s) { return s.a.round(); });
    add("unary", "trunc", OpKind::Float, 0, 0, [](const Side& s) { return s.a.trunc(); });
    add("unary", "relu", OpKind::Float, 0, 0, [](const Side& s) { return s.a.relu(); });
    add("unary", "exp", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.exp(); });
    add("unary", "exp2", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.exp2(); });
    add("unary", "sin", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.sin(); });
    add("unary", "cos", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.cos(); });
    add("unary", "tan", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.tan(); });
    add("unary", "atan", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.atan(); });
    add("unary", "sinh", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.sinh(); });
    add("unary", "cosh", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.cosh(); });
    add("unary", "tanh", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.tanh(); });
    add("unary", "sigmoid", OpKind::Float, trans_atol, trans_rtol,
        [](const Side& s) { return s.a.sigmoid(); });
    add("unary", "gelu", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.gelu(); });
    add("unary", "swish", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.a.swish(); });
    add("unary", "reciprocal", OpKind::Float, 1e-6, 1e-6, [](const Side& s) { return s.b.reciprocal(); });
    add("unary", "sqrt", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.b.sqrt(); });
    add("unary", "rsqrt", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.b.rsqrt(); });
    add("unary", "log", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.b.log(); });
    add("unary", "log2", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.b.log2(); });
    add("unary", "log10", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.b.log10(); });
    add("unary", "asin", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.u.asin(); });
    add("unary", "acos", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.u.acos(); });
    add("unary", "log1p", OpKind::Float, trans_atol, trans_rtol, [](const Side& s) { return s.u.log1p(); });
    add("unary", "isnan", OpKind::Bool, 0, 0, [](const Side& s) { return s.special.isnan(); });
    add("unary", "isinf", OpKind::Bool, 0, 0, [](const Side& s) { return s.special.isinf(); });
    add("unary", "isfinite", OpKind::Bool, 0, 0, [](const Side& s) { return s.special.isfinite(); });

    add("binary", "add", OpKind::Float, 0, 0, [](const Side& s) { return s.a.add(s.b); });
    add("binary", "sub", OpKind::Float, 0, 0, [](const Side& s) { return s.a.sub(s.b); });
    add("binary", "mul", OpKind::Float, 0, 0, [](const Side& s) { return s.a.mul(s.b); });
    add("binary", "div", OpKind::Float, 1e-6, 1e-6, [](const Side& s) { return s.a.div(s.b); });
    add("binary", "maximum", OpKind::Float, 0, 0, [](const Side& s) { return s.a.maximum(s.b); });
    add("binary", "minimum", OpKind::Float, 0, 0, [](const Side& s) { return s.a.minimum(s.b); });
    add("binary", "pow", OpKind::Float, 1e-5, 1e-4, [](const Side& s) { return s.b.pow(s.u); });
    add("binary", "mod", OpKind::Float, 1e-5, 1e-5, [](const Side& s) { return s.b.mod(s.divisor); });
    add("binary", "add scalar", OpKind::Float, 0, 0, [](const Side& s) { return s.a.add(0.5F); });
    add("binary", "sub scalar", OpKind::Float, 0, 0, [](const Side& s) { return s.a.sub(0.25F); });
    add("binary", "mul scalar", OpKind::Float, 0, 0, [](const Side& s) { return s.a.mul(1.5F); });
    add("binary", "div scalar", OpKind::Float, 1e-6, 1e-6, [](const Side& s) { return s.b.div(1.25F); });
    add("binary", "pow scalar", OpKind::Float, 1e-5, 1e-4, [](const Side& s) { return s.b.pow(1.7F); });
    add("binary", "clamp", OpKind::Float, 0, 0, [](const Side& s) { return s.a.clamp(-0.4F, 0.6F); });
    add("binary", "broadcast row", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.add(s.row); });
    add("binary", "broadcast col", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.add(s.col); });

    add("compare", "eq", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.eq(s.b); });
    add("compare", "ne", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.ne(s.b); });
    add("compare", "lt", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.lt(s.b); });
    add("compare", "le", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.le(s.b); });
    add("compare", "gt", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.gt(s.b); });
    add("compare", "ge", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.ge(s.b); });
    add("compare", "gt scalar", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.gt(0.F); });
    add("compare", "le scalar", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.le(0.5F); });
    add("compare", "eq self", OpKind::Bool, 0, 0, [](const Side& s) { return s.a.eq(s.a); });
    add("compare", "logical_not", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask.logical_not(); });
    add("compare", "logical_and", OpKind::Bool, 0, 0,
        [](const Side& s) { return s.mask.logical_and(s.mask2); });
    add("compare", "logical_or", OpKind::Bool, 0, 0,
        [](const Side& s) { return s.mask.logical_or(s.mask2); });
    add("compare", "logical_xor", OpKind::Bool, 0, 0,
        [](const Side& s) { return s.mask.logical_xor(s.mask2); });

    add("reduce", "sum", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.a.sum(); });
    add("reduce", "mean", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.a.mean(); });
    add("reduce", "max", OpKind::Float, 0, 0, [](const Side& s) { return s.a.max(); });
    add("reduce", "min", OpKind::Float, 0, 0, [](const Side& s) { return s.a.min(); });
    add("reduce", "argmax", OpKind::Index, 0, 0, [](const Side& s) { return s.a.argmax(); });
    add("reduce", "argmin", OpKind::Index, 0, 0, [](const Side& s) { return s.a.argmin(); });
    add("reduce", "grid sum", OpKind::Float, 1.0, 1e-4, [](const Side& s) { return s.grid.sum(); });
    add("reduce", "grid mean", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.grid.mean(); });
    add("reduce", "grid max", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.max(); });
    add("reduce", "grid min", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.min(); });
    add("reduce", "sum dim0", OpKind::Float, 1.0, 1e-4, [](const Side& s) { return s.grid.sum(0); });
    add("reduce", "sum dim1", OpKind::Float, 1.0, 1e-4, [](const Side& s) { return s.grid.sum(1); });
    add("reduce", "mean dim0", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.grid.mean(0); });
    add("reduce", "mean dim1", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.grid.mean(1); });
    add("reduce", "max dim0", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.max(0); });
    add("reduce", "max dim1", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.max(1); });
    add("reduce", "min dim0", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.min(0); });
    add("reduce", "min dim1", OpKind::Float, 0, 0, [](const Side& s) { return s.grid.min(1); });
    add("reduce", "argmax dim0", OpKind::Index, 0, 0,
        [](const Side& s) { return s.grid.argmax(std::array<int, 1>{0}); });
    add("reduce", "argmax dim1", OpKind::Index, 0, 0,
        [](const Side& s) { return s.grid.argmax(std::array<int, 1>{1}); });
    add("reduce", "argmin dim0", OpKind::Index, 0, 0,
        [](const Side& s) { return s.grid.argmin(std::array<int, 1>{0}); });
    add("reduce", "argmin dim1", OpKind::Index, 0, 0,
        [](const Side& s) { return s.grid.argmin(std::array<int, 1>{1}); });
    add("reduce", "var dim1", OpKind::Float, 1e-2, 1e-4, [](const Side& s) { return s.grid.var(1); });
    add("reduce", "std dim1", OpKind::Float, 1e-3, 1e-4, [](const Side& s) { return s.grid.std(1); });
    add("reduce", "prod dim1", OpKind::Float, 1e-5, 1e-5, [](const Side& s) { return s.prod.prod(1); });
    add("reduce", "cumsum", OpKind::Float, 1e-2, 1e-3, [](const Side& s) { return s.a.cumsum(0); });
    add("reduce", "cumsum dim1", OpKind::Float, 1.0, 1e-4, [](const Side& s) { return s.grid.cumsum(1); });
    add("reduce", "any", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask.any(); });
    add("reduce", "all", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask.all(); });
    add("reduce", "any all-true", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask_all.any(); });
    add("reduce", "all all-true", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask_all.all(); });
    add("reduce", "any dim1", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask_grid.any(1); });
    add("reduce", "all dim1", OpKind::Bool, 0, 0, [](const Side& s) { return s.mask_grid.all(1); });
    add_count("count_nonzero", [](const Side& s) { return s.mask.count_nonzero(); });
    add_count("count_nonzero all", [](const Side& s) { return s.mask_all.count_nonzero(); });
    add_count("count_nonzero grid", [](const Side& s) { return s.mask_grid.count_nonzero(); });

    add_sort("sort asc", OpKind::SortAsc);
    add_sort("sort desc", OpKind::SortDesc);
    add_sort("sort dim1 host", OpKind::SortRows);

    add("index", "gather 1d", OpKind::Float, 0, 0,
        [](const Side& s) { return s.b.gather(0, s.perm); });
    add("index", "gather rows", OpKind::Float, 0, 0,
        [](const Side& s) { return s.grid.gather(0, s.row_perm); });
    add("index", "gather dim1", OpKind::Float, 0, 0,
        [](const Side& s) { return s.grid.gather(1, s.gather_idx); });
    add("index", "index_copy 1d", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::zeros({k_ops_count}, s.device);
            out.index_copy_(0, s.perm, s.b);
            return out;
        },
        [](Side& s) { s.scatter_dst.index_copy_(0, s.perm, s.b); });
    add("index", "index_add 1d", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::ones({k_ops_count}, s.device);
            out.index_add_(0, s.perm, s.b);
            return out;
        },
        [](Side& s) { s.scatter_add.index_add_(0, s.perm, s.b); });
    add("index", "index_copy rows", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::zeros({k_ops_rows, k_ops_cols}, s.device);
            out.index_copy_(0, s.row_perm, s.grid);
            return out;
        },
        [](Side& s) { s.row_dst.index_copy_(0, s.row_perm, s.grid); });
    add("index", "index_add rows", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::ones({k_ops_rows, k_ops_cols}, s.device);
            out.index_add_(0, s.row_perm, s.grid);
            return out;
        },
        [](Side& s) { s.row_add.index_add_(0, s.row_perm, s.grid); });
    add("index", "index_select 4096", OpKind::Float, 0, 0,
        [](const Side& s) { return s.b.index_select(0, s.idx_small); });
    add("index", "index_select 1d", OpKind::Float, 0, 0,
        [](const Side& s) { return s.b.index_select(0, s.perm); });
    add("index", "index_select rows", OpKind::Float, 0, 0,
        [](const Side& s) { return s.grid.index_select(0, s.row_perm); });
    add("index", "scatter replace 1d", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::zeros({k_ops_count}, s.device);
            out.scatter_(0, s.perm, s.b, ScatterMode::None);
            return out;
        },
        [](Side& s) { s.scatter_dst.scatter_(0, s.perm, s.b, ScatterMode::None); });
    add("index", "scatter add 1d", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::ones({k_ops_count}, s.device);
            out.scatter_(0, s.perm, s.b, ScatterMode::Add);
            return out;
        },
        [](Side& s) { s.scatter_add.scatter_(0, s.perm, s.b, ScatterMode::Add); });
    add("index", "scatter replace rows", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::zeros({k_ops_rows, k_ops_cols}, s.device);
            out.scatter_(0, s.row_perm, s.grid, ScatterMode::None);
            return out;
        },
        [](Side& s) { s.row_dst.scatter_(0, s.row_perm, s.grid, ScatterMode::None); });
    add("index", "scatter add rows", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::ones({k_ops_rows, k_ops_cols}, s.device);
            out.scatter_(0, s.row_perm, s.grid, ScatterMode::Add);
            return out;
        },
        [](Side& s) { s.row_add.scatter_(0, s.row_perm, s.grid, ScatterMode::Add); });
    add("index", "index_fill cols", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = Tensor::zeros({k_ops_rows, k_ops_cols}, s.device);
            out.index_fill_(1, s.col_idx, 3.5F);
            return out;
        },
        [](Side& s) { s.fill_dst.index_fill_(1, s.col_idx, 3.5F); });
    add("index", "masked_select", OpKind::Float, 0, 0,
        [](const Side& s) { return s.b.masked_select(s.mask); });
    add("index", "masked_fill", OpKind::Float, 0, 0,
        [](const Side& s) {
            auto out = s.a.clone();
            out.masked_fill_(s.mask, -3.F);
            return out;
        },
        [](Side& s) { s.mask_dst.masked_fill_(s.mask, -3.F); });
    add("index", "where", OpKind::Float, 0, 0,
        [](const Side& s) { return Tensor::where(s.mask, s.a, s.b); });
    add("index", "nonzero", OpKind::Index, 0, 0,
        [](const Side& s) { return s.mask.nonzero().squeeze(1); });
    add("index", "cat halves", OpKind::Float, 0, 0, [](const Side& s) {
        return Tensor::cat({s.a.slice(0, 0, k_ops_count / 2), s.a.slice(0, k_ops_count / 2, k_ops_count)},
                           0);
    });
    add("index", "mm 1024", OpKind::Float, 1e-4, 1e-3, [](const Side& s) { return s.mm_a.mm(s.mm_b); });

    std::printf("\n%-34s %-4s %8s %12s %12s %10s %10s %8s\n", "operation", "status", "bad",
                "max_abs", "max_rel", "vulkan_ms", "cuda_ms", "vk/cuda");
    const char* current_group = "";
    int failed = 0;
    for (const Spec& spec : specs) {
        if (std::string(spec.group) != current_group) {
            current_group = spec.group;
            std::printf("-- %s --\n", current_group);
        }
        Diff diff;
        const char* status = "OK";
        try {
            if (spec.kind == OpKind::Count) {
                const std::size_t vulkan_count = spec.count(vk);
                const std::size_t cuda_count = spec.count(cu);
                diff.bad = vulkan_count == cuda_count ? 0 : 1;
                diff.max_abs = std::abs(static_cast<double>(vulkan_count) -
                                        static_cast<double>(cuda_count));
                diff.pass = diff.bad == 0;
                if (!diff.pass)
                    diff.detail = "vk=" + std::to_string(vulkan_count) +
                                  " cu=" + std::to_string(cuda_count);
            } else if (spec.kind == OpKind::SortAsc || spec.kind == OpKind::SortDesc ||
                       spec.kind == OpKind::SortRows) {
                diff = check_sort(vk, cu, host_keys, spec.kind);
            } else {
                const Tensor vulkan = spec.eval(vk);
                const Tensor cuda = spec.eval(cu);
                if (spec.kind == OpKind::Bool) diff = compare_bool(vulkan, cuda);
                else if (spec.kind == OpKind::Index) diff = compare_index(vulkan, cuda);
                else diff = compare_float(vulkan, cuda, spec.atol, spec.rtol);
            }
            if (!diff.pass) status = "FAIL";
        } catch (const std::exception& error) {
            status = "ERR";
            diff.detail = error.what();
        }

        const auto time_one = [&](Side& side) {
            if (spec.kind == OpKind::Count) {
                (void)spec.count(side);
                sync_device(side.device);
                return;
            }
            if (spec.kind == OpKind::SortAsc || spec.kind == OpKind::SortDesc) {
                auto sorted = side.keys.sort(0, spec.kind == OpKind::SortDesc);
                sync_device(side.device);
                if (!sorted.first.is_valid() || !sorted.second.is_valid())
                    throw std::runtime_error("sort returned an invalid tensor");
                return;
            }
            if (spec.kind == OpKind::SortRows) {
                auto sorted = side.grid.sort(1, false);
                sync_device(side.device);
                if (!sorted.first.is_valid() || !sorted.second.is_valid())
                    throw std::runtime_error("sort returned an invalid tensor");
                return;
            }
            if (spec.bench) {
                spec.bench(side);
                sync_device(side.device);
                return;
            }
            Tensor out = spec.eval(side);
            if (!out.is_valid()) throw std::runtime_error("op returned an invalid tensor");
            if (out.has_lazy_expr()) out = out.contiguous();
            sync_device(side.device);
        };
        double vulkan_ms = -1.0;
        double cuda_ms = -1.0;
        try {
            vulkan_ms = time_call([&] { time_one(vk); }, repeats);
        } catch (const std::exception& error) {
            status = "ERR";
            if (!diff.detail.empty()) diff.detail += "; ";
            diff.detail += std::string("vulkan timing: ") + error.what();
        }
        try {
            cuda_ms = time_call([&] { time_one(cu); }, repeats);
        } catch (const std::exception& error) {
            status = "ERR";
            if (!diff.detail.empty()) diff.detail += "; ";
            diff.detail += std::string("cuda timing: ") + error.what();
        }
        if (std::string(status) != "OK") ++failed;
        print_op_row(spec.name, status, diff, vulkan_ms, cuda_ms);
    }

    const double seconds = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - started).count();
    std::printf("\n%d ops, %d failed, %.1f s\n", static_cast<int>(specs.size()), failed, seconds);
    return failed;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!tinytensor::vulkan::available())
            throw std::runtime_error("the tinytensor Vulkan backend is not available");
        if (argc > 1 && std::string(argv[1]) == "--ops") {
            const int ops_repeats = argc > 2 ? std::stoi(argv[2]) : 5;
            if (ops_repeats < 1) throw std::runtime_error("repeats must be positive");
            return ops_sweep(ops_repeats) == 0 ? 0 : 1;
        }
        const bool million_only = argc > 1 && std::string(argv[1]) == "--million";
        const bool densify_only = argc > 1 && std::string(argv[1]) == "--densify";
        const std::size_t densify_count = densify_only && argc > 2
            ? static_cast<std::size_t>(std::stoull(argv[2])) : std::size_t{271862};
        const int repeats = million_only || densify_only ? 15 :
            (argc > 1 ? std::stoi(argv[1]) : 15);
        if (repeats < 1) throw std::runtime_error("repeats must be positive");
        const auto& info = tinytensor::vulkan::device_info();
        std::printf("device=%s api=%u subgroup=%u push_descriptors=%s\n", info.name.c_str(),
                    info.api_version, info.subgroup_size, info.push_descriptors ? "yes" : "no");
        if (million_only) {
            million_benchmark(repeats);
            return 0;
        }
        if (densify_only) {
            densify_benchmark(densify_count, 5);
            return 0;
        }

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
            const double one_op = chain_sync_ms(count, 1, repeats, Device::Vulkan);
            const double many_ops = chain_sync_ms(count, k_sweep_ops, repeats, Device::Vulkan);
            const double readback = readback_ms(count, repeats);
            const double cuda = chain_sync_ms(count, k_sweep_ops, repeats, Device::CUDA);
            const double bytes = static_cast<double>(k_sweep_ops) * static_cast<double>(count) * 12.0;
            std::printf("%12llu %11.4f %11.4f %11.4f %11.2f %12.4f %11.4f\n",
                        static_cast<unsigned long long>(count), one_op, many_ops,
                        many_ops / k_sweep_ops * 1000.0, bytes / (many_ops * 1.0e6), readback, cuda);
        }

        // 3. Training-critical primitives. Full image reductions used to run
        // in one Vulkan thread; the hierarchical path should now be close to
        // CUDA. The Adam rows use an explicit synchronization without host
        // readback and show the generic expression next to the fused Vulkan
        // primitive intended for the training backend.
        std::printf("\n=== training primitives (matched synchronization) ===\n");
        std::printf("%12s %12s %12s %9s %12s %12s %9s\n", "elements", "vk_sum_ms",
                    "cuda_sum_ms", "sum_ratio", "vk_mean_ms", "cuda_mean_ms", "mean_ratio");
        for (const std::size_t count : {std::size_t{197311}, std::size_t{1} << 20,
                                        std::size_t{1} << 22}) {
            const double vk_sum = full_reduce_ms(count, repeats, Device::Vulkan, false);
            const double cu_sum = full_reduce_ms(count, repeats, Device::CUDA, false);
            const double vk_mean = full_reduce_ms(count, repeats, Device::Vulkan, true);
            const double cu_mean = full_reduce_ms(count, repeats, Device::CUDA, true);
            std::printf("%12llu %12.4f %12.4f %8.2fx %12.4f %12.4f %8.2fx\n",
                        static_cast<unsigned long long>(count), vk_sum, cu_sum, vk_sum / cu_sum,
                        vk_mean, cu_mean, vk_mean / cu_mean);
        }
        std::printf("\n%12s %12s %12s %12s %10s %10s\n", "adam_elems", "vk_expr_ms",
                    "vk_fused_ms", "cuda_expr_ms", "expr_ratio", "fused_ratio");
        for (const std::size_t count : {std::size_t{197311}, std::size_t{197311} * 16}) {
            const double vulkan_expr = adam_expression_ms(count, repeats, Device::Vulkan);
            const double vulkan_fused = adam_fused_vulkan_ms(count, repeats);
            const double cuda_expr = adam_expression_ms(count, repeats, Device::CUDA);
            std::printf("%12llu %12.4f %12.4f %12.4f %9.2fx %9.2fx\n",
                        static_cast<unsigned long long>(count), vulkan_expr, vulkan_fused,
                        cuda_expr, vulkan_expr / cuda_expr, vulkan_fused / cuda_expr);
        }

        // 4. Correctness under the new batching and pooling: chains of varying
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

        // 5. Focused repro of the failing pattern: a readback in the middle of
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
