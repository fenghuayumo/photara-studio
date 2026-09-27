#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"
#include "vulkan/runtime/runtime.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <limits>
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
// Usage: photara_tinytensor_vulkan_bench [REPEATS | --million [REPEATS]]

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

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!tinytensor::vulkan::available())
            throw std::runtime_error("the tinytensor Vulkan backend is not available");
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
