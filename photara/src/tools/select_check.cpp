// Manual exactness check for the tinytensor Vulkan order-statistic selection
// (vulkan::select_topk_indices / vulkan::select_nth_value).
//
// The IGS refinement replaced a bitonic sort of the retained rows with two
// radix histogram passes and one emit pass, so the results have to match the
// sort exactly: same set of indices for a top-k, same value for an order
// statistic. Neither is checkable from the training log, which only reports
// counts, so this compares against a host sort on the same inputs.
//
// Usage: photara_select_check [ROWS]

#include "internal/tensor_impl.hpp"
#include "vulkan/backend.hpp"
#include "vulkan/ops.hpp"

#include <algorithm>
#include <cctype>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using tinytensor::DataType;
using tinytensor::Device;
using tinytensor::Tensor;

int failures = 0;

void check(const bool condition, const std::string& what) {
    if (!condition) {
        std::printf("  FAIL %s\n", what.c_str());
        ++failures;
    }
}

std::size_t rows_from_args(int argc, char** argv) {
    if (argc > 1) return static_cast<std::size_t>(std::stoull(argv[1]));
    return 300000;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (!tinytensor::vulkan::available()) {
            std::printf("vulkan backend unavailable\n");
            return 1;
        }
        const std::size_t rows = rows_from_args(argc, argv);
        std::mt19937 random(1234);
        // Distinct values with no ties at all, so an exact selection has
        // exactly one correct answer and a set comparison is meaningful.
        std::vector<float> values(rows);
        for (std::size_t i = 0; i < rows; ++i)
            values[i] = static_cast<float>(random() % 1000000U) + 0.5F;
        std::vector<bool> mask(rows);
        for (std::size_t i = 0; i < rows; ++i) mask[i] = (random() % 4U) != 0U;

        std::vector<std::size_t> masked;
        for (std::size_t i = 0; i < rows; ++i)
            if (mask[i]) masked.push_back(i);
        std::printf("rows=%llu masked=%llu\n",
                    static_cast<unsigned long long>(rows),
                    static_cast<unsigned long long>(masked.size()));

        auto device_values = Tensor::from_vector(values, {rows}, Device::Vulkan);
        auto device_mask = Tensor::from_vector(mask, {rows}, Device::Vulkan);
        {
            const std::size_t device_masked =
                static_cast<std::size_t>(device_mask.count_nonzero());
            check(device_masked == masked.size(),
                  "device mask transfer got=" + std::to_string(device_masked) +
                      " want=" + std::to_string(masked.size()));
        }

        const auto topk_reference = [&](const std::size_t k, const bool descending) {
            std::vector<std::size_t> order = masked;
            const std::size_t keep = std::min(k, masked.size());
            std::partial_sort(
                order.begin(), order.begin() + static_cast<std::ptrdiff_t>(keep),
                order.end(), [&](std::size_t a, std::size_t b) {
                    return descending ? values[a] > values[b] : values[a] < values[b];
                });
            order.resize(keep);
            std::sort(order.begin(), order.end());
            return order;
        };

        for (const std::size_t k : {std::size_t{1}, std::size_t{7}, std::size_t{1000},
                                    std::size_t{50000}, std::size_t{200000},
                                    masked.size(), masked.size() + 5000}) {
            for (const bool descending : {true, false}) {
                const auto expected = topk_reference(k, descending);
                std::uint32_t emitted = 0;
                auto picked = tinytensor::vulkan::select_topk_indices(
                    device_values, device_mask, static_cast<std::uint32_t>(k), descending,
                    &emitted);
                std::vector<int> got;
                if (picked.is_valid()) got = picked.to_vector_int();
                std::vector<std::size_t> sorted(got.begin(), got.end());
                std::sort(sorted.begin(), sorted.end());
                std::string label = "topk k=" + std::to_string(k) +
                    (descending ? " desc" : " asc");
                check(sorted.size() == expected.size(),
                      label + " size got=" + std::to_string(sorted.size()) +
                          " want=" + std::to_string(expected.size()));
                if (sorted.size() == expected.size()) {
                    if (!std::equal(sorted.begin(), sorted.end(), expected.begin())) {
                        std::string detail;
                        for (std::size_t i = 0; i < sorted.size(); ++i) {
                            if (sorted[i] == expected[i]) continue;
                            detail += " first_diff_at=" + std::to_string(i) +
                                " got=" + std::to_string(sorted[i]) +
                                "(v=" + std::to_string(values[sorted[i]]) + ")" +
                                " want=" + std::to_string(expected[i]) +
                                "(v=" + std::to_string(values[expected[i]]) + ")";
                            break;
                        }
                        check(false, label + " set mismatch" + detail);
                    }
                }
                if (emitted != sorted.size())
                    check(false, label + " emitted=" + std::to_string(emitted) +
                                     " size=" + std::to_string(sorted.size()));
            }
        }

        // Order statistics: the value the IGS median is built from.
        for (const std::size_t rank : {std::size_t{1}, std::size_t{2}, std::size_t{9999},
                                       masked.size() / 2, masked.size()}) {
            for (const bool descending : {true, false}) {
                if (rank > masked.size()) continue;
                std::vector<std::size_t> order = masked;
                std::sort(order.begin(), order.end(), [&](std::size_t a, std::size_t b) {
                    return descending ? values[a] > values[b] : values[a] < values[b];
                });
                const float expected = values[order[rank - 1]];
                float got = 0.0F;
                const bool ok = tinytensor::vulkan::select_nth_value(
                    device_values, device_mask, static_cast<std::uint32_t>(rank),
                    descending, &got);
                const std::string label = "nth rank=" + std::to_string(rank) +
                    (descending ? " desc" : " asc");
                check(ok, label + " not found");
                if (ok)
                    check(got == expected, label + " got=" + std::to_string(got) +
                                               " want=" + std::to_string(expected));
            }
        }

        // A rank past the end resolves to the extreme value rather than
        // failing, which is what lets the callers drop their separate
        // "how many rows qualify" readback: a selection larger than the mask
        // returns every masked row.
        {
            std::vector<float> masked_values;
            for (const std::size_t index : masked) masked_values.push_back(values[index]);
            float past_end = 0.0F;
            const bool resolved = tinytensor::vulkan::select_nth_value(
                device_values, device_mask,
                static_cast<std::uint32_t>(masked.size() + 1), true, &past_end);
            const auto extreme = std::minmax_element(masked_values.begin(),
                                                     masked_values.end());
            check(!resolved || (past_end == *extreme.first || past_end == *extreme.second),
                  "rank past the mask must clamp to an extreme");
        }

        // The exact shape the IGS refinement uses: gumbel scores over the full
        // retained array, an eligibility mask, and a cut far inside the array.
        // This is the comparison that decides whether the two paths agree.
        {
            const auto weights = device_values.clamp_min(1e-7F);
            const auto eligible = weights.isfinite().logical_and(weights.gt(0.F));
            const auto uniform = Tensor::rand(weights.shape(), Device::Vulkan)
                                     .clamp_min(1e-7F).clamp_max(1.F - 1e-7F);
            const auto scores = weights.log().sub(uniform.log().mul(-1.F).log());
            for (const std::size_t k : {std::size_t{860}, std::size_t{7439},
                                        std::size_t{20000}}) {
                const auto sorted = scores.masked_fill(
                    eligible.logical_not(), -std::numeric_limits<float>::infinity())
                    .sort(0, true);
                // Every row is finite and positive after clamp_min, so this
                // mask selects all of them, not the sampled mask above.
                const std::size_t count = std::min(k, rows);
                const auto reference = sorted.second.slice(0, 0, count).to(DataType::Int32);
                auto reference_host = reference.to_vector_int();
                std::sort(reference_host.begin(), reference_host.end());
                std::uint32_t emitted = 0;
                auto picked = tinytensor::vulkan::select_topk_indices(
                    scores, eligible, static_cast<std::uint32_t>(k), true, &emitted);
                std::vector<int> got;
                if (picked.is_valid()) got = picked.to_vector_int();
                std::sort(got.begin(), got.end());
                const std::string label = "gumbel topk k=" + std::to_string(k);
                check(got.size() == reference_host.size(),
                      label + " size got=" + std::to_string(got.size()) +
                          " want=" + std::to_string(reference_host.size()) +
                          " emitted=" + std::to_string(emitted));
                if (got.size() == reference_host.size()) {
                    const auto mismatch = std::mismatch(got.begin(), got.end(),
                                                        reference_host.begin());
                    check(mismatch.first == got.end(),
                          label + " set mismatch at " +
                              std::to_string(mismatch.first - got.begin()) +
                              " got=" + std::to_string(*mismatch.first) +
                              " want=" + std::to_string(*mismatch.second));
                }
            }
        }

        // Repeated selections must return the same rows. The key carries the
        // row index, so a tie at the cut has exactly one answer, and the data
        // path is free of the random draws the training sampler depends on
        // (nothing in the op touches the tensor RNG).
        {
            const auto first = tinytensor::vulkan::select_topk_indices(
                device_values, device_mask, 50000, true, nullptr);
            const auto second = tinytensor::vulkan::select_topk_indices(
                device_values, device_mask, 50000, true, nullptr);
            auto first_host = first.is_valid() ? first.to_vector_int() : std::vector<int>{};
            auto second_host = second.is_valid() ? second.to_vector_int() : std::vector<int>{};
            // Sets, not sequences: the emit writes each qualifying row into
            // whichever slot its ticket claims, and every caller consumes the
            // result through index_fill.
            std::sort(first_host.begin(), first_host.end());
            std::sort(second_host.begin(), second_host.end());
            check(first_host == second_host, "repeated selection differs");
        }

        std::printf("%s (%d failures)\n", failures == 0 ? "PASS" : "FAIL", failures);
        return failures == 0 ? 0 : 1;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
