#pragma once

#include <array>
#include <cstddef>
#include <vector>

namespace photara::splat::detail {

// R = diag(1,-1,-1): Photara's RDF world to a right-handed Y-up world.
// Real SH basis signs for degrees 1..3 under that rotation. R is its own inverse.
inline constexpr std::array<float, 15> k_rdf_to_rub_sh_signs{
    -1.F, -1.F, 1.F, -1.F, 1.F, 1.F, -1.F, 1.F,
    -1.F, 1.F, -1.F, -1.F, 1.F, -1.F, 1.F};

inline void convert_rdf_rub(std::vector<float>& positions,
                            std::vector<float>& quaternions,
                            std::vector<float>& sh, const std::size_t bases) {
    for (std::size_t i = 0; i < positions.size() / 3; ++i) {
        positions[3 * i + 1] = -positions[3 * i + 1];
        positions[3 * i + 2] = -positions[3 * i + 2];
        // Conjugating the orientation by R preserves covariance after the
        // frame change (principal-axis signs do not change an ellipsoid).
        quaternions[4 * i + 2] = -quaternions[4 * i + 2];
        quaternions[4 * i + 3] = -quaternions[4 * i + 3];
        for (std::size_t b = 1; b < bases; ++b)
            for (std::size_t c = 0; c < 3; ++c)
                sh[(i * bases + b) * 3 + c] *= k_rdf_to_rub_sh_signs[b - 1];
    }
}

} // namespace photara::splat::detail
