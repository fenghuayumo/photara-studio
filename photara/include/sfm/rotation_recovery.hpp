#pragma once

#include "sfm/scene.hpp"

namespace photara::sfm {

struct RotationRecoveryReport {
    std::vector<Index> candidates;
    std::vector<Index> recovered;
};

// Compare merged poses with independent verified-pair rotation averaging.
// A dominant rotation consensus supplies the gauge, never replacement poses.
// Repairs require stable-only PnP depths and held-out validation; rejected
// candidates retain their original geometry. Does not change pair activation.
RotationRecoveryReport recover_rotation_inconsistent_views(Scene& scene);

// Require two distinct strong, nonplanar, cycle-consistent neighbors and a
// dominant conflicting edge weight. This is stricter than graph nomination.
std::vector<std::uint8_t> verified_rotation_conflicts(const Scene& scene);
unsigned prune_verified_rotation_conflicts(Scene& scene);

}  // namespace photara::sfm
