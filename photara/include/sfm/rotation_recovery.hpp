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

}  // namespace photara::sfm
