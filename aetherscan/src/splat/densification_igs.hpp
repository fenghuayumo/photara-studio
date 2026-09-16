#pragma once

#include "densification_adc_plus.hpp"

namespace aetherscan::splat::densification {

// Image/world-gradient evidence, relocation and the ADC+ refinement skeleton.
// The differentiation from ADC+ is entirely in what counts as evidence and
// what ranks growth: an SSIM contrast-structure error map blended with the
// world-space position gradient, a screen-gradient gate, two distinct
// contributing cameras before replication, and score-sampled relocation.
// Geometry (split, cadence, screen cap) is shared with ADC+.
class IgsStrategy final : public AdcPlusStrategy {
protected:
    tinytensor::Tensor growth_candidates(
        const tinytensor::Tensor& eligible,
        const tinytensor::Tensor& selected) const override;
    const char* name() const override;
};

}  // namespace aetherscan::splat::densification
