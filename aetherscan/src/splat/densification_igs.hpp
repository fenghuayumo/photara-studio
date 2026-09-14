#pragma once

#include "densification_adc_plus.hpp"

namespace aetherscan::splat::densification {

// image/world-gradient evidence and anisotropic exploration.
// Retain covariance-preserving long-axis splits and ADC+ pruning; require
// repeated contribution before growth. The preset adds about 5% per 200
// optimizer steps, refining every 100 with a final convergence interval.
class IgsStrategy final : public AdcPlusStrategy {
protected:
    int split_mode() const override { return 5; }
    tinytensor::Tensor growth_candidates(
        const tinytensor::Tensor& eligible,
        const tinytensor::Tensor& selected) const override;
    const char* name() const override;
};

}  // namespace aetherscan::splat::densification
