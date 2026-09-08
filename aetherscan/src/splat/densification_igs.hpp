#pragma once

#include "densification_adc_plus.hpp"

namespace aetherscan::splat::densification {

// Retain ADC+ visibility, pruning and footprint-weighted sampling. Avoid
// duplicate budget allocation and preserve mixture moments when splitting.
class IgsStrategy final : public AdcPlusStrategy {
protected:
    int split_mode() const override { return 5; }
    tinytensor::Tensor growth_candidates(
        const tinytensor::Tensor& eligible,
        const tinytensor::Tensor& selected) const override;
    const char* name() const override;
};

}  // namespace aetherscan::splat::densification
