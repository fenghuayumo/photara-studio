#pragma once

#include "densification.hpp"

namespace aetherscan::splat::densification {

// Shared GPU topology operations keep the model and optimizer rows aligned.
class AdcPlusStrategy {
public:
    virtual ~AdcPlusStrategy() = default;
    RefinementCounts refine(
        GaussianModel& model, detail::DensificationStats& stats,
        unsigned iteration, float scene_extent, const mvs::Vec3f& scene_center,
        const TrainingOptions& options, const AdamStates& states) const;

protected:
    virtual int split_mode() const { return 2; }
    virtual tinytensor::Tensor growth_candidates(
        const tinytensor::Tensor& eligible,
        const tinytensor::Tensor& selected) const;
    virtual const char* name() const { return "adc_plus"; }
};

}  // namespace aetherscan::splat::densification
