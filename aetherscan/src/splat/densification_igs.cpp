#include "densification_igs.hpp"

namespace aetherscan::splat::densification {

tinytensor::Tensor IgsStrategy::growth_candidates(
    const tinytensor::Tensor& eligible, const tinytensor::Tensor& selected) const {
    return eligible.logical_and(!selected).nonzero().squeeze(1).to(
        tinytensor::DataType::Int32);
}

const char* IgsStrategy::name() const { return "adc_igs"; }

}  // namespace aetherscan::splat::densification
