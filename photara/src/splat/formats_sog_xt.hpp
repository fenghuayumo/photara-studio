#pragma once

#include "splat/types.hpp"

#include <cstdint>
#include <cstddef>
#include <map>
#include <string>
#include <vector>

namespace photara::splat::detail {

using SogXtFiles = std::map<std::string, std::vector<std::uint8_t>>;

// Frozen discrete structure for encoding-aware optimization. Rows in latent
// are the active image pixels in raster order, after covariance symmetry.
struct SogXtTrainingCodec {
    GaussianModel latent;
    SogXtFiles files;
    std::vector<unsigned> pixels;
    tinytensor::Tensor low, high, sh_low, sh_high, labels, counts, neighbors;
    tinytensor::Tensor centers;
    tinytensor::Tensor raw_centers;
    unsigned side{}, center_side{}, dimensions{};
};

struct DecodedSogXt {
    std::size_t count{};
    unsigned degree{};
    bool antialiased{};
    std::vector<float> means, log_scales, rotations, opacity_logits, sh;
};

// SOG-XT image payloads; packaging is handled by the existing ZIP layer.
SogXtFiles encode_sog_xt(const GaussianModel& model, SogXtTrainingCodec* training = nullptr);
GaussianModel sog_xt_training_forward(SogXtTrainingCodec& codec);
void sog_xt_training_backward(
    SogXtTrainingCodec& codec,
    const GaussianModel& decoded,
    ModelGradients& gradients,
    float tv_weight,
    float quaternion_tv_weight);
SogXtFiles encode_sog_xt_frozen(const SogXtTrainingCodec& codec, const GaussianModel& decoded);
void save_sog_xt_assets(const SogXtFiles& files, const std::filesystem::path& path);
void refresh_sog_xt_pras(SogXtTrainingCodec& codec);
DecodedSogXt decode_sog_xt(const SogXtFiles& files);

// Offline, coarse-to-fine local assignment / covariance representative search.
// Returned layout maps image pixels to source rows, including padded rows.
std::vector<std::uint32_t> sog_xt_layout(
    const std::vector<float>& features,
    unsigned side,
    unsigned dimensions);
void sog_xt_pras(
    std::vector<float>& rotations,
    std::vector<float>& log_scales,
    const std::vector<std::uint8_t>& active,
    unsigned side);

} // namespace photara::splat::detail
