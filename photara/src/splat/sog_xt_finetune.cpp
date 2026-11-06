#include "splat/trainer.hpp"
#include "splat/formats.hpp"
#include "formats_sog_xt.hpp"
#include "bilateral_grid.hpp"
#include "ppisp.hpp"
#include "training_data_loader.hpp"
#include "core/logging.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace photara::splat {
GaussianModel Trainer::finetune_sog_xt(
    const GaussianModel& input,
    const mvs::MvsScene& scene,
    const std::filesystem::path& output,
    ProgressCallback progress) const {
    if (options_.backend != TrainingBackend::cuda ||
        input.means.device() != tinytensor::Device::CUDA)
        throw std::invalid_argument("SOG-XT encoding-aware fine-tuning currently requires CUDA");
    if (input.render_method != RenderMethod::gaussian_3dgs || input.sh_degree == 0 ||
        input.size() == 0 ||
        (input.filter_mode != SplatFilterMode::classic &&
         input.filter_mode != SplatFilterMode::mip_2d))
        throw std::invalid_argument(
            "SOG-XT adaptation requires a nonempty classic/Mip2D 3DGS model with SH bands");
    if (output.empty() || !options_.sog_xt_finetune_iterations || !options_.sog_xt_batch_size ||
        options_.sog_xt_batch_size > 64)
        throw std::invalid_argument("Invalid SOG-XT adaptation output, iterations or batch size");
    if (!std::isfinite(options_.sog_xt_reference_weight) || options_.sog_xt_reference_weight < 0 ||
        !std::isfinite(options_.sog_xt_learning_rate_scale) ||
        options_.sog_xt_learning_rate_scale <= 0 || options_.sog_xt_learning_rate_scale > 1)
        throw std::invalid_argument("Invalid SOG-XT reference weight or learning-rate scale");
    auto options = options_;
    options.filter_mode = input.filter_mode;
    options.sh_degree = input.sh_degree;
    options.progressive_resolution = false;
    options.use_mvs_depth = options.use_mvs_normals = false;
    options.use_normal_field = options.use_depth_normal_loss = false;
    options.lambda_d = options.lambda_n = options.depth_weight = options.normal_weight = 0;
    options.multi_view_geo_weight = options.multi_view_ncc_weight = 0;
    std::vector<std::size_t> views;
    for (std::size_t i = 0; i < scene.views.size(); ++i)
        if (!options.evaluation_split_every || i % options.evaluation_split_every)
            views.push_back(i);
    if (views.empty())
        throw std::invalid_argument("No SOG-XT adaptation training views");
    std::vector<std::size_t> validation_views;
    for (std::size_t i = 0; i < std::min<std::size_t>(12, views.size()); ++i)
        validation_views.push_back(
            views[i * views.size() / std::min<std::size_t>(12, views.size())]);
    const unsigned steps = options.sog_xt_finetune_iterations, batch = options.sog_xt_batch_size;
    if (progress && !progress({0, steps, input.size()}))
        return input;
    detail::SogXtTrainingCodec codec;
    detail::encode_sog_xt(input, &codec);
    auto& latent = codec.latent;
    auto means = detail::make_adam_state(latent.means),
         scales = detail::make_adam_state(latent.log_scales),
         rotations = detail::make_adam_state(latent.quaternions),
         opacity = detail::make_adam_state(latent.opacity_logits),
         sh = detail::make_adam_state(latent.sh);
    auto ppisp = options.use_ppisp ? detail::make_ppisp_state(scene.views.size(), options)
                                   : detail::PpispState{};
    auto grid = options.use_bilateral_grid
                    ? detail::make_bilateral_grid_state(scene.views.size(), options)
                    : detail::BilateralGridState{};
    // Auxiliary appearance tables update per microbatch; divide their LR so
    // one optimizer batch has the original total correction step budget.
    auto appearance_options = options;
    appearance_options.ppisp_lr /= batch;
    appearance_options.bilateral_grid_lr /= batch;
    training_data::TrainingDataLoader loader(scene.views, options, 1.F);
    std::mt19937 random(options.seed);
    std::shuffle(views.begin(), views.end(), random);
    std::size_t cursor = 0;
    Rasterizer raster;
    RasterizeOptions raster_options;
    configure_rasterization(raster_options, options);
    raster_options.require_depth = false;
    raster_options.active_sh_degree = input.sh_degree;
    raster_options.scale_modifier = options.scale_modifier;
    // Selection uses training cameras only. Held-out RGB never participates
    // in either gradients or snapshot choice. Compare canonical RGB directly.
    const auto selection_score = [&](const GaussianModel& decoded) {
        double score = 0;
        auto settings = raster_options;
        settings.record_backward_state = false;
        for (auto v : validation_views) {
            const auto target = loader.get(v);
            const auto rendered = raster.forward(decoded, target.camera, settings);
            const auto diff = rendered.color.clamp(0.F, 1.F).sub(target.rgb);
            score += diff.mul(diff).mean().item();
        }
        return score / validation_views.size();
    };
    auto initial = detail::sog_xt_training_forward(codec);
    double best_score = selection_score(initial);
    auto best_files = detail::encode_sog_xt_frozen(codec, initial);
    unsigned best_step = 0;
    core::Logger::instance().info(
        "sog_xt_adaptation steps=", steps, " batch=", batch,
        " codewords=", codec.center_side * codec.center_side, " ppisp=", options.use_ppisp,
        " bilateral_grid=", options.use_bilateral_grid);
    unsigned covariance_step = 0;
    for (unsigned step = 1; step <= steps; ++step) {
        if ((step == std::max(2U, unsigned(std::lround(steps * .6)))) ||
            (step == std::max(2U, unsigned(std::lround(steps * .8))))) {
            detail::refresh_sog_xt_pras(codec);
            scales = detail::make_adam_state(latent.log_scales);
            rotations = detail::make_adam_state(latent.quaternions);
            covariance_step = 0;
        }
        auto decoded = detail::sog_xt_training_forward(codec);
        ModelGradients accumulated;
        float loss_value = 0;
        for (unsigned micro = 0; micro < batch; ++micro) {
            if (cursor == views.size()) {
                std::shuffle(views.begin(), views.end(), random);
                cursor = 0;
            }
            const auto view = views[cursor++];
            auto target = loader.get(view);
            auto rendered = raster.forward(decoded, target.camera, raster_options);
            auto loss_render = rendered;
            loss_render.context.impl.reset();
            const tinytensor::Tensor *photo = &rendered.color, *ppisp_input = nullptr,
                                     *grid_input = nullptr;
            const auto apply_ppisp = [&] {
                ppisp_input = photo;
                detail::apply_ppisp(*photo, ppisp, target.camera, view);
                photo = &ppisp.output;
            };
            if (options.use_ppisp && options.ppisp_before_bilagrid)
                apply_ppisp();
            const bool wrap = target.camera.model == CameraModel::equirectangular;
            if (options.use_bilateral_grid) {
                grid_input = photo;
                detail::apply_bilateral_grid(*photo, grid, view, wrap);
                photo = &grid.output;
            }
            if (options.use_ppisp && !options.ppisp_before_bilagrid)
                apply_ppisp();
            loss_render.color = *photo;
            auto loss = detail::compute_training_loss(
                loss_render, target, options, true, false, false, false);
            loss_value += loss.total / batch;
            const tinytensor::Tensor* gradient = &loss.color;
            const auto backward_ppisp = [&] {
                detail::backward_ppisp(ppisp, *ppisp_input, *gradient, target.camera, view);
                detail::step_ppisp(ppisp, appearance_options, (step - 1) * batch + micro + 1);
                gradient = &ppisp.input_grad;
            };
            if (options.use_ppisp && !options.ppisp_before_bilagrid)
                backward_ppisp();
            if (options.use_bilateral_grid) {
                detail::backward_bilateral_grid(grid, *grid_input, *gradient, view, wrap);
                detail::step_bilateral_grid(
                    grid, appearance_options, (step - 1) * batch + micro + 1, wrap);
                gradient = &grid.input_grad;
            }
            if (options.use_ppisp && options.ppisp_before_bilagrid)
                backward_ppisp();
            tinytensor::Tensor combined_gradient = *gradient;
            if (options.sog_xt_reference_weight > 0) {
                auto teacher_settings = raster_options;
                teacher_settings.record_backward_state = false;
                auto teacher = raster.forward(input, target.camera, teacher_settings);
                auto teacher_target = target;
                teacher_target.rgb = teacher.color;
                auto canonical_loss = rendered;
                canonical_loss.context.impl.reset();
                auto reference = detail::compute_training_loss(
                    canonical_loss, teacher_target, options, false, false, false, false);
                combined_gradient =
                    combined_gradient.add(reference.color.mul(options.sog_xt_reference_weight));
            }
            auto gradients =
                raster.backward(decoded, rendered, combined_gradient, loss.alpha, {}, {});
            const std::array<tinytensor::Tensor*, 5> dst{
                &accumulated.means, &accumulated.log_scales, &accumulated.quaternions,
                &accumulated.opacity_logits, &accumulated.sh};
            const std::array<const tinytensor::Tensor*, 5> src{
                &gradients.means, &gradients.log_scales, &gradients.quaternions,
                &gradients.opacity_logits, &gradients.sh};
            for (unsigned t = 0; t < dst.size(); ++t)
                *dst[t] = dst[t]->is_valid() ? dst[t]->add(src[t]->mul(1.F / batch))
                                             : src[t]->mul(1.F / batch);
        }
        const float fraction = float(step) / steps;
        const float warm = std::clamp((fraction - .35F) / .25F, 0.F, 1.F);
        detail::sog_xt_training_backward(codec, decoded, accumulated, 1e-4F * warm, 1e-2F * warm);
        ++covariance_step;
        // Continue with a decaying tail position rate; other attributes use
        // gentle adaptation rates rather than restarting reconstruction.
        const float lr_scale = options.sog_xt_learning_rate_scale;
        const float position_lr =
            options.means_lr * .01F * lr_scale * std::pow(.01F, float(step) / 30000.F);
        detail::adam_step(latent.means, accumulated.means, means, position_lr, step, options);
        detail::adam_step(
            latent.log_scales, accumulated.log_scales, scales, options.scales_lr * lr_scale,
            covariance_step, options);
        detail::adam_step(
            latent.quaternions, accumulated.quaternions, rotations,
            options.quaternions_lr * lr_scale, covariance_step, options);
        detail::adam_step(
            latent.opacity_logits, accumulated.opacity_logits, opacity,
            options.opacities_lr * lr_scale, step, options);
        detail::adam_step(
            latent.sh, accumulated.sh, sh, options.sh0_lr * lr_scale, step, options,
            latent.sh.shape()[1] * 3, options.sh_rest_lr * lr_scale);
        if (step == steps || step % 500 == 0) {
            auto candidate = detail::sog_xt_training_forward(codec);
            const double score = selection_score(candidate);
            core::Logger::instance().info(
                "sog_xt_adaptation_selection step=", step, " canonical_mse=", score,
                " best=", best_score);
            if (std::isfinite(score) && score < best_score) {
                best_score = score;
                best_step = step;
                best_files = detail::encode_sog_xt_frozen(codec, candidate);
            }
        }
        if (progress && (step == 1 || step == steps ||
                         (options.log_interval && step % options.log_interval == 0))) {
            TrainingProgress p;
            p.iteration = step;
            p.total_iterations = steps;
            p.gaussian_count = input.size();
            p.loss = p.rgb_loss = loss_value;
            if (!progress(p))
                return input; // No partially adapted file on cancellation.
        }
    }
    core::Logger::instance().info(
        "sog_xt_adaptation_selected step=", best_step, " canonical_mse=", best_score);
    detail::save_sog_xt_assets(best_files, output);
    return load_gaussians(output);
}
} // namespace photara::splat
