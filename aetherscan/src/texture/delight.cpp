#include "texture/delight.hpp"
#include "texture/paths.hpp"

#include "core/logging.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)
#include "features/onnx_session.hpp"
#endif

namespace aetherscan::texture {
namespace {

[[nodiscard]] std::filesystem::path default_model_dir() {
    return std::filesystem::path(AETHERSCAN_INTRINSIC_MODELS_DIR_PATH);
}

[[nodiscard]] std::filesystem::path resolve_model_dir(
    const TextureOptions& options) {
    return !options.delight_model_dir.empty() ? options.delight_model_dir
                                              : default_model_dir();
}

#if defined(AETHERSCAN_HAS_ONNXRUNTIME)

[[nodiscard]] constexpr int round32(const int x) noexcept {
    return (x + 31) & ~31;
}

struct TensorNCHW {
    std::int64_t n{1};
    std::int64_t c{0};
    std::int64_t h{0};
    std::int64_t w{0};
    std::vector<float> data;

    [[nodiscard]] std::size_t size() const noexcept { return data.size(); }
    [[nodiscard]] std::vector<std::int64_t> shape() const {
        return {n, c, h, w};
    }

    [[nodiscard]] float& at(
        const std::int64_t ch, const std::int64_t y, const std::int64_t x) {
        return data[static_cast<std::size_t>(
            (ch * h + y) * w + x)];
    }
    [[nodiscard]] float at(
        const std::int64_t ch, const std::int64_t y,
        const std::int64_t x) const {
        return data[static_cast<std::size_t>((ch * h + y) * w + x)];
    }
};

[[nodiscard]] TensorNCHW make_tensor(
    const std::int64_t c, const std::int64_t h, const std::int64_t w,
    const float fill = 0.F) {
    TensorNCHW t;
    t.c = c;
    t.h = h;
    t.w = w;
    t.data.assign(static_cast<std::size_t>(c * h * w), fill);
    return t;
}

void srgb_bytes_to_linear_nchw(
    const float* rgb_linear_or_srgb_bytes, const bool already_linear,
    const std::uint32_t width, const std::uint32_t height, TensorNCHW& out) {
    // Input TextureViewImage stores linear RGB floats in [0,1]. Delighter
    // historically expected sRGB bytes then applied x^2.2. Convert linear
    // -> approximate sRGB -> x^2.2 to match the Intrinsic training path.
    out = make_tensor(3, static_cast<std::int64_t>(height),
                      static_cast<std::int64_t>(width));
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::size_t i =
                (static_cast<std::size_t>(y) * width + x) * 3U;
            for (int c = 0; c < 3; ++c) {
                float v = rgb_linear_or_srgb_bytes[i + static_cast<std::size_t>(c)];
                if (already_linear) {
                    // linear -> sRGB
                    v = std::clamp(v, 0.F, 1.F);
                    v = v <= 0.0031308F
                            ? v * 12.92F
                            : 1.055F * std::pow(v, 1.F / 2.4F) - 0.055F;
                }
                // Intrinsic uses gamma 2.2 on sRGB-coded values.
                out.at(c, static_cast<std::int64_t>(y),
                       static_cast<std::int64_t>(x)) = std::pow(v, 2.2F);
            }
        }
    }
}

[[nodiscard]] float sample_bilinear(
    const TensorNCHW& src, const std::int64_t c, const float y,
    const float x) {
    const float y0f = std::clamp(y, 0.F, static_cast<float>(src.h - 1));
    const float x0f = std::clamp(x, 0.F, static_cast<float>(src.w - 1));
    const int y0 = static_cast<int>(y0f);
    const int x0 = static_cast<int>(x0f);
    const int y1 = std::min(y0 + 1, static_cast<int>(src.h - 1));
    const int x1 = std::min(x0 + 1, static_cast<int>(src.w - 1));
    const float ty = y0f - static_cast<float>(y0);
    const float tx = x0f - static_cast<float>(x0);
    const float v00 = src.at(c, y0, x0);
    const float v10 = src.at(c, y0, x1);
    const float v01 = src.at(c, y1, x0);
    const float v11 = src.at(c, y1, x1);
    return (v00 * (1.F - tx) + v10 * tx) * (1.F - ty) +
           (v01 * (1.F - tx) + v11 * tx) * ty;
}

[[nodiscard]] TensorNCHW resize_bilinear(
    const TensorNCHW& src, const std::int64_t new_h, const std::int64_t new_w) {
    TensorNCHW dst = make_tensor(src.c, new_h, new_w);
    if (src.h == new_h && src.w == new_w) {
        dst.data = src.data;
        return dst;
    }
    const float scale_y =
        static_cast<float>(src.h) / static_cast<float>(new_h);
    const float scale_x =
        static_cast<float>(src.w) / static_cast<float>(new_w);
    for (std::int64_t c = 0; c < src.c; ++c) {
        for (std::int64_t y = 0; y < new_h; ++y) {
            const float sy = (static_cast<float>(y) + 0.5F) * scale_y - 0.5F;
            for (std::int64_t x = 0; x < new_w; ++x) {
                const float sx =
                    (static_cast<float>(x) + 0.5F) * scale_x - 0.5F;
                dst.at(c, y, x) = sample_bilinear(src, c, sy, sx);
            }
        }
    }
    return dst;
}

[[nodiscard]] TensorNCHW concat_channels(
    const std::vector<const TensorNCHW*>& parts) {
    if (parts.empty()) throw std::invalid_argument("concat empty");
    const std::int64_t h = parts[0]->h;
    const std::int64_t w = parts[0]->w;
    std::int64_t c = 0;
    for (const auto* p : parts) {
        if (p->h != h || p->w != w)
            throw std::invalid_argument("concat size mismatch");
        c += p->c;
    }
    TensorNCHW out = make_tensor(c, h, w);
    std::int64_t offset = 0;
    for (const auto* p : parts) {
        const std::size_t n =
            static_cast<std::size_t>(p->c * p->h * p->w);
        std::copy_n(
            p->data.begin(), n,
            out.data.begin() + static_cast<std::ptrdiff_t>(offset * h * w));
        offset += p->c;
    }
    return out;
}

[[nodiscard]] TensorNCHW uninvert(const TensorNCHW& t) {
    TensorNCHW out = t;
    for (float& v : out.data) v = (1.F / v) - 1.F;
    return out;
}

[[nodiscard]] TensorNCHW calc_lum(const TensorNCHW& rgb) {
    TensorNCHW lum = make_tensor(1, rgb.h, rgb.w);
    constexpr float kr = 0.299F, kg = 0.587F, kb = 0.114F;
    for (std::int64_t y = 0; y < rgb.h; ++y) {
        for (std::int64_t x = 0; x < rgb.w; ++x) {
            lum.at(0, y, x) = kr * rgb.at(0, y, x) + kg * rgb.at(1, y, x) +
                              kb * rgb.at(2, y, x);
        }
    }
    return lum;
}

[[nodiscard]] TensorNCHW rgb_to_inv_luv(const TensorNCHW& rgb) {
    constexpr float eps = 1e-3F;
    TensorNCHW out = make_tensor(3, rgb.h, rgb.w);
    const TensorNCHW lum = calc_lum(rgb);
    for (std::int64_t y = 0; y < rgb.h; ++y) {
        for (std::int64_t x = 0; x < rgb.w; ++x) {
            const float g = std::max(rgb.at(1, y, x), eps);
            out.at(0, y, x) = 1.F / (lum.at(0, y, x) + 1.F);
            out.at(1, y, x) = 1.F / (rgb.at(0, y, x) / g + 1.F);
            out.at(2, y, x) = 1.F / (rgb.at(2, y, x) / g + 1.F);
        }
    }
    return out;
}

[[nodiscard]] TensorNCHW inv_luv_to_rgb(const TensorNCHW& inv_luv) {
    constexpr float eps = 1e-3F;
    constexpr float kr = 0.299F, kg = 0.587F, kb = 0.114F;
    TensorNCHW out = make_tensor(3, inv_luv.h, inv_luv.w);
    for (std::int64_t y = 0; y < inv_luv.h; ++y) {
        for (std::int64_t x = 0; x < inv_luv.w; ++x) {
            const float l =
                (1.F / std::max(inv_luv.at(0, y, x), eps)) - 1.F;
            const float u =
                (1.F / std::max(inv_luv.at(1, y, x), eps)) - 1.F;
            const float v =
                (1.F / std::max(inv_luv.at(2, y, x), eps)) - 1.F;
            const float g = l / (u * kr + v * kb + kg);
            out.at(0, y, x) = g * u;
            out.at(1, y, x) = g;
            out.at(2, y, x) = g * v;
        }
    }
    return out;
}

// Deterministic substitute for EqualizePredictions(..., p=0): fit scale on
// all pixels instead of a random 50% subset.
[[nodiscard]] TensorNCHW equalize_full(
    const TensorNCHW& img, const TensorNCHW& base, const TensorNCHW& full) {
    TensorNCHW full_shading = uninvert(full);
    TensorNCHW base_shading = uninvert(base);
    for (float& v : full_shading.data) v = std::max(v, 1e-5F);
    for (float& v : base_shading.data) v = std::max(v, 1e-5F);
    const TensorNCHW lum = calc_lum(img);
    double num = 0.0;
    double den = 0.0;
    for (std::size_t i = 0; i < lum.data.size(); ++i) {
        const float fa = lum.data[i] / full_shading.data[i];
        const float ba = lum.data[i] / base_shading.data[i];
        num += static_cast<double>(fa) * static_cast<double>(ba);
        den += static_cast<double>(fa) * static_cast<double>(fa);
    }
    const float scale =
        den > 1e-20 ? static_cast<float>(num / den) : 1.F;
    TensorNCHW new_full = make_tensor(1, img.h, img.w);
    for (std::size_t i = 0; i < lum.data.size(); ++i) {
        const float new_albedo =
            scale * (lum.data[i] / full_shading.data[i]);
        const float new_shading =
            lum.data[i] / std::max(new_albedo, 1e-5F);
        new_full.data[i] = 1.F / (new_shading + 1.F);
    }
    return new_full;
}

[[nodiscard]] float quantile99(std::vector<float> values) {
    if (values.empty()) return 1.F;
    const std::size_t k =
        static_cast<std::size_t>(0.99 * static_cast<double>(values.size() - 1));
    std::nth_element(values.begin(), values.begin() + static_cast<std::ptrdiff_t>(k),
                     values.end());
    return std::max(values[k], 1e-6F);
}

class OnnxDelighter {
public:
    explicit OnnxDelighter(const std::filesystem::path& model_dir, const bool use_cuda) {
        for (int i = 0; i < 4; ++i) {
            features::OnnxSessionConfig cfg;
            cfg.model_path = model_dir / ("stage_" + std::to_string(i) + ".onnx");
            cfg.use_cuda = use_cuda;
            cfg.allow_cpu_fallback = true;
            cfg.env_name = "AetherScan.Delighter";
            stages_[static_cast<std::size_t>(i)] =
                std::make_unique<features::OnnxSession>(std::move(cfg));
        }
    }

    [[nodiscard]] TensorNCHW run(
        features::OnnxSession& session, const TensorNCHW& input) {
        std::lock_guard lock(session.mutex());
        Ort::MemoryInfo memory =
            Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);
        auto shape = input.shape();
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            memory, const_cast<float*>(input.data.data()), input.data.size(),
            shape.data(), shape.size());
        auto outputs = session.session().Run(
            Ort::RunOptions{nullptr}, session.input_names().data(), &tensor, 1,
            session.output_names().data(), 1);
        const float* out_ptr = outputs[0].GetTensorData<float>();
        const auto info = outputs[0].GetTensorTypeAndShapeInfo();
        const auto out_shape = info.GetShape();
        if (out_shape.size() != 4)
            throw std::runtime_error("Unexpected delighter ONNX output rank");
        TensorNCHW out = make_tensor(out_shape[1], out_shape[2], out_shape[3]);
        std::copy_n(out_ptr, out.data.size(), out.data.begin());
        return out;
    }

    void process_view(TextureViewImage& view) {
        TensorNCHW linear_src;
        srgb_bytes_to_linear_nchw(
            view.rgb.data(), true, view.width, view.height, linear_src);
        const std::int64_t orig_h = linear_src.h;
        const std::int64_t orig_w = linear_src.w;
        constexpr int base_dim = 384;

        TensorNCHW img = resize_bilinear(
            linear_src, round32(static_cast<int>(orig_h)),
            round32(static_cast<int>(orig_w)));
        const std::int64_t fh = img.h;
        const std::int64_t fw = img.w;

        // BaseResize: longest side -> base_dim, rounded to 32.
        const float base_scale =
            static_cast<float>(base_dim) /
            static_cast<float>((std::max)(img.w, img.h));
        TensorNCHW base_in = resize_bilinear(
            img,
            round32(static_cast<int>(static_cast<float>(img.h) * base_scale)),
            round32(static_cast<int>(static_cast<float>(img.w) * base_scale)));
        TensorNCHW base_out =
            resize_bilinear(run(*stages_[0], base_in), fh, fw);
        TensorNCHW full_out = run(*stages_[0], img);
        TensorNCHW ordinal_full = equalize_full(img, base_out, full_out);
        TensorNCHW iid_in = concat_channels({&img, &base_out, &ordinal_full});
        TensorNCHW inv_shading = run(*stages_[1], iid_in);
        for (float& v : inv_shading.data) v = std::max(v, 1e-3F);
        TensorNCHW shading = uninvert(inv_shading);
        TensorNCHW gray_albedo = make_tensor(3, fh, fw);
        for (std::int64_t c = 0; c < 3; ++c)
            for (std::int64_t y = 0; y < fh; ++y)
                for (std::int64_t x = 0; x < fw; ++x)
                    gray_albedo.at(c, y, x) =
                        img.at(c, y, x) / std::max(shading.at(0, y, x), 1e-5F);

        const float color_scale =
            static_cast<float>(base_dim) /
            static_cast<float>((std::max)(fh, fw));
        const std::int64_t color_h =
            round32(static_cast<int>(static_cast<float>(fh) * color_scale));
        const std::int64_t color_w =
            round32(static_cast<int>(static_cast<float>(fw) * color_scale));

        TensorNCHW inv_img_luv =
            resize_bilinear(rgb_to_inv_luv(img), color_h, color_w);
        TensorNCHW inv_albedo_luv =
            resize_bilinear(rgb_to_inv_luv(gray_albedo), color_h, color_w);
        TensorNCHW inv_base_gray =
            resize_bilinear(inv_shading, color_h, color_w);
        TensorNCHW col_in =
            concat_channels({&inv_img_luv, &inv_base_gray, &inv_albedo_luv});
        TensorNCHW inv_uv =
            resize_bilinear(run(*stages_[2], col_in), fh, fw);

        TensorNCHW inv_luv = concat_channels({&inv_shading, &inv_uv});
        TensorNCHW rough_shading = inv_luv_to_rgb(inv_luv);
        TensorNCHW rough_albedo = make_tensor(3, fh, fw);
        for (std::size_t i = 0; i < rough_albedo.data.size(); ++i)
            rough_albedo.data[i] =
                img.data[i] / std::max(rough_shading.data[i], 1e-5F);
        const float q = quantile99(rough_albedo.data);
        const float gain = 0.75F / q;
        for (float& v : rough_albedo.data) v = std::max(v * gain, 1e-3F);

        TensorNCHW inv_rough = make_tensor(3, fh, fw);
        for (std::size_t i = 0; i < inv_rough.data.size(); ++i) {
            const float shading_v =
                img.data[i] / std::max(rough_albedo.data[i], 1e-5F);
            inv_rough.data[i] = 1.F / (shading_v + 1.F);
        }
        TensorNCHW alb_in =
            concat_channels({&img, &inv_rough, &rough_albedo});
        TensorNCHW pred = run(*stages_[3], alb_in);
        TensorNCHW high = resize_bilinear(pred, orig_h, orig_w);

        view.rgb.resize(static_cast<std::size_t>(orig_h * orig_w * 3));
        for (std::int64_t y = 0; y < orig_h; ++y) {
            for (std::int64_t x = 0; x < orig_w; ++x) {
                const std::size_t dst =
                    static_cast<std::size_t>(y * orig_w + x) * 3U;
                for (int c = 0; c < 3; ++c) {
                    // Network albedo is linear-ish after sigmoid; convert
                    // with Intrinsic's gamma (1/2.2) then treat as sRGB and
                    // store linear for the baker.
                    float srgb = std::pow(
                        std::clamp(high.at(c, y, x), 0.F, 1.F), 1.F / 2.2F);
                    srgb = std::clamp(srgb, 0.F, 1.F);
                    const float linear =
                        srgb <= 0.04045F
                            ? srgb / 12.92F
                            : std::pow((srgb + 0.055F) / 1.055F, 2.4F);
                    view.rgb[dst + static_cast<std::size_t>(c)] = linear;
                }
            }
        }
    }

private:
    std::array<std::unique_ptr<features::OnnxSession>, 4> stages_;
};

#endif  // AETHERSCAN_HAS_ONNXRUNTIME

}  // namespace

bool delighter_available(const TextureOptions& options) {
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)options;
    return false;
#else
    const auto dir = resolve_model_dir(options);
    for (int i = 0; i < 4; ++i) {
        if (!std::filesystem::exists(
                dir / ("stage_" + std::to_string(i) + ".onnx")))
            return false;
    }
    return true;
#endif
}

void delight_texture_views(
    std::vector<TextureViewImage>& views, const TextureOptions& options) {
    if (views.empty()) return;
#if !defined(AETHERSCAN_HAS_ONNXRUNTIME)
    (void)options;
    throw std::runtime_error(
        "Delighter requires ONNX Runtime. Reconfigure with "
        "-DAETHERSCAN_ENABLE_ONNX=ON and place stage_0..3.onnx under "
        "AETHERSCAN_INTRINSIC_MODELS_DIR");
#else
    if (!delighter_available(options)) {
        throw std::runtime_error(
            "Delighter ONNX models not found under " +
            resolve_model_dir(options).string() +
            " (need stage_0.onnx .. stage_3.onnx). Set "
            "AETHERSCAN_INTRINSIC_MODELS_DIR or TextureOptions::delight_model_dir");
    }
    core::StageScope stage("texture.delight");
    OnnxDelighter delighter(
        resolve_model_dir(options), options.delight_use_cuda);
    core::ProgressReporter progress("texture.delight", views.size());
    for (auto& view : views) {
        delighter.process_view(view);
        progress.advance();
    }
    stage.finish();
#endif
}

}  // namespace aetherscan::texture
