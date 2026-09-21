#include "texture/mesh_preview.hpp"

#include "photara_drender/photara_drender.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace photara::texture {
namespace {

constexpr float k_near_eps = 1e-4F;

std::array<float, 16> world_to_clip(
    const MeshPreviewCamera& camera, const float near_z, const float far_z) {
    const float* m = camera.world_to_camera.data();
    const float r00 = m[0], r10 = m[1], r20 = m[2];
    const float r01 = m[4], r11 = m[5], r21 = m[6];
    const float r02 = m[8], r12 = m[9], r22 = m[10];
    const float tx = m[12], ty = m[13], tz = m[14];
    const float width = static_cast<float>(std::max(1U, camera.width));
    const float height = static_cast<float>(std::max(1U, camera.height));
    const float x_offset = 2.F * (camera.cx + 0.5F) / width - 1.F;
    const float y_offset = 2.F * (camera.cy + 0.5F) / height - 1.F;
    const float depth_scale = (far_z + near_z) / (far_z - near_z);
    const float depth_offset = -2.F * far_z * near_z / (far_z - near_z);
    const float fx_term = 2.F * camera.fx / width;
    const float fy_term = 2.F * camera.fy / height;

    std::array<float, 16> clip{};
    clip[0 * 4 + 0] = fx_term * r00 + x_offset * r20;
    clip[0 * 4 + 1] = fx_term * r01 + x_offset * r21;
    clip[0 * 4 + 2] = fx_term * r02 + x_offset * r22;
    clip[0 * 4 + 3] = fx_term * tx + x_offset * tz;
    clip[1 * 4 + 0] = fy_term * r10 + y_offset * r20;
    clip[1 * 4 + 1] = fy_term * r11 + y_offset * r21;
    clip[1 * 4 + 2] = fy_term * r12 + y_offset * r22;
    clip[1 * 4 + 3] = fy_term * ty + y_offset * tz;
    clip[2 * 4 + 0] = depth_scale * r20;
    clip[2 * 4 + 1] = depth_scale * r21;
    clip[2 * 4 + 2] = depth_scale * r22;
    clip[2 * 4 + 3] = depth_scale * tz + depth_offset;
    clip[3 * 4 + 0] = r20;
    clip[3 * 4 + 1] = r21;
    clip[3 * 4 + 2] = r22;
    clip[3 * 4 + 3] = tz;
    return clip;
}

void camera_depth_range(
    const std::vector<float>& positions, const MeshPreviewCamera& camera,
    float& near_z, float& far_z) {
    const float* m = camera.world_to_camera.data();
    const float r20 = m[2], r21 = m[6], r22 = m[10], tz = m[14];
    near_z = 0.F;
    far_z = 0.F;
    const std::size_t count = positions.size() / 3U;
    for (std::size_t i = 0; i < count; ++i) {
        const float z = r20 * positions[3U * i] + r21 * positions[3U * i + 1U] +
                        r22 * positions[3U * i + 2U] + tz;
        if (!std::isfinite(z) || z <= k_near_eps) continue;
        if (near_z <= 0.F) {
            near_z = z;
            far_z = z;
        } else {
            near_z = std::min(near_z, z);
            far_z = std::max(far_z, z);
        }
    }
    if (!(far_z > near_z) || near_z <= 0.F) {
        near_z = 0.05F;
        far_z = 100.F;
        return;
    }
    near_z = std::max(near_z * 0.5F, k_near_eps);
    far_z = std::max(far_z * 1.5F, near_z + 1e-3F);
}

io::RgbImage solid_image(
    const std::uint32_t width, const std::uint32_t height,
    const std::array<float, 3>& rgb) {
    io::RgbImage image;
    image.width = std::max(1U, width);
    image.height = std::max(1U, height);
    image.pixels.assign(
        static_cast<std::size_t>(image.width) * image.height * 3U, 0);
    const auto r = static_cast<std::uint8_t>(
        std::lround(std::clamp(rgb[0], 0.F, 1.F) * 255.F));
    const auto g = static_cast<std::uint8_t>(
        std::lround(std::clamp(rgb[1], 0.F, 1.F) * 255.F));
    const auto b = static_cast<std::uint8_t>(
        std::lround(std::clamp(rgb[2], 0.F, 1.F) * 255.F));
    for (std::size_t i = 0; i < image.pixels.size(); i += 3) {
        image.pixels[i] = r;
        image.pixels[i + 1] = g;
        image.pixels[i + 2] = b;
    }
    return image;
}

}  // namespace

class MeshPreviewRasterizer::Impl {
public:
    Impl() : rasterizer_(context_) {}

    io::RgbImage render(
        const std::vector<float>& positions,
        const std::vector<float>& normals,
        const std::vector<float>& colours,
        const std::vector<std::uint32_t>& indices,
        const MeshPreviewCamera& camera,
        const MeshPreviewOptions& options) {
        const std::uint32_t width = std::max(1U, camera.width);
        const std::uint32_t height = std::max(1U, camera.height);
        if (positions.size() < 9 || indices.size() < 3)
            return solid_image(width, height, options.background);

        const std::size_t vertex_count = positions.size() / 3U;
        float near_z = 0.F;
        float far_z = 0.F;
        camera_depth_range(positions, camera, near_z, far_z);
        const auto clip = world_to_clip(camera, near_z, far_z);
        clip_positions_.resize(vertex_count * 4U);
        for (std::size_t i = 0; i < vertex_count; ++i) {
            const float x = positions[3U * i];
            const float y = positions[3U * i + 1U];
            const float z = positions[3U * i + 2U];
            for (int row = 0; row < 4; ++row) {
                clip_positions_[4U * i + static_cast<std::size_t>(row)] =
                    clip[static_cast<std::size_t>(row) * 4U] * x +
                    clip[static_cast<std::size_t>(row) * 4U + 1U] * y +
                    clip[static_cast<std::size_t>(row) * 4U + 2U] * z +
                    clip[static_cast<std::size_t>(row) * 4U + 3U];
            }
        }

        photara_drender::RasterizeOptions raster_options;
        raster_options.width = width;
        raster_options.height = height;
        raster_options.cull_mode = photara_drender::CullMode::none;
        raster_options.output_barycentric_derivatives = false;
        const photara_drender::RasterizeOutput rendered = rasterizer_.forward(
            clip_positions_, indices, raster_options);
        const std::size_t pixels =
            static_cast<std::size_t>(width) * height;
        if (rendered.raster.size() != pixels * 4U)
            throw std::runtime_error(
                "photara_drender returned an unexpected mesh raster size");

        const bool have_normals = normals.size() == positions.size();
        const bool have_colours =
            options.vertex_colour && colours.size() == positions.size();
        std::vector<float> attributes;
        std::uint32_t attribute_count = 0;
        if (have_normals) {
            attribute_count += 3;
            attributes.insert(attributes.end(), normals.begin(), normals.end());
        }
        if (have_colours) {
            if (attributes.empty()) {
                attributes = colours;
            } else {
                std::vector<float> interleaved(vertex_count * 6U);
                for (std::size_t i = 0; i < vertex_count; ++i) {
                    interleaved[6U * i] = normals[3U * i];
                    interleaved[6U * i + 1U] = normals[3U * i + 1U];
                    interleaved[6U * i + 2U] = normals[3U * i + 2U];
                    interleaved[6U * i + 3U] = colours[3U * i];
                    interleaved[6U * i + 4U] = colours[3U * i + 1U];
                    interleaved[6U * i + 5U] = colours[3U * i + 2U];
                }
                attributes = std::move(interleaved);
            }
            attribute_count += 3;
        }
        std::vector<float> interpolated;
        if (attribute_count > 0) {
            auto out = rasterizer_.interpolate_forward(
                attributes, attribute_count, indices, rendered);
            interpolated = std::move(out.values);
        }

        const float* w2c = camera.world_to_camera.data();
        io::RgbImage image = solid_image(width, height, options.background);
        for (std::size_t pixel = 0; pixel < pixels; ++pixel) {
            const float triangle_id = rendered.raster[4U * pixel + 3U];
            if (!(triangle_id > 0.F)) continue;
            const float a0 = rendered.raster[4U * pixel];
            const float a1 = rendered.raster[4U * pixel + 1U];
            const float a2 = 1.F - a0 - a1;
            float nx = 0.F, ny = 0.F, nz = -1.F;
            float cr = options.clay[0], cg = options.clay[1], cb = options.clay[2];
            if (!interpolated.empty()) {
                const std::size_t base = pixel * attribute_count;
                if (have_normals) {
                    nx = interpolated[base];
                    ny = interpolated[base + 1U];
                    nz = interpolated[base + 2U];
                }
                if (have_colours) {
                    const std::size_t colour_off = have_normals ? 3U : 0U;
                    cr = interpolated[base + colour_off];
                    cg = interpolated[base + colour_off + 1U];
                    cb = interpolated[base + colour_off + 2U];
                }
            }
            // World normal -> camera space; two-sided Lambert along view +Z.
            float cxn = w2c[0] * nx + w2c[4] * ny + w2c[8] * nz;
            float cyn = w2c[1] * nx + w2c[5] * ny + w2c[9] * nz;
            float czn = w2c[2] * nx + w2c[6] * ny + w2c[10] * nz;
            const float nlen =
                std::sqrt(cxn * cxn + cyn * cyn + czn * czn);
            if (nlen > 1e-8F) {
                cxn /= nlen;
                cyn /= nlen;
                czn /= nlen;
            }
            if (czn > 0.F) {
                cxn = -cxn;
                cyn = -cyn;
                czn = -czn;
            }
            const float light = 0.22F + 0.78F * std::abs(czn);
            float r = cr * light;
            float g = cg * light;
            float b = cb * light;
            if (options.wireframe) {
                const float edge = std::min(a0, std::min(a1, a2));
                if (edge < 0.02F) {
                    r *= 0.22F;
                    g *= 0.22F;
                    b *= 0.22F;
                }
            }
            image.pixels[3U * pixel] = static_cast<std::uint8_t>(
                std::lround(std::clamp(r, 0.F, 1.F) * 255.F));
            image.pixels[3U * pixel + 1U] = static_cast<std::uint8_t>(
                std::lround(std::clamp(g, 0.F, 1.F) * 255.F));
            image.pixels[3U * pixel + 2U] = static_cast<std::uint8_t>(
                std::lround(std::clamp(b, 0.F, 1.F) * 255.F));
        }
        return image;
    }

private:
    photara_drender::Context context_{};
    photara_drender::Rasterizer rasterizer_;
    std::vector<float> clip_positions_;
};

MeshPreviewRasterizer::MeshPreviewRasterizer()
    : impl_(std::make_unique<Impl>()) {}

MeshPreviewRasterizer::~MeshPreviewRasterizer() = default;
MeshPreviewRasterizer::MeshPreviewRasterizer(
    MeshPreviewRasterizer&&) noexcept = default;
MeshPreviewRasterizer& MeshPreviewRasterizer::operator=(
    MeshPreviewRasterizer&&) noexcept = default;

io::RgbImage MeshPreviewRasterizer::render(
    const std::vector<float>& positions, const std::vector<float>& normals,
    const std::vector<float>& colours,
    const std::vector<std::uint32_t>& indices,
    const MeshPreviewCamera& camera, const MeshPreviewOptions& options) {
    return impl_->render(
        positions, normals, colours, indices, camera, options);
}

}  // namespace photara::texture
