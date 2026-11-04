#pragma once
#include "splat/prior.hpp"
#include "inference/geometry_prior.hpp"
#include "io/image.hpp"
#include <Eigen/Core>
#include <Eigen/Geometry>
#include <algorithm>
#include <array>
#include <cmath>
#include <stdexcept>

namespace photara::splat::detail {
// Six overlapping 120-degree pinhole faces. R maps face coordinates to the
// source camera; every source ray lies inside at least one face, including poles.
inline Eigen::Matrix3d prior_face_rotation(unsigned face) {
    const std::array<Eigen::Vector3d, 6> forward{
        Eigen::Vector3d{0, 0, 1}, {1, 0, 0}, {0, 0, -1}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}};
    const auto z = forward.at(face);
    const Eigen::Vector3d up =
        std::abs(z.y()) > .9 ? Eigen::Vector3d{0, 0, 1} : Eigen::Vector3d{0, 1, 0};
    const Eigen::Vector3d x = up.cross(z).normalized(), y = z.cross(x);
    Eigen::Matrix3d r;
    r.col(0) = x;
    r.col(1) = y;
    r.col(2) = z;
    return r;
}
inline Eigen::Vector3d prior_source_ray(const Camera& c, unsigned x, unsigned y) {
    CameraRay r =
        c.model == CameraModel::equirectangular
            ? unproject_equirectangular_camera(x, y, c.width, c.height)
            : unproject_fisheye_camera(x, y, c.fx, c.fy, c.cx, c.cy, c.k1, c.k2, c.k3, c.k4);
    return r.valid ? Eigen::Vector3d{r.x, r.y, r.z} : Eigen::Vector3d::Zero();
}
inline Camera prior_face_camera(const Camera& source, const Eigen::Matrix3d& r, unsigned side) {
    Camera c;
    c.width = c.height = side;
    c.fx = c.fy = side / (2.F * std::sqrt(3.F));
    c.cx = c.cy = (side - 1) * .5F;
    c.position = source.position;
    // T_face_world = R_face_source * T_source_world, including translation.
    for (unsigned col = 0; col < 4; ++col)
        for (unsigned row = 0; row < 3; ++row) {
            c.world_to_camera[4 * col + row] = 0;
            for (unsigned k = 0; k < 3; ++k)
                c.world_to_camera[4 * col + row] +=
                    float(r(k, row)) * source.world_to_camera[4 * col + k];
        }
    c.world_to_camera[15] = 1;
    return c;
}
// Return zero scale when a face lacks reliable SfM support. Normal-only
// inference never requires a depth scale. Neither validity map masks RGB loss.
template <class Predict, class Scale>
TrainingPrior predict_native_prior(
    const io::RgbImage& image,
    const io::GrayImage& mask,
    const Camera& camera,
    bool need_depth,
    bool need_normals,
    Predict predict,
    Scale align,
    unsigned side = 512) {
    if (!uses_native_splat_projection(camera.model) || !side || image.width != camera.width ||
        image.height != camera.height ||
        image.pixels.size() != std::size_t(image.width) * image.height * 3 ||
        (!mask.pixels.empty() && (mask.width != image.width || mask.height != image.height ||
                                  mask.pixels.size() != std::size_t(image.width) * image.height)))
        throw std::invalid_argument("Invalid native prior source");
    const std::size_t count = std::size_t(camera.width) * camera.height;
    TrainingPrior out;
    out.camera = camera;
    if (need_depth)
        out.depth.resize(count);
    if (need_normals)
        out.normals.resize(3 * count);
    std::vector<double> dw(need_depth ? count : 0), nw(need_normals ? count : 0);
    std::vector<Eigen::Vector3d> rays(count);
    for (unsigned y = 0; y < camera.height; ++y)
        for (unsigned x = 0; x < camera.width; ++x)
            rays[std::size_t(y) * camera.width + x] = prior_source_ray(camera, x, y);
    for (unsigned face = 0; face < 6; ++face) {
        const auto rotation = prior_face_rotation(face);
        const auto tile = prior_face_camera(camera, rotation, side);
        io::RgbImage input{side, side, std::vector<std::uint8_t>(std::size_t(side) * side * 3)};
        std::vector<std::uint8_t> valid(std::size_t(side) * side);
        for (unsigned y = 0; y < side; ++y)
            for (unsigned x = 0; x < side; ++x) {
                const Eigen::Vector3d r =
                    rotation * Eigen::Vector3d{(x - tile.cx) / tile.fx, (y - tile.cy) / tile.fy, 1};
                const auto uv = camera.model == CameraModel::equirectangular
                                    ? project_equirectangular_camera(
                                          r.x(), r.y(), r.z(), camera.width, camera.height)
                                    : project_fisheye_camera(
                                          r.x(), r.y(), r.z(), camera.fx, camera.fy, camera.cx,
                                          camera.cy, camera.k1, camera.k2, camera.k3, camera.k4);
                if (!uv.valid || !std::isfinite(uv.u + uv.v) || uv.v < 0 ||
                    uv.v > camera.height - 1)
                    continue;
                double u = uv.u;
                if (camera.model == CameraModel::equirectangular)
                    u -= std::floor(u / camera.width) * camera.width;
                else if (u < 0 || u > camera.width - 1)
                    continue;
                const unsigned x0 = unsigned(u), y0 = unsigned(uv.v),
                               x1 = camera.model == CameraModel::equirectangular
                                        ? (x0 + 1) % camera.width
                                        : std::min(x0 + 1, camera.width - 1),
                               y1 = std::min(y0 + 1, camera.height - 1);
                const std::array<std::size_t, 4> ids{
                    std::size_t(y0) * camera.width + x0, std::size_t(y0) * camera.width + x1,
                    std::size_t(y1) * camera.width + x0, std::size_t(y1) * camera.width + x1};
                if (!mask.pixels.empty() && std::any_of(ids.begin(), ids.end(), [&](auto i) {
                        return mask.pixels[i] == 0;
                    }))
                    continue;
                const double dx = u - x0, dy = uv.v - y0;
                const std::array<double, 4> w{
                    (1 - dx) * (1 - dy), dx * (1 - dy), (1 - dx) * dy, dx * dy};
                const auto dst = std::size_t(y) * side + x;
                valid[dst] = 1;
                for (unsigned ch = 0; ch < 3; ++ch) {
                    double value = 0;
                    for (unsigned k = 0; k < 4; ++k)
                        value += w[k] * image.pixels[3 * ids[k] + ch];
                    input.pixels[3 * dst + ch] =
                        std::uint8_t(std::clamp(std::lround(value), 0L, 255L));
                }
            }
        if (std::count(valid.begin(), valid.end(), 1) < std::max<std::size_t>(1, valid.size() / 32))
            continue;
        auto prediction = predict(input, tile);
        const std::size_t tile_count = std::size_t(side) * side;
        if (prediction.width != side || prediction.height != side ||
            prediction.valid.size() != tile_count ||
            (need_depth && prediction.depth.size() != tile_count) ||
            (need_normals && prediction.normals.size() != tile_count) ||
            prediction.depth_representation != inference::DepthRepresentation::camera_z)
            throw std::runtime_error("Native prior predictor returned incompatible geometry");
        for (std::size_t i = 0; i < tile_count; ++i)
            prediction.valid[i] &= valid[i];
        const double scale = need_depth ? align(tile, prediction) : 0;
        for (std::size_t i = 0; i < count; ++i) {
            if (rays[i].isZero() || (!mask.pixels.empty() && mask.pixels[i] == 0))
                continue;
            const Eigen::Vector3d q = rotation.transpose() * rays[i];
            if (!(q.z() > 0))
                continue;
            const double u = tile.fx * q.x() / q.z() + tile.cx,
                         v = tile.fy * q.y() / q.z() + tile.cy;
            if (u < 0 || v < 0 || u > side - 1 || v > side - 1)
                continue;
            // Nearest prediction sampling avoids mixing foreground/background depths.
            const auto j = std::size_t(std::lround(v)) * side + std::lround(u);
            if (!prediction.valid[j])
                continue;
            const double weight = std::pow(q.z(), 4);
            if (need_depth && scale > 0 && std::isfinite(scale)) {
                const double d = prediction.depth[j] * scale / q.z() *
                                 (camera.model == CameraModel::equirectangular ? 1 : rays[i].z());
                if (d > 0 && std::isfinite(d)) {
                    out.depth[i] += float(weight * d);
                    dw[i] += weight;
                }
            }
            if (need_normals) {
                const auto& pn = prediction.normals[j];
                const Eigen::Vector3d normal = rotation * Eigen::Vector3d{pn[0], pn[1], pn[2]};
                if (!normal.allFinite() || normal.norm() < .5)
                    continue;
                for (unsigned ch = 0; ch < 3; ++ch)
                    out.normals[ch * count + i] += float(weight * normal[ch]);
                nw[i] += weight;
            }
        }
    }
    std::size_t valid_depth = 0, valid_normal = 0;
    for (std::size_t i = 0; i < count; ++i) {
        if (need_depth && dw[i] > 0) {
            out.depth[i] /= float(dw[i]);
            ++valid_depth;
        }
        if (need_normals && nw[i] > 0) {
            Eigen::Vector3d n{out.normals[i], out.normals[count + i], out.normals[2 * count + i]};
            if (n.norm() > 1e-6) {
                n.normalize();
                ++valid_normal;
            } else
                n.setZero();
            for (unsigned ch = 0; ch < 3; ++ch)
                out.normals[ch * count + i] = float(n[ch]);
        }
    }
    if ((need_depth && !valid_depth) || (need_normals && !valid_normal))
        throw std::runtime_error(
            "Native prior has no valid requested geometry; check masks and sparse scale support");
    out.validate();
    return out;
}
} // namespace photara::splat::detail
