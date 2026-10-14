#include "formats_usdc.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#ifdef PHOTARA_HAS_USDC
#include "next/load-usd.hh"
#include "next/writer/usdc-writer.hh"
#endif

namespace photara::splat::detail {
#ifdef PHOTARA_HAS_USDC
namespace {
namespace usd = lightusd::next;
constexpr const char *sh_name = "radiance:sphericalHarmonicsCoefficients";
constexpr const char *degree_name = "radiance:sphericalHarmonicsDegree";
void finite(const std::vector<float> &values) {
    for (float value : values)
        if (!std::isfinite(value))
            throw std::runtime_error("USDC refuses non-finite Gaussian values");
}
void attribute(usd::PrimSpec &prim, const std::string &name, const std::string &type,
               usd::Value value, bool vertex = false, int elements = 1) {
    prim.add_property(name, std::move(value),
                      name == degree_name ? usd::PropSlot::kFlagUniform : 0);
    prim.set_property_type_name(name, type);
    if (vertex) {
        auto &meta = prim.ensure_property_meta(name);
        meta.interpolation = "vertex";
        meta.elementSize = elements;
        meta.authored |= usd::PropMeta::kInterpolation | usd::PropMeta::kElementSize;
    }
}
std::vector<float> array(const usd::PrimSpec &p, const char *name, std::size_t components) {
    const auto *value = p.property_value(name);
    if (!value || !value->is_array())
        throw std::runtime_error(std::string("USDC missing ") + name);
    const auto *data = value->as_float_array();
    if (!data || data->empty() || data->size() % components)
        throw std::runtime_error(std::string("Invalid USDC array ") + name);
    finite(*data);
    return *data;
}
} // namespace
#endif

std::vector<std::uint8_t> save_usdc(const GaussianModel &model) {
#ifdef PHOTARA_HAS_USDC
    const auto n = model.size();
    const auto degree = model.sh_degree;
    const std::size_t bases = (degree + 1) * (degree + 1);
    if (!n || degree > 3 || n > std::numeric_limits<std::uint32_t>::max() / bases)
        throw std::runtime_error("Invalid USDC Gaussian count or SH degree");
    auto positions = model.means.to_vector(), scales = model.log_scales.to_vector();
    auto quats = model.quaternions.to_vector(), opacity = model.opacity_logits.to_vector();
    auto sh = model.sh.to_vector();
    if (positions.size() != n * 3 || scales.size() != n * 3 || quats.size() != n * 4 ||
        opacity.size() != n || sh.size() != n * bases * 3)
        throw std::runtime_error("Invalid USDC Gaussian model shape");
    for (const auto *values : {&positions, &scales, &quats, &opacity, &sh})
        finite(*values);
    Eigen::Vector3f low = Eigen::Vector3f::Constant(std::numeric_limits<float>::infinity());
    Eigen::Vector3f high = -low;
    std::vector<float> normalized_quats(n * 4), color(n * 3);
    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Quaternionf q(quats[4 * i], quats[4 * i + 1], quats[4 * i + 2], quats[4 * i + 3]);
        if (!std::isfinite(q.norm()) || q.norm() < 1e-12F)
            throw std::runtime_error("Invalid USDC quaternion");
        q.normalize();
        normalized_quats[i * 4] = q.w();
        normalized_quats[i * 4 + 1] = q.x();
        normalized_quats[i * 4 + 2] = q.y();
        normalized_quats[i * 4 + 3] = q.z();
        Eigen::Vector3f s;
        for (unsigned k = 0; k < 3; ++k) {
            scales[i * 3 + k] = std::exp(scales[i * 3 + k]);
            s[k] = scales[i * 3 + k];
            if (!std::isfinite(s[k]) || s[k] <= 0)
                throw std::runtime_error("Invalid USDC Gaussian scale");
            color[i * 3 + k] =
                std::clamp(0.5F + 0.28209479177387814F * sh[(i * bases) * 3 + k], 0.F, 1.F);
        }
        // The crate's GfQuatf tuple is scalar-first. LightUSD's flat-array
        // writer passes these lanes through; do not use its scalar xyzw API.
        Eigen::Matrix3f rs = q.toRotationMatrix() * s.asDiagonal();
        Eigen::Vector3f extent = 3.F * rs.cwiseProduct(rs).rowwise().sum().cwiseSqrt();
        Eigen::Vector3f p(positions[i * 3], positions[i * 3 + 1], positions[i * 3 + 2]);
        if (!extent.allFinite())
            throw std::runtime_error("USDC Gaussian bounds overflow");
        low = low.cwiseMin(p - extent);
        high = high.cwiseMax(p + extent);
        float v = opacity[i];
        opacity[i] = v >= 0 ? 1.F / (1.F + std::exp(-v)) : std::exp(v) / (1.F + std::exp(v));
    }
    if (!low.allFinite() || !high.allFinite())
        throw std::runtime_error("USDC scene bounds overflow");
    usd::Layer layer;
    auto id = layer.define_prim_at_path("/Gaussians", "ParticleField3DGaussianSplat");
    auto *p = layer.prim_mutable(id);
    if (!p)
        throw std::runtime_error("Cannot create USD Gaussian prim");
    attribute(*p, "positions", "point3f[]",
              usd::Value::MakeFloatCompArray(std::move(positions), usd::TypeId::Point3f, 3));
    attribute(*p, "scales", "float3[]", usd::Value::MakeFloat3Array(std::move(scales)));
    attribute(*p, "orientations", "quatf[]",
              usd::Value::MakeFloatCompArray(std::move(normalized_quats), usd::TypeId::Quatf, 4));
    attribute(*p, "opacities", "float[]", usd::Value::MakeFloatArray(opacity));
    attribute(*p, degree_name, "int", usd::Value(static_cast<std::int32_t>(degree)));
    attribute(*p, sh_name, "float3[]", usd::Value::MakeFloat3Array(std::move(sh)), true,
              static_cast<int>(bases));
    attribute(*p, "primvars:displayColor", "color3f[]",
              usd::Value::MakeFloatCompArray(std::move(color), usd::TypeId::Color3f, 3), true);
    attribute(*p, "primvars:displayOpacity", "float[]",
              usd::Value::MakeFloatArray(std::move(opacity)), true);
    attribute(
        *p, "extent", "float3[]",
        usd::Value::MakeFloat3Array({low.x(), low.y(), low.z(), high.x(), high.y(), high.z()}));
    layer.meta().defaultPrim = "Gaussians";
    layer.meta().defaultPrim_set = true;
    layer.meta().upAxis = "Y";
    layer.meta().upAxis_set = true;
    layer.meta().metersPerUnit = 1;
    layer.meta().metersPerUnit_set = true;
    layer.meta().comment = "Photara native C++ 3D Gaussian export";
    layer.finalize();
    std::vector<std::uint8_t> bytes;
    auto result = usd::WriteLayerToUSDCMemory(bytes, layer);
    if (!result.success)
        throw std::runtime_error("USDC encoding failed: " + result.error);
    if (bytes.size() < 88 || std::string(bytes.begin(), bytes.begin() + 8) != "PXR-USDC")
        throw std::runtime_error("USDC encoder did not produce a crate file");
    return bytes;
#else
    (void)model;
    throw std::runtime_error("USDC support requires PHOTARA_ENABLE_USDC=ON");
#endif
}

GaussianModel load_usdc(std::span<const std::uint8_t> bytes, tinytensor::Device device) {
#ifdef PHOTARA_HAS_USDC
    usd::Stage stage;
    std::string warn, error;
    if (!usd::LoadUSDFromMemory(bytes.data(), bytes.size(), &stage, &warn, &error))
        throw std::runtime_error("Cannot read USDC: " + error);
    const auto *layer = stage.GetRootLayer();
    const auto *p = layer ? layer->prim_at_path("/Gaussians") : nullptr;
    if (!p || p->type_name() != "ParticleField3DGaussianSplat" || layer->prim_count() != 1 ||
        p->property("xformOpOrder"))
        throw std::runtime_error("USDC import supports a single untransformed /Gaussians prim");
    auto positions = array(*p, "positions", 3), scales = array(*p, "scales", 3),
         q = array(*p, "orientations", 4), op = array(*p, "opacities", 1),
         sh = array(*p, sh_name, 3);
    const auto *d = p->property_value(degree_name);
    const auto *degree = d ? d->as_int() : nullptr;
    if (!degree || *degree < 0 || *degree > 3)
        throw std::runtime_error("Invalid USDC SH degree");
    const auto n = positions.size() / 3;
    std::size_t bases = (*degree + 1) * (*degree + 1);
    if (scales.size() != n * 3 || q.size() != n * 4 || op.size() != n || sh.size() != n * bases * 3)
        throw std::runtime_error("USDC Gaussian array counts disagree");
    for (float &s : scales) {
        if (s <= 0)
            throw std::runtime_error("USDC scale must be positive");
        s = std::log(s);
    }
    std::vector<float> wxyz(q.size());
    for (std::size_t i = 0; i < n; ++i) {
        Eigen::Quaternionf rotation(q[i * 4], q[i * 4 + 1], q[i * 4 + 2], q[i * 4 + 3]);
        if (rotation.norm() < 1e-12F)
            throw std::runtime_error("Invalid USDC quaternion");
        rotation.normalize();
        wxyz[i * 4] = rotation.w();
        wxyz[i * 4 + 1] = rotation.x();
        wxyz[i * 4 + 2] = rotation.y();
        wxyz[i * 4 + 3] = rotation.z();
    }
    for (float &o : op) {
        if (o < 0 || o > 1)
            throw std::runtime_error("USDC opacity must be in [0,1]");
        o = std::clamp(o, 1e-7F, 1.F - 1e-7F);
        o = std::log(o / (1 - o));
    }
    GaussianModel m;
    m.sh_degree = static_cast<unsigned>(*degree);
    m.means = tinytensor::Tensor::from_vector(positions, {n, 3}, device);
    m.log_scales = tinytensor::Tensor::from_vector(scales, {n, 3}, device);
    m.quaternions = tinytensor::Tensor::from_vector(wxyz, {n, 4}, device);
    m.opacity_logits = tinytensor::Tensor::from_vector(op, {n, 1}, device);
    m.sh = tinytensor::Tensor::from_vector(sh, {n, bases, 3}, device);
    return m;
#else
    (void)bytes;
    (void)device;
    throw std::runtime_error("USDC support requires PHOTARA_ENABLE_USDC=ON");
#endif
}
} // namespace photara::splat::detail
