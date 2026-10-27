#include "splat/formats.hpp"

#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <iostream>
#include <string>
#include <stdexcept>
#include <vector>

using namespace photara::splat;

void require(bool ok, const char* message) {
    if (!ok) throw std::runtime_error(message);
}

GaussianModel fixture(
    const std::size_t count, const unsigned degree,
    const std::vector<float>& means) {
    constexpr auto cpu = tinytensor::Device::CPU;
    GaussianModel model;
    model.sh_degree = degree;
    const std::size_t bases = (degree + 1U) * (degree + 1U);
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, cpu);
    model.log_scales =
        tinytensor::Tensor::from_vector(std::vector<float>(count * 3U, -1.F), {count, 3U}, cpu);
    std::vector<float> rotations(count * 4U, 0.F);
    for (std::size_t index = 0; index < count; ++index) rotations[index * 4U] = 1.F;
    model.quaternions = tinytensor::Tensor::from_vector(rotations, {count, 4U}, cpu);
    model.opacity_logits = tinytensor::Tensor::from_vector(
        std::vector<float>(count, -2.F), {count, 1U}, cpu);
    std::vector<float> sh(count * bases * 3U);
    for (std::size_t index = 0; index < sh.size(); ++index)
        sh[index] = static_cast<float>(static_cast<int>(index % 17) - 8) * 0.03F;
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, cpu);
    return model;
}

void near(
    const std::vector<float>& actual, const std::vector<float>& expected,
    const float tolerance, const char* message) {
    require(actual.size() == expected.size(), message);
    for (std::size_t index = 0; index < actual.size(); ++index)
        require(std::abs(actual[index] - expected[index]) <= tolerance, message);
}

std::array<double,16> test_sh_basis(double x, double y, double z) {
    const double xx=x*x, yy=y*y, zz=z*z;
    return {0.28209479177387814,-0.4886025119029199*y,0.4886025119029199*z,-0.4886025119029199*x,
        1.0925484305920792*x*y,-1.0925484305920792*y*z,0.31539156525252005*(2*zz-xx-yy),
        -1.0925484305920792*x*z,0.5462742152960396*(xx-yy),-0.5900435899266435*y*(3*xx-yy),
        2.890611442640554*x*y*z,-0.4570457994644658*y*(4*zz-xx-yy),
        0.3731763325901154*z*(2*zz-3*xx-3*yy),-0.4570457994644658*x*(4*zz-xx-yy),
        1.445305721320277*z*(xx-yy),-0.5900435899266435*x*(xx-3*yy)};
}

void test_rotation_preserves_sh_appearance() {
    const auto model=fixture(1,3,{1.F,2.F,3.F});
    GaussianTransform turn; turn.rotation_degrees={0.F,0.F,90.F};
    const auto transformed=transform_gaussians(model,turn);
    const auto original=model.sh.to_vector(), rotated=transformed.sh.to_vector();
    for (const auto& direction : std::array<std::array<double,3>,4>{{
            {1.,0.,0.}, {0.,1.,0.}, {0.,0.,1.}, {0.2672612419,0.5345224838,0.8017837257}}}) {
        const auto world=test_sh_basis(direction[0],direction[1],direction[2]);
        const auto local=test_sh_basis(direction[1],-direction[0],direction[2]);
        for (int c=0;c<3;++c) {
            double a=0,b=0;
            for (int i=0;i<16;++i) { a+=world[i]*rotated[3*i+c]; b+=local[i]*original[3*i+c]; }
            require(std::abs(a-b)<1e-5, "SH appearance must follow 3D rotation through degree three");
        }
    }
    turn.rotation_degrees={30.F,-22.F,87.F};
    const auto arbitrary=transform_gaussians(model,turn).sh.to_vector();
    constexpr double pi=3.14159265358979323846;
    const double a=30.*pi/180., b=-22.*pi/180., c=87.*pi/180.;
    const double x=.2672612419, y=.5345224838, z=.8017837257;
    const double zx=std::cos(c)*x+std::sin(c)*y, zy=-std::sin(c)*x+std::cos(c)*y;
    const double yx=std::cos(b)*zx-std::sin(b)*z, yz=std::sin(b)*zx+std::cos(b)*z;
    const auto world=test_sh_basis(x,y,z);
    const auto local=test_sh_basis(yx,std::cos(a)*zy+std::sin(a)*yz,-std::sin(a)*zy+std::cos(a)*yz);
    for (int channel=0;channel<3;++channel) {
        double actual=0, expected=0;
        for(int i=0;i<16;++i) { actual+=world[i]*arbitrary[3*i+channel]; expected+=local[i]*original[3*i+channel]; }
        require(std::abs(actual-expected)<1e-5, "Arbitrary rotation must preserve directional appearance");
    }
}

void test_transform_and_merge() {
    // Degree-0 model (2 gaussians) merged after a scaled, translated
    // degree-1 model (3 gaussians).
    GaussianModel first = fixture(2, 0, {0.F, 0.F, 0.F, 1.F, 2.F, 3.F});
    const std::vector<float> second_means{
        5.F, 6.F, 7.F, -1.F, -2.F, -3.F, 0.5F, 1.5F, 2.5F};
    GaussianModel second = fixture(3, 1, second_means);
    const std::vector<float> filter{0.25F, 0.5F, 1.F};
    constexpr auto cpu = tinytensor::Device::CPU;
    second.filter_3d =
        tinytensor::Tensor::from_vector(filter, {3U, 1U}, cpu);

    GaussianTransform transform;
    transform.translation = {10.F, -2.F, 4.F};
    transform.scale = 2.F;
    const GaussianModel placed = transform_gaussians(second, transform);

    near(
        placed.means.to_vector(),
        {20.F, 10.F, 18.F, 8.F, -6.F, -2.F, 11.F, 1.F, 9.F}, 1e-5F,
        "Transformed means are wrong");
    const auto placed_scales = placed.log_scales.to_vector();
    for (const float value : placed_scales)
        require(std::abs(value - (-1.F + std::log(2.F))) <= 1e-5F,
                "Transformed log scales are wrong");
    near(
        placed.filter_3d.to_vector(), {0.5F, 1.F, 2.F}, 1e-5F,
        "Transformed filter radii are wrong");
    require(transform_gaussians(second, GaussianTransform{}).means.to_vector() ==
                second.means.to_vector(),
            "Identity transform copied instead of returning the model");

    test_rotation_preserves_sh_appearance();
    // A rigid rotation applies to positions, ellipsoid orientation, normal
    // features and view-dependent SH together; colour should rotate with the object.
    GaussianModel oriented = fixture(1, 1, {1.F, 0.F, 0.F});
    oriented.normal_features = tinytensor::Tensor::from_vector({1.F,0.F,0.F,0.7F}, {1U,4U}, cpu);
    GaussianTransform turn; turn.rotation_degrees = {0.F, 0.F, 90.F};
    const auto turned = transform_gaussians(oriented, turn);
    near(turned.means.to_vector(), {0.F, 1.F, 0.F}, 1e-5F, "Rotation must move Gaussian centres");
    const float h=std::sqrt(0.5F);
    near(turned.quaternions.to_vector(), {h,0.F,0.F,h}, 1e-5F, "Rotation must rotate Gaussian ellipsoids");
    near(turned.normal_features.to_vector(), {0.F,1.F,0.F,0.7F}, 1e-5F, "Rotation must rotate stored normals");
    const auto before_sh=oriented.sh.to_vector(), after_sh=turned.sh.to_vector();
    for (int c=0; c<3; ++c) {
        require(std::abs(after_sh[c]-before_sh[c])<1e-6F, "DC colour must stay unchanged");
        require(std::abs(after_sh[3+c]-before_sh[9+c])<1e-5F &&
                std::abs(after_sh[6+c]-before_sh[6+c])<1e-5F &&
                std::abs(after_sh[9+c]+before_sh[3+c])<1e-5F,
                "View-dependent colour must rotate with the model");
    }
    const GaussianModel merged = merge_gaussians({first, placed});
    require(merged.size() == 5, "Merged Gaussian count is wrong");
    require(merged.sh_degree == 1, "Merged SH degree is wrong");
    require(merged.sh.shape()[1] == 4, "Merged SH was not padded to 4 bases");
    const auto merged_means = merged.means.to_vector();
    const auto merged_opacities = merged.opacity_logits.to_vector();
    const auto merged_sh = merged.sh.to_vector();
    require(merged_means[0] == 0.F && merged_means[3] == 1.F,
            "Merged means were reordered");
    require(std::abs(merged_means[6] - 20.F) <= 1e-4F,
            "Merged means were reordered");
    require(std::abs(merged_opacities[0] + 2.F) <= 1e-6F &&
                std::abs(merged_opacities[4] + 2.F) <= 1e-6F,
            "Merged opacities were reordered");
    // The degree-0 model keeps its DC colour in the padded layout while its
    // higher bands stay zero; the degree-1 rows follow unchanged.
    const auto first_sh = first.sh.to_vector();
    const auto placed_sh = placed.sh.to_vector();
    for (std::size_t gaussian = 0; gaussian < 2; ++gaussian) {
        for (std::size_t component = 0; component < 3U; ++component)
            require(std::abs(merged_sh[(gaussian * 4U) * 3U + component] -
                             first_sh[gaussian * 3U + component]) <= 1e-6F,
                    "Merged DC colour changed");
        for (std::size_t base = 1; base < 4U; ++base)
            for (std::size_t component = 0; component < 3U; ++component)
                require(merged_sh[(gaussian * 4U + base) * 3U + component] ==
                            0.F,
                        "Merged higher SH bands were not zero-padded");
    }
    near(placed_sh,
         std::vector<float>(
             merged_sh.begin() + static_cast<std::ptrdiff_t>(2 * 4U * 3U),
             merged_sh.end()),
         0.F, "Merged higher-degree rows changed");
    // Only one input carried filter_3d, so the merged model must drop it.
    require(!merged.filter_3d.is_valid(),
            "Merged model kept a partial filter field");

    bool rejected = false;
    try { (void)merge_gaussians({}); } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "Merging an empty list was accepted");
    rejected = false;
    GaussianTransform bad;
    bad.scale = 0.F;
    try { (void)transform_gaussians(second, bad); }
    catch (const std::runtime_error&) { rejected = true; }
    require(rejected, "A non-positive transform scale was accepted");
}

void test_merge_optional_fields_and_roundtrip() {
    constexpr auto cpu = tinytensor::Device::CPU;
    GaussianModel first = fixture(2, 1, {0.F, 0.F, 0.F, 1.F, 1.F, 1.F});
    GaussianModel second = fixture(3, 2, {2.F, 0.F, 0.F, 3.F, 0.F, 0.F, 4.F, 0.F, 0.F});
    first.normal_features = tinytensor::Tensor::from_vector(
        std::vector<float>(2 * 4U, 0.5F), {2U, 4U}, cpu);
    second.normal_features = tinytensor::Tensor::from_vector(
        std::vector<float>(3 * 4U, 0.25F), {3U, 4U}, cpu);
    first.filter_3d = tinytensor::Tensor::from_vector(
        std::vector<float>{0.1F, 0.2F}, {2U, 1U}, cpu);
    second.filter_3d = tinytensor::Tensor::from_vector(
        std::vector<float>(3, 0.3F), {3U, 1U}, cpu);

    const GaussianModel merged = merge_gaussians({first, second});
    require(merged.size() == 5, "Merged count with optional fields is wrong");
    require(merged.sh_degree == 2, "Merged degree with SH2 is wrong");
    require(merged.sh.shape()[1] == 9, "Merged bases were not padded to 9");
    require(merged.normal_features.is_valid() &&
                merged.normal_features.numel() == 5 * 4U,
            "Merged normal features were dropped unexpectedly");
    require(merged.filter_3d.is_valid() && merged.filter_3d.numel() == 5,
            "Merged filter radii were dropped unexpectedly");
    const auto normals = merged.normal_features.to_vector();
    require(normals[0] == 0.5F && normals[2 * 4U] == 0.25F,
            "Merged normal features were reordered");

    const auto folder = std::filesystem::temp_directory_path() /
                        ("photara-merge-" +
                         std::to_string(
                             std::chrono::steady_clock::now()
                                 .time_since_epoch()
                                 .count()));
    std::filesystem::create_directories(folder);
    struct Cleanup {
        std::filesystem::path path;
        ~Cleanup() {
            std::error_code error;
            std::filesystem::remove_all(path, error);
        }
    } cleanup{folder};
    const auto file = folder / "merged.ply";
    save_gaussians(merged, file);
    const GaussianModel loaded = load_gaussians(file, GaussianFormat::ply, cpu);
    require(loaded.size() == 5, "Merged PLY round trip changed the count");
    require(loaded.sh_degree == 2, "Merged PLY round trip changed the degree");
    near(loaded.means.to_vector(), merged.means.to_vector(), 1e-5F,
         "Merged PLY round trip changed means");
}

// Two visually distinct synthetic blobs for manual editor testing:
// `photara_splat_merge_test --write <folder>` dumps them as PLY files.
GaussianModel demo_model(
    const std::size_t count, const unsigned degree,
    const std::array<float, 3>& centre, const float radius,
    const std::array<float, 3>& colour) {
    constexpr auto cpu = tinytensor::Device::CPU;
    constexpr float c0 = 0.282094791773878F;
    const std::size_t bases = (degree + 1U) * (degree + 1U);
    std::vector<float> means, scales, rotations, opacities, sh;
    means.reserve(count * 3U);
    scales.reserve(count * 3U);
    rotations.reserve(count * 4U);
    opacities.reserve(count);
    sh.reserve(count * bases * 3U);
    // Golden-angle spiral: an even, deterministic shell of centres.
    const float golden = 3.14159265358979F * (3.F - std::sqrt(5.F));
    for (std::size_t index = 0; index < count; ++index) {
        const float y = 1.F - 2.F * (static_cast<float>(index) + 0.5F) /
                             static_cast<float>(count);
        const float r = std::sqrt(std::max(0.F, 1.F - y * y));
        const float theta = golden * static_cast<float>(index);
        means.push_back(centre[0] + radius * r * std::cos(theta));
        means.push_back(centre[1] + radius * y);
        means.push_back(centre[2] + radius * r * std::sin(theta));
        scales.insert(scales.end(), 3, std::log(0.035F));
        rotations.insert(rotations.end(), {1.F, 0.F, 0.F, 0.F});
        opacities.push_back(2.5F);
        for (std::size_t base = 0; base < bases; ++base)
            for (std::size_t component = 0; component < 3U; ++component) {
                if (base == 0) {
                    sh.push_back((colour[component] - 0.5F) / c0);
                } else {
                    // Small deterministic higher-band detail.
                    sh.push_back(
                        0.6F * std::sin(
                                   static_cast<float>(index % 31) +
                                   static_cast<float>(base * 3U + component)));
                }
            }
    }
    GaussianModel model;
    model.means = tinytensor::Tensor::from_vector(means, {count, 3U}, cpu);
    model.log_scales =
        tinytensor::Tensor::from_vector(scales, {count, 3U}, cpu);
    model.quaternions =
        tinytensor::Tensor::from_vector(rotations, {count, 4U}, cpu);
    model.opacity_logits =
        tinytensor::Tensor::from_vector(opacities, {count, 1U}, cpu);
    model.sh = tinytensor::Tensor::from_vector(sh, {count, bases, 3U}, cpu);
    model.sh_degree = degree;
    return model;
}

void write_demo_models(const std::filesystem::path& folder) {
    std::filesystem::create_directories(folder);
    // Red SH0 blob at the origin and a teal SH1 blob beside it, so the two
    // models stay recognizable after import, placement, and merge.
    save_gaussians(
        demo_model(3000, 0, {0.F, 0.F, 0.F}, 0.8F, {0.92F, 0.16F, 0.10F}),
        folder / "demo_red_sh0.ply", GaussianFormat::ply);
    save_gaussians(
        demo_model(3000, 1, {2.2F, 0.3F, 0.F}, 0.5F, {0.10F, 0.62F, 0.92F}),
        folder / "demo_teal_sh1.ply", GaussianFormat::ply);
}

int main(const int argc, char** argv) {
    try {
        if (argc == 3 && std::string(argv[1]) == "--write") {
            const auto folder = std::filesystem::path(argv[2]);
            write_demo_models(folder);
            std::cout << "Demo models written to "
                      << std::filesystem::absolute(folder).string() << '\n';
            return 0;
        }
        test_transform_and_merge();
        test_merge_optional_fields_and_roundtrip();
        std::cout << "Gaussian transform/merge tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
