#include "splat_render/renderer.hpp"

#include <cmath>
#include <iostream>
#include <stdexcept>

void require(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

int main() {
    try {
        // A front-facing flattened Gaussian at known camera depth.
        const float means[]{0.F, 0.F, 3.F};
        const float scales[]{std::log(0.4F), std::log(0.4F), std::log(0.04F)};
        const float rotations[]{1.F, 0.F, 0.F, 0.F};
        const float opacity[]{10.F};
        const float sh[]{1.F, 0.F, 0.F};
        splat_render::GaussianCloud cloud;
        cloud.count = 1;
        cloud.means = means; cloud.log_scales = scales;
        cloud.quaternions = rotations; cloud.opacity_logits = opacity; cloud.sh = sh;
        photara::vk::DeviceRequest request;
        request.want.graphics = true;
        auto device = photara::vk::Device::create(request);
        splat_render::Renderer renderer;
        renderer.attach(device);
        if (!renderer.upload(cloud, "display-test")) throw std::runtime_error(renderer.failure());
        splat_render::Camera camera;
        camera.world_to_camera = {1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1};
        camera.width = camera.height = 64;
        camera.fx = camera.fy = 64.F; camera.cx = camera.cy = 32.F;
        splat_render::DisplayOptions options;
        splat_render::FrameTarget target;
        auto draw = [&](splat_render::Shading shading) {
            if (!renderer.draw(camera, target, shading, options)) throw std::runtime_error(renderer.failure());
        };
        auto center = [&]() {
            std::vector<std::uint8_t> rgb;
            std::uint32_t width, height;
            require(renderer.download_rgb(rgb, width, height), "download failed");
            require(width == 64 && height == 64 && rgb.size() == 64 * 64 * 3, "image extent");
            const auto p = (32 * 64 + 32) * 3;
            return std::array<int, 3>{rgb[p], rgb[p + 1], rgb[p + 2]};
        };
        draw(splat_render::Shading::gaussian);
        const auto color = center();
        draw(splat_render::Shading::depth);
        const auto depth = center();
        require(color != depth, "depth must differ from RGB");
        const auto range = renderer.depth_range();
        require(range[0] > 2.F && range[0] < 4.F && range[1] > range[0], "automatic depth range");
        const int cached_slot = target.slot;
        draw(splat_render::Shading::depth);
        require(target.slot == cached_slot, "unchanged diagnostic frame must be cached");
        options.automatic_depth = false; options.depth_near = 0.F; options.depth_far = 6.F;
        options.grayscale_depth = true;
        draw(splat_render::Shading::depth);
        const auto gray = center();
        require(gray[0] == gray[1] && gray[1] == gray[2] && gray[0] > 100 && gray[0] < 155,
                "manual grayscale depth should put depth 3 midway in [0,6]");
        draw(splat_render::Shading::normal);
        const auto normal = center();
        require(std::abs(normal[0] - 128) < 10 && std::abs(normal[1] - 128) < 10 &&
                (normal[2] < 10 || normal[2] > 245), "normal must encode the plane's Z axis");
        options.world_normals = false;
        draw(splat_render::Shading::normal);
        require(center() == normal, "identity camera has identical world and camera normals");
        const auto original_camera = camera;
        const float c = std::sqrt(0.5F);
        // Rotate the camera and translate so the Gaussian remains centered.
        camera.world_to_camera = {c,0,-c,0, 0,1,0,0, c,0,c,0, -3*c,0,3-3*c,1};
        camera.position = {3*c, 0, 3-3*c};
        draw(splat_render::Shading::normal);
        const auto camera_normal = center();
        options.world_normals = true;
        draw(splat_render::Shading::normal);
        const auto world_normal = center();
        require(std::abs(world_normal[0] - 128) < 10 &&
                std::abs(camera_normal[0] - world_normal[0]) > 50,
                "rotated camera normals must transform back to the world Z axis");
        camera = original_camera;
        draw(splat_render::Shading::rings);
        draw(splat_render::Shading::gaussian);
        require(center() == color, "switching back must restore RGB");
        std::cout << "RGB, depth, normal, display settings and frame cache passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
