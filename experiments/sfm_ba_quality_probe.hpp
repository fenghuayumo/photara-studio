#pragma once

#include "sfm/appearance.hpp"
#include "sfm/asfm.hpp"
#include "sfm/bundle.hpp"
#include "sfm/export_colmap.hpp"

#include <cmath>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>

// Offline self-calibration ablations. No reference camera/point data enters
// these solves; all variants start from the same native reconstruction.
inline void probe_ba_quality(photara::sfm::Scene& scene,
                             const std::filesystem::path& output,
                             const std::string& mode) {
    using namespace photara::sfm;
    const bool square = mode == "polish-square" || mode == "polish-pinhole" ||
                        mode == "polish-per-view";
    const bool pinhole = mode == "polish-pinhole" || mode == "polish-per-view";
    if (square) for (auto& camera : scene.cameras) {
        camera.fx = camera.fy = std::sqrt(camera.fx * camera.fy);
        if (pinhole) camera.k1 = camera.k2 = camera.p1 = camera.p2 = 0;
    }
    if (mode == "polish-per-view") {
        const auto shared = scene.cameras;
        scene.cameras.clear();
        for (auto& image : scene.images) {
            auto camera = shared.at(image.camera_id);
            camera.id = static_cast<Index>(scene.cameras.size());
            image.camera_id = camera.id;
            scene.cameras.push_back(camera);
        }
    }
    BundleOptions options;
    options.optimizer.maximum_iterations = 100;
    options.optimizer.huber_delta = 1;
    options.optimizer.optimize_focal = true;
    options.optimizer.optimize_aspect_ratio = !square;
    options.optimizer.optimize_distortion = !pinhole;
    options.optimizer.focal_prior_weight = mode == "polish" ? 5.0 : 0.05;
    options.gate_intrinsics_by_observability = mode != "polish-per-view";
    const auto result = run_bundle_adjustment(scene, options);
    if (!result.success) throw std::runtime_error("Quality probe BA failed");
    std::cout << "quality_probe_mode=" << mode << '\n'
              << "quality_probe_initial_cost=" << result.optimizer.initial_cost << '\n'
              << "quality_probe_final_cost=" << result.optimizer.final_cost << '\n'
              << "quality_probe_iterations=" << result.optimizer.iterations.size() << '\n';
    colour_triangulated_tracks(scene);
    save_asfm(scene, output.string()+".asfm");
    save_colmap_text(scene, output.string()+"_colmap");
}
