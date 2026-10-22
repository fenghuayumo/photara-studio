// Refine an existing native map without changing calibration or registration.
#include "sfm/asfm.hpp"
#include "sfm/bundle.hpp"
#include "sfm/triangulation.hpp"
#include "core/logging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <cmath>
#include <string>

int main(int argc, char** argv) {
    using namespace photara::sfm;
    try {
        if (argc < 3) throw std::runtime_error("input-full.asfm new-output.asfm [--precision] [--huber=positive-pixels] [--retriangulate]");
        bool retriangulate = false;
        BundleOptions options;
        options.optimizer.maximum_iterations = 100;
        options.optimizer.huber_delta = 1;
        for (int i = 3; i < argc; ++i) {
            const std::string arg(argv[i]);
            if (arg == "--retriangulate") {
                retriangulate = true;
            } else if (arg == "--precision") {
                options.optimizer.maximum_iterations = 150;
                options.optimizer.maximum_pcg_iterations = 500;
                options.optimizer.pcg_tolerance = 1e-6;
                options.optimizer.function_tolerance = 1e-9;
            } else if (arg.starts_with("--huber=")) {
                const auto value = arg.substr(8);
                std::size_t consumed = 0;
                const double delta = std::stod(value, &consumed);
                if (consumed != value.size() || !std::isfinite(delta) || delta <= 0)
                    throw std::runtime_error("Huber threshold must be finite and positive");
                options.optimizer.huber_delta = delta;
            } else throw std::runtime_error("unknown option: " + arg);
        }
        const std::filesystem::path input(argv[1]), output(argv[2]);
        auto diagnostics = output;
        diagnostics.replace_extension();
        diagnostics += "_sfm_diagnostics.csv";
        if (std::filesystem::exists(output) || std::filesystem::exists(diagnostics))
            throw std::runtime_error("output already exists");
        photara::core::Logger::instance().configure(output.parent_path(), "map-refinement");
        Scene scene = load_asfm(input);
        const auto registered = scene.registered_count();
        if (retriangulate) {
            TriangulationOptions triangulation;
            triangulation.reproj_threshold_px = 2;
            triangulation.min_angle_deg = 1;
            triangulation.split_tracks = false;
            std::size_t before = 0, after = 0;
            for (const auto& track : scene.tracks) before += track.num_inliers;
            triangulate_tracks(scene, false, triangulation);
            for (const auto& track : scene.tracks) after += track.num_inliers;
            photara::core::Logger::instance().info("refinement retriangulation: inliers_before=",
                before, " inliers_after=", after);
        }
        // The retained self-calibration is already established. Weak branch
        // constraints must improve geometry rather than change the lens model.
        options.write_intrinsics = false;
        const auto result = run_bundle_adjustment(scene, options);
        if (!result.success || result.optimizer.final_cost > result.optimizer.initial_cost)
            throw std::runtime_error("joint refinement did not produce a usable lower-cost map");
        if (scene.registered_count() != registered)
            throw std::runtime_error("refinement changed registration");
        std::ofstream csv(diagnostics);
        csv << std::setprecision(17)
            << "image_id,name,registered,camera_id,width,height,fx,fy,cx,cy,k1,k2,p1,p2,center_x,center_y,center_z,qw,qx,qy,qz,camera_model\n";
        for (const auto& im : scene.images) {
            const auto& c = scene.camera_of(im);
            const auto q = im.pose.quaternion();
            csv << im.id << ',' << std::quoted(im.path.filename().string()) << ',' << im.registered << ','
                << im.camera_id << ',' << c.width << ',' << c.height << ',' << c.fx << ',' << c.fy << ','
                << c.cx << ',' << c.cy << ',' << c.k1 << ',' << c.k2 << ',' << c.p1 << ',' << c.p2 << ','
                << im.pose.C.x() << ',' << im.pose.C.y() << ',' << im.pose.C.z() << ','
                << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
                << (c.model == photara::CameraModel::opencv_fisheye ? "opencv_fisheye" :
                    c.model == photara::CameraModel::equirectangular ? "equirectangular" : "pinhole") << '\n';
        }
        if (!csv) throw std::runtime_error("cannot write refinement diagnostics");
        save_asfm(scene, output);
        std::cout << std::setprecision(17) << "registered=" << registered
                  << " huber=" << options.optimizer.huber_delta
                  << " pcg_limit=" << options.optimizer.maximum_pcg_iterations
                  << " termination=" << static_cast<int>(result.optimizer.termination)
                  << " initial_cost=" << result.optimizer.initial_cost
                  << " final_cost=" << result.optimizer.final_cost
                  << " iterations=" << result.optimizer.iterations.size() << '\n';
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
