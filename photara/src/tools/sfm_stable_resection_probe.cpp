#include "core/logging.hpp"
#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/submap_recovery.hpp"
#include <fstream>
#include <iostream>

// Development-only bounded ancestry experiment. Each pass freezes its anchor
// mask; newly accepted views are promoted only for the NEXT pass. Original
// stable poses/intrinsics remain fixed and no reference model is read.
int main(int argc, char** argv) {
    try {
        using namespace photara::sfm;
        if (argc != 3) throw std::invalid_argument("Usage: photara_sfm_stable_resection_probe reconstruction-HEX.bin NEW_OUTPUT.asfm");
        const std::filesystem::path input(argv[1]), output(argv[2]);
        const auto reports = std::filesystem::path(output.string() + ".rounds.csv");
        if (std::filesystem::exists(output) || std::filesystem::exists(reports))
            throw std::invalid_argument("Resection output already exists");
        const auto stem = input.stem().string();
        if (!stem.starts_with("reconstruction-")) throw std::invalid_argument("A final graph checkpoint is required");
        std::size_t consumed = 0;
        const auto key = std::stoull(stem.substr(15), &consumed, 16);
        if (consumed != stem.size() - 15) throw std::invalid_argument("Invalid reconstruction checkpoint key");
        CheckpointOptions checkpoint;
        checkpoint.directory = input.parent_path(); checkpoint.write = false;
        Scene scene;
        if (!CheckpointStore(checkpoint).load_scene(CheckpointStage::reconstruction, key, scene))
            throw std::runtime_error("Cannot load final graph checkpoint");
        if (scene.pairs.empty()) throw std::runtime_error("Resection requires pair geometry");
        photara::core::Logger::instance().configure({}, "photara", photara::core::LogLevel::debug);
        const auto audit = analyze_position_observability(scene);
        if (audit.reliable_views < 3) throw std::runtime_error("No frozen stable map available");
        auto stable = audit.reliable;
        const auto cameras = scene.cameras;
        std::vector<Pose3D> original;
        for (const auto& image : scene.images) original.push_back(image.pose);
        std::ofstream csv(reports);
        if (!csv) throw std::runtime_error("Cannot write resection reports");
        csv << "round,frozen_anchors,accepted,names\n";
        unsigned total = 0;
        for (unsigned round = 0; round < 3; ++round) {
            const auto recovered = recover_stable_resections(scene, stable);
            unsigned anchors = 0;
            for (auto flag : stable) anchors += flag != 0;
            csv << round << ',' << anchors << ',' << recovered.size() << ",\"";
            for (auto id : recovered) {
                for (char c : scene.images[id].path.filename().string()) { if (c == '"') csv << '"'; csv << c; }
                csv << ';';
            }
            csv << "\"\n"; csv.flush();
            for (Index id = 0; id < scene.images.size(); ++id)
                if (audit.reliable[id] && ((scene.images[id].pose.C - original[id].C).norm() != 0 ||
                                          (scene.images[id].pose.R - original[id].R).norm() != 0))
                    throw std::runtime_error("Original stable pose changed during resection");
            for (Index id = 0; id < scene.cameras.size(); ++id) {
                const auto& a = cameras[id]; const auto& b = scene.cameras[id];
                if (a.fx != b.fx || a.fy != b.fy || a.cx != b.cx || a.cy != b.cy ||
                    a.k1 != b.k1 || a.k2 != b.k2 || a.p1 != b.p1 || a.p2 != b.p2 || a.model != b.model)
                    throw std::runtime_error("Camera intrinsics changed during resection");
            }
            std::cout << "round=" << round << " frozen_anchors=" << anchors << " accepted=" << recovered.size() << '\n';
            if (recovered.empty()) break;
            total += static_cast<unsigned>(recovered.size());
            for (auto id : recovered) stable[id] = 1;
        }
        audit_position_support(scene);
        save_asfm(scene, output);
        if (!csv) throw std::runtime_error("Failed writing resection reports");
        std::cout << "recovered_views=" << total << " original_anchors_unchanged=1 intrinsics_unchanged=1\n";
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
