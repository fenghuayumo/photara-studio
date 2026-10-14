#include "core/logging.hpp"
#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/submap_recovery.hpp"
#include <fstream>
#include <iostream>
#include <queue>

// Development-only probe. It accepts no reference poses and never modifies the
// input checkpoint; existing recovery gates preserve the frozen stable map.
int main(int argc, char** argv) {
    try {
        using namespace photara::sfm;
        if (argc != 3) throw std::invalid_argument("Usage: photara_sfm_position_recovery_probe reconstruction-HEX.bin NEW_OUTPUT.asfm");
        const std::filesystem::path input(argv[1]), output(argv[2]);
        const auto reports = std::filesystem::path(output.string() + ".recovery.csv");
        if (std::filesystem::exists(output) || std::filesystem::exists(reports))
            throw std::invalid_argument("Recovery output already exists");
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
        if (scene.pairs.empty()) throw std::runtime_error("Recovery requires pair geometry");
        photara::core::Logger::instance().configure({}, "photara", photara::core::LogLevel::debug);
        const auto audit = analyze_position_observability(scene);
        if (audit.reliable_views < 3) throw std::runtime_error("No frozen stable map available");
        std::vector<std::vector<Index>> adjacency(scene.images.size());
        std::vector<float> strongest(scene.images.size());
        for (const auto& pair : scene.pairs)
            if (pair.active && pair.relative_pose && !pair.degenerate_planar && !pair.zero_baseline &&
                pair.id1 < scene.images.size() && pair.id2 < scene.images.size() && pair.matches.size() >= 30 &&
                std::isfinite(pair.composite_weight()))
                for (auto id : {pair.id1, pair.id2}) strongest[id] = std::max(strongest[id], pair.composite_weight());
        for (const auto& pair : scene.pairs) {
            if (!pair.active || !pair.relative_pose || pair.degenerate_planar || pair.zero_baseline || pair.matches.size() < 30 ||
                pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size() ||
                !std::isfinite(pair.composite_weight()) || pair.composite_weight() <= 0 ||
                pair.composite_weight() < 0.1F * std::min(strongest[pair.id1], strongest[pair.id2]) ||
                audit.reliable[pair.id1] || audit.reliable[pair.id2] ||
                !scene.images[pair.id1].registered || !scene.images[pair.id2].registered) continue;
            adjacency[pair.id1].push_back(pair.id2); adjacency[pair.id2].push_back(pair.id1);
        }
        std::ofstream csv(reports);
        if (!csv) throw std::runtime_error("Cannot write recovery reports");
        csv << "group,views,reconstructed,landmarks,candidate_tracks,stable_multiview_tracks,stable_triangulated_tracks,shared_points,fit_inliers,validation_points,validation_inliers,accepted,reason,names\n";
        std::vector<bool> visited(scene.images.size());
        unsigned groups = 0, recovered = 0;
        for (Index seed = 0; seed < scene.images.size(); ++seed) {
            if (visited[seed] || !scene.images[seed].registered || audit.reliable[seed]) continue;
            std::vector<Index> images;
            std::queue<Index> pending; pending.push(seed); visited[seed] = true;
            while (!pending.empty()) {
                const auto id = pending.front(); pending.pop(); images.push_back(id);
                for (auto next : adjacency[id]) if (!visited[next]) { visited[next] = true; pending.push(next); }
            }
            const auto report = recover_independent_submap(scene, images, audit.reliable);
            ++groups; if (report.accepted) recovered += static_cast<unsigned>(images.size());
            csv << groups << ',' << images.size() << ',' << report.reconstructed_views << ',' << report.local_landmarks
                << ',' << report.candidate_tracks << ',' << report.stable_multiview_tracks << ',' << report.stable_triangulated_tracks
                << ',' << report.shared_points << ',' << report.fit_inliers << ',' << report.validation_points
                << ',' << report.validation_inliers << ',' << report.accepted << ',' << report.reason << ",\"";
            for (auto id : images) {
                for (char c : scene.images[id].path.filename().string()) { if (c == '"') csv << '"'; csv << c; }
                csv << ';';
            }
            csv << "\"\n"; csv.flush();
            std::cout << "group=" << groups << " views=" << images.size() << " accepted=" << report.accepted
                      << " reason=" << report.reason << " shared_points=" << report.shared_points << '\n';
        }
        audit_position_support(scene);
        save_asfm(scene, output);
        if (!csv) throw std::runtime_error("Failed writing recovery reports");
        std::cout << "groups=" << groups << " recovered_views=" << recovered << '\n';
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
