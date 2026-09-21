#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/submap_recovery.hpp"
#include "core/logging.hpp"
#include <charconv>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <queue>
#include <stdexcept>
#include <unordered_set>

using namespace photara::sfm;

int main(int argc, char** argv) {
    try {
        photara::core::Logger::instance().configure({});
        if (argc != 3 && argc != 4) {
            std::cerr << "Usage: photara_sfm_realign reconstruction-HEX.bin NEW_OUTPUT_DIRECTORY [ADDITIONAL_RISK_NAMES.txt|--resection]\n";
            return 1;
        }
        const std::filesystem::path input(argv[1]), output(argv[2]);
        const std::string name = input.stem().string();
        const std::string prefix = "reconstruction-";
        if (!name.starts_with(prefix)) throw std::runtime_error("expected reconstruction checkpoint");
        const std::string hex = name.substr(prefix.size());
        std::uint64_t fingerprint{};
        const auto parsed = std::from_chars(hex.data(), hex.data() + hex.size(), fingerprint, 16);
        if (parsed.ec != std::errc{} || parsed.ptr != hex.data() + hex.size())
            throw std::runtime_error("invalid checkpoint fingerprint");
        if (std::filesystem::exists(output)) throw std::runtime_error("output already exists");
        Scene scene;
        CheckpointOptions options;
        options.directory = input.parent_path();
        options.write = false;
        if (!CheckpointStore(options).load_scene(CheckpointStage::reconstruction, fingerprint, scene))
            throw std::runtime_error("cannot load checkpoint");
        prune_unsupported_registrations(scene);
        if (argc == 4 && std::string_view(argv[3]) == "--resection") {
            const auto before_count = scene.registered_count();
            recover_weakly_connected_views(scene);
            std::vector<std::uint8_t> stable(scene.images.size());
            for (const auto& image : scene.images) stable[image.id] = image.registered;
            const auto recovered = recover_stable_resections(scene, stable);
            std::filesystem::create_directories(output);
            save_asfm(scene, output / "reconstruction.asfm");
            std::cout << "resection probe: registered=" << before_count << " -> "
                      << scene.registered_count() << " additional=" << recovered.size() << '\n';
            return 0;
        }
        auto audit = analyze_alignment_observability(scene);
        // Prior risk evidence must not disappear solely because added pair
        // edges close a graph cycle. The optional list adds candidates; it
        // cannot promote any camera to the stable mask or supply coordinates.
        if (argc == 4) {
            std::ifstream names_file(argv[3]);
            if (!names_file) throw std::runtime_error("cannot open risk image list");
            std::unordered_set<std::string> requested;
            std::string line;
            while (std::getline(names_file, line)) {
                if (!line.empty() && line.back() == '\r') line.pop_back();
                if (!line.empty() && !requested.insert(line).second)
                    throw std::runtime_error("duplicate risk image name");
            }
            if (requested.empty()) throw std::runtime_error("empty risk image list");
            for (const auto& image : scene.images) {
                if (!requested.erase(image.path.filename().string())) continue;
                if (!image.registered) throw std::runtime_error("risk image is not registered");
                if (audit.reliable[image.id]) {
                    audit.reliable[image.id] = 0;
                    --audit.reliable_views;
                    ++audit.unreliable_views;
                }
            }
            if (!requested.empty()) throw std::runtime_error("unknown risk image name");
        }
        std::vector<Pose3D> before;
        for (const auto& image : scene.images) before.push_back(image.pose);
        std::vector<std::vector<Index>> adjacency(scene.images.size());
        for (const auto& pair : scene.pairs) {
            if (!pair.active || !pair.relative_pose || pair.zero_baseline ||
                pair.id1 >= scene.images.size() || pair.id2 >= scene.images.size() ||
                !scene.images[pair.id1].registered || !scene.images[pair.id2].registered ||
                audit.reliable[pair.id1] || audit.reliable[pair.id2]) continue;
            adjacency[pair.id1].push_back(pair.id2);
            adjacency[pair.id2].push_back(pair.id1);
        }
        std::filesystem::create_directories(output);
        std::ofstream reports(output / "submaps.csv");
        reports << "group,views,reconstructed,local_landmarks,boundary_matches,candidate_tracks,stable_multiview_tracks,stable_triangulated_tracks,shared,fit_inliers,validation,validation_inliers,accepted,reason,seconds\n";
        std::ofstream members(output / "members.csv");
        members << "group,image_id,name\n";
        std::vector<bool> visited(scene.images.size(), false);
        unsigned group = 0, accepted = 0;
        for (Index root = 0; root < scene.images.size(); ++root) {
            if (visited[root] || !scene.images[root].registered || audit.reliable[root]) continue;
            std::vector<Index> ids;
            std::queue<Index> pending;
            pending.push(root); visited[root] = true;
            while (!pending.empty()) {
                Index id = pending.front(); pending.pop(); ids.push_back(id);
                for (Index next : adjacency[id]) if (!visited[next]) {
                    visited[next] = true; pending.push(next);
                }
            }
            std::sort(ids.begin(), ids.end());
            for (Index id : ids) members << group << ',' << id << ','
                << std::quoted(scene.images[id].path.filename().string()) << '\n';
            members.flush();
            const auto start = std::chrono::steady_clock::now();
            SubmapRecoveryOptions recovery;
            recovery.independent_model_callback = [&](const HierarchicalSubscene& sub) {
                save_asfm(sub.scene, output / ("independent_" + std::to_string(group) + ".asfm"));
            };
            auto result = recover_independent_submap(scene, ids, audit.reliable, recovery);
            const double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            reports << group++ << ',' << ids.size() << ',' << result.reconstructed_views << ','
                << result.local_landmarks << ',' << result.boundary_matches << ',' << result.candidate_tracks << ','
                << result.stable_multiview_tracks << ',' << result.stable_triangulated_tracks << ','
                << result.shared_points << ',' << result.fit_inliers << ',' << result.validation_points << ','
                << result.validation_inliers << ',' << result.accepted << ',' << result.reason << ',' << seconds << '\n';
            reports.flush();
            accepted += result.accepted ? static_cast<unsigned>(ids.size()) : 0;
            std::cout << "submap: views=" << ids.size() << " reason=" << result.reason
                << " shared=" << result.shared_points << " seconds=" << seconds << '\n';
        }
        std::ofstream poses(output / "poses.csv");
        poses << std::setprecision(17) << "image_id,name,stable,registered,before_x,before_y,before_z,after_x,after_y,after_z,displacement\n";
        for (const auto& image : scene.images) {
            poses << image.id << ',' << std::quoted(image.path.filename().string()) << ','
                << unsigned(audit.reliable[image.id]) << ',' << image.registered;
            for (int j = 0; j < 3; ++j) poses << ',' << before[image.id].C[j];
            for (int j = 0; j < 3; ++j) poses << ',' << image.pose.C[j];
            poses << ',' << (image.pose.C - before[image.id].C).norm() << '\n';
        }
        save_asfm(scene, output / "reconstruction.asfm");
        std::cout << "recovery: flagged=" << audit.unreliable_views << " accepted=" << accepted
                  << " (checkpoint-based geometry experiment; not an end-to-end benchmark)\n";
        return accepted == audit.unreliable_views ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
