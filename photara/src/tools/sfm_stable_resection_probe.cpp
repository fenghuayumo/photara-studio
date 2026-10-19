#include "core/logging.hpp"
#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/submap_recovery.hpp"
#include <fstream>
#include <iostream>
#include <cstdlib>
#include "../../../experiments/sfm_missing_group_probe.hpp"
#include "../../../experiments/sfm_boundary_support_probe.hpp"
#include "../../../experiments/sfm_local_boundary_audit.hpp"
#include "../../../experiments/sfm_structureless_probe.hpp"
#include "../../../experiments/sfm_cross_anchor_precision.hpp"

// Development-only bounded ancestry experiment. Each pass freezes its anchor
// mask; newly accepted views are promoted only for the NEXT pass. Original
// stable poses/intrinsics remain fixed and no reference model is read.
int main(int argc, char** argv) {
    try {
        using namespace photara::sfm;
        if (argc != 3 && !(argc == 4 && (std::string_view(argv[3]) == "--cross-anchor-precision" || std::string_view(argv[3]) == "--structureless" || std::string_view(argv[3]) == "--audit-local-boundary" || std::string_view(argv[3]) == "--descriptor-map" || std::string_view(argv[3]) == "--groups" || std::string_view(argv[3]) == "--audit-pairs" || std::string_view(argv[3]) == "--inactive-links" || std::string_view(argv[3]) == "--retained-poses")))
            throw std::invalid_argument("Usage: photara_sfm_stable_resection_probe reconstruction-HEX.bin NEW_OUTPUT.asfm [--cross-anchor-precision|--structureless|--descriptor-map|--groups|--audit-pairs|--audit-local-boundary|--inactive-links|--retained-poses]");
        const bool groups = argc == 4 && std::string_view(argv[3]) == "--groups";
        const bool descriptor_map = argc == 4;
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
        if (argc == 4 && std::string_view(argv[3]) == "--cross-anchor-precision") {
            const auto* cache=std::getenv("PHOTARA_GROUP_FRONTEND_CACHE");
            if(!cache)throw std::invalid_argument("Precision probe requires PHOTARA_GROUP_FRONTEND_CACHE");
            std::ofstream csv(reports);
            if(!csv)throw std::runtime_error("Cannot write precision report");
            photara::sfm::experiment::probe_cross_anchor_precision(scene,cache,csv);
            std::cout<<"precision_audit_only=1 parent_geometry_unchanged=1 registered="<<scene.registered_count()<<'\n';
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "--structureless") {
            const auto* cache=std::getenv("PHOTARA_GROUP_FRONTEND_CACHE");
            if (!cache) throw std::invalid_argument("Structure-less probe requires PHOTARA_GROUP_FRONTEND_CACHE");
            const auto anchors_path=std::filesystem::path(output.string()+".anchors.csv");
            if (std::filesystem::exists(anchors_path)) throw std::invalid_argument("Anchor report already exists");
            std::ofstream csv(reports),anchors_csv(anchors_path);
            std::ofstream completion_csv;
            if(std::getenv("PHOTARA_TRACK_COMPLETION")) {
                const auto path=std::filesystem::path(output.string()+".tracks.csv");
                if(std::filesystem::exists(path))throw std::invalid_argument("Completion report already exists");
                completion_csv.open(path);
                if(!completion_csv)throw std::runtime_error("Cannot write completion report");
            }
            std::filesystem::path windows,exports;
            if(completion_csv.is_open()) {
                exports=output.string()+".candidates";
                if(!std::filesystem::create_directory(exports))throw std::invalid_argument("Candidates output already exists");
            }
            if(std::getenv("PHOTARA_BOUNDARY_WINDOWS")) {
                windows=output.string()+".windows";
                if(!std::filesystem::create_directory(windows))throw std::invalid_argument("Windows output already exists");
            }
            photara::sfm::experiment::probe_structureless_caches(scene,cache,csv,anchors_csv,
                completion_csv.is_open()?&completion_csv:nullptr,windows,exports);
            std::cout << "pose_hypotheses_only=1 parent_geometry_unchanged=1 registered="<<scene.registered_count()<<'\n';
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "--audit-local-boundary") {
            const auto* cache=std::getenv("PHOTARA_GROUP_FRONTEND_CACHE");
            if (!cache) throw std::invalid_argument("Local audit requires PHOTARA_GROUP_FRONTEND_CACHE");
            std::ofstream csv(reports);
            audit_local_boundary(scene,cache,csv);
            std::cout << "audit_only=1 parent_geometry_unchanged=1\n";
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "--retained-poses") {
            // Evaluation-only snapshot. These flags are not registrations:
            // preserve unsupported retained poses to assess why they failed.
            Scene snapshot;
            snapshot.cameras = scene.cameras;
            std::ofstream csv(reports);
            csv << "name,originally_registered,retained_pose_available\n";
            for (const auto& source : scene.images) {
                Image image;
                image.id = source.id; image.camera_id = source.camera_id;
                image.path = source.path; image.pose = source.pose;
                image.registered = source.registered || (source.pose.C.allFinite() &&
                    source.pose.R.allFinite() && source.pose.C.norm() > 1e-8);
                csv << source.path.filename().string() << ',' << source.registered << ',' << image.registered << '\n';
                snapshot.images.push_back(std::move(image));
            }
            save_asfm(snapshot,output);
            std::cout << "evaluation_snapshot_only=1 geometry_unchanged=1\n";
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "--audit-pairs") {
            std::ofstream csv(reports);
            if (!csv) throw std::runtime_error("Cannot write support audit");
            probe_boundary_support(scene,csv);
            if (!csv) throw std::runtime_error("Failed writing support audit");
            std::cout << "topology_only=1 geometry_unchanged=1 registered_views=" << scene.registered_count() << '\n';
            return 0;
        }
        if (argc == 4 && std::string_view(argv[3]) == "--inactive-links") {
            std::vector<std::uint8_t> stable;
            std::vector<Pose3D> originals;
            for (const auto& image : scene.images) { stable.push_back(image.registered); originals.push_back(image.pose); }
            std::vector<RawPairMatches> links;
            for (const auto& pair : scene.pairs) {
                if (!pair.relative_pose || pair.matches.empty() || stable[pair.id1] == stable[pair.id2]) continue;
                RawPairMatches link;
                link.id1 = pair.id1; link.id2 = pair.id2;
                for (const auto& match : pair.matches) link.matches.push_back({match.query,match.train});
                links.push_back(std::move(link));
            }
            const auto recovered = recover_stable_resections(scene,stable,links);
            std::ofstream csv(reports);
            csv << "recovered_image\n";
            for (Index id : recovered) csv << scene.images[id].path.filename().string() << '\n';
            for (Index id=0;id<scene.images.size();++id)
                if (stable[id] && (scene.images[id].pose.C != originals[id].C || scene.images[id].pose.R != originals[id].R))
                    throw std::runtime_error("Frozen original pose changed");
            save_asfm(scene,output);
            if (const auto* export_path = std::getenv("PHOTARA_PROBE_EXPORT_COLMAP")) save_colmap_text(scene,export_path);
            std::cout << "inactive_link_recovered=" << recovered.size() << " frozen_poses_unchanged=1 registered=" << scene.registered_count() << '\n';
            return 0;
        }
        if (groups) {
            std::vector<Pose3D> poses; std::vector<bool> registered;
            for (const auto& image:scene.images) { poses.push_back(image.pose); registered.push_back(image.registered); }
            const auto cameras=scene.cameras; const auto old_tracks=scene.tracks.size();
            const auto fingerprint=[&] {
                FingerprintBuilder hash;
                for (std::size_t t=0;t<old_tracks;++t) {
                    const auto& track=scene.tracks.at(t); hash.append_bytes(track.position.data(),3*sizeof(double));
                    hash.append(track.num_inliers); hash.append(track.observations.size());
                    for (const auto& obs:track.observations) { hash.append(obs.image_id); hash.append(obs.feature_id); }
                }
                return hash.value();
            };
            const auto original=fingerprint();
            std::ofstream csv(reports);
            const auto recovered = probe_missing_groups(scene, csv);
            for (Index id=0;id<scene.images.size();++id)
                if (registered[id] && (!scene.images[id].registered || scene.images[id].pose.C!=poses[id].C || scene.images[id].pose.R!=poses[id].R))
                    throw std::runtime_error("Group recovery changed a frozen camera");
            if (fingerprint()!=original) throw std::runtime_error("Group recovery changed original landmarks");
            if (scene.cameras.size()!=cameras.size()) throw std::runtime_error("Group recovery changed calibration groups");
            for (Index id=0;id<cameras.size();++id) {
                const auto& a=cameras[id]; const auto& b=scene.cameras[id];
                if (a.model!=b.model || a.fx!=b.fx || a.fy!=b.fy || a.cx!=b.cx || a.cy!=b.cy || a.k1!=b.k1 || a.k2!=b.k2 || a.p1!=b.p1 || a.p2!=b.p2)
                    throw std::runtime_error("Group recovery changed frozen intrinsics");
            }
            std::cout << "original_poses_unchanged=1 original_landmarks_unchanged=1 intrinsics_unchanged=1\n";
            save_asfm(scene, output);
            if (const auto* export_path = std::getenv("PHOTARA_PROBE_EXPORT_COLMAP"))
                save_colmap_text(scene, export_path);
            std::cout << "recovered_views=" << recovered << " registered_views=" << scene.registered_count() << '\n';
            return 0;
        }
        const auto audit = analyze_position_observability(scene);
        if (audit.reliable_views < 3) throw std::runtime_error("No frozen stable map available");
        auto stable = audit.reliable;
        if (descriptor_map)
            for (Index id = 0; id < scene.images.size(); ++id) stable[id] = scene.images[id].registered;
        const auto cameras = scene.cameras;
        std::vector<Pose3D> original;
        for (const auto& image : scene.images) original.push_back(image.pose);
        const auto original_track_count = scene.tracks.size();
        const auto map_fingerprint = [&] {
            FingerprintBuilder hash;
            for (std::size_t t = 0; t < original_track_count; ++t) {
                const auto& track = scene.tracks.at(t);
                hash.append_bytes(track.position.data(), 3 * sizeof(double));
                hash.append(track.num_inliers);
                hash.append(track.observations.size());
                for (const auto& obs : track.observations) { hash.append(obs.image_id); hash.append(obs.feature_id); }
            }
            return hash.value();
        };
        const auto original_map = descriptor_map ? map_fingerprint() : 0;
        std::ofstream csv(reports);
        if (!csv) throw std::runtime_error("Cannot write resection reports");
        csv << "round,frozen_anchors,accepted,names\n";
        unsigned total = 0;
        for (unsigned round = 0; round < 3; ++round) {
            std::vector<Index> recovered;
            if (descriptor_map) {
                FrontEndOptions recovery;
                if (const auto* relaxed = std::getenv("PHOTARA_PROBE_DENSE_RECOVERY");
                    relaxed && std::string_view(relaxed) == "1") {
                    recovery.mutual_check = false;
                    recovery.match_ratio = 0.9F;
                }
                recover_missing_views(scene, recovery);
                if (map_fingerprint() != original_map)
                    throw std::runtime_error("Original landmarks or observations changed during missing-view recovery");
                for (Index id = 0; id < scene.images.size(); ++id)
                    if (scene.images[id].registered && !stable[id]) recovered.push_back(id);
            } else recovered = recover_stable_resections(scene, stable);
            unsigned anchors = 0;
            for (auto flag : stable) anchors += flag != 0;
            csv << round << ',' << anchors << ',' << recovered.size() << ",\"";
            for (auto id : recovered) {
                for (char c : scene.images[id].path.filename().string()) { if (c == '"') csv << '"'; csv << c; }
                csv << ';';
            }
            csv << "\"\n"; csv.flush();
            for (Index id = 0; id < scene.images.size(); ++id)
                if ((descriptor_map ? stable[id] : audit.reliable[id]) && ((scene.images[id].pose.C - original[id].C).norm() != 0 ||
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
            if (descriptor_map) break;
        }
        audit_position_support(scene);
        save_asfm(scene, output);
        if (!csv) throw std::runtime_error("Failed writing resection reports");
        std::cout << "recovered_views=" << total << " original_anchors_unchanged=1 intrinsics_unchanged=1\n";
        if (descriptor_map) std::cout << "original_landmarks_unchanged=1\n";
        if (const auto* export_path = std::getenv("PHOTARA_PROBE_EXPORT_COLMAP"))
            save_colmap_text(scene, export_path);
        return 0;
    } catch (const std::exception& error) { std::cerr << error.what() << '\n'; return 1; }
}
