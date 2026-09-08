#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/preview.hpp"
#include "sfm/asfm.hpp"
#include "core/logging.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <stdexcept>

using namespace aetherscan::sfm;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Scene fixture(bool shared = true) {
    Scene scene;
    scene.thread_count = 1;
    PinholeCamera camera;
    camera.id = 0; camera.width = 1000; camera.height = 800;
    camera.fx = camera.fy = 700; camera.cx = 500; camera.cy = 400;
    scene.cameras.push_back(camera);
    for (Index i = 0; i < 8; ++i) {
        Image image;
        image.id = i; image.camera_id = 0; image.registered = true;
        image.pose.C = Vec3(0.25 * i, 0.12 * (i % 2), 0);
        image.features.keypoints.resize(160);
        scene.images.push_back(image);
    }
    std::mt19937 rng(81);
    std::uniform_real_distribution<double> xy(-1.5, 1.5), z(4.0, 6.0);
    for (Index p = 0; p < 160; ++p) {
        Vec3 point(xy(rng), xy(rng), z(rng));
        Track track;
        track.position = Vec3(99, -40, 3);  // must never supply alignment depth
        for (auto& image : scene.images) {
            Vec2 pixel = camera.project(image.pose.transform_world_to_camera(point));
            image.features.keypoints[p].x = static_cast<float>(pixel.x());
            image.features.keypoints[p].y = static_cast<float>(pixel.y());
            if (shared || image.id < 4) track.observations.push_back({image.id, p});
        }
        track.num_inliers = 4;
        scene.tracks.push_back(track);
        if (!shared) {
            Track local;
            for (Index i = 4; i < 8; ++i) local.observations.push_back({i, p});
            scene.tracks.push_back(local);
        }
    }
    for (Index a = 0; a < 8; ++a) for (Index b = a + 1; b < 8; ++b) {
        if (a < 4 && b >= 4 && !(shared && a == 3 && b == 4)) continue;
        ImagePair pair(a, b);
        pair.relative_pose = scene.images[b].pose / scene.images[a].pose;
        pair.relative_pose->C.normalize(); pair.weight_spatial = 1;
        for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
        scene.pairs.push_back(pair);
    }
    for (Index i = 4; i < 8; ++i) {
        scene.images[i].pose.C = 3 * scene.images[i].pose.C + Vec3(10, -2, 1);
        scene.images[i].pose.R = Eigen::AngleAxisd(0.4, Vec3::UnitY()).toRotationMatrix();
    }
    return scene;
}

int main() {
    try {
        aetherscan::core::Logger::instance().configure({});
        const std::vector<Index> ids{4, 5, 6, 7};
        const std::vector<std::uint8_t> stable{1,1,1,1,0,0,0,0};
        Scene scene = fixture();
        const auto preview = make_alignment_preview(scene, 17);
        expect(preview.tracks.size() <= 17 && !preview.tracks.empty(), "preview must bound sampled points");
        expect(preview.images.size() == scene.images.size() && preview.pairs.empty(), "preview keeps camera IDs without graph");
        const auto decoded_preview = decode_asfm(encode_asfm(preview));
        expect(decoded_preview.tracks.size() == preview.tracks.size(), "preview observation remapping must round-trip");
        const auto preview_path = std::filesystem::temp_directory_path() /
            ("aetherscan-preview-test-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + ".asfm");
        save_alignment_preview(scene, preview_path);
        expect(load_asfm(preview_path).registered_count() == 8, "published preview must load");
        auto next_preview = scene;
        next_preview.images[7].registered = false;
        save_alignment_preview(next_preview, preview_path);
        expect(load_asfm(preview_path).registered_count() == 7,
            "preview publication must replace an existing snapshot");
        std::filesystem::remove(preview_path);
        Scene resected = fixture();
        // Remove local edges: singletons can be anchored by stable 2D-3D
        // evidence even though they cannot independently triangulate depth.
        resected.pairs.erase(std::remove_if(resected.pairs.begin(), resected.pairs.end(),
            [](const ImagePair& p) { return p.id1 >= 4; }), resected.pairs.end());
        const auto recovered = recover_stable_resections(resected);
        expect(recovered.size() == 4, "stable resection must recover isolated drifting poses");
        for (Index i = 4; i < 8; ++i)
            expect((resected.images[i].pose.C - Vec3(0.25*i,0.12*(i%2),0)).norm() < 0.01,
                "resection must not reuse drifting depths");
        expect(analyze_alignment_observability(resected).unreliable_views == 0,
            "independent stable-depth support must certify corrected singletons");
        Scene no_anchor = fixture(false);
        expect(recover_stable_resections(no_anchor).empty(), "resection must reject missing stable depths");
        Scene heldout_mismatch = fixture();
        for (Index i = 4; i < 8; ++i) {
            const auto original_features = heldout_mismatch.images[i].features.keypoints;
            for (Index p = 0; p < 160; p += 5)
                heldout_mismatch.images[i].features.keypoints[p] = original_features[(p + 75) % 160];
        }
        const auto rejected_poses = heldout_mismatch.images;
        expect(recover_stable_resections(heldout_mismatch).empty(),
            "resection must reject inconsistent held-out correspondences despite a perfect fit subset");
        for (Index i = 0; i < 8; ++i)
            expect(heldout_mismatch.images[i].pose.C == rejected_poses[i].pose.C &&
                   heldout_mismatch.images[i].pose.R == rejected_poses[i].pose.R,
                "rejected resection must preserve all poses");
        // Neither global registration nor inherited shared landmarks may hide
        // a dense four-camera branch attached through one verified bridge.
        auto risk = find_structural_pair_risks(scene);
        expect(risk[4] && risk[5] && risk[6] && risk[7], "dense bridge branch must trigger frontend rescue");
        expect(!risk[0] && !risk[1] && !risk[2] && !risk[3], "retain main pair block");
        Scene closed = scene;
        ImagePair closure(2, 5);
        closure.relative_pose = Pose3D::identity();
        closed.pairs.push_back(closure);
        risk = find_structural_pair_risks(closed);
        for (auto value : risk) expect(!value, "second independent graph link closes the bridge");
        auto sub = make_independent_submap(scene, ids);
        expect(sub.scene.tracks.empty(), "must discard inherited points and tracks");
        expect(sub.scene.pairs.size() == 6, "must exclude boundary pairs");
        for (const auto& image : sub.scene.images)
            expect(!image.registered && image.pose.C.isZero() && image.pose.R.isIdentity(), "must discard old poses");
        const auto original = scene;
        auto report = recover_independent_submap(scene, ids, stable);
        std::cout << "positive: " << report.reason << " shared=" << report.shared_points << '\n';
        expect(report.accepted, "independent geometry should correct the drift");
        expect(report.validation_inliers >= 3, "must validate held-out points");
        for (Index i = 0; i < 4; ++i)
            expect(scene.images[i].pose.C == original.images[i].pose.C &&
                scene.images[i].pose.R == original.images[i].pose.R, "stable poses must remain exact");
        for (Index i : ids)
            expect((scene.images[i].pose.C - Vec3(0.25 * i, 0.12 * (i % 2), 0)).norm() < 0.01,
                "must recover known coordinates without using parent points");
        Scene rejected = fixture(false);
        const auto before = rejected;
        report = recover_independent_submap(rejected, ids, stable);
        expect(!report.accepted && report.reason == "insufficient_independent_shared_points",
            "disconnected maps must not invent alignment");
        for (Index i = 0; i < 8; ++i)
            expect(rejected.images[i].pose.C == before.images[i].pose.C, "rejection must preserve poses");
        for (Index i = 0; i < before.tracks.size(); ++i)
            expect(rejected.tracks[i].position == before.tracks[i].position &&
                rejected.tracks[i].num_inliers == before.tracks[i].num_inliers,
                "rejection must preserve tracks");
        Scene rollback = fixture();
        SubmapRecoveryOptions strict;
        strict.minimum_camera_observations = 1000;
        report = recover_independent_submap(rollback, ids, stable, strict);
        expect(!report.accepted && report.reason == "insufficient_final_camera_support", "must check final camera support");
        for (Index i : ids) expect(rollback.images[i].pose.C == original.images[i].pose.C,
            "late rejection must roll back candidate poses");
        Scene circular = fixture();
        for (Index i = 1; i < 4; ++i) {
            circular.images[i].pose = circular.images[0].pose;
            circular.images[i].features.keypoints = circular.images[0].features.keypoints;
        }
        report = recover_independent_submap(circular, ids, stable);
        expect(!report.accepted && report.shared_points == 0,
            "coincident stable rays must not certify independent depth");
        Scene inconsistent = fixture();
        for (Index i = 0; i < 4; ++i) for (Index p = 0; p < 160; ++p) {
            // Preserve stable multiview consistency but destroy the cross-map
            // correspondences: no single similarity explains this permutation.
            inconsistent.images[i].features.keypoints[p] =
                original.images[i].features.keypoints[(p * 37) % 160];
        }
        report = recover_independent_submap(inconsistent, ids, stable);
        expect(!report.accepted && report.reason == "similarity_consensus_failed",
            "wrong cross-map matches must fail geometric consensus");
        Scene collinear = fixture();
        for (Index i = 0; i < 4; ++i) for (Index p = 0; p < 160; ++p) {
            const Vec3 point(-0.8 + 0.01 * p, 0, 5);
            const Vec2 pixel = collinear.cameras[0].project(
                collinear.images[i].pose.transform_world_to_camera(point));
            collinear.images[i].features.keypoints[p].x = static_cast<float>(pixel.x());
            collinear.images[i].features.keypoints[p].y = static_cast<float>(pixel.y());
        }
        report = recover_independent_submap(collinear, ids, stable);
        expect(!report.accepted && report.reason == "ill_conditioned_shared_geometry",
            "collinear shared points cannot constrain similarity");
        Scene heldout = fixture();
        auto ordering = make_independent_submap(heldout, ids);
        build_tracks(ordering.scene);
        for (Index t = 0; t < ordering.scene.tracks.size(); t += 5) {
            const Index p = ordering.scene.tracks[t].observations.front().feature_id;
            for (Index i = 0; i < 4; ++i)
                heldout.images[i].features.keypoints[p] = original.images[i].features.keypoints[(p + 71) % 160];
        }
        report = recover_independent_submap(heldout, ids, stable);
        expect(!report.accepted && report.reason == "heldout_geometry_failed",
            "good fit must not bypass failed held-out geometry");
        bool invalid = false;
        try { make_independent_submap(original, {4, 4}); }
        catch (const std::invalid_argument&) { invalid = true; }
        expect(invalid, "duplicate image IDs must be rejected");
        std::cout << "submap recovery tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
