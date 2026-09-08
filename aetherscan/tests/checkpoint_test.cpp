#include "sfm/checkpoint.hpp"

#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>

using namespace aetherscan;

int main() {
    const auto suffix =
        std::chrono::steady_clock::now().time_since_epoch().count();
    const std::filesystem::path directory =
        std::filesystem::temp_directory_path() /
        ("aetherscan-checkpoint-test-" + std::to_string(suffix));
    try {
        sfm::CheckpointOptions options;
        options.directory = directory;
        sfm::CheckpointStore store(options);

        sfm::Scene scene;
        scene.thread_count = 7;
        scene.resection_progress.since_full_ba = 3;
        scene.resection_progress.bundle_adjustment_stage = 1;
        scene.resection_progress.last_registered = {0};
        scene.resection_progress.recent_inlier_ratios = {0.75};
        scene.cameras.push_back(
            {0, 1280, 720, 900.0, 900.0, 640.0, 360.0});
        scene.cameras[0].focal_prior = 875.0;
        scene.cameras[0].model = CameraModel::opencv_fisheye;
        sfm::Image image;
        image.id = 0;
        image.camera_id = 0;
        image.path = std::filesystem::path(u8"测试-image.jpg");
        image.registered = true;
        image.pose.C = sfm::Vec3(1.0, 2.0, 3.0);
        image.features.image_width = 1280;
        image.features.image_height = 720;
        image.features.descriptor_dimension = 2;
        image.features.extractor_name = "test";
        image.features.keypoints.push_back({10.F, 20.F, 2.F, 0.5F, 3.F});
        image.features.descriptors = {0.25F, 0.75F};
        scene.images.push_back(std::move(image));
        sfm::Image second_image;
        second_image.id = 1;
        second_image.camera_id = 0;
        second_image.path = "second-image.jpg";
        second_image.features.image_width = 1280;
        second_image.features.image_height = 720;
        scene.images.push_back(std::move(second_image));
        sfm::ImagePair pair(0, 1);
        pair.weight_spatial = 0.8F;
        pair.weight_geometry = 0.7F;
        pair.weight_connectivity = 0.6F;
        pair.weight_triplet = 0.4F;
        pair.weight_cycle = 0.2F;
        pair.estimated_focal = 880.0;
        pair.zero_baseline = true;
        scene.pairs.push_back(pair);
        sfm::Track track;
        track.position = sfm::Vec3(4.0, 5.0, 6.0);
        track.observations.push_back({0, 0});
        track.split_generation = 17;
        scene.tracks.push_back(std::move(track));
        scene.registration_generation = 17;

        constexpr std::uint64_t scene_key = 0x12345678ULL;
        store.save_scene(sfm::CheckpointStage::features, scene_key, scene);
        sfm::Scene restored;
        if (!store.load_scene(
                sfm::CheckpointStage::features, scene_key, restored) ||
            restored.thread_count != 7 || restored.images.size() != 2 ||
            restored.images[0].path != scene.images[0].path ||
            restored.images[0].features.descriptors.size() != 2 ||
            restored.cameras[0].focal_prior != 875.0 ||
            restored.cameras[0].model != CameraModel::opencv_fisheye ||
            restored.pairs.size() != 1 ||
            restored.pairs[0].estimated_focal != pair.estimated_focal ||
            restored.pairs[0].weight_connectivity != 0.6F ||
            restored.pairs[0].weight_triplet != 0.4F ||
            restored.pairs[0].weight_cycle != 0.2F ||
            !restored.pairs[0].zero_baseline ||
            restored.resection_progress.since_full_ba != 3 ||
            restored.resection_progress.last_registered !=
                scene.resection_progress.last_registered) {
            std::cerr << "feature checkpoint round trip failed\n";
            std::filesystem::remove_all(directory);
            return 1;
        }
        scene.images[0].features.compress_descriptors_u8();
        store.save_scene(
            sfm::CheckpointStage::features, scene_key + 2, scene);
        if (!store.load_scene(
                sfm::CheckpointStage::features, scene_key + 2, restored) ||
            restored.images[0].features.storage !=
                features::DescriptorStorage::uint8 ||
            restored.images[0].features.descriptors_u8.size() != 2 ||
            !restored.images[0].features.descriptors.empty()) {
            std::cerr << "compressed feature checkpoint round trip failed\n";
            std::filesystem::remove_all(directory);
            return 9;
        }
        store.save_scene(sfm::CheckpointStage::tracks, scene_key, scene);
        if (!store.load_scene(
                sfm::CheckpointStage::tracks, scene_key, restored) ||
            !restored.images[0].features.descriptors.empty() ||
            restored.tracks.size() != 1 ||
            restored.registration_generation != 0 ||
            restored.tracks[0].split_generation != 0 ||
            restored.image_tracks.size() != 2 ||
            restored.image_tracks[0].size() != 1) {
            std::cerr << "scene checkpoint round trip failed\n";
            std::filesystem::remove_all(directory);
            return 1;
        }
        if (store.load_scene(
                sfm::CheckpointStage::tracks, scene_key + 1, restored)) {
            std::cerr << "checkpoint key mismatch was accepted\n";
            std::filesystem::remove_all(directory);
            return 2;
        }
        scene.thread_count = 9;
        store.save_scene(
            sfm::CheckpointStage::tracks, scene_key + 1, scene);
        sfm::Scene first_variant;
        sfm::Scene second_variant;
        if (!store.load_scene(
                sfm::CheckpointStage::tracks, scene_key, first_variant) ||
            !store.load_scene(
                sfm::CheckpointStage::tracks, scene_key + 1,
                second_variant) ||
            first_variant.thread_count != 7 ||
            second_variant.thread_count != 9) {
            std::cerr << "content-addressed checkpoint variants collided\n";
            std::filesystem::remove_all(directory);
            return 3;
        }

        std::vector<sfm::RawPairMatches> matches(1);
        matches[0].id1 = 2;
        matches[0].id2 = 5;
        matches[0].matches.push_back({3, 4, 0.125F});
        store.save_matches(scene_key, matches);
        std::vector<sfm::RawPairMatches> restored_matches;
        if (!store.load_matches(scene_key, restored_matches) ||
            restored_matches.size() != 1 ||
            restored_matches[0].matches.size() != 1 ||
            restored_matches[0].matches[0].train != 4) {
            std::cerr << "match checkpoint round trip failed\n";
            std::filesystem::remove_all(directory);
            return 4;
        }

        // A partial/trailing write must be rejected instead of being restored.
        std::filesystem::path matches_path;
        for (const auto& entry : std::filesystem::directory_iterator(directory))
            if (entry.path().filename().string().starts_with("matches-"))
                matches_path = entry.path();
        std::ofstream corrupt(matches_path, std::ios::binary | std::ios::app);
        corrupt.put('\0');
        corrupt.close();
        if (store.load_matches(scene_key, restored_matches)) {
            std::cerr << "corrupt checkpoint was accepted\n";
            std::filesystem::remove_all(directory);
            return 5;
        }
        const std::filesystem::path image_path = directory / "identity.jpg";
        {
            std::ofstream image(image_path, std::ios::binary);
            image << "AAAA";
        }
        const auto timestamp = std::filesystem::last_write_time(image_path);
        const sfm::ImageSetFingerprint first_fingerprint =
            sfm::fingerprint_image_set({image_path});
        {
            std::ofstream image(
                image_path, std::ios::binary | std::ios::trunc);
            image << "BBBB";
        }
        std::filesystem::last_write_time(image_path, timestamp);
        sfm::verify_image_snapshot(
            {image_path}, first_fingerprint, sfm::ImageSnapshotCheck::identity);
        if (first_fingerprint.value == sfm::fingerprint_images({image_path})) {
            std::cerr << "same-size image replacement was not invalidated\n";
            std::filesystem::remove_all(directory);
            return 6;
        }
        bool content_caught = false;
        try {
            sfm::verify_image_snapshot(
                {image_path}, first_fingerprint, sfm::ImageSnapshotCheck::content);
        } catch (const std::runtime_error&) {
            content_caught = true;
        }
        if (!content_caught) {
            std::cerr << "content snapshot check missed replacement\n";
            std::filesystem::remove_all(directory);
            return 8;
        }
        std::filesystem::remove_all(directory);
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        std::filesystem::remove_all(directory);
        return 7;
    }
}
