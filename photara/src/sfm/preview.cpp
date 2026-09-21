#include "sfm/preview.hpp"
#include "sfm/asfm.hpp"
#include <stdexcept>
#include <unordered_map>
#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <Windows.h>
#endif
namespace photara::sfm {
Scene make_alignment_preview(const Scene& source, const std::size_t maximum_points) {
    Scene result;
    result.cameras = source.cameras;
    for (const auto& image : source.images) {
        Image view;
        view.id = image.id;
        view.camera_id = image.camera_id;
        view.path = image.path;
        view.pose = image.pose;
        view.registered = image.registered && image.pose.C.allFinite() && image.pose.R.allFinite();
        result.images.push_back(std::move(view));
    }
    if (maximum_points == 0) return result;
    std::size_t count = 0;
    for (const auto& track : source.tracks)
        if (track.is_triangulated() && track.position.allFinite()) ++count;
    const std::size_t stride = std::max<std::size_t>(1, count / maximum_points + (count % maximum_points != 0));
    std::vector<std::unordered_map<Index, Index>> feature_map(source.images.size());
    std::size_t ordinal = 0;
    for (const auto& track : source.tracks) {
        if (!track.is_triangulated() || !track.position.allFinite()) continue;
        if (ordinal++ % stride != 0) continue;
        Track point;
        point.position = track.position;
        for (std::size_t i = 0; i < std::min<std::size_t>(track.num_inliers, track.observations.size()); ++i) {
            const auto& obs = track.observations[i];
            if (obs.image_id >= source.images.size() || !result.images[obs.image_id].registered) continue;
            const auto& features = source.images[obs.image_id].features.keypoints;
            if (obs.feature_id >= features.size()) continue;
            auto& keypoints = result.images[obs.image_id].features.keypoints;
            auto [it, inserted] = feature_map[obs.image_id].emplace(obs.feature_id, static_cast<Index>(keypoints.size()));
            if (inserted) keypoints.push_back(features[obs.feature_id]);
            point.observations.push_back({obs.image_id, it->second});
        }
        if (point.observations.size() < 2) continue;
        point.num_inliers = static_cast<std::uint8_t>(point.observations.size());
        result.tracks.push_back(std::move(point));
    }
    return result;
}
void save_alignment_preview(const Scene& source, const std::filesystem::path& path) {
    auto temporary = path;
    temporary += ".tmp";
    if (!path.parent_path().empty()) std::filesystem::create_directories(path.parent_path());
    save_asfm(make_alignment_preview(source), temporary);
#if defined(_WIN32)
    if (!MoveFileExW(temporary.c_str(), path.c_str(), MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH))
        throw std::runtime_error("Cannot publish alignment preview");
#else
    std::filesystem::rename(temporary, path);
#endif
}
}
