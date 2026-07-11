#include "sfm/scene.hpp"

#include <algorithm>

namespace aetherscan::sfm {

void Scene::clear() {
    cameras.clear();
    images.clear();
    pairs.clear();
    tracks.clear();
}

ImagePair* Scene::find_pair(Index a, Index b) {
    if (a > b) std::swap(a, b);
    for (auto& pair : pairs) {
        if (pair.id1 == a && pair.id2 == b) return &pair;
    }
    return nullptr;
}

const ImagePair* Scene::find_pair(Index a, Index b) const {
    if (a > b) std::swap(a, b);
    for (const auto& pair : pairs) {
        if (pair.id1 == a && pair.id2 == b) return &pair;
    }
    return nullptr;
}

unsigned Scene::registered_count() const {
    unsigned count = 0;
    for (const auto& image : images) {
        if (image.registered) ++count;
    }
    return count;
}

bool Scene::invalidate_image(const Index image_id) {
    if (image_id >= images.size()) return false;
    images[image_id].registered = false;
    images[image_id].pose = Pose3D::identity();
    for (auto& track : tracks) {
        auto& obs = track.observations;
        obs.erase(
            std::remove_if(obs.begin(), obs.end(),
                [image_id](const Observation& o) { return o.image_id == image_id; }),
            obs.end());
        track.num_inliers = static_cast<std::uint8_t>(
            std::min<std::size_t>(track.num_inliers, obs.size()));
        if (track.num_inliers < 2) track.num_inliers = 0;
    }
    return true;
}

}  // namespace aetherscan::sfm
