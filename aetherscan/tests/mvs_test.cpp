#include "mvs/densify.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <iostream>
#include <set>
#include <stdexcept>

namespace {

using namespace aetherscan::mvs;

void require(const bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

MvsView make_plane_view(const Index id, const std::uint32_t size) {
    MvsView view;
    view.id = id;
    view.sfm_image_id = id;
    view.path = "this-file-does-not-exist.png";
    view.pose = aetherscan::sfm::Pose3D::identity();
    view.width = size;
    view.height = size;
    view.fx = 100.F;
    view.fy = 100.F;
    view.cx = static_cast<float>(size - 1) * 0.5F;
    view.cy = static_cast<float>(size - 1) * 0.5F;
    view.src_fx = view.fx;
    view.src_fy = view.fy;
    view.src_cx = view.cx;
    view.src_cy = view.cy;
    view.src_width = size;
    view.src_height = size;
    view.depth_map.view_id = id;
    view.depth_map.resize(size, size);
    view.depth_map.depth_min = 1.F;
    view.depth_map.depth_max = 3.F;
    std::fill(view.depth_map.depth.begin(), view.depth_map.depth.end(), 2.F);
    std::fill(
        view.depth_map.normal.begin(), view.depth_map.normal.end(),
        Vec3f{0.F, 0.F, -1.F});
    std::fill(
        view.depth_map.confidence.begin(), view.depth_map.confidence.end(), 0.1F);
    return view;
}

MvsScene make_plane_scene() {
    constexpr std::uint32_t size = 16;
    MvsScene scene;
    scene.thread_count = 4;
    scene.views.push_back(make_plane_view(0, size));
    scene.views.push_back(make_plane_view(1, size));
    scene.views[0].neighbors.push_back(NeighborScore{1, 1.F, 10.F, 1.F, 10});
    scene.views[1].neighbors.push_back(NeighborScore{0, 1.F, 10.F, 1.F, 10});
    SparsePoint sparse;
    sparse.position = Vec3f{0.F, 0.F, 2.F};
    sparse.view_ids = {0, 1};
    scene.sparse_points.push_back(std::move(sparse));
    return scene;
}

void test_parallel_fusion() {
    MvsScene scene = make_plane_scene();
    DensifyOptions options;
    options.speckle_size = 1;
    options.min_views_fuse = 2;
    options.thread_count = 4;
    fuse_depth_maps(scene, options);
    require(!scene.dense_cloud.points.empty(), "plane fusion produced no points");
    require(
        scene.dense_cloud.points.size() >= 220 &&
            scene.dense_cloud.points.size() <= 280,
        "plane fusion did not weld duplicate views at pixel scale");
    for (const DensePoint& point : scene.dense_cloud.points) {
        require(point.position.allFinite(), "fused position is non-finite");
        require(point.normal.allFinite(), "fused normal is non-finite");
        require(point.views.size() >= 2, "fused point lacks multi-view support");
    }
}

void test_projective_mesh() {
    MvsScene scene = make_plane_scene();
    DensifyOptions options;
    options.speckle_size = 1;
    options.min_views_fuse = 2;
    fuse_depth_maps(scene, options);
    options.mesh_method = MeshMethod::depth_projective;
    options.mesh_pixel_step = 1;
    options.mesh_min_component_faces = 1;
    reconstruct_mesh(scene, options);

    require(!scene.mesh.vertices.empty(), "plane meshing produced no vertices");
    require(scene.mesh.faces.size() == 450, "unexpected plane face count");
    require(
        scene.mesh.normals.size() == scene.mesh.vertices.size(),
        "mesh normals are incomplete");
    std::set<std::array<int, 3>> unique;
    for (const Eigen::Vector3i& face : scene.mesh.faces) {
        std::array<int, 3> key{face[0], face[1], face[2]};
        std::sort(key.begin(), key.end());
        require(key[0] >= 0, "negative mesh index");
        require(
            key[2] < static_cast<int>(scene.mesh.vertices.size()),
            "mesh index out of range");
        require(unique.insert(key).second, "duplicate projective face");
    }
}

void test_quality_presets() {
    DensifyOptions options;
    apply_quality_preset(options, DensifyQuality::high);
    require(options.resolution_level == 0, "high preset is not full resolution");
    require(options.min_patch_views == 3, "high preset patch support is weak");
    require(options.min_views_filter == 2, "high preset filter support is weak");
    require(options.min_views_fuse == 3, "high preset fusion support is weak");
    require(options.mask_border_px == 2, "high preset has no silhouette guard");
    require(
        options.min_viewing_incidence_cos >= 0.2F,
        "high preset accepts unstable grazing samples");
    require(options.mesh_pixel_step == 2, "high preset mesh is too large by default");

    apply_quality_preset(options, DensifyQuality::preview);
    require(options.resolution_level == 2, "preview preset resolution mismatch");
    require(
        std::abs(options.depth_diff_threshold - 0.01F) < 1e-6F,
        "preset application leaked high-quality thresholds");
}

}  // namespace

int main() {
    try {
        test_parallel_fusion();
        test_projective_mesh();
        test_quality_presets();
        std::cout << "mvs tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mvs test failed: " << error.what() << '\n';
        return 1;
    }
}
