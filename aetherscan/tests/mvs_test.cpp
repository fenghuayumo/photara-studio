#include "mvs/densify.hpp"
#include "mvs/export.hpp"
#include "mvs/internal.hpp"
#include "mvs/maxflow.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <numbers>
#include <random>
#include <set>
#include <stdexcept>
#include <vector>

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
        require(
            point.view_weights.size() == point.views.size(),
            "fused per-view weights are incomplete");
        require(
            std::all_of(
                point.view_weights.begin(), point.view_weights.end(),
                [](const float weight) { return weight > 0.F; }),
            "fused per-view weight is not positive");
    }
}

void test_mask_and_roi_constrained_fusion() {
    MvsScene scene = make_plane_scene();
    for (auto& view : scene.views) {
        view.foreground_mask.assign(view.depth_map.size(), 0);
        for (std::uint32_t y = 0; y < view.height; ++y)
            for (std::uint32_t x = 0; x < view.width / 2; ++x)
                view.foreground_mask[view.depth_map.index(x, y)] = 1;
    }
    scene.roi.valid = true;
    scene.roi.center = Vec3f{-0.08F, 0.F, 2.F};
    scene.roi.half_extent = Vec3f{0.08F, 0.2F, 0.1F};
    DensifyOptions options;
    options.speckle_size = 1;
    options.min_views_fuse = 2;
    fuse_depth_maps(scene, options);
    require(!scene.dense_cloud.points.empty(), "masked fusion produced no points");
    require(scene.dense_cloud.points.size() < 150, "fusion ignored foreground mask");
    for (const DensePoint& point : scene.dense_cloud.points)
        require(scene.roi.contains(point.position), "fusion emitted point outside ROI");
}

void test_projected_mesh_mask() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.mesh.vertices = {
        Vec3f{-0.12F, -0.12F, 2.F}, Vec3f{0.12F, -0.12F, 2.F},
        Vec3f{0.12F, 0.12F, 2.F}, Vec3f{-0.12F, 0.12F, 2.F}};
    scene.mesh.faces = {Eigen::Vector3i{0,1,2}, Eigen::Vector3i{0,2,3}};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    const auto& mask = scene.views[0].foreground_mask;
    require(mask.size() == 32U * 32U, "projected mask has wrong size");
    require(mask[16U * 32U + 16U] != 0, "projected mesh missed image center");
    require(mask[0] == 0, "projected mesh mask filled background corner");
}

void test_input_mask_is_not_cut_by_coarse_mesh_holes() {
    MvsScene scene;
    scene.views.push_back(make_plane_view(0, 32));
    scene.views[0].foreground_mask.assign(32U * 32U, 255);
    // A deliberately incomplete coarse mesh that does not cover the center.
    scene.mesh.vertices = {
        Vec3f{-0.20F, -0.20F, 2.F}, Vec3f{-0.08F, -0.20F, 2.F},
        Vec3f{-0.20F, -0.08F, 2.F}};
    scene.mesh.faces = {Eigen::Vector3i{0,1,2}};
    scene.roi.valid = true;
    scene.roi.center = Vec3f{0.F, 0.F, 2.F};
    scene.roi.half_extent = Vec3f{0.25F, 0.25F, 0.25F};
    DensifyOptions options;
    options.auto_roi_mask_dilate_px = 0;
    detail::build_projected_foreground_masks(scene, options);
    require(
        scene.views[0].foreground_mask[16U * 32U + 16U] != 0,
        "coarse mesh hole was baked into the final input mask");
    require(
        scene.views[0].foreground_mask[0] == 0,
        "projected OBB envelope failed to reject distant background");
}

void test_manual_obb_file() {
    const auto path = std::filesystem::temp_directory_path() /
                      "aetherscan-mvs-test-roi.txt";
    {
        std::ofstream output(path);
        output << "1 2 3  1 0.01 0  0 1 0  0 0 1  4 5 6\n";
    }
    OrientedBoundingBox roi;
    const bool loaded = detail::load_manual_roi(path, roi);
    std::filesystem::remove(path);
    require(loaded && roi.valid, "manual OBB file did not load");
    require((roi.axes.transpose() * roi.axes - Mat3f::Identity()).norm() < 1e-4F,
            "manual OBB axes were not orthonormalized");
    require(roi.contains(Vec3f{1.F,2.F,3.F}), "manual OBB missed its center");
    require(!roi.contains(Vec3f{20.F,2.F,3.F}), "manual OBB accepted far point");
    save_roi(roi, path);
    OrientedBoundingBox roundtrip;
    require(detail::load_manual_roi(path, roundtrip), "saved OBB did not reload");
    std::filesystem::remove(path);
    require((roundtrip.center - roi.center).norm() < 1e-5F,
            "OBB save/load changed its center");
}

void test_automatic_ground_and_subject_roi() {
    MvsScene scene;
    const Vec3f target{0.F, 0.5F, 0.F};
    const std::array<Vec3f, 4> cameras{
        Vec3f{2.F, 1.5F, 0.F}, Vec3f{-2.F, 1.5F, 0.F},
        Vec3f{0.F, 1.5F, 2.F}, Vec3f{0.F, 1.5F, -2.F}};
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        MvsView view;
        view.id = static_cast<Index>(i);
        view.pose.C = cameras[i].cast<double>();
        const Vec3f forward = (target - cameras[i]).normalized();
        Vec3f right = Vec3f::UnitY().cross(forward).normalized();
        const Vec3f down = forward.cross(right).normalized();
        view.pose.R.row(0) = right.cast<double>();
        view.pose.R.row(1) = down.cast<double>();
        view.pose.R.row(2) = forward.cast<double>();
        scene.views.push_back(view);
    }
    for (int z = -10; z <= 10; ++z)
        for (int x = -10; x <= 10; ++x) {
            DensePoint p;
            p.position = Vec3f{0.1F * x, 0.F, 0.1F * z};
            p.views = {0,1};
            scene.dense_cloud.points.push_back(p);
        }
    for (int z = -4; z <= 4; ++z)
        for (int y = 2; y <= 8; ++y)
            for (int x = -4; x <= 4; ++x) {
                DensePoint p;
                p.position = Vec3f{0.05F*x, 0.1F*y, 0.05F*z};
                p.views = {0,1,2};
                scene.dense_cloud.points.push_back(p);
            }
    DensifyOptions options;
    options.auto_roi_component_voxel_fraction = 0.04F;
    options.roi_margin_fraction = 0.05F;
    require(detail::estimate_automatic_roi(scene, options), "automatic ROI failed");
    require(scene.has_ground_plane, "automatic ROI missed dominant ground plane");
    require(scene.roi.contains(target), "automatic ROI missed subject target");
    require(!scene.roi.contains(Vec3f{0.9F, 0.F, 0.9F}),
            "automatic ROI retained distant ground");
    for (const auto& point : scene.dense_cloud.points)
        require(point.position.y() > 0.05F, "subject component retained ground");
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
    require(
        options.mesh_method == MeshMethod::delaunay_cut,
        "high preset does not select global meshing");
    require(options.min_patch_views == 3, "high preset patch support is weak");
    require(options.min_views_filter == 2, "high preset filter support is weak");
    require(options.min_views_fuse == 3, "high preset fusion support is weak");
    require(options.mask_border_px == 1, "high preset silhouette guard mismatch");
    require(
        options.grazing_weight_floor > 0.F &&
            options.grazing_weight_floor < 0.2F,
        "high preset grazing samples are not softly weighted");
    require(options.mesh_pixel_step == 2, "high preset mesh is too large by default");
    require(
        std::abs(options.mesh_k_behind - 1.F) < 1e-6F,
        "global mesh surface thickness is not one sigma");
    require(
        options.descriptor_min_magnitude > 0.F &&
            options.low_texture_prior_magnitude >
                options.descriptor_min_magnitude,
        "weighted NCC texture thresholds are invalid");
    require(
        std::abs(options.mesh_dist_insert_px - 0.75F) < 1e-6F,
        "high preset global mesh spacing is too coarse");

    apply_quality_preset(options, DensifyQuality::default_quality);
    require(
        std::abs(options.mesh_dist_insert_px - 0.75F) < 1e-6F,
        "default preset global mesh spacing mismatch");

    apply_quality_preset(options, DensifyQuality::preview);
    require(options.resolution_level == 2, "preview preset resolution mismatch");
    require(
        options.mesh_method == MeshMethod::depth_projective,
        "preview preset does not select projective meshing");
    require(
        std::abs(options.depth_diff_threshold - 0.01F) < 1e-6F,
        "preset application leaked high-quality thresholds");
}

#if !defined(AETHERSCAN_HAS_CGAL)
void test_missing_cgal_fails_before_densify() {
    MvsScene scene;
    DensifyOptions options;
    options.mesh_method = MeshMethod::delaunay_cut;
    bool rejected = false;
    try {
        densify(scene, options);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    require(rejected, "missing CGAL did not reject global meshing early");
}
#endif

void test_scalable_maxflow_cut() {
    maxflow::Graph graph(2);
    graph.add_tweights(0, 5.F, 0.F);
    graph.add_tweights(1, 0.F, 5.F);
    graph.add_edge(0, 1, 2.F, 2.F);
    require(std::abs(graph.maxflow() - 2.F) < 1e-5F, "maxflow value mismatch");
    require(graph.is_source_side(0), "source-constrained node crossed cut");
    require(!graph.is_source_side(1), "sink-constrained node crossed cut");
    maxflow::Graph unary_overlap(1);
    unary_overlap.add_tweights(0, 7.F, 3.F);
    require(
        std::abs(unary_overlap.maxflow() - 3.F) < 1e-5F,
        "maxflow omitted the constant unary offset");

    struct Arc {
        int a{};
        int b{};
        float forward{};
        float reverse{};
    };
    std::mt19937 random(41);
    std::uniform_int_distribution<int> capacity(0, 5);
    for (int trial = 0; trial < 40; ++trial) {
        constexpr int nodes = 7;
        std::array<float, nodes> source{};
        std::array<float, nodes> sink{};
        std::vector<Arc> arcs;
        maxflow::Graph candidate(nodes);
        for (int i = 0; i < nodes; ++i) {
            source[static_cast<std::size_t>(i)] =
                static_cast<float>(capacity(random));
            sink[static_cast<std::size_t>(i)] =
                static_cast<float>(capacity(random));
            candidate.add_tweights(
                static_cast<std::size_t>(i),
                source[static_cast<std::size_t>(i)],
                sink[static_cast<std::size_t>(i)]);
        }
        for (int i = 0; i < nodes; ++i) {
            for (int j = i + 1; j < nodes; ++j) {
                if ((capacity(random) & 1) == 0) continue;
                const Arc arc{
                    i, j, static_cast<float>(capacity(random)),
                    static_cast<float>(capacity(random))};
                arcs.push_back(arc);
                candidate.add_edge(i, j, arc.forward, arc.reverse);
            }
        }
        const auto cut_energy = [&](const unsigned mask) {
            float energy = 0.F;
            for (int i = 0; i < nodes; ++i) {
                const bool source_side = (mask & (1U << i)) != 0;
                energy += source_side
                    ? sink[static_cast<std::size_t>(i)]
                    : source[static_cast<std::size_t>(i)];
            }
            for (const Arc& arc : arcs) {
                const bool a_source = (mask & (1U << arc.a)) != 0;
                const bool b_source = (mask & (1U << arc.b)) != 0;
                if (a_source && !b_source) energy += arc.forward;
                if (b_source && !a_source) energy += arc.reverse;
            }
            return energy;
        };
        float optimum = std::numeric_limits<float>::infinity();
        for (unsigned mask = 0; mask < (1U << nodes); ++mask)
            optimum = std::min(optimum, cut_energy(mask));
        candidate.maxflow();
        unsigned result_mask = 0;
        for (int i = 0; i < nodes; ++i)
            if (candidate.is_source_side(static_cast<std::size_t>(i)))
                result_mask |= 1U << i;
        require(
            std::abs(cut_energy(result_mask) - optimum) < 1e-4F,
            "push-relabel cut differs from brute-force optimum");
    }
}

void test_mesh_clean() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.F, 0.F, 0.F}, Vec3f{1.F, 0.F, 0.F},
        Vec3f{1.F, 1.F, 0.F}, Vec3f{0.F, 1.F, 0.F},
        Vec3f{5.F, 5.F, 5.F}};
    mesh.faces = {
        Eigen::Vector3i{0, 1, 2}, Eigen::Vector3i{0, 3, 2},
        Eigen::Vector3i{2, 1, 0}, Eigen::Vector3i{0, 0, 1}};
    DensifyOptions options;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options);
    require(mesh.faces.size() == 2, "mesh clean kept invalid faces");
    require(mesh.vertices.size() == 4, "mesh clean did not compact vertices");
    require(mesh.normals.size() == 4, "mesh clean did not rebuild normals");

    int shared_forward = 0;
    int shared_reverse = 0;
    for (const Eigen::Vector3i& face : mesh.faces) {
        for (int i = 0; i < 3; ++i) {
            const int a = face[i];
            const int b = face[(i + 1) % 3];
            if (a == 0 && b == 2) ++shared_forward;
            if (a == 2 && b == 0) ++shared_reverse;
        }
    }
    require(
        shared_forward == 1 && shared_reverse == 1,
        "mesh clean did not orient the shared edge consistently");
}

void test_roi_aware_mesh_clean() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{-0.2F, -0.2F, 0.F}, Vec3f{0.2F, -0.2F, 0.F},
        Vec3f{0.F, 0.2F, 0.F}, Vec3f{2.F, 0.F, 0.F},
        Vec3f{-0.2F, -0.2F, -0.2F}, Vec3f{0.2F, -0.2F, -0.2F},
        Vec3f{0.F, 0.2F, -0.2F}, Vec3f{0.F, 0.F, 0.2F}};
    mesh.faces = {
        Eigen::Vector3i{0,1,2}, Eigen::Vector3i{1,3,2},
        // Tetrahedron with the (4, 6, 5) face missing: this is an internal
        // reconstruction hole and must still close under an active ROI.
        Eigen::Vector3i{4,5,7}, Eigen::Vector3i{5,6,7},
        Eigen::Vector3i{6,4,7}};
    OrientedBoundingBox roi;
    roi.valid = true;
    roi.half_extent = Vec3f{0.5F,0.5F,0.5F};
    DensifyOptions options;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 8;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options, &roi);
    require(
        mesh.faces.size() == 7,
        "ROI Clean did not distinguish an ROI cut from an internal hole");
    for (const Vec3f& vertex : mesh.vertices)
        require(roi.contains(vertex), "ROI Clean kept outside vertex");
}

void test_bow_tie_holes_are_split_and_closed() {
    Mesh mesh;
    mesh.vertices = {
        Vec3f{0.F, 0.F, 0.F}, Vec3f{1.F, 0.F, 0.F},
        Vec3f{0.F, 1.F, 0.F}, Vec3f{0.F, 0.F, 1.F},
        Vec3f{-1.F, 0.F, 0.F}, Vec3f{0.F, -1.F, 0.F},
        Vec3f{0.F, 0.F, -1.F}};
    mesh.faces = {
        Eigen::Vector3i{0,3,1}, Eigen::Vector3i{1,3,2},
        Eigen::Vector3i{2,3,0}, Eigen::Vector3i{0,6,4},
        Eigen::Vector3i{4,6,5}, Eigen::Vector3i{5,6,0}};
    DensifyOptions options;
    options.mesh_method = MeshMethod::depth_projective;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 8;
    options.mesh_spurious_factor = 0.F;
    options.mesh_remove_spikes = false;
    detail::clean_mesh(mesh, options);
    require(
        mesh.faces.size() == 12,
        "bow-tie boundary fans were not split and closed independently");
    require(
        mesh.vertices.size() == 10,
        "bow-tie cleanup did not duplicate the shared vertex and add caps");
}

#if defined(AETHERSCAN_HAS_CGAL)
void test_global_delaunay_mesh() {
    MvsScene scene;
    const std::array<Vec3f, 6> cameras{
        Vec3f{3.F, 0.F, 0.F}, Vec3f{-3.F, 0.F, 0.F},
        Vec3f{0.F, 3.F, 0.F}, Vec3f{0.F, -3.F, 0.F},
        Vec3f{0.F, 0.F, 3.F}, Vec3f{0.F, 0.F, -3.F}};
    for (std::size_t i = 0; i < cameras.size(); ++i) {
        MvsView view;
        view.id = static_cast<Index>(i);
        view.pose = aetherscan::sfm::Pose3D::identity();
        view.pose.C = cameras[i].cast<double>();
        view.fx = view.fy = 500.F;
        scene.views.push_back(std::move(view));
    }
    for (int latitude = 1; latitude < 12; ++latitude) {
        const float phi = std::numbers::pi_v<float> *
                          static_cast<float>(latitude) / 12.F;
        for (int longitude = 0; longitude < 24; ++longitude) {
            const float theta = 2.F * std::numbers::pi_v<float> *
                                static_cast<float>(longitude) / 24.F;
            DensePoint point;
            point.position = Vec3f{
                std::sin(phi) * std::cos(theta),
                std::sin(phi) * std::sin(theta), std::cos(phi)};
            point.normal = point.position;
            point.color = (point.position.array() * 0.5F + 0.5F).matrix();
            point.weight = 1.F;
            for (std::size_t i = 0; i < cameras.size(); ++i)
                if (cameras[i].dot(point.position) > 0.F)
                    point.views.push_back(static_cast<Index>(i));
            scene.dense_cloud.points.push_back(std::move(point));
        }
    }
    DensifyOptions options;
    options.mesh_method = MeshMethod::delaunay_cut;
    options.mesh_max_points = 0;
    options.mesh_min_component_faces = 1;
    options.mesh_close_hole_edges = 0;
    options.mesh_k_inf = 1.0e4F;
    options.mesh_k_qual = 0.05F;
    options.mesh_k_behind = 1.F;
    // A per-facet edge cutoff must not puncture the closed graph-cut surface.
    options.mesh_max_edge_voxels = 1.F;
    require(
        detail::reconstruct_mesh_global_cgal(scene, options),
        "global Delaunay backend rejected the sphere");
    std::map<std::pair<int, int>, unsigned> edge_counts;
    for (const Eigen::Vector3i& face : scene.mesh.faces) {
        for (int edge = 0; edge < 3; ++edge) {
            int a = face[edge];
            int b = face[(edge + 1) % 3];
            if (a > b) std::swap(a, b);
            ++edge_counts[{a, b}];
        }
    }
    require(
        std::none_of(
            edge_counts.begin(), edge_counts.end(),
            [](const auto& item) { return item.second == 1; }),
        "global graph-cut surface was punctured by facet filtering");
    detail::clean_mesh(scene.mesh, options);
    require(!scene.mesh.faces.empty(), "global Delaunay mesh is empty");
    require(
        scene.mesh.normals.size() == scene.mesh.vertices.size(),
        "global Delaunay mesh normals are incomplete");
}
#endif

}  // namespace

int main() {
    try {
        test_parallel_fusion();
        test_mask_and_roi_constrained_fusion();
        test_projected_mesh_mask();
        test_input_mask_is_not_cut_by_coarse_mesh_holes();
        test_manual_obb_file();
        test_automatic_ground_and_subject_roi();
        test_projective_mesh();
        test_quality_presets();
#if !defined(AETHERSCAN_HAS_CGAL)
        test_missing_cgal_fails_before_densify();
#endif
        test_scalable_maxflow_cut();
        test_mesh_clean();
        test_roi_aware_mesh_clean();
        test_bow_tie_holes_are_split_and_closed();
#if defined(AETHERSCAN_HAS_CGAL)
        test_global_delaunay_mesh();
#endif
        std::cout << "mvs tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "mvs test failed: " << error.what() << '\n';
        return 1;
    }
}
