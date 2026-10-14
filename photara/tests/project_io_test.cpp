#include "project/archive.hpp"
#include "project/document.hpp"
#include "io/format_version.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_colmap.hpp"
#include "sfm/export_nerfstudio.hpp"
#include "sfm/scene.hpp"
#include "splat/training_preset.hpp"
#include "mvs/mesh_postprocess.hpp"

#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace photara::sfm;
using namespace photara::project;

int failures = 0;
std::vector<std::uint8_t> encode_settings_v6(
    const Settings& settings, const std::filesystem::path& project) {
    auto bytes = encode_settings(settings, project);
    // Version 7 appends two ints, two booleans and two uint64 budgets.
    bytes.resize(bytes.size() - sizeof(std::uint64_t) - 2 * sizeof(std::int32_t) -
        2 * sizeof(std::uint8_t) - 2 * sizeof(std::uint64_t));
    const std::uint32_t version = 6;
    std::memcpy(bytes.data(), &version, sizeof(version));
    return bytes;
}
void expect(bool condition, const char* message) {
    if (!condition) {
        std::cerr << "FAIL: " << message << '\n';
        ++failures;
    }
}

Scene make_scene() {
    Scene scene;
    PinholeCamera camera;
    camera.id = 0;
    camera.width = 100;
    camera.height = 80;
    camera.fx = camera.fy = 50;
    camera.cx = 50;
    camera.cy = 40;
    camera.k1 = 0.1;
    scene.cameras.push_back(camera);

    for (Index i = 0; i < 2; ++i) {
        Image image;
        image.id = i;
        image.camera_id = 0;
        image.path = "img" + std::to_string(i) + ".jpg";
        image.registered = true;
        image.pose.C = Vec3(static_cast<double>(i), 0, 0);
        image.features.image_width = 100;
        image.features.image_height = 80;
        image.features.keypoints.resize(1);
        image.features.keypoints[0].x = 50;
        image.features.keypoints[0].y = 40;
        scene.images.push_back(std::move(image));
    }

    Track track;
    track.position = Vec3(0.5, 0, 2);
    track.observations = {{0, 0}, {1, 0}};
    track.num_inliers = 2;
    scene.tracks.push_back(track);
    return scene;
}

void patch_u32(
    const std::filesystem::path& path, const std::uint64_t offset,
    const std::uint32_t value) {
    std::fstream stream(path, std::ios::in | std::ios::out | std::ios::binary);
    if (!stream) throw std::runtime_error("Failed to patch " + path.string());
    stream.seekp(static_cast<std::streamoff>(offset));
    stream.write(reinterpret_cast<const char*>(&value), sizeof(value));
    if (!stream) throw std::runtime_error("Failed while patching " + path.string());
}

bool throws_on_open(const std::filesystem::path& path) {
    try {
        Archive::open(path);
        return false;
    } catch (const std::exception&) {
        return true;
    }
}

}  // namespace

int main() {
    const auto dir = std::filesystem::temp_directory_path();
    const auto asfm_path = dir / "photara_project_io.asfm";
    const auto ascan_path = dir / "photara_project_io.ascan";
    std::error_code error;
    std::filesystem::remove(asfm_path, error);
    std::filesystem::remove(ascan_path, error);

    const Scene source = make_scene();
    save_asfm(source, asfm_path);
    const auto signature = sfm_source_signature(asfm_path, dir, dir / "signature.ascan");
    expect(signature == sfm_source_signature(asfm_path, dir, dir / "signature.ascan"),
        "unchanged working SfM has a stable source signature");
    expect(signature != sfm_source_signature(asfm_path, dir / "other_images", dir / "signature.ascan"),
        "changing image path base invalidates source signature");
    expect(signature != sfm_source_signature(asfm_path, dir, dir / "other_project" / "signature.ascan"),
        "save-as path base invalidates source signature");
    auto signed_archive = Archive::create();
    signed_archive.set_chunk(ChunkType::sfm_source, signature);
    write_sfm(signed_archive, source, dir / "signature.ascan");
    expect(!signed_archive.has(ChunkType::sfm_source), "replacing SfM invalidates its source signature");
    auto changed_source = source;
    changed_source.images[0].pose.C.x() += 1;
    save_asfm(changed_source, asfm_path);
    expect(signature != sfm_source_signature(asfm_path, dir, dir / "signature.ascan"),
        "changed working SfM invalidates source signature");
    save_asfm(source, asfm_path);
    const Scene restored = load_asfm(asfm_path);
    expect(restored.cameras.size() == 1, "asfm camera count");
    expect(restored.cameras[0].k1 == 0.1, "asfm distortion");
    expect(restored.images.size() == 2, "asfm image count");
    expect(restored.images[0].registered, "asfm registered flag");
    expect(restored.images[0].features.keypoints.size() == 1, "asfm keypoints");
    expect(restored.images[0].features.descriptors.empty(), "asfm omits descriptors");
    expect(restored.tracks.size() == 1, "asfm track count");
    expect(restored.tracks[0].num_inliers == 2, "asfm inliers");
    expect(!restored.tracks[0].has_color, "asfm colour absent by default");
    expect(restored.pairs.empty(), "asfm omits pairs");
    Scene audited=source;
    audited.images[0].position_audit=PositionAudit::unverified;
    audited.images[1].position_audit=PositionAudit::graph_supported;
    const auto audit_bytes=encode_asfm(audited);
    const auto audit_restored=decode_asfm(audit_bytes);
    expect(audit_restored.images[0].position_audit==PositionAudit::unverified &&
           audit_restored.images[1].position_audit==PositionAudit::graph_supported,
           "position uncertainty survives ASFM round trip without changing registration");
    std::uint32_t minimum_reader{};
    std::memcpy(&minimum_reader,audit_bytes.data()+12,sizeof(minimum_reader));
    expect(minimum_reader==k_asfm_position_audit_min_reader,
           "unverified poses require a reader that preserves the warning");
    expect(restored.images[0].position_audit==PositionAudit::not_audited,
           "unaudited models are not implicitly promoted to verified");

    const auto ply_path = dir / "photara_project_io_sparse.ply";
    std::filesystem::remove(ply_path, error);
    save_sparse_ply(source, ply_path);
    std::ifstream ply(ply_path);
    expect(static_cast<bool>(ply), "sparse ply opens");
    std::string ply_line;
    bool saw_vertex = false;
    while (std::getline(ply, ply_line)) {
        if (ply_line.find("element vertex 1") != std::string::npos)
            saw_vertex = true;
    }
    expect(saw_vertex, "sparse ply writes triangulated tracks");
    ply.close();
    std::filesystem::remove(ply_path, error);

    const auto colmap_dir = dir / "photara_project_io_colmap";
    std::filesystem::remove_all(colmap_dir, error);
    save_colmap_text(source, colmap_dir, {}, true);
    std::ifstream cameras_txt(colmap_dir / "cameras.txt");
    expect(static_cast<bool>(cameras_txt), "colmap cameras.txt opens");
    std::string cameras_body(
        (std::istreambuf_iterator<char>(cameras_txt)),
        std::istreambuf_iterator<char>());
    expect(cameras_body.find("OPENCV") != std::string::npos, "colmap OPENCV camera");
    std::ifstream images_txt(colmap_dir / "images.txt");
    expect(static_cast<bool>(images_txt), "colmap images.txt opens");
    std::string images_body(
        (std::istreambuf_iterator<char>(images_txt)),
        std::istreambuf_iterator<char>());
    expect(images_body.find("50") != std::string::npos, "colmap writes 2D observations");
    std::ifstream points_txt(colmap_dir / "points3D.txt");
    expect(static_cast<bool>(points_txt), "colmap points3D.txt opens");
    std::string points_body(
        (std::istreambuf_iterator<char>(points_txt)),
        std::istreambuf_iterator<char>());
    expect(points_body.find("0.5") != std::string::npos, "colmap writes triangulated points");
    std::filesystem::remove_all(colmap_dir, error);

    const auto json_path = dir / "photara_project_io_transforms.json";
    std::filesystem::remove(json_path, error);
    save_nerfstudio_transforms(source, json_path);
    std::ifstream json(json_path);
    expect(static_cast<bool>(json), "nerfstudio json opens");
    std::string json_body(
        (std::istreambuf_iterator<char>(json)),
        std::istreambuf_iterator<char>());
    expect(json_body.find("OPENCV") != std::string::npos, "nerfstudio camera model");
    expect(
        json_body.find("transform_matrix") != std::string::npos,
        "nerfstudio transform_matrix");
    expect(json_body.find("\"fl_x\":50") != std::string::npos, "nerfstudio fl_x");
    json.close();
    std::filesystem::remove(json_path, error);

    Scene colored = source;
    colored.tracks[0].has_color = true;
    colored.tracks[0].color_r = 12;
    colored.tracks[0].color_g = 34;
    colored.tracks[0].color_b = 56;
    const Scene colored_restored = decode_asfm(encode_asfm(colored));
    expect(colored_restored.tracks[0].has_color, "asfm colour round trip flag");
    expect(colored_restored.tracks[0].color_r == 12, "asfm colour r");
    expect(colored_restored.tracks[0].color_g == 34, "asfm colour g");
    expect(colored_restored.tracks[0].color_b == 56, "asfm colour b");

    Scene fish_source = source;
    fish_source.cameras[0].model = photara::CameraModel::opencv_fisheye;
    fish_source.cameras[0].p1 = 0.0003;
    fish_source.cameras[0].p2 = -0.00002;
    const auto fish_restored = decode_asfm(encode_asfm(fish_source));
    expect(fish_restored.cameras[0].model == photara::CameraModel::opencv_fisheye,
           "asfm fish model round trip");
    expect(fish_restored.cameras[0].p2 == -0.00002, "asfm fish k4 round trip");
    expect(restored.cameras[0].model == photara::CameraModel::pinhole,
           "asfm pinhole default");

    Settings settings;
    expect(!settings.focus_region, "new projects default focus region off");
    expect(settings.iterations == 30'000 && settings.densification_cap == 3'000'000 &&
           settings.training_mode == 1 && !settings.speedy_pruning,
           "new projects default to standard 3DGS training");
    expect(!settings.sam_use_mask_for_sfm, "SAM alignment masks default off");
    settings.name = "demo";
    settings.image_directory = dir / "images";
    settings.dataset_source = dir / "colmap";
    settings.dataset_format = "colmap";
    settings.dataset_initial_cloud = dir / "seed.ply";
    settings.splat_output_format = "glb";
    settings.splat_model_source = dir / "imported.glb";
    settings.video_frames_dir = dir / "clip" / "images";
    settings.video_fps = 3.5F;
    settings.video_sharp_window = 5;
    settings.video_max_frames = 400;
    settings.video_quality = 90;
    settings.video_scale = 0.5F;
    settings.video_rotate = 90;
    settings.camera_model = 2;
    settings.strategy = 1;
    settings.densification_cap = 2'500'000;
    settings.training_mode = 0;
    settings.speedy_pruning = true;
    settings.sh_degree = 2;
    settings.sfm_mode = 2;
    settings.max_features = 4096;
    settings.build_mesh = true;
    settings.mesh_quality = 2;
    settings.mesh_postprocess_quality = 0;
    settings.mesh_simplify = true;
    settings.mesh_target_faces = 730'000;
    settings.mesh_remesh = true;
    settings.mesh_remesh_target_faces = 1'500'000;
    settings.mesh_remesh_smooth_iterations = 3;
    settings.mesh_source = 1;
    settings.texture_quality = 2;
    settings.atlas_resolution = 4096;
    settings.texture_delight = true;
    settings.texture_optimize = false;
    settings.texture_photometric_steps = 4500;
    settings.texture_seam_steps = 90;
    settings.texture_blend_linear = true;
    settings.mask_mode = 1;
    settings.sam_masks = true;
    settings.sam_text = "glass cup";
    settings.sam_use_mask_for_sfm = true;

    Archive archive = Archive::create();
    replace_sfm_stage(archive, source, settings, ascan_path);
    archive.set_chunk(ChunkType::gaussians, {1, 2, 3, 4});
    archive.save(ascan_path);

    Archive loaded = Archive::open(ascan_path);
    expect(loaded.has(ChunkType::settings), "ascan settings chunk");
    expect(loaded.has(ChunkType::sfm), "ascan sfm chunk");
    expect(loaded.has(ChunkType::gaussians), "ascan gaussians chunk");
    const Settings round_trip = read_settings(loaded);
    expect(round_trip.name == "demo", "settings name");
    expect(
        round_trip.dataset_source == settings.dataset_source,
        "external dataset source");
    expect(
        round_trip.dataset_format == "colmap",
        "external dataset format");
    expect(
        round_trip.dataset_initial_cloud == settings.dataset_initial_cloud,
        "external dataset initializer");
    expect(
        round_trip.splat_output_format == settings.splat_output_format,
        "splat output format");
    expect(
        round_trip.splat_model_source == settings.splat_model_source,
        "splat model source");
    expect(
        round_trip.video_frames_dir == settings.video_frames_dir,
        "video frames dir");
    expect(round_trip.video_fps == 3.5F, "video fps");
    expect(round_trip.video_sharp_window == 5, "video sharp window");
    expect(round_trip.video_max_frames == 400, "video max frames");
    expect(round_trip.video_quality == 90, "video quality");
    expect(round_trip.video_scale == 0.5F, "video scale");
    expect(round_trip.video_rotate == 90, "video rotate");
    expect(round_trip.camera_model == 2, "settings camera model");
    expect(round_trip.strategy == 1, "settings splat strategy");
    expect(
        round_trip.densification_cap == 2'500'000,
        "settings densification cap");
    expect(round_trip.sh_degree == 2, "settings SH degree");
    expect(round_trip.training_mode == 0 && round_trip.speedy_pruning,
           "training mode and pruning survive project roundtrip");
    expect(round_trip.mask_mode == 1, "settings mask mode");
    expect(round_trip.sam_use_mask_for_sfm, "SAM alignment mask opt-in round trip");
    expect(round_trip.sam_masks && round_trip.sam_text == "glass cup",
           "SAM generation settings preserved with alignment opt-in");
    expect(round_trip.sfm_mode == 2, "settings sfm mode");
    auto retired_sfm_settings = settings;
    retired_sfm_settings.sfm_mode = 1;
    expect(
        decode_settings(encode_settings(retired_sfm_settings, ascan_path),
                        ascan_path).sfm_mode == 2,
        "legacy incremental SfM projects migrate to hierarchical");
    auto global_sfm_settings = settings;
    global_sfm_settings.sfm_mode = 0;
    expect(
        decode_settings(encode_settings(global_sfm_settings, ascan_path),
                        ascan_path).sfm_mode == 0,
        "global SfM project mode is preserved");
    expect(round_trip.max_features == 4096, "settings max features");
    expect(round_trip.build_mesh, "settings build mesh");
    expect(round_trip.mesh_quality == 2, "settings mesh quality");
    expect(round_trip.mesh_postprocess_quality == 0 && round_trip.mesh_simplify &&
        round_trip.mesh_target_faces == 730'000 && round_trip.mesh_remesh &&
        round_trip.mesh_remesh_target_faces == 1'500'000 &&
        round_trip.mesh_remesh_smooth_iterations == 3,
        "independent postprocessing budgets and smoothing survive project roundtrip");
    const auto old_postprocess = decode_settings(encode_settings_v6(settings, ascan_path), ascan_path);
    expect(!old_postprocess.mesh_simplify && !old_postprocess.mesh_remesh &&
        old_postprocess.mesh_target_faces == 1'000'000,
        "old projects do not automatically simplify or remesh existing geometry");
    {
        auto truncated = encode_settings(settings, ascan_path);
        truncated.pop_back();
        bool rejected = false;
        try { (void)decode_settings(truncated, ascan_path); }
        catch (const std::exception&) { rejected = true; }
        expect(rejected, "truncated version 7 postprocessing settings are rejected");
        auto invalid = settings;
        invalid.mesh_postprocess_quality = 10;
        invalid.mesh_target_faces = 0;
        invalid.mesh_remesh_target_faces = 3;
        invalid.mesh_remesh_smooth_iterations = -1;
        const auto restored = decode_settings(encode_settings(invalid, ascan_path), ascan_path);
        expect(restored.mesh_postprocess_quality == 1 && restored.mesh_target_faces == 1'000'000 &&
            restored.mesh_remesh_target_faces == 1'000'000 && restored.mesh_remesh_smooth_iterations == 1,
            "invalid postprocessing settings receive valid budgets");
    }
    for (int quality : {0, 1, 2}) {
        const auto preset = photara::mvs::mesh_postprocess_preset(quality);
        auto selected = settings;
        selected.mesh_postprocess_quality = quality;
        selected.mesh_target_faces = preset.target_faces;
        selected.mesh_remesh_target_faces = preset.target_faces;
        selected.mesh_remesh_smooth_iterations = preset.smooth_iterations;
        const auto restored = decode_settings(encode_settings(selected, ascan_path), ascan_path);
        expect(restored.mesh_quality == 2 && restored.mesh_postprocess_quality == quality &&
            restored.mesh_target_faces == preset.target_faces &&
            restored.mesh_remesh_smooth_iterations == preset.smooth_iterations,
            "postprocessing presets stay independent of extraction quality");
    }
    auto legacy_settings = encode_settings_v6(settings, ascan_path);
    legacy_settings.resize(legacy_settings.size() - 4 * sizeof(std::int32_t) - 2 * sizeof(std::uint8_t));
    expect(decode_settings(legacy_settings, ascan_path).mesh_quality == 1,
           "legacy project defaults to standard mesh quality");
    for (int quality : {0, 1, 2}) {
        auto selected = settings;
        selected.mesh_quality = quality;
        expect(decode_settings(encode_settings(selected, ascan_path), ascan_path).mesh_quality == quality,
               "mesh quality modes survive project roundtrip");
    }
    for (int mode : {0, 1, 2}) {
        const auto preset = photara::splat::training_preset(
            static_cast<photara::splat::TrainingMode>(mode));
        auto selected = settings;
        selected.training_mode = mode;
        selected.iterations = static_cast<int>(preset.iterations);
        selected.densification_cap = static_cast<int>(preset.gaussian_cap);
        selected.speedy_pruning = preset.speedy_pruning;
        const auto restored = decode_settings(encode_settings(selected, ascan_path), ascan_path);
        expect(restored.training_mode == mode && restored.iterations == selected.iterations &&
               restored.densification_cap == selected.densification_cap &&
               restored.speedy_pruning == selected.speedy_pruning,
               "3DGS presets survive project roundtrip");
    }
    auto old_training = encode_settings_v6(settings, ascan_path);
    old_training.resize(old_training.size() - 3 * sizeof(std::int32_t) - 2 * sizeof(std::uint8_t));
    const auto old_restored = decode_settings(old_training, ascan_path);
    expect(!old_restored.speedy_pruning && old_restored.densification_cap == 2'500'000,
           "legacy projects preserve their cap without enabling Speedy pruning");
    expect(round_trip.mesh_source == 1, "settings mesh source");
    expect(round_trip.texture_quality == 2, "settings texture quality");
    expect(round_trip.atlas_resolution == 4096, "settings atlas resolution");
    expect(round_trip.texture_delight, "settings texture delight");
    expect(!round_trip.texture_optimize, "settings texture optimize");
    expect(round_trip.texture_photometric_steps == 4500 && round_trip.texture_seam_steps == 90,
           "custom texture iteration counts survive project roundtrip");
    for (int quality : {0, 1, 2}) {
        auto selected = settings;
        selected.texture_quality = quality;
        auto old_texture = encode_settings_v6(selected, ascan_path);
        old_texture.resize(old_texture.size() - 2 * sizeof(std::int32_t) - sizeof(std::uint8_t));
        const auto restored = decode_settings(old_texture, ascan_path);
        expect(restored.texture_photometric_steps == (quality == 0 ? 1000 : (quality == 2 ? 6000 : 3000)) &&
               restored.texture_seam_steps == (quality == 0 ? 30 : (quality == 2 ? 120 : 60)),
               "old projects receive texture iteration defaults for their quality preset");
    }
    {
        auto selected = settings;
        selected.texture_seam_steps = 0;
        expect(decode_settings(encode_settings(selected, ascan_path), ascan_path).texture_seam_steps == 0,
               "disabled seam optimization survives project roundtrip");
        selected.texture_photometric_steps = -1;
        selected.texture_seam_steps = -1;
        const auto restored = decode_settings(encode_settings(selected, ascan_path), ascan_path);
        expect(restored.texture_photometric_steps == 3000 && restored.texture_seam_steps == 60,
               "invalid texture iteration counts receive safe defaults");
    }
    expect(round_trip.texture_blend_linear, "settings texture blend linear");
    const auto loaded_scene = read_sfm(loaded);
    expect(loaded_scene.has_value(), "read sfm from ascan");
    expect(loaded_scene && loaded_scene->cameras[0].k1 == 0.1, "ascan sfm distortion");
    expect(loaded.chunk(ChunkType::gaussians).size() == 4, "opaque gaussian blob");

    Settings updated = round_trip;
    updated.iterations = 1234;
    write_settings(loaded, updated, ascan_path);
    loaded.save(ascan_path);
    Archive reopened = Archive::open(ascan_path);
    expect(read_settings(reopened).iterations == 1234, "settings-only rewrite");
    expect(reopened.has(ChunkType::gaussians), "settings rewrite keeps gaussians");
    expect(reopened.chunk(ChunkType::gaussians)[2] == 3, "gaussian bytes preserved");

    const auto ascan_copy = dir / "photara_project_io_copy.ascan";
    std::filesystem::remove(ascan_copy, error);
    Archive copied = Archive::open(ascan_path);
    copied.save(ascan_copy);
    Archive copy_loaded = Archive::open(ascan_copy);
    expect(copy_loaded.has(ChunkType::sfm), "save-as keeps sfm");
    expect(copy_loaded.has(ChunkType::gaussians), "save-as keeps gaussians");
    expect(copy_loaded.chunk(ChunkType::gaussians)[2] == 3, "save-as copies bytes");
    std::filesystem::remove(ascan_copy, error);

    Archive rebuilt = Archive::open(ascan_path);
    replace_sfm_stage(rebuilt, source, updated, ascan_path);
    rebuilt.save(ascan_path);
    Archive after_align = Archive::open(ascan_path);
    expect(after_align.has(ChunkType::sfm), "realign keeps sfm");
    expect(!after_align.has(ChunkType::gaussians), "realign drops gaussians");
    expect(after_align.writer_version() == k_ascan_version, "ascan writer version");
    expect(
        after_align.min_reader_version() == k_ascan_min_reader,
        "ascan min reader version");

    auto settings_bytes = encode_settings(updated, ascan_path);
    {
        auto legacy_bytes = encode_settings_v6(updated, ascan_path);
        // Remove texture steps, training mode/pruning, mesh quality, SAM
        // alignment opt-in, and the Focus Region byte.
        legacy_bytes.resize(legacy_bytes.size() - 4 * sizeof(std::int32_t) - 3 * sizeof(std::uint8_t));
        const Settings legacy = decode_settings(legacy_bytes, ascan_path);
        expect(!legacy.sam_use_mask_for_sfm,
               "old projects default to full-image alignment");
        expect(legacy.sam_masks && legacy.sam_text == "glass cup",
               "old projects preserve SAM generation and training masks");
        Settings disabled = updated;
        disabled.sam_use_mask_for_sfm = false;
        expect(!decode_settings(encode_settings(disabled, ascan_path), ascan_path)
                    .sam_use_mask_for_sfm,
               "SAM alignment mask opt-out round trip");
    }
    {
        Settings toggled = updated;
        toggled.scene_mode = true;
        toggled.focus_region = false;
        const Settings restored =
            decode_settings(encode_settings(toggled, ascan_path), ascan_path);
        expect(restored.scene_mode && !restored.focus_region,
               "focus region is independent of capture mode and round trips");
        auto pre_region = encode_settings_v6(toggled, ascan_path);
        pre_region.resize(pre_region.size() - sizeof(std::uint8_t));
        const Settings legacy_region = decode_settings(pre_region, ascan_path);
        expect(legacy_region.scene_mode && !legacy_region.focus_region,
               "projects saved before focus region default it off");
        for (bool scene_mode : {false, true}) {
            toggled.scene_mode = scene_mode;
            toggled.focus_region = true;
            expect(decode_settings(encode_settings(toggled, ascan_path), ascan_path)
                       .focus_region,
                   "explicit focus region opt-in survives project roundtrip");
            auto legacy = encode_settings_v6(toggled, ascan_path);
            legacy.pop_back();
            expect(!decode_settings(legacy, ascan_path).focus_region,
                   "legacy capture modes do not implicitly enable focus region");
        }
    }
    settings_bytes.insert(settings_bytes.end(), {0x11, 0x22, 0x33, 0x44});
    const Settings additive = decode_settings(settings_bytes, ascan_path);
    expect(additive.name == "demo", "settings ignore trailing fields");
    expect(additive.iterations == 1234, "settings prefix still loads");

    {
        Settings invalid = updated;
        invalid.strategy = 3;
        bool threw = false;
        try {
            (void)decode_settings(
                encode_settings(invalid, ascan_path), ascan_path);
        } catch (const std::invalid_argument&) {
            threw = true;
        }
        expect(threw, "invalid splat strategy index is rejected");
    }

    {
        auto future = settings_bytes;
        std::uint32_t newer_writer = 2;
        std::memcpy(future.data(), &newer_writer, 4);
        const Settings future_settings = decode_settings(future, ascan_path);
        expect(future_settings.name == "demo", "newer additive settings load");
    }
    {
        auto breaking = encode_settings(updated, ascan_path);
        std::uint32_t required = 99;
        std::memcpy(breaking.data(), &required, 4);
        std::memcpy(breaking.data() + 4, &required, 4);
        bool refused = false;
        try {
            decode_settings(breaking, ascan_path);
        } catch (const std::exception&) {
            refused = true;
        }
        expect(refused, "settings min_reader refuses older builds");
    }

    Archive with_unknown = Archive::create();
    write_settings(with_unknown, updated, ascan_path);
    with_unknown.set_chunk(static_cast<ChunkType>(100), {9, 8, 7});
    with_unknown.save(ascan_path);
    Archive unknown_loaded = Archive::open(ascan_path);
    expect(
        unknown_loaded.has(static_cast<ChunkType>(100)),
        "unknown optional chunk is preserved");
    expect(
        unknown_loaded.chunk(static_cast<ChunkType>(100)).size() == 3,
        "unknown chunk bytes copied");
    unknown_loaded.set_chunk(ChunkType::training_checkpoint, {1, 2, 3, 4});
    unknown_loaded.save(ascan_path);
    expect(
        unknown_loaded.chunk(static_cast<ChunkType>(100)) == std::vector<std::uint8_t>({9, 8, 7}),
        "same-file save refreshes offsets of untouched lazy chunks");
    unknown_loaded.erase_chunk(ChunkType::training_checkpoint);
    unknown_loaded.save(ascan_path);
    expect(
        Archive::open(ascan_path).chunk(static_cast<ChunkType>(100)) == std::vector<std::uint8_t>({9, 8, 7}),
        "sequential stage saves retain untouched chunks");
    const auto encoded = unknown_loaded.encode();
    expect(
        Archive::from_bytes(encoded).chunk(static_cast<ChunkType>(100)) == std::vector<std::uint8_t>({9, 8, 7}),
        "in-memory archive round trip preserves optional chunks");
    auto damaged = encoded;
    damaged.back() ^= 1;
    bool checksum_refused = false;
    try { (void)Archive::from_bytes(damaged); }
    catch (const std::exception&) { checksum_refused = true; }
    expect(checksum_refused, "in-memory archive validates payload checksums");

    const auto stream_source = dir / "photara_io_stream_source.ascan";
    const auto stream_target = dir / "photara_io_stream_target.ascan";
    auto stream_archive = Archive::create();
    std::vector<std::uint8_t> large_chunk(1024 * 1024 + 17, 83);
    stream_archive.set_chunk(ChunkType::gaussians, large_chunk);
    stream_archive.save(stream_source);
    auto stream_lazy = Archive::open(stream_source);
    stream_lazy.save(stream_target);
    expect(Archive::open(stream_target).chunk(ChunkType::gaussians) == large_chunk,
        "streamed archive copies a chunk across buffer boundaries");
    {
        std::fstream corrupt(stream_source, std::ios::binary | std::ios::in | std::ios::out);
        corrupt.seekp(-1, std::ios::end);
        corrupt.put(0);
    }
    bool stream_refused = false;
    try { stream_lazy = Archive::open(stream_source); stream_lazy.save(stream_target); }
    catch (const std::exception&) { stream_refused = true; }
    expect(stream_refused, "streamed save validates untouched chunk checksums");
    expect(Archive::open(stream_target).chunk(ChunkType::gaussians) == large_chunk,
        "failed streamed save preserves the destination");
    expect(!std::filesystem::exists(stream_target.string() + ".tmp"),
        "failed streamed save removes its temporary output");

    Archive required_unknown = Archive::create();
    write_settings(required_unknown, updated, ascan_path);
    required_unknown.set_chunk(
        static_cast<ChunkType>(101), {1},
        photara::io::k_chunk_must_understand);
    required_unknown.save(ascan_path);
    expect(throws_on_open(ascan_path), "must_understand unknown chunk refused");

    Archive compat = Archive::create();
    write_settings(compat, updated, ascan_path);
    compat.save(ascan_path);
    patch_u32(ascan_path, 8, 2);
    Archive newer_writer = Archive::open(ascan_path);
    expect(
        newer_writer.writer_version() == 2,
        "newer ascan writer with min_reader 1 still opens");
    patch_u32(ascan_path, 8, 99);
    patch_u32(ascan_path, 20, 99);
    expect(throws_on_open(ascan_path), "ascan min_reader 99 refused");

    save_asfm(source, asfm_path);
    patch_u32(asfm_path, 8, 2);
    const Scene future_asfm = load_asfm(asfm_path);
    expect(future_asfm.cameras.size() == 1, "newer asfm writer still loads");
    patch_u32(asfm_path, 8, 99);
    patch_u32(asfm_path, 12, 99);
    bool asfm_refused = false;
    try {
        load_asfm(asfm_path);
    } catch (const std::exception&) {
        asfm_refused = true;
    }
    expect(asfm_refused, "asfm min_reader 99 refused");

    std::filesystem::remove(asfm_path, error);
    std::filesystem::remove(ascan_path, error);
    if (failures == 0) {
        std::filesystem::remove(stream_source, error);
        std::filesystem::remove(stream_target, error);
        std::cout << "project io tests passed\n";
        return 0;
    }
    return 1;
}
