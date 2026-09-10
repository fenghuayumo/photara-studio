#include "project/archive.hpp"
#include "project/document.hpp"
#include "io/format_version.hpp"
#include "sfm/asfm.hpp"
#include "sfm/scene.hpp"

#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using namespace aetherscan::sfm;
using namespace aetherscan::project;

int failures = 0;
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
    const auto asfm_path = dir / "aetherscan_project_io.asfm";
    const auto ascan_path = dir / "aetherscan_project_io.ascan";
    std::error_code error;
    std::filesystem::remove(asfm_path, error);
    std::filesystem::remove(ascan_path, error);

    const Scene source = make_scene();
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
    fish_source.cameras[0].model = aetherscan::CameraModel::opencv_fisheye;
    fish_source.cameras[0].p1 = 0.0003;
    fish_source.cameras[0].p2 = -0.00002;
    const auto fish_restored = decode_asfm(encode_asfm(fish_source));
    expect(fish_restored.cameras[0].model == aetherscan::CameraModel::opencv_fisheye,
           "asfm fish model round trip");
    expect(fish_restored.cameras[0].p2 == -0.00002, "asfm fish k4 round trip");
    expect(restored.cameras[0].model == aetherscan::CameraModel::pinhole,
           "asfm pinhole default");

    Settings settings;
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
    settings.sfm_mode = 2;
    settings.max_features = 4096;
    settings.build_mesh = true;

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
    expect(round_trip.sfm_mode == 2, "settings sfm mode");
    expect(round_trip.max_features == 4096, "settings max features");
    expect(round_trip.build_mesh, "settings build mesh");
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

    const auto ascan_copy = dir / "aetherscan_project_io_copy.ascan";
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
    settings_bytes.insert(settings_bytes.end(), {0x11, 0x22, 0x33, 0x44});
    const Settings additive = decode_settings(settings_bytes, ascan_path);
    expect(additive.name == "demo", "settings ignore trailing fields");
    expect(additive.iterations == 1234, "settings prefix still loads");

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

    Archive required_unknown = Archive::create();
    write_settings(required_unknown, updated, ascan_path);
    required_unknown.set_chunk(
        static_cast<ChunkType>(101), {1},
        aetherscan::io::k_chunk_must_understand);
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
        std::cout << "project io tests passed\n";
        return 0;
    }
    return 1;
}
