#include "sfm/export_mvs.hpp"
#include "sfm/scene.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>

namespace {

using namespace photara::sfm;

int failures = 0;
void expect(bool c, const char* m) {
    if (!c) {
        std::cerr << "FAIL: " << m << '\n';
        ++failures;
    }
}

}  // namespace

int main() {
    Scene scene;
    PinholeCamera camera;
    camera.width = 100;
    camera.height = 100;
    camera.fx = camera.fy = 50;
    camera.cx = camera.cy = 50;
    scene.cameras = {camera, camera};

    for (Index i = 0; i < 2; ++i) {
        Image image;
        image.id = i;
        image.camera_id = i;
        image.path = "img" + std::to_string(i) + ".jpg";
        image.registered = true;
        image.pose.C = Vec3(static_cast<double>(i), 0, 0);
        image.features.keypoints.resize(1);
        image.features.keypoints[0].x = 50;
        image.features.keypoints[0].y = 50;
        scene.images.push_back(std::move(image));
    }

    Track track;
    track.position = Vec3(0.5, 0, 2);
    track.observations = {{0, 0}, {1, 0}};
    track.num_inliers = 2;
    scene.tracks.push_back(track);

    const auto path = std::filesystem::path("photara_export_test.mvs");
    ExportMvsOptions options;
    options.sample_colors = false;
    try {
        export_openmvs_interface(scene, path, options);
    } catch (const std::exception& error) {
        std::cerr << "export threw: " << error.what() << '\n';
        ++failures;
    }

    std::ifstream in(path, std::ios::binary);
    expect(static_cast<bool>(in), "mvs file created");
    if (in) {
        char magic[4]{};
        in.read(magic, 4);
        expect(
            magic[0] == 'M' && magic[1] == 'V' && magic[2] == 'S' && magic[3] == 'I',
            "MVSI magic");
    }
    scene.cameras[0].model = photara::CameraModel::opencv_fisheye;
    bool refused_fisheye = false;
    try { export_openmvs_interface(scene, path, options); }
    catch (const std::runtime_error&) { refused_fisheye = true; }
    expect(refused_fisheye, "unrectified fisheye cannot be mislabeled as pinhole MVS");
    std::error_code ec;
    std::filesystem::remove(path, ec);
    if (failures == 0) {
        std::cout << "sfm export_mvs tests passed\n";
        return 0;
    }
    return 1;
}
