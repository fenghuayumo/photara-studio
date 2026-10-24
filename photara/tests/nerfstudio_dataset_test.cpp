#include "splat/dataset.hpp"
#include "io/image.hpp"
#include <fstream>
#include <iostream>
#include <cmath>
#include <stdexcept>
using namespace photara;
void check(bool value, const char* message) { if (!value) throw std::runtime_error(message); }
int main(int argc, char** argv) {
 try {
    if (argc > 1) {
        splat::DatasetLoadRequest request;
        request.source = argv[1];
        auto result = splat::load_splat_dataset(request);
        std::cout << splat::dataset_format_name(result.format) << " views=" << result.scene.views.size()
                  << " points=" << result.scene.dense_cloud.points.size() << '\n';
        return 0;
    }
    const auto root = std::filesystem::temp_directory_path() / "photara_nerfstudio_test";
    std::filesystem::create_directories(root / "images");
    io::save_rgb_png(io::RgbImage{8,6,std::vector<std::uint8_t>(8*6*3,127)},root/"images/a.png");
    auto write = [&](const std::string& text) { std::ofstream(root/"transforms.json") << text; };
    const std::string pose = R"("transform_matrix":[[1,0,0,1],[0,1,0,2],[0,0,1,3],[0,0,0,1]])";
    write(R"({"fl_x":12,"fl_y":14,"w":8,"h":6,"frames":[{"file_path":"images/a.png",)" + pose +
          R"(},{"file_path":"images/a.png","fl_x":16,"cx":3,)" + pose + "}]}");
    splat::DatasetLoadRequest request;
    request.source = root;
    request.random_initial_point_count = 16;
    auto loaded = splat::load_splat_dataset(request);
    check(loaded.format == splat::DatasetFormat::nerfstudio && loaded.scene.views.size()==2,"format/frame detection");
    check(loaded.generated_initial_points && loaded.scene.dense_cloud.points.size()==16,"random initialization");
    const auto& v = loaded.scene.views[0];
    check(v.fx==12 && v.fy==14 && v.cx==4 && v.cy==3,"global intrinsics");
    check(loaded.scene.views[1].fx==16 && loaded.scene.views[1].cx==3,"per-frame intrinsics");
    check((v.pose.C-Eigen::Vector3d(1,2,3)).norm()<1e-9,"camera center");
    check((v.pose.R.transpose()*Eigen::Vector3d::UnitZ()-Eigen::Vector3d(0,0,-1)).norm()<1e-9,"OpenGL conversion");
    write(R"({"camera_angle_x":1.5707963267948966,"frames":[{"file_path":"images/a",)" + pose + "}]}");
    request.source = root/"transforms.json";
    loaded = splat::load_splat_dataset(request);
    check(std::abs(loaded.scene.views[0].fx-4)<1e-5,"Blender angle/extension fallback");
    write(R"({"fl_x":12,"w":16,"h":12,"frames":[{"file_path":"images/a.png",)" + pose + "}]}");
    loaded = splat::load_splat_dataset(request);
    check(loaded.scene.views[0].fx==6,"resized image focal");
    write(R"({"camera_model":"OPENCV_FISHEYE","frames":[{"file_path":"images/a.png",)" + pose + "}]}");
    bool rejected=false;
    try { (void)splat::load_splat_dataset(request); } catch(const std::exception&) { rejected=true; }
    check(rejected,"unsupported cameras must fail");
    write(R"({"fl_x":12,"frames":[{"file_path":"images/a.png","transform_matrix":[[2,0,0,0],[0,1,0,0],[0,0,1,0],[0,0,0,1]]}]})");
    rejected=false;
    try { (void)splat::load_splat_dataset(request); } catch(const std::exception&) { rejected=true; }
    check(rejected,"scaled pose must fail");
    std::filesystem::remove_all(root);
    std::cout << "Nerfstudio dataset checks passed\n";
 } catch (const std::exception& e) { std::cerr << e.what() << '\n'; return 1; }
}
