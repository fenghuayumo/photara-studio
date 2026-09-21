// Offline recovery experiment. Reads a retained checkpoint; never overwrites it.
#include "sfm/checkpoint.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/asfm.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include "core/logging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <algorithm>
#include <cmath>
#include <string>
#include "sfm_focal_probe.hpp"

int main(int argc, char** argv) {
    using namespace photara::sfm;
    if (argc != 3 && !(argc == 4 && (std::string(argv[3]) == "reattach" || std::string(argv[3]) == "focal"))) {
        std::cerr << "checkpoint.bin output_prefix [reattach|focal]\n"; return 1;
    }
    const std::filesystem::path input(argv[1]), output(argv[2]);
    photara::core::Logger::instance().configure(output.parent_path(), "recovery-probe");
    if (std::filesystem::exists(output.string()+".asfm")) return 2;
    const auto stem = input.stem().string();
    const auto key = std::stoull(stem.substr(stem.find('-')+1), nullptr, 16);
    CheckpointOptions options; options.directory=input.parent_path(); options.write=false;
    Scene scene;
    if (!CheckpointStore(options).load_scene(CheckpointStage::reconstruction,key,scene)) return 3;
    // A rejected final checkpoint retains candidate poses but clears their
    // registration flags. Offline component experiments need the original
    // candidate set back to exercise recovery from the mapping state.
    if (argc == 4 && std::string(argv[3]) == "reattach")
        for (auto& image : scene.images) image.registered = true;
    if (argc == 4 && std::string(argv[3]) == "focal") probe_small_group_focals(scene);
    auto audit=analyze_alignment_observability(scene);
    std::cout << "Original reliable " << audit.reliable_views << '\n';
    recover_weakly_connected_views(scene);
    recover_stable_resections(scene);
    audit=analyze_alignment_observability(scene);
    std::vector<unsigned> counts(scene.images.size());
    std::vector<double> squared(scene.images.size());
    std::vector<std::vector<double>> errors(scene.images.size());
    for (const auto& track:scene.tracks) if (track.is_triangulated())
        for (std::size_t o=0;o<std::min<std::size_t>(track.num_inliers,track.observations.size());++o) {
            const auto& obs=track.observations[o]; const auto& im=scene.images[obs.image_id];
            if (!im.registered) continue;
            const auto p=im.pose.transform_world_to_camera(track.position);
            if (p.z()<=0) continue;
            const auto& kp=im.features.keypoints[obs.feature_id];
            const double error=(scene.camera_of(im).project(p)-Vec2(kp.x,kp.y)).squaredNorm();
            if (std::isfinite(error)) { ++counts[obs.image_id]; squared[obs.image_id]+=error; errors[obs.image_id].push_back(std::sqrt(error)); }
        }
    std::ofstream csv(output.string()+"_sfm_diagnostics.csv"); csv << std::setprecision(15);
    csv << "name,registered,center_x,center_y,center_z,qw,qx,qy,qz,observations,reprojection_rms_px,reprojection_p95_px,alignment_reliable\n";
    for (Index i=0;i<scene.images.size();++i) {
        const auto& im=scene.images[i];const auto q=im.pose.quaternion();
        auto& values=errors[i];
        std::sort(values.begin(),values.end());
        const double rank=values.empty()?0:0.95*(values.size()-1);
        const auto lower=static_cast<std::size_t>(rank);
        const double p95=values.empty()?0:values[lower]+(rank-lower)*(values[std::min(lower+1,values.size()-1)]-values[lower]);
        csv << im.path.filename().string() << ',' << im.registered << ','
            << im.pose.C.x()<<','<<im.pose.C.y()<<','<<im.pose.C.z()<<','
            << q.w()<<','<<q.x()<<','<<q.y()<<','<<q.z()<<','<<counts[i]<<','
            << (counts[i]?std::sqrt(squared[i]/counts[i]):0)<<','<<p95<<','<<int(audit.reliable[i])<<'\n';
    }
    save_asfm(scene,output.string()+".asfm");
}
