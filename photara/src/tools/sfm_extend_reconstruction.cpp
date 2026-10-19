// Extend a retained full-feature ASFM with newly verified correspondences.
// Candidate poses/points never enter the solver; the retained map is frozen.
#include "sfm/asfm.hpp"
#include "sfm/checkpoint.hpp"
#include "sfm/submap_recovery.hpp"
#include "core/logging.hpp"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <stdexcept>

int main(int argc, char** argv) {
    using namespace photara::sfm;
    try {
        if (argc != 4) throw std::runtime_error(
            "baseline-full.asfm correspondence-checkpoint.bin output.asfm");
        const std::filesystem::path input(argv[2]), output(argv[3]);
        auto csv_path = output;
        csv_path.replace_extension();
        csv_path += "_sfm_diagnostics.csv";
        if (std::filesystem::exists(output) || std::filesystem::exists(csv_path))
            throw std::runtime_error("output already exists");
        photara::core::Logger::instance().configure(output.parent_path(), "frozen-extension");
        Scene baseline = load_asfm(argv[1]), correspondences;
        const auto stem = input.stem().string();
        const auto key = std::stoull(stem.substr(stem.find('-') + 1), nullptr, 16);
        CheckpointOptions options;
        options.directory = input.parent_path();
        options.write = false;
        if (!CheckpointStore(options).load_scene(CheckpointStage::reconstruction, key, correspondences))
            throw std::runtime_error("cannot read correspondence checkpoint");
        if (baseline.images.size() != correspondences.images.size())
            throw std::runtime_error("image sets differ");
        std::vector<std::uint8_t> stable(baseline.images.size());
        std::vector<Pose3D> frozen;
        for (Index i = 0; i < baseline.images.size(); ++i) {
            auto& old = baseline.images[i];
            const auto& fresh = correspondences.images[i];
            if (old.path.lexically_normal() != fresh.path.lexically_normal() ||
                old.features.image_width != fresh.features.image_width ||
                old.features.image_height != fresh.features.image_height ||
                old.features.keypoints.size() > fresh.features.keypoints.size())
                throw std::runtime_error("incompatible image/features: " + old.path.string());
            // Reject compact/remapped feature indices and extraction changes.
            for (std::size_t j = 0; j < old.features.keypoints.size(); ++j) {
                const auto& a = old.features.keypoints[j];
                const auto& b = fresh.features.keypoints[j];
                if (a.x != b.x || a.y != b.y || a.scale != b.scale ||
                    a.orientation != b.orientation || a.response != b.response)
                    throw std::runtime_error("feature indices differ: " + old.path.string());
            }
            stable[i] = old.registered;
            frozen.push_back(old.pose);
            if (!stable[i]) old.features = fresh.features;
        }
        // Do not import candidate registration, poses, intrinsics or tracks.
        for (auto& pair : correspondences.pairs) {
            if (!pair.active || pair.zero_baseline || pair.id1 >= stable.size() ||
                pair.id2 >= stable.size() || stable[pair.id1] == stable[pair.id2]) continue;
            const auto& a = baseline.images[pair.id1].features.keypoints;
            const auto& b = baseline.images[pair.id2].features.keypoints;
            std::erase_if(pair.matches, [&](const auto& m) { return m.query >= a.size() || m.train >= b.size(); });
            if (!pair.matches.empty()) baseline.pairs.push_back(std::move(pair));
        }
        correspondences.clear();
        const auto before = baseline.registered_count();
        const auto recovered = recover_stable_resections(baseline, stable);
        for (Index i = 0; i < stable.size(); ++i) {
            if (!stable[i]) continue;
            if (!baseline.images[i].registered ||
                !(baseline.images[i].pose.R.array() == frozen[i].R.array()).all() ||
                !(baseline.images[i].pose.C.array() == frozen[i].C.array()).all())
                throw std::runtime_error("frozen pose changed");
        }
        for (Index i : recovered) baseline.images[i].position_audit = PositionAudit::unverified;
        std::ofstream csv(csv_path);
        csv << std::setprecision(17)
            << "image_id,name,registered,camera_id,width,height,fx,fy,cx,cy,k1,k2,p1,p2,center_x,center_y,center_z,qw,qx,qy,qz,camera_model\n";
        for (const auto& im : baseline.images) {
            const auto& c = baseline.camera_of(im);
            const auto q = im.pose.quaternion();
            csv << im.id << ',' << std::quoted(im.path.filename().string()) << ',' << im.registered << ','
                << im.camera_id << ',' << c.width << ',' << c.height << ',' << c.fx << ',' << c.fy << ','
                << c.cx << ',' << c.cy << ',' << c.k1 << ',' << c.k2 << ',' << c.p1 << ',' << c.p2 << ','
                << im.pose.C.x() << ',' << im.pose.C.y() << ',' << im.pose.C.z() << ','
                << q.w() << ',' << q.x() << ',' << q.y() << ',' << q.z() << ','
                << (c.model == photara::CameraModel::opencv_fisheye ? "opencv_fisheye" :
                    c.model == photara::CameraModel::equirectangular ? "equirectangular" : "pinhole") << '\n';
        }
        if (!csv) throw std::runtime_error("cannot write diagnostics");
        save_asfm(baseline, output);
        std::cout << "before=" << before << " added=" << recovered.size()
                  << " after=" << baseline.registered_count() << " frozen_poses_equal=1\n";
        for (Index i : recovered) std::cout << "added_image=" << baseline.images[i].path.filename() << '\n';
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
