#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include "sfm/reconstruct.hpp"
#include "sfm/preview.hpp"
#include "sfm/asfm.hpp"
#include "sfm/rotation_recovery.hpp"
#include "core/logging.hpp"
#include "../src/sfm/resection_seeding.hpp"
#include "../../experiments/sfm_resection_refinement.hpp"
#include "../../experiments/sfm_pair_group_init.hpp"
#include "../../experiments/sfm_local_proposals.hpp"
#include "../../experiments/sfm_ray_group_alignment.hpp"
#include "../../experiments/sfm_structureless_probe.hpp"
#include "../../experiments/sfm_cross_anchor_precision.hpp"
#include <iostream>
#include <chrono>
#include <random>
#include <numeric>
#include <stdexcept>

using namespace photara::sfm;

void expect(bool condition, const char* message) {
    if (!condition) throw std::runtime_error(message);
}

Scene fixture(bool shared = true, bool fisheye = false) {
    Scene scene;
    scene.thread_count = 1;
    PinholeCamera camera;
    camera.id = 0; camera.width = 1000; camera.height = 800;
    camera.fx = camera.fy = 700; camera.cx = 500; camera.cy = 400;
    if (fisheye) {
        camera.model = photara::CameraModel::opencv_fisheye;
        camera.fx = camera.fy = 400;
        camera.k1 = 0.025; camera.k2 = -0.002;
        camera.p1 = 0.0002; camera.p2 = -0.00001;
    }
    scene.cameras.push_back(camera);
    for (Index i = 0; i < 8; ++i) {
        Image image;
        image.id = i; image.camera_id = 0; image.registered = true;
        image.pose.C = Vec3(0.25 * i, 0.12 * (i % 2), 0);
        image.features.keypoints.resize(160);
        scene.images.push_back(image);
    }
    std::mt19937 rng(81);
    const double extent = fisheye ? 4.0 : 1.5;
    std::uniform_real_distribution<double> xy(-extent, extent), z(4.0, 6.0);
    for (Index p = 0; p < 160; ++p) {
        Vec3 point(xy(rng), xy(rng), z(rng));
        Track track;
        track.position = Vec3(99, -40, 3);  // must never supply alignment depth
        for (auto& image : scene.images) {
            Vec2 pixel = camera.project(image.pose.transform_world_to_camera(point));
            image.features.keypoints[p].x = static_cast<float>(pixel.x());
            image.features.keypoints[p].y = static_cast<float>(pixel.y());
            if (shared || image.id < 4) track.observations.push_back({image.id, p});
        }
        track.num_inliers = 4;
        scene.tracks.push_back(track);
        if (!shared) {
            Track local;
            for (Index i = 4; i < 8; ++i) local.observations.push_back({i, p});
            scene.tracks.push_back(local);
        }
    }
    for (Index a = 0; a < 8; ++a) for (Index b = a + 1; b < 8; ++b) {
        if (a < 4 && b >= 4 && !(shared && a == 3 && b == 4)) continue;
        ImagePair pair(a, b);
        pair.relative_pose = scene.images[b].pose / scene.images[a].pose;
        pair.relative_pose->C.normalize(); pair.weight_spatial = 1;
        for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
        scene.pairs.push_back(pair);
    }
    for (Index i = 4; i < 8; ++i) {
        scene.images[i].pose.C = 3 * scene.images[i].pose.C + Vec3(10, -2, 1);
        scene.images[i].pose.R = Eigen::AngleAxisd(0.4, Vec3::UnitY()).toRotationMatrix();
    }
    return scene;
}

Scene corridor_fixture() {
    Scene scene;
    scene.thread_count = 1;
    PinholeCamera camera;
    camera.id = 0; camera.width = 1000; camera.height = 800;
    camera.fx = camera.fy = 700; camera.cx = 500; camera.cy = 400;
    scene.cameras.push_back(camera);
    constexpr Index n = 12;
    for (Index i = 0; i < n; ++i) {
        Image image;
        image.id = i; image.camera_id = 0; image.registered = true;
        image.pose.C = Vec3(0.25 * i, 0.12 * (i % 2), 0);
        image.features.keypoints.resize(160);
        scene.images.push_back(image);
    }
    std::mt19937 rng(81);
    std::uniform_real_distribution<double> xy(-1.5, 1.5), z(4.0, 6.0);
    for (Index p = 0; p < 160; ++p) {
        Vec3 point(xy(rng), xy(rng), z(rng));
        Track track;
        track.position = Vec3(99, -40, 3);
        for (auto& image : scene.images) {
            Vec2 pixel = camera.project(image.pose.transform_world_to_camera(point));
            image.features.keypoints[p].x = static_cast<float>(pixel.x());
            image.features.keypoints[p].y = static_cast<float>(pixel.y());
            if (image.id < 4 || image.id >= 8)
                track.observations.push_back({image.id, p});
        }
        track.num_inliers = 4;
        scene.tracks.push_back(track);
    }
    const auto add_pair = [&](Index a, Index b) {
        ImagePair pair(a, b);
        pair.relative_pose = scene.images[b].pose / scene.images[a].pose;
        pair.relative_pose->C.normalize();
        pair.weight_spatial = 1;
        for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
        scene.pairs.push_back(pair);
    };
    for (Index a = 0; a < 8; ++a)
        for (Index b = a + 1; b < 8; ++b) add_pair(a, b);
    for (Index a = 8; a < n; ++a)
        for (Index b = a + 1; b < n; ++b) add_pair(a, b);
    add_pair(7, 8);
    add_pair(1, 8);
    scene.pairs.back().weight_geometry = 0;
    for (Index i = 4; i < 8; ++i)
        scene.images[i].pose.C.z() += 1.5;
    for (Index i = 8; i < n; ++i) {
        scene.images[i].pose.C = 3 * scene.images[i].pose.C + Vec3(10, -2, 1);
        scene.images[i].pose.R = Eigen::AngleAxisd(0.4, Vec3::UnitY()).toRotationMatrix();
    }
    return scene;
}

void test_rotation_recovery() {
    Scene coherent = fixture();
    for (Index id = 0; id < coherent.images.size(); ++id) {
        coherent.images[id].pose.C = Vec3(0.25 * id, 0.12 * (id % 2), 0);
        coherent.images[id].pose.R.setIdentity();
    }
    triangulate_tracks(coherent, false, 2.F, 1.F);
    expect(recover_rotation_inconsistent_views(coherent).candidates.empty(),
           "coherent poses must not be flagged by rotation recovery");
    for (bool transformed : {false, true}) {
        Scene scene = coherent;
        const Mat3 world_rotation = transformed
            ? Eigen::AngleAxisd(0.9, Vec3(1, 2, 3).normalized()).toRotationMatrix()
            : Mat3::Identity();
        const Vec3 offset = transformed ? Vec3(10, -4, 3) : Vec3::Zero();
        for (auto& image : scene.images) {
            image.pose.C = world_rotation * image.pose.C + offset;
            image.pose.R *= world_rotation.transpose();
        }
        for (auto& track : scene.tracks)
            track.position = world_rotation * track.position + offset;
        const Scene before = scene;
        scene.images[7].pose.R = Eigen::AngleAxisd(0.4, Vec3::UnitY()).toRotationMatrix()
                                 * scene.images[7].pose.R;
        scene.images[7].pose.C += Vec3(3, -2, 1);
        const auto report = recover_rotation_inconsistent_views(scene);
        expect(report.candidates == std::vector<Index>{7} &&
               report.recovered == std::vector<Index>{7},
               "rotation drift must recover from independent stable depths in either gauge");
        expect((scene.images[7].pose.C - before.images[7].pose.C).norm() < 1e-4 &&
               (scene.images[7].pose.R - before.images[7].pose.R).norm() < 1e-4,
               "rotation recovery must restore the actual pose");
        for (Index id = 0; id < 7; ++id)
            expect((scene.images[id].pose.C - before.images[id].pose.C).norm() == 0 &&
                   (scene.images[id].pose.R - before.images[id].pose.R).norm() == 0,
                   "rotation recovery must leave stable poses unchanged");
        for (Index id = 0; id < scene.pairs.size(); ++id)
            expect(scene.pairs[id].active == before.pairs[id].active,
                   "rotation audit must not mutate verified pair activation");
    }
    Scene invalid = coherent;
    invalid.images[7].pose.R = Eigen::AngleAxisd(0.4, Vec3::UnitY()).toRotationMatrix();
    const auto pixels = invalid.images[7].features.keypoints;
    for (Index p = 0; p < pixels.size(); ++p)
        invalid.images[7].features.keypoints[p] = pixels[(p * 37) % pixels.size()];
    const Scene before = invalid;
    const auto rejected = recover_rotation_inconsistent_views(invalid);
    expect(rejected.candidates == std::vector<Index>{7} && rejected.recovered.empty(),
           "rotation disagreement alone must not accept invalid PnP correspondences");
    for (Index id = 0; id < invalid.images.size(); ++id)
        expect((invalid.images[id].pose.C - before.images[id].pose.C).norm() == 0 &&
               (invalid.images[id].pose.R - before.images[id].pose.R).norm() == 0,
               "rejected recovery must preserve every original pose");
    Scene ambiguous = fixture();
    const Scene ambiguous_before = ambiguous;
    expect(recover_rotation_inconsistent_views(ambiguous).candidates.empty(),
           "rotation recovery requires a dominant common gauge");
    for (Index id = 0; id < ambiguous.images.size(); ++id)
        expect((ambiguous.images[id].pose.C - ambiguous_before.images[id].pose.C).norm() == 0 &&
               (ambiguous.images[id].pose.R - ambiguous_before.images[id].pose.R).norm() == 0,
               "ambiguous rotation consensus must leave the reconstruction unchanged");
}

void test_native_local_proposals() {
    for (bool fisheye : {false,true}) {
        Scene scene=fixture(true,fisheye);
        const auto original=scene;
        const std::vector<std::uint8_t> stable{1,1,1,1,0,0,0,0};
        std::vector<RawPairMatches> raw;
        for (const auto [a,b] : {std::pair{0u,4u},std::pair{0u,5u},std::pair{4u,5u}}) {
            RawPairMatches pair; pair.id1=a;pair.id2=b;
            for (Index f=0;f<(a==0 ? 24u : 160u);++f) pair.matches.push_back({f,f});
            raw.push_back(std::move(pair));
        }
        // There are deliberately no surviving frontend pairs. Only the raw
        // proposals can restore this weak but cyclic boundary.
        scene.pairs.clear();
        const auto summary=verify_local_raw_proposals(scene,stable,raw);
        expect(summary.boundary_pairs==2 && summary.weak_boundary_pairs==2 &&
               summary.cycle_supported_matches==48,
               "native raw verification must restore discarded cyclic weak proposals");
        expect(scene.cameras[0].fx==original.cameras[0].fx && scene.cameras[0].k1==original.cameras[0].k1,
               "proposal verification cannot change calibration");
        for (Index id=0;id<scene.images.size();++id)
            expect(scene.images[id].pose.C==original.images[id].pose.C &&
                   scene.images[id].pose.R==original.images[id].pose.R &&
                   scene.images[id].registered==original.images[id].registered,
                   "proposal verification cannot register or move a camera");
        auto disconnected=raw; disconnected.erase(disconnected.begin()+1);
        const auto no_cycle=verify_local_raw_proposals(scene,stable,disconnected);
        expect(no_cycle.boundary_pairs==0 && no_cycle.rejected_weak_pairs==1,
               "a weak pair without an independent feature cycle remains excluded");
        auto wrong=raw;
        for (auto& match:wrong[1].matches) match.train=(match.train+71)%160;
        expect(verify_local_raw_proposals(scene,stable,wrong).boundary_pairs==0,
               "wrong/repeated boundary associations must not create feature cycles");
        const auto before=scene.pairs.size();
        wrong=raw; wrong.back().matches.front().query=10000;
        bool invalid=false;
        try { verify_local_raw_proposals(scene,stable,wrong); }
        catch (const std::invalid_argument&) { invalid=true; }
        expect(invalid && scene.pairs.size()==before,"invalid raw proposals must fail before commit");
    }
}

Scene ray_fixture(bool fisheye=false) {
    Scene scene=fixture(true,fisheye);
    std::erase_if(scene.pairs,[](const auto& p) {return p.id1<4 && p.id2>=4;});
    // Each local point is observed by just ONE original anchor. No landmark
    // can supply a stable-only triangulated depth, but the segment jointly
    // connects to four distinct cameras.
    for (Index a=0;a<4;++a) for (Index b=4;b<8;++b) {
        ImagePair pair(a,b);pair.relative_pose=Pose3D{};pair.weight_spatial=1;
        for (Index f=a;f<160;f+=4) pair.matches.push_back({f,f});
        scene.pairs.push_back(std::move(pair));
    }
    for (Index t=0;t<scene.tracks.size();++t) {
        scene.tracks[t].observations={{t%4,t}};scene.tracks[t].num_inliers=0;
    }
    return scene;
}

void test_ray_group_alignment() {
    const std::vector<Index> targets{4,5,6,7};
    const std::vector<std::uint8_t> stable{1,1,1,1,0,0,0,0};
    for (bool fisheye:{false,true}) {
        auto scene=ray_fixture(fisheye);const auto original=scene;
        const auto report=experiment::recover_ray_group(scene,targets,stable);
        if (!report.accepted) std::cerr << "ray fixture rejection: " << report.reason << " fit=" << report.fit_inliers << " heldout=" << report.heldout_inliers << '\n';
        expect(report.accepted && report.fit_tracks>=100 && report.heldout_tracks>=30,
               "joint ray alignment must validate without any stable-only triangulated depths");
        for(Index id:targets)
            expect((scene.images[id].pose.C-Vec3(0.25*id,0.12*(id%2),0)).norm()<0.01,
                   "ray alignment must recover the metric camera locations");
        for(Index id=0;id<4;++id)
            expect(scene.images[id].pose.C==original.images[id].pose.C && scene.images[id].pose.R==original.images[id].pose.R,
                   "ray alignment/segment BA must freeze original cameras exactly");
        expect(scene.cameras[0].fx==original.cameras[0].fx && scene.cameras[0].k1==original.cameras[0].k1,
               "segment BA must keep native calibration fixed");
    }
    auto single=ray_fixture();
    std::erase_if(single.pairs,[](const auto& p) {return p.id1<4 && p.id2>=4 && p.id1!=0;});
    expect(!experiment::recover_ray_group(single,targets,stable).accepted,"one anchor cannot certify segment scale");
    auto negligible=ray_fixture();for(Index id=0;id<4;++id) negligible.images[id].pose.C.setZero();
    expect(!experiment::recover_ray_group(negligible,targets,stable).accepted,"coincident anchors cannot certify scale");
    auto planar=ray_fixture();
    for(Index f=0;f<160;++f) {
        const auto& kp=planar.images[0].features.keypoints[f];
        const Vec3 ray=planar.cameras[0].unproject_normalized(Vec2(kp.x,kp.y));
        const Vec3 point=(5/ray.z())*ray;
        for(Index id=0;id<8;++id) {
            const Vec2 pixel=planar.cameras[0].project(point-Vec3(0.25*id,0.12*(id%2),0));
            planar.images[id].features.keypoints[f].x=static_cast<float>(pixel.x());
            planar.images[id].features.keypoints[f].y=static_cast<float>(pixel.y());
        }
    }
    expect(!experiment::recover_ray_group(planar,targets,stable).accepted,"planar local geometry remains conservatively rejected");
    auto ambiguous=ray_fixture();
    for(auto& pair:ambiguous.pairs) if(pair.id1<4 && pair.id2>=4) {
        const auto matches=pair.matches;
        for(const auto& match:matches) pair.matches.push_back({match.query,(match.train+1)%160});
    }
    expect(!experiment::recover_ray_group(ambiguous,targets,stable).accepted,"ambiguous anchor features must not become ray votes");
    auto wrong=ray_fixture();const auto original=wrong;
    for(auto& pair:wrong.pairs) if(pair.id1<4 && pair.id2>=4)
        for(auto& m:pair.matches) m.train=(m.train+71)%160;
    expect(!experiment::recover_ray_group(wrong,targets,stable).accepted,"wrong boundary links must fail ray consensus");
    for(Index id=0;id<wrong.images.size();++id)
        expect(wrong.images[id].pose.C==original.images[id].pose.C && wrong.images[id].pose.R==original.images[id].pose.R,
               "rejected ray candidate cannot change original or target poses");
    auto withheld=ray_fixture();
    auto independent=make_independent_submap(withheld,targets);build_tracks(independent.scene);
    expect(initialize_missing_group_pairs(independent.scene),"holdout test needs independent initialization");
    for(Index t=0;t<independent.scene.tracks.size();t+=5) {
        const Index feature=independent.scene.tracks[t].observations.front().feature_id;
        for(Index id=0;id<4;++id)
            withheld.images[id].features.keypoints[feature]=original.images[id].features.keypoints[(feature+71)%160];
    }
    const auto heldout=experiment::recover_ray_group(withheld,targets,stable);
    expect(!heldout.accepted && heldout.reason=="heldout_ray_geometry_failed",
           "strong fit cannot bypass corrupted disjoint held-out ray tracks");
}

void test_geometric_track_completion() {
    Scene scene;PinholeCamera camera;camera.width=1000;camera.height=800;
    camera.fx=camera.fy=600;camera.cx=500;camera.cy=400;scene.cameras.push_back(camera);
    for(Index a=0;a<6;++a) {
        Image image;image.id=a;image.camera_id=0;image.registered=a<5;image.pose.C=Vec3(.4*a,0,0);
        image.features.keypoints.resize(4);
        for(Index f=0;f<4;++f) {
            const auto p=camera.project(image.pose.transform_world_to_camera(Vec3(.2*f,.3*f,5.+f)));
            image.features.keypoints[f].x=float(p.x());image.features.keypoints[f].y=float(p.y());
        }
        scene.images.push_back(image);
    }
    for(Index f=0;f<3;++f)for(Index a:{0u,2u}) {
        Track track;track.position=Vec3(.2*f,.3*f,5.+f);track.observations={{a,f},{a+1,f}};track.num_inliers=2;scene.tracks.push_back(track);
    }
    Track deferred;deferred.observations={{0,3},{1,3},{2,3}};scene.tracks.push_back(deferred);
    for(Index a=0;a<5;++a) {
        ImagePair pair(a,a+1);pair.relative_pose=Pose3D{};
        for(Index f=0;f<3;++f)pair.matches.push_back({f,f});
        // A cross-point union must fail its geometric check.
        if(a==1)pair.matches.push_back({0,2});
        scene.pairs.push_back(pair);
    }
    const auto poses=scene.images;
    expect(experiment::stable_depth_support(scene,5)==0,"control target has no depth through final anchor");
    const auto report=experiment::complete_stable_tracks(scene);
    expect(report.merged==3 && report.added==3 && report.retriangulated==1,"completion must merge, extend and retriangulate with fixed cameras");
    expect(report.projection_rejected+report.conflicts>0,"inconsistent cross-point matches must be rejected");
    expect(experiment::stable_depth_support(scene,5)==3,"completion must expose newly supported target features");
    for(Index i=0;i<scene.images.size();++i)
        expect(scene.images[i].pose.C==poses[i].pose.C && scene.images[i].registered==poses[i].registered,"completion cannot change camera poses or registration");
    for(const auto& track:scene.tracks)if(track.is_triangulated()) {
        std::set<Index> ids;
        for(unsigned o=0;o<track.num_inliers;++o)expect(ids.insert(track.observations[o].image_id).second &&
            experiment::completion_projects(scene,track.observations[o],track.position),"completed inliers must be unique and pass actual projection");
    }
}

void test_cross_anchor_precision() {
    for(bool fisheye:{false,true}) {
        Scene scene;PinholeCamera camera;camera.width=1000;camera.height=800;
        camera.fx=camera.fy=500;camera.cx=500;camera.cy=400;
        if(fisheye){camera.model=photara::CameraModel::opencv_fisheye;camera.k1=.02;}
        scene.cameras.push_back(camera);
        std::mt19937 rng(721);std::uniform_real_distribution<double> xy(-2.5,2.5),z(4,8);
        std::vector<Vec3> points;for(unsigned f=0;f<240;++f)points.emplace_back(xy(rng),xy(rng),z(rng));
        for(Index a=0;a<4;++a) {
            Image image;image.id=a;image.camera_id=0;image.registered=a<3;image.pose.C=Vec3(.6*a,.15*(a%2),0);
            for(const auto& X:points) {
                const auto p=camera.project(image.pose.transform_world_to_camera(X));
                photara::features::Keypoint kp;kp.x=float(p.x());kp.y=float(p.y());image.features.keypoints.push_back(kp);
            }
            scene.images.push_back(image);
        }
        for(Index a=0;a<3;++a) {
            ImagePair pair(a,a+1);pair.relative_pose=Pose3D{};
            for(Index f=0;f<240;++f)pair.matches.push_back({f,f});scene.pairs.push_back(pair);
        }
        const std::vector<std::uint8_t> stable{1,1,1,0};
        const auto split=experiment::partition_precision(scene,stable);
        expect(split.samples.size()==48,"reserved components must supply independent stable-only depths");
        expect(experiment::score_precision(scene,3,scene.images[3].pose,split).accepted,"accurate native poses must pass cross-anchor precision");
        auto bad=scene.images[3].pose;bad.C.x()+=.1;
        const auto biased=experiment::score_precision(scene,3,bad,split);
        expect(!biased.accepted && biased.count==48 && biased.p95>2,"biased poses must fail without hiding bad evaluation pixels");
        auto behind=scene.images[3].pose;behind.C.z()=100;
        expect(!experiment::score_precision(scene,3,behind,split).accepted,"negative depth cannot pass precision");
        auto fit=scene;experiment::exclude_precision_partition(fit,split);
        for(const auto& track:fit.tracks)for(const auto& obs:track.observations)
            expect(!split.reserved.contains({obs.image_id,obs.feature_id}),"reserved ancestry cannot leak into rebuilt fit tracks");
        for(const auto& pair:fit.pairs)for(const auto& m:pair.matches)
            expect(!split.reserved.contains({pair.id1,m.query}) && !split.reserved.contains({pair.id2,m.train}),"reserved pixels cannot leak through alternative edges");
        auto conflict=scene;conflict.pairs[0].matches.push_back({0,1});
        const auto ambiguous=experiment::partition_precision(conflict,stable);
        expect(ambiguous.ambiguous>0 && ambiguous.reserved.contains({0,0}) && ambiguous.reserved.contains({0,1}),"conflicting components must be held out together");
        auto weak=scene;weak.pairs.erase(weak.pairs.begin());
        expect(experiment::partition_precision(weak,stable).samples.empty(),"two-anchor depths lack an independent anchor check");
        auto corrupted=scene;
        for(auto& kp:corrupted.images[2].features.keypoints)kp.y+=40;
        expect(experiment::partition_precision(corrupted,stable).samples.empty(),"bad withheld stable anchors cannot certify depth");
    }
}

void test_structureless_pose() {
    for(bool fisheye:{false,true}) {
        Scene scene;PinholeCamera camera;
        camera.width=1000;camera.height=800;camera.fx=camera.fy=450;camera.cx=500;camera.cy=400;
        camera.trust_intrinsics=true;
        if(fisheye){camera.model=photara::CameraModel::opencv_fisheye;camera.k1=0.02;}
        scene.cameras.push_back(camera);
        for(Index a=0;a<4;++a) {
            Image image;image.id=a;image.camera_id=0;image.registered=a<3;
            image.pose.C=Vec3(0.7*a,0.2*(a%2),0);
            image.features.keypoints.resize(180);scene.images.push_back(image);
        }
        Pose3D truth;truth.R=Eigen::AngleAxisd(0.13,Vec3::UnitY()).toRotationMatrix();truth.C=Vec3(.5,-.3,.2);
        std::mt19937 rng(813);std::uniform_real_distribution<double> xy(-2.5,2.5),z(4.,9.);
        for(Index a=0;a<3;++a) {ImagePair pair(a,3);pair.relative_pose=Pose3D{};scene.pairs.push_back(pair);}
        for(Index f=0;f<180;++f) {
            const Vec3 point(xy(rng),xy(rng),z(rng));const Index a=f%3;
            const Vec2 p=camera.project(scene.images[a].pose.transform_world_to_camera(point));
            const Vec2 q=camera.project(truth.transform_world_to_camera(point));
            scene.images[a].features.keypoints[f].x=float(p.x());scene.images[a].features.keypoints[f].y=float(p.y());
            scene.images[3].features.keypoints[f].x=float(q.x());scene.images[3].features.keypoints[f].y=float(q.y());
            scene.pairs[a].matches.push_back({f,f});
        }
        // Target retained pose is deliberately unrelated; there are no 3D tracks.
        scene.images[3].pose.C=Vec3(900,-400,700);
        const auto original=scene.images[3].pose;
        auto report=experiment::estimate_structureless(scene,3,512);
        if(!report.accepted)std::cerr<<"structureless positive: "<<report.reason<<" fit="<<report.fit_inliers<<" condition="<<report.condition<<'\n';
        expect(report.accepted,"structure-less pose must recover without existing 3D or target initialization");
        expect((report.pose.C-truth.C).norm()<.001 && (report.pose.R-truth.R).norm()<.001,"structure-less coordinate convention must recover absolute pose");
        expect(scene.images[3].pose.C==original.C && !scene.images[3].registered,"pose hypothesis cannot mutate input");
        auto one=scene;one.pairs.resize(1);
        expect(!experiment::estimate_structureless(one,3,128).accepted,"single-anchor scale cannot be accepted");
        auto coincident=scene;for(Index a=0;a<3;++a)coincident.images[a].pose.C.setZero();
        expect(!experiment::estimate_structureless(coincident,3,128).accepted,"coincident anchors cannot certify metric translation");
        auto wrong=scene;
        // Keep feature IDs and the split fixed while corrupting every held-out
        // target measurement; renumbering matches would change the split itself.
        for(Index f=0;f<180;f+=5)wrong.images[3].features.keypoints[f]=scene.images[3].features.keypoints[(f+27)%180];
        expect(!experiment::estimate_structureless(wrong,3,512).accepted,"corrupted whole held-out features cannot be accepted");
        auto ambiguous=scene;
        for(auto& pair:ambiguous.pairs) {const auto matches=pair.matches;for(const auto& m:matches)pair.matches.push_back({m.query,(m.train+3)%180});}
        expect(!experiment::estimate_structureless(ambiguous,3,128).accepted,"ambiguous anchor-feature associations cannot vote twice");
        auto planar=scene;
        for(Index f=0;f<180;++f) {
            const Vec3 X(-2.+.2*(f%20),-1.+.25*(f/20),6.);const Index a=f%3;
            const Vec2 p=camera.project(planar.images[a].pose.transform_world_to_camera(X)),q=camera.project(truth.transform_world_to_camera(X));
            planar.images[a].features.keypoints[f].x=float(p.x());planar.images[a].features.keypoints[f].y=float(p.y());
            planar.images[3].features.keypoints[f].x=float(q.x());planar.images[3].features.keypoints[f].y=float(q.y());
        }
        expect(!experiment::estimate_structureless(planar,3,256).accepted,"planar geometry must not certify the experimental pose");
    }
}

int main() {
    try {
        test_rotation_recovery();
        test_native_local_proposals();
        test_ray_group_alignment();
        test_structureless_pose();
        test_cross_anchor_precision();
        test_geometric_track_completion();
        photara::core::Logger::instance().configure({});
        const std::vector<Index> ids{4, 5, 6, 7};
        const std::vector<std::uint8_t> stable{1,1,1,1,0,0,0,0};
        {
            Scene scene = fixture();
            std::mt19937 rng(81);
            std::uniform_real_distribution<double> xy(-1.5,1.5), z(4.0,6.0);
            std::vector<detail::ResectionFitPoint> fit;
            for (Index f = 0; f < 80; ++f) {
                const Vec3 truth(xy(rng),xy(rng),z(rng));
                fit.push_back({f,f,truth+Vec3(0.01,0,0.04),{0,1,2,3}});
            }
            Pose3D initial;
            initial.C = Vec3(1.04,0.01,0.02);
            initial.R = Eigen::AngleAxisd(0.01,Vec3::UnitY()).toRotationMatrix();
            const auto before = scene;
            const auto refined = detail::refine_resection_depths(scene,4,initial,fit);
            expect((refined.C-Vec3(1,0,0)).norm() < 1e-4 &&
                   (refined.R-Mat3::Identity()).norm() < 1e-4,
                   "joint fit-only refinement removes depth-induced camera pose bias");
            for (Index f = 80; f < 160; ++f) {
                scene.images[4].features.keypoints[f].x += 200;
                scene.images[4].features.keypoints[f].y -= 100;
            }
            const auto excluded = detail::refine_resection_depths(scene,4,initial,fit);
            expect((excluded.C-refined.C).norm() < 1e-9 && (excluded.R-refined.R).norm() < 1e-9,
                   "held-out pixels cannot enter the refinement optimizer");
            for (Index id = 0; id < scene.images.size(); ++id)
                expect(scene.images[id].pose.C == before.images[id].pose.C &&
                       scene.images[id].pose.R == before.images[id].pose.R,
                       "disposable depth refinement leaves all source camera poses exact");
            for (Index t = 0; t < scene.tracks.size(); ++t)
                expect(scene.tracks[t].position == before.tracks[t].position,
                       "disposable depth refinement cannot rewrite source landmarks");
        }
        {
            Scene search = fixture();
            search.pairs.clear();
            search.images[4].registered = false;
            Pose3D truth;
            truth.C = Vec3(1, 0, 0);
            for (Index anchor : {0U, 3U}) {
                ImagePair pair(anchor, 4);
                pair.relative_pose = truth/search.images[anchor].pose;
                pair.matches.resize(20);
                search.pairs.push_back(pair);
            }
            std::vector<Vec3> points, bearings;
            std::mt19937 rng(81);
            std::uniform_real_distribution<double> xy(-1.5, 1.5), z(4.0, 6.0);
            for (Index f = 0; f < 160; ++f) {
                points.emplace_back(xy(rng), xy(rng), z(rng));
                bearings.push_back(truth.transform_world_to_camera(points.back()).normalized());
            }
            const auto seed = detail::relative_resection_search_seed(search, 4, bearings, points);
            expect(seed.success && seed.num_inliers == 160 && (seed.pose.C-truth.C).norm() < 1e-8,
                "independent boundary rays can seed search without the inherited target pose");
            search.pairs.back().relative_pose->C *= -1;
            expect(!detail::relative_resection_search_seed(search, 4, bearings, points).success,
                "negative relative ray depths cannot seed recovery");
            search.pairs.pop_back();
            expect(!detail::relative_resection_search_seed(search, 4, bearings, points).success,
                "one boundary ray cannot determine a search center");
        }
        {
            Scene missing = fixture();
            for (Index id = 4; id < 8; ++id) missing.images[id].registered = false;
            for (auto& track : missing.tracks)
                std::erase_if(track.observations, [](const auto& obs) { return obs.image_id >= 4; });
            std::erase_if(missing.pairs, [](const auto& pair) { return pair.id2 >= 4; });
            const auto before = missing;
            std::vector<RawPairMatches> tentative;
            for (Index id = 4; id < 8; ++id) for (Index anchor = 0; anchor < 3; ++anchor) {
                RawPairMatches pair;
                pair.id1 = id; pair.id2 = anchor;
                for (Index f = 0; f < 160; ++f) pair.matches.push_back({f, f});
                tentative.push_back(std::move(pair));
            }
            expect(recover_stable_resections(missing, stable, tentative).size() == 4,
                "tentative map links must recover views without inherited pair geometry");
            {
                Scene distant = before;
                for (Index anchor = 0; anchor < 4; ++anchor)
                    distant.images[anchor].pose.C *= 0.16;
                std::mt19937 depth_rng(81);
                std::uniform_real_distribution<double> xy(-1.5, 1.5), z(4.0, 6.0);
                for (Index f = 0; f < 160; ++f) {
                    const Vec3 point(xy(depth_rng), xy(depth_rng), z(depth_rng));
                    for (Index anchor = 0; anchor < 4; ++anchor) {
                        const auto pixel = distant.cameras[0].project(
                            distant.images[anchor].pose.transform_world_to_camera(point));
                        distant.images[anchor].features.keypoints[f].x = static_cast<float>(pixel.x());
                        distant.images[anchor].features.keypoints[f].y = static_cast<float>(pixel.y());
                    }
                }
                const auto frozen = distant;
                expect(recover_stable_resections(distant, stable, tentative).size() == 4,
                    "independent multi-view depths below two degrees retain ordinary resection support");
                for (Index id = 4; id < 8; ++id)
                    expect((distant.images[id].pose.C - Vec3(0.25*id, 0.12*(id%2), 0)).norm() < 1e-3,
                        "low-parallax multi-view recovery must recover the actual camera centers");
                for (Index id = 0; id < 4; ++id)
                    expect(distant.images[id].pose.C == frozen.images[id].pose.C,
                        "low-parallax recovery keeps anchor poses frozen");
                Scene two_depths = frozen;
                for (auto& track : two_depths.tracks)
                    std::erase_if(track.observations, [](const auto& obs) { return obs.image_id >= 2; });
                expect(recover_stable_resections(two_depths, stable, tentative).empty(),
                    "two-view local depths still require the wider parallax gate");
            }
            for (Index id = 0; id < 4; ++id)
                expect(missing.images[id].pose.C == before.images[id].pose.C &&
                       missing.images[id].pose.R == before.images[id].pose.R,
                       "map recovery must preserve every frozen camera exactly");
            for (Index id = 4; id < 8; ++id)
                expect((missing.images[id].pose.C - Vec3(0.25 * id, 0.12 * (id % 2), 0)).norm() < 0.01,
                       "map recovery must solve missing poses from independent depths");
            Scene false_matches = before;
            for (auto& pair : tentative) for (auto& m : pair.matches)
                m.train = (m.query * 71 + 13) % 160;
            expect(recover_stable_resections(false_matches, stable, tentative).empty(),
                "consistent descriptor aliases must still pass geometric validation");
            for (unsigned count : {29U, 37U}) {
                Scene sparse = before;
                sparse.tracks.resize(count);
                // Each depth has only two frozen views, but independent
                // landmarks cover three cameras in both fit and validation.
                for (Index t = 0; t < sparse.tracks.size(); ++t)
                    std::erase_if(sparse.tracks[t].observations, [t](const auto& obs) {
                        return obs.image_id == 3 || obs.image_id == t % 3;
                    });
                auto links = tentative;
                for (auto& pair : links) {
                    pair.matches.clear();
                    for (Index f = 0; f < count; ++f) pair.matches.push_back({f, f});
                }
                const auto recovered = recover_stable_resections(sparse, stable, links);
                expect(recovered.size() == (count == 37 ? 4 : 0),
                    "sparse frozen-map recovery must retain the 30-observation final minimum");
                Scene two_view_only = before;
                for (auto& track : two_view_only.tracks)
                    std::erase_if(track.observations, [](const auto& obs) { return obs.image_id >= 2; });
                expect(recover_stable_resections(two_view_only, stable, links).empty(),
                    "tentative recovery cannot certify itself from only two anchor cameras");
            }
            expect(recover_missing_views(missing) == 0,
                "a complete reconstruction must bypass re-extraction entirely");
        }
        {
            const auto add_witnesses = [](Scene& source) {
                const Pose3D target{Mat3::Identity(), Vec3(1.5, 0, 0)};
                for (Index anchor : {0u, 1u}) {
                    ImagePair pair(anchor, 6);
                    pair.relative_pose = target / source.images[anchor].pose;
                    pair.relative_pose->C.normalize(); pair.weight_spatial = 1;
                    for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
                    source.pairs.push_back(std::move(pair));
                }
            };
            Scene conflict = fixture();
            add_witnesses(conflict);
            conflict.images[6].pose.R = Mat3::Identity();
            const auto original = conflict.images[6].pose;
            const Mat3 aliased_rotation = Eigen::AngleAxisd(0.14, Vec3::UnitY()).toRotationMatrix();
            std::vector<Vec3> alias_bearings, alias_points;
            for (Index p = 0; p < 160; ++p) {
                Track anchor;
                for (Index id = 0; id < 4; ++id) anchor.observations.push_back({id, p});
                expect(triangulate_track(anchor, conflict) >= 3, "rotation test uses independently triangulated anchor depths");
                const Vec2 pixel = conflict.cameras[0].project(aliased_rotation * (anchor.position - Vec3(1.5, 0, 0)));
                conflict.images[6].features.keypoints[p].x = static_cast<float>(pixel.x());
                conflict.images[6].features.keypoints[p].y = static_cast<float>(pixel.y());
                alias_bearings.push_back(conflict.cameras[0].unproject(pixel));
                alias_points.push_back(anchor.position);
            }
            const auto pixel_only = estimate_absolute_pose(alias_bearings, alias_points, conflict.cameras[0]);
            expect(pixel_only.success && pixel_only.num_inliers >= 150,
                   "ambiguous hypothesis passes pixel consensus before independent rotation auditing");
            const auto rejected = recover_stable_resections(conflict, stable);
            expect(std::find(rejected.begin(), rejected.end(), 6) == rejected.end() &&
                   conflict.images[6].pose.C == original.C && conflict.images[6].pose.R == original.R,
                   "pixel-consistent aliases cannot override two frozen relative-rotation witnesses");
            Scene correctable = fixture();
            add_witnesses(correctable);
            correctable.images[6].pose.R = Eigen::AngleAxisd(0.3, Vec3::UnitY()).toRotationMatrix();
            const auto recovered = recover_stable_resections(correctable, stable);
            expect(std::find(recovered.begin(), recovered.end(), 6) != recovered.end() &&
                   (correctable.images[6].pose.R - Mat3::Identity()).norm() < 1e-3,
                   "rotation witnesses allow correcting an inaccurate inherited orientation");
        }
        Scene weak_loop=fixture();
        ImagePair weak_closure(2,5);
        weak_closure.relative_pose=Pose3D::identity();
        weak_closure.weight_spatial=1;
        weak_closure.weight_geometry=1e-6F;
        for (Index p=0;p<160;++p) weak_closure.matches.push_back({p,p});
        weak_loop.pairs.push_back(weak_closure);
        expect(analyze_alignment_observability(weak_loop).reliable_views==8,
            "unweighted connectivity alone misses the weak loop");
        const auto fixed_poses=weak_loop.images;
        expect(recover_weakly_connected_views(weak_loop)==4,
            "weak loop must trigger independent pose recovery");
        for (Index i=0;i<4;++i)
            expect(weak_loop.images[i].pose.C==fixed_poses[i].pose.C &&
                   weak_loop.images[i].pose.R==fixed_poses[i].pose.R,
                   "weak branch recovery freezes main poses exactly");
        for (Index i=4;i<8;++i)
            expect(weak_loop.images[i].registered &&
                   (weak_loop.images[i].pose.C-Vec3(0.25*i,0.12*(i%2),0)).norm()<0.01,
                   "weak loop recovery corrects drift without reference poses");
        Scene unsupported=fixture(false);
        expect(recover_weakly_connected_views(unsupported)==4,
            "unsupported branch must be withheld rather than reported aligned");
        for (Index i=4;i<8;++i)
            expect(!unsupported.images[i].registered,"unanchored poses cannot remain registered");
        bool bad_mask=false;
        try { recover_stable_resections(unsupported,{1}); }
        catch (const std::invalid_argument&) { bad_mask=true; }
        expect(bad_mask,"explicit anchor masks must cover all views");
        Scene scene = fixture();
        const auto preview = make_alignment_preview(scene, 17);
        expect(preview.tracks.size() <= 17 && !preview.tracks.empty(), "preview must bound sampled points");
        expect(preview.images.size() == scene.images.size() && preview.pairs.empty(), "preview keeps camera IDs without graph");
        const auto decoded_preview = decode_asfm(encode_asfm(preview));
        expect(decoded_preview.tracks.size() == preview.tracks.size(), "preview observation remapping must round-trip");
        const auto preview_path = std::filesystem::temp_directory_path() /
            ("photara-preview-test-" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()) + ".asfm");
        save_alignment_preview(scene, preview_path);
        expect(load_asfm(preview_path).registered_count() == 8, "published preview must load");
        auto next_preview = scene;
        next_preview.images[7].registered = false;
        save_alignment_preview(next_preview, preview_path);
        expect(load_asfm(preview_path).registered_count() == 7,
            "preview publication must replace an existing snapshot");
        std::filesystem::remove(preview_path);
        Scene resected = fixture();
        // Remove local edges: singletons can be anchored by stable 2D-3D
        // evidence even though they cannot independently triangulate depth.
        resected.pairs.erase(std::remove_if(resected.pairs.begin(), resected.pairs.end(),
            [](const ImagePair& p) { return p.id1 >= 4; }), resected.pairs.end());
        const auto recovered = recover_stable_resections(resected);
        expect(recovered.size() == 4, "stable resection must recover isolated drifting poses");
        for (Index i = 4; i < 8; ++i)
            expect((resected.images[i].pose.C - Vec3(0.25*i,0.12*(i%2),0)).norm() < 0.01,
                "resection must not reuse drifting depths");
        expect(analyze_alignment_observability(resected).unreliable_views == 0,
            "independent stable-depth support must certify corrected singletons");
        for (const std::size_t count : {41, 42, 50}) {
            Scene small = fixture();
            small.tracks.resize(count);
            for (auto& image : small.images) image.features.keypoints.resize(count);
            for (auto& pair : small.pairs) pair.matches.resize(count);
            const auto original = small.images;
            const auto recovered = recover_stable_resections(small, stable);
            expect(recovered.size() == (count >= 42 ? 4 : 0),
                "small resection must reserve at least 30 fit and 12 validation observations");
            for (Index i = 0; i < 8; ++i) {
                if (i < 4 || count < 42)
                    expect(small.images[i].pose.C == original[i].pose.C &&
                           small.images[i].pose.R == original[i].pose.R,
                        "stable and rejected small-sample poses must remain unchanged");
                else
                    expect((small.images[i].pose.C - Vec3(0.25*i, 0.12*(i%2), 0)).norm() < 0.01,
                        "small-sample resection must recover accurate poses");
            }
        }
        Scene no_anchor = fixture(false);
        {
            StableResectionOptions complete;
            complete.minimum_consensus_ratio = 1;
            {
                auto guard = complete;
                guard.require_pose_improvement = true;
                Scene drifting = fixture();
                expect(recover_stable_resections(drifting, stable, {}, guard).size() == 4,
                    "independent improvement gate admits correction of drifting poses");
                Scene correct = fixture();
                for (Index i = 4; i < 8; ++i) {
                    correct.images[i].pose.R.setIdentity();
                    correct.images[i].pose.C = Vec3(0.25*i, 0.12*(i%2), 0);
                }
                const auto previous = correct.images;
                expect(recover_stable_resections(correct, stable, {}, guard).empty(),
                    "independent improvement gate keeps an already accurate pose");
                for (Index i = 0; i < 8; ++i)
                    expect(correct.images[i].pose.C == previous[i].pose.C &&
                           correct.images[i].pose.R == previous[i].pose.R,
                        "rejected needless pose changes preserve every old pose");
            }
            Scene clean = fixture();
            expect(recover_stable_resections(clean, stable, {}, complete).size() == 4,
                "consensus floor accepts completely corroborated native recovery");
            Scene outliers = fixture();
            const auto pixels = outliers.images[4].features.keypoints;
            for (Index p = 0; p < pixels.size(); p += 8)
                outliers.images[4].features.keypoints[p] = pixels[(p + 67) % pixels.size()];
            Scene strict = outliers;
            const auto original = strict.images[4].pose;
            const auto ordinary = recover_stable_resections(outliers, stable);
            expect(std::find(ordinary.begin(), ordinary.end(), 4) != ordinary.end(),
                "native fixture has an ordinary robust pose consensus");
            const auto gated = recover_stable_resections(strict, stable, {}, complete);
            expect(std::find(gated.begin(), gated.end(), 4) == gated.end() &&
                strict.images[4].pose.C == original.C && strict.images[4].pose.R == original.R,
                "stronger consensus floor rejects without changing the candidate pose");
            bool invalid = false;
            complete.minimum_consensus_ratio = 1.1;
            try { recover_stable_resections(strict, stable, {}, complete); }
            catch (const std::invalid_argument&) { invalid = true; }
            expect(invalid, "consensus floor rejects invalid ratios");
        }
        expect(recover_stable_resections(no_anchor).empty(), "resection must reject missing stable depths");
        const auto patch_fixture = [&] {
            Scene patch = fixture();
            patch.cameras[0].width = 4000; patch.cameras[0].height = 3200;
            for (Index id = 4; id < 8; ++id) {
                patch.images[id].registered = false;
                for (Index anchor = 0; anchor < 3; ++anchor) {
                    ImagePair pair(anchor, id);
                    Pose3D target; target.C = Vec3(0.25 * id, 0.12 * (id % 2), 0);
                    pair.relative_pose = target / patch.images[anchor].pose;
                    pair.relative_pose->C.normalize(); pair.weight_spatial = 1;
                    for (Index feature = 0; feature < 160; ++feature)
                        pair.matches.push_back({feature, feature});
                    patch.pairs.push_back(std::move(pair));
                }
            }
            return patch;
        };
        Scene patch = patch_fixture();
        expect(recover_stable_resections(patch, stable).size() == 4,
            "well-conditioned localized support with independent pair witnesses must recover missing views");
        for (Index id = 4; id < 8; ++id)
            expect((patch.images[id].pose.C - Vec3(0.25 * id, 0.12 * (id % 2), 0)).norm() < 0.01,
                "localized recovery must retain pose accuracy");
        for (const bool bad_rotation : {false, true}) {
            Scene rejected_patch = patch_fixture();
            if (bad_rotation) {
                for (auto& pair : rejected_patch.pairs)
                    if (pair.id1 < 4 && pair.id2 >= 4)
                        pair.relative_pose->R = Eigen::AngleAxisd(0.2, Vec3::UnitY()).toRotationMatrix();
            } else {
                rejected_patch.cameras[0].width = 100000;
                rejected_patch.cameras[0].height = 100000;
            }
            const auto before = rejected_patch.images;
            expect(recover_stable_resections(rejected_patch, stable).empty(),
                "localized evidence requires image extent and independent absolute rotation agreement");
            for (Index id = 0; id < 8; ++id)
                expect(rejected_patch.images[id].registered == before[id].registered &&
                       rejected_patch.images[id].pose.C == before[id].pose.C &&
                       rejected_patch.images[id].pose.R == before[id].pose.R,
                    "rejected localized recovery must preserve registration and poses");
        }
        Scene noisy = fixture();
        for (Index i=4;i<8;++i) {
            const auto pixels=noisy.images[i].features.keypoints;
            for (Index p=0;p<160;p+=3)
                noisy.images[i].features.keypoints[p]=pixels[(p+67)%160];
        }
        expect(recover_stable_resections(noisy,stable).size()==4,
            "robust fit and held-out consensus must tolerate candidate outliers");
        for (Index i=4;i<8;++i)
            expect((noisy.images[i].pose.C-Vec3(0.25*i,0.12*(i%2),0)).norm()<0.01,
                "outlier-tolerant recovery must preserve pose accuracy");
        Scene direction_seed = fixture();
        const auto seed_pixels = direction_seed.images[4].features.keypoints;
        for (Index p = 0; p < 160; ++p)
            if (p % 5 >= 2)
                direction_seed.images[4].features.keypoints[p] =
                    seed_pixels[(p + 67) % 160];
        for (Index anchor : {Index{0}, Index{1}, Index{2}}) {
            ImagePair pair(anchor, 4);
            pair.relative_pose =
                Pose3D{Mat3::Identity(), Vec3(1.0, 0.0, 0.0)} /
                direction_seed.images[anchor].pose;
            pair.relative_pose->C.normalize();
            pair.weight_spatial = 1;
            for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
            direction_seed.pairs.push_back(std::move(pair));
        }
        Scene reversed_seed = direction_seed;
        std::reverse(reversed_seed.images.begin(), reversed_seed.images.end());
        for (Index i = 0; i < 8; ++i) reversed_seed.images[i].id = i;
        for (auto& track : reversed_seed.tracks)
            for (auto& observation : track.observations)
                observation.image_id = 7 - observation.image_id;
        for (auto& pair : reversed_seed.pairs) {
            const Index first = 7 - pair.id2;
            pair.id2 = 7 - pair.id1;
            pair.id1 = first;
            if (pair.relative_pose) pair.relative_pose = pair.relative_pose->inverse();
            for (auto& match : pair.matches) std::swap(match.query, match.train);
        }
        const auto reverse_recovered = recover_stable_resections(
            reversed_seed, {0,0,0,0,1,1,1,1});
        expect(std::find(reverse_recovered.begin(), reverse_recovered.end(), 3) !=
                   reverse_recovered.end() &&
               (reversed_seed.images[3].pose.C - Vec3(1.0, 0.0, 0)).norm() < 0.01,
            "baseline certification must be invariant to image-pair ordering");
        const auto seeded_recovered =
            recover_stable_resections(direction_seed, stable);
        expect(std::find(seeded_recovered.begin(), seeded_recovered.end(), 4) !=
                   seeded_recovered.end(),
            "multiple verified baselines may certify a minority pixel consensus");
        expect((direction_seed.images[4].pose.C -
                Vec3(1.0, 0.0, 0)).norm() < 0.01,
            "baseline-certified resection must recover the actual pose");
        Scene minority_consensus = fixture();
        for (Index i = 4; i < 8; ++i) {
            const auto pixels = minority_consensus.images[i].features.keypoints;
            for (Index p = 0; p < 160; ++p)
                if (p % 3 != 0)
                    minority_consensus.images[i].features.keypoints[p] = pixels[(p + 67) % 160];
        }
        expect(recover_stable_resections(minority_consensus, stable).empty(),
            "a minority pose consensus must not seed subsequent recovery waves");
        Scene boundary_only = fixture();
        for (auto& track : boundary_only.tracks) track.observations.resize(4);
        const auto boundary_recovered = recover_stable_resections(boundary_only, stable);
        expect(boundary_recovered.size() == 1 && boundary_recovered.front() == 4,
            "valid boundary links must recover a camera absent from the source tracks");
        Scene geometry_only = fixture();
        for (auto& track : geometry_only.tracks) track.observations.resize(4);
        for (auto& pair : geometry_only.pairs) {
            if (pair.id1 == 3 && pair.id2 == 4) {
                pair.active = false;
                pair.relative_pose.reset();
                pair.H = Mat3::Identity();
            }
        }
        Scene false_geometry = geometry_only;
        for (auto& pair : false_geometry.pairs)
            if (pair.id1 == 3 && pair.id2 == 4)
                for (auto& match : pair.matches)
                    match.train = (match.query * 37 + 13) % 160;
        expect(recover_stable_resections(false_geometry, stable).empty(),
            "inactive geometry proposals must not bypass independent PnP validation");
        const auto geometry_recovered = recover_stable_resections(geometry_only, stable);
        expect(geometry_recovered.size() == 1 && geometry_recovered.front() == 4 &&
               (geometry_only.images[4].pose.C - Vec3(1.0, 0, 0)).norm() < 0.01,
            "geometry-only inactive matches can propose independently validated resection");
        Scene rejected_anchor = fixture();
        const auto anchor_pixels = rejected_anchor.images[3].features.keypoints;
        for (Index p = 0; p < 160; ++p)
            rejected_anchor.images[3].features.keypoints[p] = anchor_pixels[(p + 67) % 160];
        for (auto& pair : rejected_anchor.pairs)
            if (pair.id1 == 3 && pair.id2 == 4)
                for (auto& match : pair.matches) match.train = (match.query + 67) % 160;
        expect(recover_stable_resections(rejected_anchor, stable).size() == 4,
            "a rejected depth observation must not introduce ambiguous target correspondences");
        for (Index i = 4; i < 8; ++i)
            expect((rejected_anchor.images[i].pose.C - Vec3(0.25*i, 0.12*(i%2), 0)).norm() < 0.01,
                "rejected anchor links must not corrupt recovered poses");
        Scene conflicting_links = fixture(true, true);
        for (auto& pair : conflicting_links.pairs)
            if (pair.id1 == 3 && pair.id2 == 4)
                for (auto& match : pair.matches) match.train = (match.query + 67) % 160;
        for (Index anchor : {Index{0}, Index{1}}) {
            ImagePair pair(anchor, 4);
            const Pose3D target_pose{Mat3::Identity(), Vec3(1, 0, 0)};
            pair.relative_pose = target_pose / conflicting_links.images[anchor].pose;
            pair.relative_pose->C.normalize();
            for (Index p = 0; p < 160; ++p) pair.matches.push_back({p, p});
            conflicting_links.pairs.push_back(std::move(pair));
        }
        Scene tied_links = conflicting_links;
        ImagePair conflicting_pair(2, 4);
        conflicting_pair.relative_pose = Pose3D{Mat3::Identity(), Vec3(1, 0, 0)} / tied_links.images[2].pose;
        conflicting_pair.relative_pose->C.normalize();
        for (Index p = 0; p < 160; ++p) conflicting_pair.matches.push_back({p, (p + 67) % 160});
        tied_links.pairs.push_back(std::move(conflicting_pair));
        // Duplicate records from one anchor must never become independent votes.
        const auto wrong_pair = *std::find_if(conflicting_links.pairs.begin(), conflicting_links.pairs.end(),
            [](const ImagePair& pair) { return pair.id1 == 3 && pair.id2 == 4; });
        for (unsigned duplicate = 0; duplicate < 5; ++duplicate)
            conflicting_links.pairs.push_back(wrong_pair);
        const auto tied_recovered = recover_stable_resections(tied_links, stable);
        expect(std::find(tied_recovered.begin(), tied_recovered.end(), 4) == tied_recovered.end(),
            "equal independent correspondence support must remain ambiguous");
        expect(recover_stable_resections(conflicting_links, stable).size() == 4,
            "independent multiview agreement must resolve a conflicting single-view link");
        expect((conflicting_links.images[4].pose.C - Vec3(1, 0, 0)).norm() < 0.01,
            "multiview correspondence resolution must recover the actual pose");
        Scene two_anchors=fixture();
        for (auto& track:two_anchors.tracks)
            track.observations.erase(std::remove_if(track.observations.begin(),track.observations.end(),
                [](const Observation& obs){return obs.image_id==2 || obs.image_id==3;}),track.observations.end());
        expect(recover_stable_resections(two_anchors,stable).empty(),
            "two-view depths need a third stable observation before anchoring recovery");
        Scene inconsistent_direction=fixture();
        const Pose3D false_pose{Mat3::Identity(),Vec3(1.0,1.2,0)};
        for (Index p=0;p<160;++p) {
            // Triangulate the fixture's real landmark independently, then make
            // a perfectly fitting but pair-inconsistent candidate camera.
            Track landmark=inconsistent_direction.tracks[p];
            landmark.observations.resize(4);
            expect(triangulate_track(landmark,inconsistent_direction,2.F,1.F)>=3,
                "fixture landmark must have independent depth");
            const auto pixel=inconsistent_direction.cameras[0].project(false_pose.transform_world_to_camera(landmark.position));
            inconsistent_direction.images[4].features.keypoints[p].x=static_cast<float>(pixel.x());
            inconsistent_direction.images[4].features.keypoints[p].y=static_cast<float>(pixel.y());
        }
        const auto direction_recovered=recover_stable_resections(inconsistent_direction,stable);
        expect(std::find(direction_recovered.begin(),direction_recovered.end(),4)==direction_recovered.end(),
            "pixel consensus must not override an inconsistent stable pair baseline");
        Scene heldout_mismatch = fixture();
        std::vector<std::size_t> heldout_order(160);
        std::iota(heldout_order.begin(), heldout_order.end(), 0);
        std::mt19937 split_rng(0x53464d);
        std::shuffle(heldout_order.begin(), heldout_order.end(), split_rng);
        for (Index i = 4; i < 8; ++i) {
            const auto original_features = heldout_mismatch.images[i].features.keypoints;
            for (std::size_t j = 0; j < 53; ++j) {
                const auto p = heldout_order[j];
                heldout_mismatch.images[i].features.keypoints[p] = original_features[(p + 75) % 160];
            }
        }
        const auto rejected_poses = heldout_mismatch.images;
        expect(recover_stable_resections(heldout_mismatch).empty(),
            "resection must reject inconsistent held-out correspondences despite a perfect fit subset");
        for (Index i = 0; i < 8; ++i)
            expect(heldout_mismatch.images[i].pose.C == rejected_poses[i].pose.C &&
                   heldout_mismatch.images[i].pose.R == rejected_poses[i].pose.R,
                "rejected resection must preserve all poses");
        // Neither global registration nor inherited shared landmarks may hide
        // a dense four-camera branch attached through one verified bridge.
        auto risk = find_structural_pair_risks(scene);
        expect(risk[4] && risk[5] && risk[6] && risk[7], "dense bridge branch must trigger frontend rescue");
        expect(!risk[0] && !risk[1] && !risk[2] && !risk[3], "retain main pair block");
        Scene closed = scene;
        ImagePair closure(2, 5);
        closure.relative_pose = Pose3D::identity();
        closed.pairs.push_back(closure);
        risk = find_structural_pair_risks(closed);
        for (auto value : risk) expect(!value, "second independent graph link closes the bridge");
        auto sub = make_independent_submap(scene, ids);
        expect(sub.scene.tracks.empty(), "must discard inherited points and tracks");
        expect(sub.scene.pairs.size() == 6, "must exclude boundary pairs");
        for (const auto& image : sub.scene.images)
            expect(!image.registered && image.pose.C.isZero() && image.pose.R.isIdentity(), "must discard old poses");
        const auto original = scene;
        auto report = recover_independent_submap(scene, ids, stable);
        std::cout << "positive: " << report.reason << " shared=" << report.shared_points << '\n';
        expect(report.accepted, "independent geometry should correct the drift");
        expect(report.validation_inliers >= 3, "must validate held-out points");
        for (Index i = 0; i < 4; ++i)
            expect(scene.images[i].pose.C == original.images[i].pose.C &&
                scene.images[i].pose.R == original.images[i].pose.R, "stable poses must remain exact");
        for (Index i : ids)
            expect((scene.images[i].pose.C - Vec3(0.25 * i, 0.12 * (i % 2), 0)).norm() < 0.01,
                "must recover known coordinates without using parent points");
        Scene rejected = fixture(false);
        const auto before = rejected;
        report = recover_independent_submap(rejected, ids, stable);
        expect(!report.accepted && report.reason == "insufficient_independent_shared_points",
            "disconnected maps must not invent alignment");
        for (Index i = 0; i < 8; ++i)
            expect(rejected.images[i].pose.C == before.images[i].pose.C, "rejection must preserve poses");
        for (Index i = 0; i < before.tracks.size(); ++i)
            expect(rejected.tracks[i].position == before.tracks[i].position &&
                rejected.tracks[i].num_inliers == before.tracks[i].num_inliers,
                "rejection must preserve tracks");
        Scene rollback = fixture();
        SubmapRecoveryOptions strict;
        strict.minimum_camera_observations = 1000;
        report = recover_independent_submap(rollback, ids, stable, strict);
        expect(!report.accepted && report.reason == "insufficient_final_camera_support", "must check final camera support");
        for (Index i : ids) expect(rollback.images[i].pose.C == original.images[i].pose.C,
            "late rejection must roll back candidate poses");
        Scene circular = fixture();
        for (Index i = 1; i < 4; ++i) {
            circular.images[i].pose = circular.images[0].pose;
            circular.images[i].features.keypoints = circular.images[0].features.keypoints;
        }
        report = recover_independent_submap(circular, ids, stable);
        expect(!report.accepted && report.shared_points == 0,
            "coincident stable rays must not certify independent depth");
        Scene inconsistent = fixture();
        for (Index i = 0; i < 4; ++i) for (Index p = 0; p < 160; ++p) {
            // Preserve stable multiview consistency but destroy the cross-map
            // correspondences: no single similarity explains this permutation.
            inconsistent.images[i].features.keypoints[p] =
                original.images[i].features.keypoints[(p * 37) % 160];
        }
        report = recover_independent_submap(inconsistent, ids, stable);
        expect(!report.accepted && report.reason == "similarity_consensus_failed",
            "wrong cross-map matches must fail geometric consensus");
        Scene collinear = fixture();
        for (Index i = 0; i < 4; ++i) for (Index p = 0; p < 160; ++p) {
            const Vec3 point(-0.8 + 0.01 * p, 0, 5);
            const Vec2 pixel = collinear.cameras[0].project(
                collinear.images[i].pose.transform_world_to_camera(point));
            collinear.images[i].features.keypoints[p].x = static_cast<float>(pixel.x());
            collinear.images[i].features.keypoints[p].y = static_cast<float>(pixel.y());
        }
        report = recover_independent_submap(collinear, ids, stable);
        expect(!report.accepted && report.reason == "ill_conditioned_shared_geometry",
            "collinear shared points cannot constrain similarity");
        Scene heldout = fixture();
        auto ordering = make_independent_submap(heldout, ids);
        build_tracks(ordering.scene);
        for (Index t = 0; t < ordering.scene.tracks.size(); t += 5) {
            const Index p = ordering.scene.tracks[t].observations.front().feature_id;
            for (Index i = 0; i < 4; ++i)
                heldout.images[i].features.keypoints[p] = original.images[i].features.keypoints[(p + 71) % 160];
        }
        report = recover_independent_submap(heldout, ids, stable);
        expect(!report.accepted && report.reason == "heldout_geometry_failed",
            "good fit must not bypass failed held-out geometry");
        bool invalid = false;
        try { make_independent_submap(original, {4, 4}); }
        catch (const std::invalid_argument&) { invalid = true; }
        expect(invalid, "duplicate image IDs must be rejected");
        for (const bool new_depths : {false, true}) {
            Scene corridor = corridor_fixture();
            // Parallel rays cannot triangulate. Cache their failed solve while
            // other tracks propagate, and invalidate incident tracks as new
            // anchors register just as in the full-rebuild reference.
            for (auto& image : corridor.images) {
                if (new_depths && image.id >= 8) continue;
                for (Index feature = 0; feature < 5; ++feature) {
                    image.features.keypoints[feature].x = 500;
                    image.features.keypoints[feature].y = 400;
                }
            }
            Scene full_rebuild = corridor;
            WeakBranchRecoveryOptions reference;
            reference.reuse_verified_geometry = false;
            const auto reference_count = recover_weakly_connected_views(full_rebuild, reference);
            const auto cached_count = recover_weakly_connected_views(corridor);
            expect(cached_count == reference_count && cached_count >= 4,
                "weak branch beyond a locally rigid clique must still recover");
            expect(corridor.tracks.size() == full_rebuild.tracks.size(),
                   "cached recovery must preserve split track topology");
            for (Index i=0;i<corridor.images.size();++i)
                expect(corridor.images[i].registered == full_rebuild.images[i].registered &&
                       (corridor.images[i].pose.C-full_rebuild.images[i].pose.C).norm()<1e-10 &&
                       (corridor.images[i].pose.R-full_rebuild.images[i].pose.R).norm()<1e-10,
                       "cached recovery must match full-rebuild poses");
            for (Index i=0;i<corridor.tracks.size();++i) {
                const auto& a=corridor.tracks[i];const auto& b=full_rebuild.tracks[i];
                expect(a.num_inliers==b.num_inliers && a.observations.size()==b.observations.size() &&
                       (a.position-b.position).norm()<1e-10,
                       "cached recovery must match full-rebuild triangulation");
                for (std::size_t o=0;o<a.observations.size();++o)
                    expect(a.observations[o].image_id==b.observations[o].image_id &&
                           a.observations[o].feature_id==b.observations[o].feature_id,
                           "cached recovery must preserve observation ordering");
            }
            for (Index i = 8; i < 12; ++i)
                expect(corridor.images[i].registered &&
                       (corridor.images[i].pose.C - Vec3(0.25 * i, 0.12 * (i % 2), 0)).norm() < 0.05,
                       "weak branch on the far side of the corridor must stay recovered");
            for (Index i = 0; i < 4; ++i)
                expect(corridor.images[i].registered,
                       "core cameras must remain registered");
        }
        {
            Scene chain = fixture();
            const auto frozen = chain;
            // Every local camera has at most two neighbors: no four-view star.
            std::erase_if(chain.pairs, [](const auto& p) {
                return p.id1 >= 4 && p.id2 != p.id1 + 1;
            });
            auto local_chain = make_independent_submap(chain, ids);
            build_tracks(local_chain.scene);
            expect(!star_initialize(local_chain.scene), "chain control must fail star initialization");
            SubmapRecoveryOptions pair_options;
            pair_options.independent_initializer = [](Scene& independent) {
                for (const auto& image : independent.images)
                    expect(!image.registered && image.pose.C.isZero(), "initializer must not inherit parent poses");
                for (const auto& track : independent.tracks)
                    expect(!track.is_triangulated(), "initializer must not inherit parent depth");
                return initialize_missing_group_pairs(independent);
            };
            const auto chain_report = recover_independent_submap(chain, ids, stable, pair_options);
            expect(chain_report.accepted && chain_report.reconstructed_views == 4,
                   "independent pair growth must reconstruct and validate the full chain");
            for (Index i=0;i<4;++i)
                expect((chain.images[i].pose.C-frozen.images[i].pose.C).norm()==0 &&
                       (chain.images[i].pose.R-frozen.images[i].pose.R).norm()==0,
                       "joint chain initialization must preserve stable cameras exactly");
            Scene bad_chain = fixture(false);
            std::erase_if(bad_chain.pairs, [](const auto& p) { return p.id1 >= 4 && p.id2 != p.id1+1; });
            expect(!recover_independent_submap(bad_chain, ids, stable, pair_options).accepted,
                   "a reconstructed chain without independent boundary depth must stay rejected");
        }
        std::cout << "submap recovery tests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n'; return 1;
    }
}
