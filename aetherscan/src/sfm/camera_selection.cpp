#include "sfm/camera_selection.hpp"
#include "ba/optimizer.hpp"
#include "core/logging.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace aetherscan::sfm {
namespace {
struct Candidate {
    PinholeCamera camera;
    std::vector<RelativePoseResult> fits;
    std::vector<double> scores;
    double score{};
};

bool triangulate(const PinholeCamera& camera, const Pose3D& pose,
                 const Vec2& first, const Vec2& second, Vec3& point) {
    const Vec3 a = camera.unproject_normalized(first);
    const Vec3 b = pose.R.transpose()*camera.unproject_normalized(second);
    if (!a.allFinite() || !b.allFinite()) return false;
    const double cosine = a.dot(b), denominator = 1-cosine*cosine;
    if (denominator < 1e-5) return false;
    const double ac = a.dot(pose.C), bc = b.dot(pose.C);
    const double depth1 = (ac-cosine*bc)/denominator;
    const double depth2 = (cosine*ac-bc)/denominator;
    if (depth1 <= 0 || depth2 <= 0) return false;
    point = 0.5*(depth1*a+pose.C+depth2*b);
    return point.allFinite() && point.z()>1e-8 && pose.transform_world_to_camera(point).z()>1e-8;
}

// Validation matches are never used by RANSAC or calibration BA. Native pixel
// errors put both camera models and all focal hypotheses on the same scale.
void score_candidate(Candidate& candidate, const std::vector<CameraModelProbe>& probes) {
    candidate.scores.assign(probes.size(),0.0);
    for (std::size_t i=0; i<probes.size(); ++i) {
        if (!candidate.fits[i].success) continue;
        const auto& probe=probes[i];
        unsigned count=0;
        for (std::size_t j=0; j<probe.first.size(); j+=3) {
            ++count;
            Vec3 point;
            const auto& pose=candidate.fits[i].pose;
            if (!triangulate(candidate.camera,pose,probe.first[j],probe.second[j],point)) continue;
            const double error2=0.5*((candidate.camera.project(point)-probe.first[j]).squaredNorm()+
                (candidate.camera.project(pose.transform_world_to_camera(point))-probe.second[j]).squaredNorm());
            if (std::isfinite(error2)) candidate.scores[i]+=std::max(0.0,1.0-error2/9.0);
        }
        if (count) candidate.scores[i]/=count;
    }
    candidate.score=std::accumulate(candidate.scores.begin(),candidate.scores.end(),0.0)/probes.size();
}

Candidate fit_candidate(const PinholeCamera& camera, const std::vector<CameraModelProbe>& probes) {
    Candidate candidate; candidate.camera=camera; candidate.fits.resize(probes.size());
    RelativePoseOptions options;
    options.max_iterations=1500; options.min_iterations=100;
    options.min_inliers=20; options.max_epipolar_error_px=3;
    options.max_reproj_error_px=4; options.min_ray_angle_deg=0.5;
    for (std::size_t i=0; i<probes.size(); ++i) {
        std::vector<Vec2> first,second;
        for (std::size_t j=0; j<probes[i].first.size(); ++j) {
            if (j%3==0 || !camera.unproject(probes[i].first[j]).allFinite() ||
                !camera.unproject(probes[i].second[j]).allFinite()) continue;
            first.push_back(probes[i].first[j]); second.push_back(probes[i].second[j]);
        }
        if (first.size()>=30)
            candidate.fits[i]=estimate_relative_pose(first,second,camera,camera,options);
    }
    score_candidate(candidate,probes);
    return candidate;
}

Candidate refine_calibration(Candidate candidate, const std::vector<CameraModelProbe>& probes,
                             const bool fixed_focal) {
    ba::Problem problem;
    const auto& camera=candidate.camera;
    problem.intrinsics.push_back({camera.fx,camera.fy,camera.cx,camera.cy,
        camera.k1,camera.k2,camera.p1,camera.p2,camera.model});
    std::vector<std::size_t> groups;
    for (std::size_t i=0; i<probes.size(); ++i) {
        const auto& fit=candidate.fits[i];
        if (!fit.success || fit.degenerate_planar) continue;
        const auto id=static_cast<ba::Index>(problem.poses.size());
        const auto q=fit.pose.quaternion();
        problem.poses.push_back({});
        problem.poses.push_back({q.w(),q.x(),q.y(),q.z(),fit.pose.C.x(),fit.pose.C.y(),fit.pose.C.z()});
        problem.pose_constant.insert(problem.pose_constant.end(),{1,0});
        problem.pose_intrinsic.insert(problem.pose_intrinsic.end(),{0,0});
        groups.push_back(i);
        for (std::size_t j=0; j<probes[i].first.size(); ++j) {
            if (j%3==0) continue;
            Vec3 point;
            const auto& first=probes[i].first[j]; const auto& second=probes[i].second[j];
            if (!triangulate(camera,fit.pose,first,second,point)) continue;
            if ((camera.project(point)-first).norm()>4 ||
                (camera.project(fit.pose.transform_world_to_camera(point))-second).norm()>4) continue;
            const auto pid=static_cast<ba::Index>(problem.points.size());
            problem.points.push_back({point.x(),point.y(),point.z()});
            problem.observations.push_back(id,pid,first.x(),first.y());
            problem.observations.push_back(id+1,pid,second.x(),second.y());
        }
    }
    if (groups.size()<2 || problem.points.size()<60) return candidate;
    ba::OptimizerOptions options;
    options.maximum_iterations=80; options.maximum_pcg_iterations=150;
    options.optimize_focal=!fixed_focal; options.optimize_distortion=true;
    options.focal_prior_weight=0.05; options.huber_delta=1.5;
    options.fix_first_point=false;
    const auto summary=ba::optimize_cpu(problem,options);
    if (!summary.usable()) return candidate;
    const auto& intr=problem.intrinsics[0];
    Candidate refined=candidate;
    refined.camera.fx=intr.fx; refined.camera.fy=intr.fy;
    refined.camera.k1=intr.k1; refined.camera.k2=intr.k2;
    refined.camera.p1=intr.p1; refined.camera.p2=intr.p2;
    for (std::size_t i=0; i<groups.size(); ++i) {
        const auto& p=problem.poses[2*i+1];
        refined.fits[groups[i]].pose.R=Quat(p.qw,p.qx,p.qy,p.qz).normalized().toRotationMatrix();
        refined.fits[groups[i]].pose.C=Vec3(p.cx,p.cy,p.cz);
    }
    score_candidate(refined,probes);
    // Intrinsics are shared by every probe, including pairs excluded from BA.
    // Refit training matches so those pairs are not scored with stale poses
    // from a different focal/distortion hypothesis. Keep validation held out.
    auto refitted = fit_candidate(refined.camera, probes);
    if (refitted.score > refined.score) refined = std::move(refitted);
    return refined.score>candidate.score ? refined : candidate;
}
} // namespace

CameraModelSelection select_camera_model(
    const PinholeCamera& source, const std::vector<CameraModelProbe>& input,
    const double supplied_focal, const bool fisheye_lens_hint, const CameraModel requested) {
    CameraModelSelection result;
    const double size=static_cast<double>(std::max(source.width,source.height));
    if (requested==CameraModel::opencv_fisheye) result.model=requested;
    result.focal_pixels=supplied_focal>0 ? supplied_focal :
        (result.model==CameraModel::opencv_fisheye ? 0.5 : 1.2)*size;
    result.reason="insufficient geometry; retain initial camera";
    std::vector<CameraModelProbe> probes;
    for (const auto& probe:input)
        if (probe.first.size()>=45 && probe.first.size()==probe.second.size()) probes.push_back(probe);
    if (size<=0 || probes.size()<2) return result;
    const std::array<double,8> ratios{0.25,0.35,0.5,0.7,1.0,1.4,2.0,2.8};
    std::array<Candidate,2> best;
    for (int model=0; model<2; ++model) {
        if (requested!=CameraModel::automatic && static_cast<int>(requested)!=model) continue;
        auto fit=[&](double focal) {
            auto camera=source; camera.model=static_cast<CameraModel>(model);
            camera.fx=camera.fy=focal; camera.k1=camera.k2=camera.p1=camera.p2=0;
            camera.trust_intrinsics=true;
            return fit_candidate(camera,probes);
        };
        std::vector<Candidate> seeds;
        if (supplied_focal>0) seeds.push_back(fit(supplied_focal));
        else for (double ratio:ratios) seeds.push_back(fit(size*ratio));
        std::sort(seeds.begin(),seeds.end(),[](const Candidate& a,const Candidate& b) {
            return a.score>b.score;
        });
        for (auto& seed:seeds) {
            if (seed.score<0.1) break;
            // A zero-distortion ranking can put every top seed in the same
            // wrong focal basin. Give every coarse seed a shared-distortion
            // fit before local focal search can move it into that basin.
            Candidate coarse = refine_calibration(seed, probes, supplied_focal>0);
            Candidate candidate=std::move(seed);
            if (supplied_focal<=0) {
                for (double span:{0.30,0.10,0.035}) {
                    const double center=candidate.camera.focal();
                    for (int step:{-2,-1,1,2}) {
                        auto trial=fit(center*std::exp(span*step/2));
                        if (trial.score>candidate.score) candidate=std::move(trial);
                    }
                }
            }
            candidate=refine_calibration(std::move(candidate),probes,supplied_focal>0);
            if (coarse.score>candidate.score) candidate=std::move(coarse);
            core::Logger::instance().info("calibration hypothesis: model=",model,
                " focal=",candidate.camera.focal()," validation=",candidate.score);
            if (candidate.score>best[model].score) best[model]=std::move(candidate);
        }
    }
    if (requested==CameraModel::automatic) {
        // Planarity depends on the undistortion hypothesis. Counting a planar
        // pinhole fit as zero while accepting its warped fisheye counterpart
        // manufactures evidence for fisheye (e.g. an object turntable). Exclude
        // the pair symmetrically from model comparison if either fit is planar.
        std::size_t comparable_pairs=0;
        for (std::size_t i=0; i<probes.size(); ++i) {
            const bool planar = (i<best[0].fits.size() && best[0].fits[i].degenerate_planar) ||
                                (i<best[1].fits.size() && best[1].fits[i].degenerate_planar);
            if (!planar) { ++comparable_pairs; continue; }
            for (auto& candidate:best)
                if (i<candidate.scores.size()) candidate.scores[i]=0;
        }
        for (auto& candidate:best)
            candidate.score=std::accumulate(candidate.scores.begin(),candidate.scores.end(),0.0)/
                std::max<std::size_t>(1,comparable_pairs);
    }
    result.pinhole_score=best[0].score; result.fisheye_score=best[1].score;
    const int winner=best[1].score>best[0].score ? 1 : 0;
    if (best[winner].scores.empty()) return result;
    // Near a perfect fit an absolute four-percent margin can reject a model
    // that halves the validation loss. Retain the conservative absolute
    // margin for noisy pairs, and a one-percent floor for numerical ties.
    const auto margin = [](double other_score) {
        return std::clamp(0.5*(1.0-other_score), 0.01, 0.04);
    };
    unsigned wins=0;
    for (std::size_t i=0; i<probes.size(); ++i) {
        const double score=best[winner].scores[i];
        if (score>=0.25) ++result.informative_pairs;
        const double other=best[1-winner].scores.empty() ? 0 : best[1-winner].scores[i];
        if (score>=other+margin(other)) ++wins;
    }
    result.confident=result.informative_pairs>=2 && wins>=2 && best[winner].score>=0.3 &&
        best[winner].score-best[1-winner].score>=margin(best[1-winner].score);
    const bool explicit_model=requested!=CameraModel::automatic;
    int selected=winner;
    if (!explicit_model && !result.confident) {
        selected=(fisheye_lens_hint && best[1].score>=0.3 && best[1].score+0.02>=best[0].score) ? 1 : 0;
    }
    result.informative_pairs=static_cast<unsigned>(std::count_if(
        best[selected].scores.begin(),best[selected].scores.end(),[](double score) {return score>=0.25;}));
    if (best[selected].score<0.25 || result.informative_pairs<2) return result;
    // Ambiguous pinhole/fisheye fits can trade extreme focal against distortion.
    // Preserve the ordinary pinhole view-graph self-calibration instead of
    // injecting those unsupported coefficients into the full reconstruction.
    if (!explicit_model && !result.confident && selected==0) {
        result.reason="ambiguous model; retain pinhole self-calibration";
        return result;
    }
    result.model=static_cast<CameraModel>(selected);
    result.focal_pixels=best[selected].camera.focal();
    result.distortion={best[selected].camera.k1,best[selected].camera.k2,
        best[selected].camera.p1,best[selected].camera.p2};
    result.reason=explicit_model ? "requested model, validated calibration" : result.confident
        ? "held-out geometry after calibration" : "ambiguous model; calibrated fallback";
    return result;
}
} // namespace aetherscan::sfm
