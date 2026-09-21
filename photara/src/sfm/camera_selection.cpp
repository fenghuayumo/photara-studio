#include "sfm/camera_selection.hpp"
#include "ba/optimizer.hpp"
#include "core/logging.hpp"
#include "parallel/thread_pool.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <numeric>

namespace photara::sfm {
bool is_equirectangular_image_size(
    const std::uint32_t width, const std::uint32_t height) {
    if (width == 0 || height == 0) return false;
    const double expected = 2.0 * static_cast<double>(height);
    return std::abs(static_cast<double>(width) - expected) <=
           0.01 * static_cast<double>(width);
}

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
    if (!point.allFinite()) return false;
    // Positive depth along each observed ray is the cheirality test for every
    // central camera; the pinhole z>0 form is its front-hemisphere special case.
    if (camera.is_equirectangular())
        return pose.transform_world_to_camera(point).dot(b) > 0.0;
    return point.z()>1e-8 && pose.transform_world_to_camera(point).z()>1e-8;
}

// Panorama validation error: the tangent-plane residual is the pixel metric that
// stays valid across the azimuth seam and at the poles, where a pixel difference
// between an image coordinate and a projected ray is meaningless.
double reprojection_error_sq(const PinholeCamera& camera, const Vec3& camera_point,
                             const Vec2& observed) {
    if (camera.is_equirectangular()) {
        const auto local = camera.local_reprojection(camera_point, observed);
        return local.valid ? local.residual.squaredNorm() : 1e12;
    }
    return (camera.project(camera_point)-observed).squaredNorm();
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
            const double error2=0.5*(reprojection_error_sq(candidate.camera,point,probe.first[j])+
                reprojection_error_sq(candidate.camera,pose.transform_world_to_camera(point),probe.second[j]));
            if (std::isfinite(error2)) candidate.scores[i]+=std::max(0.0,1.0-error2/9.0);
        }
        if (count) candidate.scores[i]/=count;
    }
    candidate.score=std::accumulate(candidate.scores.begin(),candidate.scores.end(),0.0)/probes.size();
}

// Fits several focal/model hypotheses over the same probe set in one parallel
// sweep. Each (hypothesis, probe) task runs one independent PoseLib RANSAC;
// per-task results are deterministic, and scoring stays serial afterwards, so
// this reproduces fit_candidate() exactly while removing the serial RANSAC
// queue that dominated cold-start calibration time.
std::vector<Candidate> fit_candidates(
    const std::vector<PinholeCamera>& cameras,
    const std::vector<CameraModelProbe>& probes) {
    std::vector<Candidate> candidates(cameras.size());
    for (std::size_t h=0; h<cameras.size(); ++h) {
        candidates[h].camera=cameras[h];
        candidates[h].fits.assign(probes.size(),{});
    }
    if (cameras.empty() || probes.empty()) return candidates;

    RelativePoseOptions options;
    options.max_iterations=1500; options.min_iterations=100;
    options.min_inliers=20; options.max_epipolar_error_px=3;
    options.max_reproj_error_px=4; options.min_ray_angle_deg=0.5;

    const std::size_t hypothesis_count=cameras.size();
    const std::size_t probe_count=probes.size();
    const std::size_t task_count=hypothesis_count*probe_count;
    parallel::parallel_for(
        task_count, parallel::resolve_thread_count(0),
        [&](const std::size_t task) {
            const std::size_t h=task/probe_count, i=task%probe_count;
            const PinholeCamera& camera=cameras[h];
            std::vector<Vec2> first,second;
            for (std::size_t j=0; j<probes[i].first.size(); ++j) {
                if (j%3==0 || !camera.unproject(probes[i].first[j]).allFinite() ||
                    !camera.unproject(probes[i].second[j]).allFinite()) continue;
                first.push_back(probes[i].first[j]); second.push_back(probes[i].second[j]);
            }
            if (first.size()>=30)
                candidates[h].fits[i]=
                    estimate_relative_pose(first,second,camera,camera,options);
        });
    for (auto& candidate:candidates) score_candidate(candidate,probes);
    return candidates;
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
    auto refitted = std::move(fit_candidates({refined.camera}, probes)[0]);
    if (refitted.score > refined.score) refined = std::move(refitted);
    return refined.score>candidate.score ? refined : candidate;
}
} // namespace

CameraModelSelection select_camera_model(
    const PinholeCamera& source, const std::vector<CameraModelProbe>& input,
    const double supplied_focal, const bool fisheye_lens_hint,
    const CameraModel requested, const bool panorama_size_hint) {
    CameraModelSelection result;
    // An explicitly requested panorama needs no calibration search: the chart is
    // fixed by the image size, so there is no focal or distortion to fit.
    if (requested == CameraModel::equirectangular) {
        PinholeCamera panorama = source;
        panorama.set_equirectangular_intrinsics();
        result.model = CameraModel::equirectangular;
        result.focal_pixels = panorama.fx;
        result.distortion = {0.0, 0.0, 0.0, 0.0};
        result.confident = true;
        result.reason = "requested equirectangular panorama";
        return result;
    }
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
        const auto make_camera=[&](const double focal) {
            auto camera=source; camera.model=static_cast<CameraModel>(model);
            camera.fx=camera.fy=focal; camera.k1=camera.k2=camera.p1=camera.p2=0;
            camera.trust_intrinsics=true;
            return camera;
        };
        std::vector<Candidate> seeds;
        {
            std::vector<PinholeCamera> seed_cameras;
            if (supplied_focal>0) {
                seed_cameras.push_back(make_camera(supplied_focal));
            } else {
                for (double ratio:ratios)
                    seed_cameras.push_back(make_camera(size*ratio));
            }
            seeds=fit_candidates(seed_cameras,probes);
        }
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
                    std::vector<PinholeCamera> trial_cameras;
                    for (int step:{-2,-1,1,2}) {
                        trial_cameras.push_back(
                            make_camera(center*std::exp(span*step/2)));
                    }
                    for (auto& trial : fit_candidates(trial_cameras,probes)) {
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
    // Panorama hypothesis: only for 2:1 imagery. The equirectangular chart has no
    // intrinsics to fit, so its held-out score is directly comparable with the
    // perspective candidates'.
    std::string panorama_veto;
    if (requested == CameraModel::automatic && panorama_size_hint) {
        PinholeCamera panorama = source;
        panorama.set_equirectangular_intrinsics();
        Candidate pano = fit_candidates({panorama}, probes)[0];
        result.equirect_score = pano.score;
        const double pano_informative = static_cast<double>(std::count_if(
            pano.scores.begin(), pano.scores.end(),
            [](double score) { return score >= 0.25; }));
        const double perspective = std::max(best[0].score, best[1].score);
        // The 2:1 image size is decisive here, and deliberately so: the
        // projection of a 360 capture cannot be recovered from geometry when the
        // matched features sit inside a narrow forward cone, because any
        // monotone radial warp of a central camera stays nearly epipolar-
        // consistent at the small baselines a hand-held panorama produces.
        // The guard below is therefore a sanity bound rather than a competition:
        // a genuine panorama still scores *lower* than a narrow perspective fit
        // (the comparison is per pair, and a video-length baseline leaves a
        // single pair's translation under-determined - which the honest
        // spherical chart reports and a warped perspective fit does not). What
        // it rejects is the case where the perspective model explains the
        // matches several times better, which is what a 2:1 *rectilinear* photo
        // looks like.
        constexpr double k_panorama_relative_score = 0.25;
        if (pano_informative >= 2.0 &&
            pano.score >= k_panorama_relative_score * perspective) {
            result.model = CameraModel::equirectangular;
            result.focal_pixels = panorama.fx;
            result.distortion = {0.0, 0.0, 0.0, 0.0};
            result.confident = true;
            result.informative_pairs = static_cast<unsigned>(pano_informative);
            result.reason = "2:1 image size with geometrically sound panorama "
                            "hypotheses";
            return result;
        }
        panorama_veto =
            " (2:1 image size, but the perspective fit explains the kept-out "
            "matches much better: not a panorama chart)";
    }
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
    if (best[selected].score<0.25 || result.informative_pairs<2) {
        result.reason += panorama_veto;
        return result;
    }
    // Ambiguous pinhole/fisheye fits can trade extreme focal against distortion.
    // Preserve the ordinary pinhole view-graph self-calibration instead of
    // injecting those unsupported coefficients into the full reconstruction.
    if (!explicit_model && !result.confident && selected==0) {
        result.reason="ambiguous model; retain pinhole self-calibration" + panorama_veto;
        return result;
    }
    result.model=static_cast<CameraModel>(selected);
    result.focal_pixels=best[selected].camera.focal();
    result.distortion={best[selected].camera.k1,best[selected].camera.k2,
        best[selected].camera.p1,best[selected].camera.p2};
    result.reason=explicit_model ? "requested model, validated calibration" : result.confident
        ? "held-out geometry after calibration" : "ambiguous model; calibrated fallback";
    result.reason += panorama_veto;
    return result;
}
} // namespace photara::sfm
