#pragma once

#include "sfm_pair_group_init.hpp"
#include "sfm/submap_recovery.hpp"
#include "sfm/tracks.hpp"
#include "sfm/triangulation.hpp"
#include <PoseLib/solvers/gp4ps.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <map>
#include <set>
#include <iomanip>

// Opt-in experiment. This has its own multi-camera pixel validator; it does not
// claim to satisfy the shared-3D-point Sim(3) validator. No default caller.
namespace photara::sfm::experiment {
struct RayGroupReport {
    bool accepted{};
    std::string reason;
    unsigned tracks{}, fit_tracks{}, heldout_tracks{}, anchors{}, fit_inliers{}, heldout_inliers{};
    unsigned fit_observations{}, heldout_observations{};
    double heldout_mean{}, heldout_after_mean{}, condition_ratio{};
};
struct RayLink { Index track{},anchor{},feature{}; Vec3 origin,ray,point; double angular{}; };

inline bool ray_nonplanar(const std::vector<Vec3>& points) {
    if (points.size()<4) return false;
    Vec3 mean=Vec3::Zero(); for (const auto& p:points) mean+=p;
    mean/=static_cast<double>(points.size());
    Mat3 covariance=Mat3::Zero(); for (const auto& p:points) covariance+=(p-mean)*(p-mean).transpose();
    Eigen::SelfAdjointEigenSolver<Mat3> eig(covariance);
    return eig.info()==Eigen::Success && eig.eigenvalues()[0]>1e-5*eig.eigenvalues()[2];
}
struct RayCost {
    RayLink link;
    template<class T> bool operator()(const T* const state,T* residual) const {
        T point[3]{T(link.point.x()),T(link.point.y()),T(link.point.z())},rotated[3];
        ceres::AngleAxisRotatePoint(state,point,rotated);
        T delta[3]; T squared=T(1e-20);
        for (unsigned j=0;j<3;++j) { delta[j]=exp(state[6])*rotated[j]+state[3+j]-T(link.origin[j]); squared+=delta[j]*delta[j]; }
        const T length=sqrt(squared);
        for (unsigned j=0;j<3;++j) residual[j]=(delta[j]/length-T(link.ray[j]))/T(link.angular);
        return true;
    }
};
inline bool ray_support(const Scene& scene,const RayLink& link,const Vec3& point,double* error=nullptr) {
    const auto& image=scene.images[link.anchor]; const auto& camera=scene.camera_of(image);
    const Vec3 projected=image.pose.transform_world_to_camera(point);
    const auto& feature=image.features.keypoints[link.feature];
    const double e=camera.checked_reprojection_error(projected,Vec2(feature.x,feature.y));
    if (error) *error=e;
    return std::isfinite(e) && e<=2.0;
}

inline RayGroupReport recover_ray_group(Scene& parent,const std::vector<Index>& targets,
    const std::vector<std::uint8_t>& stable,bool refine_segment=true,
    std::ostream* diagnostics=nullptr,Index excluded_anchor=k_invalid) {
    RayGroupReport report;
    const auto reject=[&](const char* reason) { report.reason=reason;return report; };
    if (stable.size()!=parent.images.size() || targets.size()<4) return reject("invalid_group");
    for (Index id:targets) if (id>=stable.size() || stable[id]) return reject("invalid_target");
    auto sub=make_independent_submap(parent,targets); build_tracks(sub.scene);
    if (!initialize_missing_group_pairs(sub.scene) || sub.scene.registered_count()!=targets.size())
        return reject("independent_initialization_failed");
    // Map feature observations to one independent local landmark. Ambiguous
    // links never become multiple votes for the same track/anchor.
    using Key=std::pair<Index,Index>;
    std::map<Key,Index> features;
    for (Index t=0;t<sub.scene.tracks.size();++t) {
        const auto& track=sub.scene.tracks[t];
        if (!track.is_triangulated()) continue;
        for (unsigned o=0;o<track.num_inliers;++o) {
            const auto& obs=track.observations[o];
            const Key key{targets[obs.image_id],obs.feature_id};
            auto [it,added]=features.emplace(key,t); if (!added && it->second!=t) it->second=k_invalid;
        }
    }
    std::map<Key,std::set<Index>> associations;
    for (const auto& pair:parent.pairs) {
        if (pair.id1>=stable.size() || pair.id2>=stable.size()) return reject("invalid_pair");
        if (!pair.active || !pair.relative_pose || pair.zero_baseline) continue;
        for (const auto& m:pair.matches) {
            if (m.query>=parent.images[pair.id1].features.keypoints.size() ||
                m.train>=parent.images[pair.id2].features.keypoints.size()) return reject("invalid_feature");
            const auto add=[&](Index target,Index feature,Index anchor,Index observed) {
                const auto found=features.find({target,feature});
                if (found!=features.end() && found->second!=k_invalid && stable[anchor] && anchor!=excluded_anchor && parent.images[anchor].registered)
                    associations[{anchor,observed}].insert(found->second);
            };
            add(pair.id1,m.query,pair.id2,m.train); add(pair.id2,m.train,pair.id1,m.query);
        }
    }
    std::map<Key,std::set<Index>> unique;
    for (const auto& [obs,tracks]:associations) if (tracks.size()==1)
        unique[{*tracks.begin(),obs.first}].insert(obs.second);
    std::map<Index,std::vector<RayLink>> grouped;
    for (const auto& [key,observations]:unique) {
        if (observations.size()!=1) continue;
        const auto [track,anchor]=key; const auto feature=*observations.begin();
        const auto& image=parent.images[anchor]; const auto& camera=parent.camera_of(image);
        const auto& kp=image.features.keypoints[feature];
        grouped[track].push_back({track,anchor,feature,image.pose.C,
            image.pose.R.transpose()*camera.unproject_normalized(Vec2(kp.x,kp.y)),
            sub.scene.tracks[track].position,camera.pixel_error_to_angular(2)});
    }
    std::vector<RayLink> fit,heldout;
    std::vector<Vec3> points; std::set<Index> anchor_ids;
    std::size_t ordinal=0;
    for (const auto& [id,links]:grouped) {
        auto& set=(ordinal++%5==0) ? heldout : fit;
        set.insert(set.end(),links.begin(),links.end());
        if ((ordinal-1)%5!=0) points.push_back(sub.scene.tracks[id].position);
        for (const auto& link:links) anchor_ids.insert(link.anchor);
    }
    report.tracks=static_cast<unsigned>(grouped.size()); report.fit_tracks=static_cast<unsigned>(points.size());
    report.heldout_tracks=report.tracks-report.fit_tracks; report.anchors=static_cast<unsigned>(anchor_ids.size());
    report.fit_observations=static_cast<unsigned>(fit.size());report.heldout_observations=static_cast<unsigned>(heldout.size());
    if (report.fit_tracks<30 || report.heldout_tracks<12 || anchor_ids.size()<3)
        return reject("insufficient_ray_support");
    if (!ray_nonplanar(points)) return reject("degenerate_local_geometry");
    std::set<Index> fit_anchors,validation_anchors;
    for (const auto& link:fit) fit_anchors.insert(link.anchor);
    for (const auto& link:heldout) validation_anchors.insert(link.anchor);
    if (fit_anchors.size()<3 || validation_anchors.size()<3) return reject("insufficient_anchor_diversity");
    const auto spread=[&](const std::vector<RayLink>& links) {
        std::map<Index,std::vector<Vec2>> pixels;
        for (const auto& link:links) {
            const auto& kp=parent.images[link.anchor].features.keypoints[link.feature];
            pixels[link.anchor].emplace_back(kp.x,kp.y);
        }
        unsigned supported=0;
        for (const auto& [anchor,observations]:pixels)
            if (observations.size()>=3 && compute_spatial_weight(observations,parent.camera_of(parent.images[anchor]),4)>=3.F/16.F)
                ++supported;
        return supported>=3;
    };
    if (!spread(fit) || !spread(heldout)) return reject("insufficient_spatial_support");
    // Normalize both coordinate frames using fit observations only. gp4ps:
    // scale * anchor_center + depth * world_ray = R * local_point + t.
    Vec3 local_mean=Vec3::Zero(),anchor_mean=Vec3::Zero();
    for (const auto& p:points) local_mean+=p; local_mean/=static_cast<double>(points.size());
    for (Index id:fit_anchors) anchor_mean+=parent.images[id].pose.C;
    anchor_mean/=static_cast<double>(fit_anchors.size());
    double local_scale=0,anchor_scale=0;
    for (const auto& p:points) local_scale+=(p-local_mean).squaredNorm();
    local_scale=std::sqrt(local_scale/points.size());
    for (Index id:fit_anchors) anchor_scale+=(parent.images[id].pose.C-anchor_mean).squaredNorm();
    anchor_scale=std::sqrt(anchor_scale/fit_anchors.size());
    if (!(local_scale>1e-8 && anchor_scale>1e-8)) return reject("negligible_baseline");
    auto normalized=fit;
    for (auto& link:normalized) { link.origin=(link.origin-anchor_mean)/anchor_scale; link.point=(link.point-local_mean)/local_scale; }
    const auto to_global=[&](const Similarity3& transform) {
        Similarity3 global=transform; global.scale*=anchor_scale/local_scale;
        global.t=anchor_mean+anchor_scale*transform.t-global.scale*global.R*local_mean;
        return global;
    };
    Similarity3 best; std::vector<std::size_t> inliers;
    std::mt19937 rng(0x497A1u); std::uniform_int_distribution<std::size_t> sample(0,fit.size()-1);
    double best_error=std::numeric_limits<double>::infinity();
    for (unsigned iteration=0;iteration<2048;++iteration) {
        std::vector<Vec3> origins,rays,landmarks; std::set<Index> distinct_tracks,distinct_anchors;
        for (unsigned attempt=0;attempt<64 && origins.size()<4;++attempt) {
            const auto& link=normalized[sample(rng)];
            if (origins.size()==1 && (link.origin-origins.front()).norm()<1e-6) continue;
            if (!distinct_tracks.insert(link.track).second) continue;
            origins.push_back(link.origin);rays.push_back(link.ray);landmarks.push_back(link.point);distinct_anchors.insert(link.anchor);
        }
        if (origins.size()!=4 || distinct_anchors.size()<2) continue;
        if (!ray_nonplanar(landmarks)) continue;
        Eigen::Matrix4d linear;
        for (unsigned i=0;i<2;++i) {
            const auto& x=rays[i];const auto& p=origins[i];
            linear.row(2*i)<<x.z(),0,-x.x(),-p.x()*x.z()+p.z()*x.x();
            linear.row(2*i+1)<<0,x.z(),-x.y(),-p.y()*x.z()+p.z()*x.y();
        }
        if (linear.fullPivLu().rcond()<1e-8) continue;
        std::vector<poselib::CameraPose> poses; std::vector<double> scales;
        poselib::gp4ps(origins,rays,landmarks,&poses,&scales);
        for (std::size_t c=0;c<poses.size();++c) {
            if (!(std::isfinite(scales[c]) && scales[c]>1e-6 && scales[c]<1e6)) continue;
            Similarity3 transform; transform.scale=1/scales[c];transform.R=poses[c].R();transform.t=poses[c].t/scales[c];
            const auto global=to_global(transform);
            std::vector<std::size_t> consensus; double error=0;
            for (std::size_t i=0;i<fit.size();++i) {
                double e=0;
                if (ray_support(parent,fit[i],global.apply(fit[i].point),&e)) { consensus.push_back(i);error+=e; }
            }
            if (consensus.size()>inliers.size() || (consensus.size()==inliers.size() && error<best_error)) {
                best=transform;inliers=std::move(consensus);best_error=error;
            }
        }
    }
    report.fit_inliers=static_cast<unsigned>(inliers.size());
    if(diagnostics && !inliers.empty()) {
        *diagnostics<<"track,anchor,feature,split,target_first,target_last,target_observations,local_max_px,anchor_parallax_deg,stable_depth_inliers,depth_difference_normalized,anchor_error_px,similarity_scale\n";
        const auto transform=to_global(best);
        for(const auto& [id,links]:grouped) {
            const auto& track=sub.scene.tracks[id];double internal=0,parallax=0;
            Index first=k_invalid,last=0;
            for(unsigned o=0;o<track.num_inliers;++o) {
                const auto& obs=track.observations[o];const auto& image=sub.scene.images[obs.image_id];
                const auto& kp=image.features.keypoints[obs.feature_id];
                internal=std::max(internal,sub.scene.camera_of(image).checked_reprojection_error(image.pose.transform_world_to_camera(track.position),Vec2(kp.x,kp.y)));
                first=std::min(first,targets[obs.image_id]);last=std::max(last,targets[obs.image_id]);
            }
            Track depth;
            for(const auto& l:links)depth.observations.push_back({l.anchor,l.feature});
            for(const auto& a:links)for(const auto& b:links)
                parallax=std::max(parallax,std::acos(std::clamp(a.ray.dot(b.ray),-1.,1.))*180/3.141592653589793);
            TriangulationOptions tri;tri.reproj_threshold_px=2;tri.min_angle_deg=1;tri.split_tracks=false;
            if(depth.observations.size()>=2)triangulate_track(depth,parent,tri);
            const double difference=depth.is_triangulated()?(depth.position-transform.apply(track.position)).norm()/anchor_scale:-1;
            const bool withheld=std::any_of(heldout.begin(),heldout.end(),[&](const auto& l){return l.track==id;});
            for(const auto& link:links) {
                double error=0;ray_support(parent,link,transform.apply(track.position),&error);
                *diagnostics<<id<<','<<std::quoted(parent.images[link.anchor].path.filename().string())<<','<<link.feature<<','<<(withheld?"holdout":"fit")<<','
                    <<std::quoted(parent.images[first].path.filename().string())<<','<<std::quoted(parent.images[last].path.filename().string())<<','
                    <<unsigned(track.num_inliers)<<','<<internal<<','<<parallax<<','<<unsigned(depth.num_inliers)<<','<<difference<<','<<error<<','<<transform.scale<<'\n';
            }
        }
        if(!*diagnostics)throw std::runtime_error("Cannot write ray residual diagnostic");
    }
    if (inliers.size()<30 || inliers.size()<0.8*fit.size()) return reject("ray_consensus_failed");
    // Refine seven normalized transform parameters on fit inliers, never holdout.
    double state[7]; const auto q=Quat(best.R);
    const double quaternion[4]{q.w(),q.x(),q.y(),q.z()};ceres::QuaternionToAngleAxis(quaternion,state);
    for (unsigned j=0;j<3;++j) state[3+j]=best.t[j];state[6]=std::log(best.scale);
    ceres::Problem problem;
    for (auto i:inliers) problem.AddResidualBlock(new ceres::AutoDiffCostFunction<RayCost,3,7>(new RayCost{normalized[i]}),new ceres::HuberLoss(1),state);
    problem.SetParameterLowerBound(state,6,-14);problem.SetParameterUpperBound(state,6,14);
    ceres::Solver::Options options;options.max_num_iterations=50;options.linear_solver_type=ceres::DENSE_QR;options.num_threads=1;
    ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
    if (!summary.IsSolutionUsable()) return reject("ray_refinement_failed");
    double rotation[9];ceres::AngleAxisToRotationMatrix(state,rotation);
    best.R=Eigen::Map<Mat3>(rotation);best.scale=std::exp(state[6]);best.t=Vec3(state[3],state[4],state[5]);
    // The undamped fit Hessian must constrain all seven parameters, especially
    // scale. Damping or a single camera cannot manufacture observability.
    Eigen::Matrix<double,7,7> hessian=Eigen::Matrix<double,7,7>::Zero();
    std::vector<std::size_t> refined;
    const auto global=to_global(best);
    std::set<Index> supported_tracks,supported_anchors;
    for (std::size_t i=0;i<fit.size();++i) if (ray_support(parent,fit[i],global.apply(fit[i].point))) {
        RayCost residual{normalized[i]}; double base[3];residual(state,base);
        Eigen::Matrix<double,3,7> jacobian;
        for (unsigned j=0;j<7;++j) { state[j]+=1e-6;double next[3];residual(state,next);state[j]-=1e-6;
            for (unsigned r=0;r<3;++r) jacobian(r,j)=(next[r]-base[r])/1e-6; }
        hessian+=jacobian.transpose()*jacobian;refined.push_back(i);
        supported_tracks.insert(fit[i].track);supported_anchors.insert(fit[i].anchor);
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,7,7>> eig(hessian);
    report.condition_ratio=eig.eigenvalues()[0]/eig.eigenvalues()[6];
    report.fit_inliers=static_cast<unsigned>(refined.size());
    if (supported_tracks.size()<30 || supported_anchors.size()<3 || refined.size()<0.8*fit.size())
        return reject("refined_ray_consensus_failed");
    std::map<Index,bool> whole_tracks;
    for (const auto& link:fit) {
        auto [it,inserted]=whole_tracks.emplace(link.track,true);
        it->second=it->second && ray_support(parent,link,global.apply(link.point));
    }
    const auto agreeing=std::count_if(whole_tracks.begin(),whole_tracks.end(),[](const auto& entry){return entry.second;});
    if (agreeing<30 || agreeing<0.8*report.fit_tracks) return reject("inconsistent_ray_tracks");
    if (eig.info()!=Eigen::Success || !std::isfinite(report.condition_ratio) || report.condition_ratio<1e-8)
        return reject("unobservable_similarity");
    // Track-level validation: all boundary observations of a held-out landmark
    // must pass, so repeated observations never inflate independent support.
    const auto validate=[&](const Scene& geometry,bool after) {
        unsigned good=0;double error=0;std::set<Index> anchors_seen,views_seen;std::vector<Vec3> supported;
        for (const auto& [t,links]:grouped) {
            if (std::none_of(heldout.begin(),heldout.end(),[&](const auto& link){return link.track==t;})) continue;
            const Vec3 point=after ? geometry.tracks[t].position : global.apply(sub.scene.tracks[t].position);
            bool valid=true;double sum=0;
            for (const auto& link:links) { double e=0;valid=ray_support(parent,link,point,&e)&&valid;sum+=std::min(1000.0,e); }
            const auto& track=sub.scene.tracks[t];
            for (unsigned o=0;o<track.num_inliers;++o) {
                const auto& obs=track.observations[o];const Index id=targets[obs.image_id];
                const Pose3D pose=after ? geometry.images[id].pose : Pose3D{
                    sub.scene.images[obs.image_id].pose.R*global.R.transpose(),
                    global.apply(sub.scene.images[obs.image_id].pose.C)};
                const auto& kp=parent.images[id].features.keypoints[obs.feature_id];
                valid=valid && parent.camera_of(parent.images[id]).checked_reprojection_error(
                    pose.transform_world_to_camera(point),Vec2(kp.x,kp.y))<=2.0;
            }
            error+=sum/links.size();
            if (!valid) continue;
            ++good;supported.push_back(point);for(const auto& link:links) anchors_seen.insert(link.anchor);
            for (unsigned o=0;o<track.num_inliers;++o) views_seen.insert(track.observations[o].image_id);
        }
        report.heldout_inliers=good;
        if (after) report.heldout_after_mean=error/report.heldout_tracks;else report.heldout_mean=error/report.heldout_tracks;
        return good>=12 && good>=0.8*report.heldout_tracks && anchors_seen.size()>=3 && views_seen.size()>=2 && ray_nonplanar(supported);
    };
    if (!validate(sub.scene,false)) return reject("heldout_ray_geometry_failed");
    Scene trial;trial.thread_count=1;trial.cameras=parent.cameras;trial.images=parent.images;
    for (Index id=0;id<trial.images.size();++id) trial.images[id].registered=stable[id]!=0;
    for (Index id=0;id<targets.size();++id) {
        auto& image=trial.images[targets[id]];image.registered=true;
        image.pose={sub.scene.images[id].pose.R*global.R.transpose(),global.apply(sub.scene.images[id].pose.C)};
    }
    trial.tracks=sub.scene.tracks;
    for (auto& track:trial.tracks) {
        if (track.is_triangulated()) track.position=global.apply(track.position);
        for (auto& obs:track.observations) obs.image_id=targets[obs.image_id];
    }
    for (auto i:refined) {
        auto& track=trial.tracks[fit[i].track];
        if (track.num_inliers==255) continue;
        track.observations.insert(track.observations.begin()+track.num_inliers,{fit[i].anchor,fit[i].feature});++track.num_inliers;
    }
    rebuild_track_index(trial);
    if (refine_segment) {
        BundleOptions ba;ba.free_image_ids=targets;ba.optimize_all_registered=false;
        ba.fixed_image_ids.assign(fit_anchors.begin(),fit_anchors.end());
        ba.optimizer.optimize_focal=false;ba.optimizer.optimize_aspect_ratio=false;ba.optimizer.optimize_distortion=false;
        ba.optimizer.maximum_iterations=50;ba.optimizer.huber_delta=1.0;ba.write_intrinsics=false;
        if (!run_bundle_adjustment(trial,ba).success) return reject("segment_refinement_failed");
    }
    if (!validate(trial,true) || report.heldout_after_mean>report.heldout_mean+0.05)
        return reject("segment_holdout_regressed");
    std::map<Index,std::set<Index>> observations;
    for (auto& track:trial.tracks) {
        if (!track.is_triangulated()) continue;
        // Recheck every retained internal observation after BA; holdout boundary
        // observations can only be attached AFTER all validation is finished.
        const auto end=track.observations.begin()+track.num_inliers;
        const auto last=std::stable_partition(track.observations.begin(),end,[&](const auto& obs) {
            RayLink link;link.anchor=obs.image_id;link.feature=obs.feature_id;
            return ray_support(trial,link,track.position);
        });
        track.num_inliers=static_cast<std::uint8_t>(last-track.observations.begin());
        if (!track.is_triangulated()) continue;
        for (unsigned o=0;o<track.num_inliers;++o)
            observations[track.observations[o].image_id].insert(track.observations[o].feature_id);
    }
    for (Index id:targets) if (observations[id].size()<30) return reject("insufficient_final_camera_support");
    for (Index id=0;id<stable.size();++id) if (stable[id] &&
        (parent.images[id].pose.C!=trial.images[id].pose.C || parent.images[id].pose.R!=trial.images[id].pose.R))
        return reject("frozen_anchor_changed");
    for (Index id:targets) { parent.images[id].pose=trial.images[id].pose;parent.images[id].registered=true; }
    parent.tracks=std::move(trial.tracks);rebuild_track_index(parent);
    report.accepted=true;report.reason="accepted";return report;
}
} // namespace photara::sfm::experiment
