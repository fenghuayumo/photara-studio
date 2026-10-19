#pragma once

#include "sfm_local_proposals.hpp"
#include "sfm_track_completion.hpp"
#include "sfm_boundary_windows.hpp"
#include "sfm/asfm.hpp"
#include "sfm/export_colmap.hpp"
#include <PoseLib/solvers/gen_relpose_5p1pt.h>
#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <Eigen/Eigenvalues>
#include <random>
#include <iomanip>
#include <fstream>

// Developer-only, immutable-input pose hypotheses. No reference poses or target
// retained poses are used. This is not a production registration entry point.
namespace photara::sfm::experiment {
struct StructurelessLink {
    Index feature{}, anchor{}, observed{};
    Vec3 origin, ray, query;
    double anchor_angle{}, query_angle{};
};
struct StructurelessReport {
    bool accepted{};
    std::string reason;
    unsigned constraints{}, features{}, ambiguous{}, anchors{}, multi_anchor_features{};
    unsigned fit{}, holdout{}, proposal_inliers{}, fit_inliers{}, holdout_inliers{}, fit_features{}, holdout_features{};
    unsigned fit_anchor_count{}, holdout_anchor_count{}, occupied_cells{};
    double baseline{}, condition{}, heldout_mean{};
    Pose3D pose;
};

inline std::vector<StructurelessLink> structureless_links(const Scene& scene, Index target,
    StructurelessReport& report) {
    using Key=std::pair<Index,Index>;
    std::map<Key,std::set<Index>> forward,reverse;
    for (const auto& pair:scene.pairs) {
        if (!pair.active || !pair.relative_pose || pair.zero_baseline) continue;
        const bool first=pair.id1==target,second=pair.id2==target;
        if (!first && !second) continue;
        const Index anchor=first ? pair.id2 : pair.id1;
        if (!scene.images.at(anchor).registered) continue;
        for (const auto& m:pair.matches) {
            const Index feature=first?m.query:m.train,observed=first?m.train:m.query;
            if (feature>=scene.images[target].features.keypoints.size() ||
                observed>=scene.images[anchor].features.keypoints.size())
                throw std::invalid_argument("Invalid structure-less feature");
            forward[{feature,anchor}].insert(observed);
            reverse[{anchor,observed}].insert(feature);
        }
    }
    std::vector<StructurelessLink> links;
    std::set<Index> anchors; std::map<Index,unsigned> features;
    const auto& query_camera=scene.camera_of(scene.images.at(target));
    for (const auto& [key,observations]:forward) {
        const auto [feature,anchor]=key;
        if (observations.size()!=1 || reverse.at({anchor,*observations.begin()}).size()!=1) {
            ++report.ambiguous;continue;
        }
        const Index observed=*observations.begin();const auto& image=scene.images[anchor];
        const auto& camera=scene.camera_of(image);
        const auto& p=scene.images[target].features.keypoints[feature];
        const auto& q=image.features.keypoints[observed];
        const Vec3 u=image.pose.R.transpose()*camera.unproject_normalized(Vec2(q.x,q.y));
        const Vec3 v=query_camera.unproject_normalized(Vec2(p.x,p.y));
        if (!u.allFinite() || !v.allFinite() || !image.pose.C.allFinite()) continue;
        links.push_back({feature,anchor,observed,image.pose.C,u,v,
            camera.pixel_error_to_angular(2),query_camera.pixel_error_to_angular(2)});
        anchors.insert(anchor);++features[feature];
    }
    report.constraints=static_cast<unsigned>(links.size());
    report.features=static_cast<unsigned>(features.size());report.anchors=static_cast<unsigned>(anchors.size());
    for (const auto& [f,n]:features) report.multi_anchor_features+=n>1;
    for (Index a:anchors) for (Index b:anchors)
        report.baseline=std::max(report.baseline,(scene.images[a].pose.C-scene.images[b].pose.C).norm());
    std::set<std::pair<int,int>> cells;
    for (const auto& [f,n]:features) {
        const auto& p=scene.images[target].features.keypoints[f];
        cells.emplace(std::clamp(int(4*p.x/std::max(1u,query_camera.width)),0,3),
                      std::clamp(int(4*p.y/std::max(1u,query_camera.height)),0,3));
    }
    report.occupied_cells=static_cast<unsigned>(cells.size());return links;
}

struct StructurelessEpipolarCost {
    StructurelessLink link;
    template<class T> bool operator()(const T* state,T* residual) const {
        T origin[3]{T(link.origin.x()),T(link.origin.y()),T(link.origin.z())};
        T ray[3]{T(link.ray.x()),T(link.ray.y()),T(link.ray.z())},a[3],b[3];
        ceres::AngleAxisRotatePoint(state,origin,b);ceres::AngleAxisRotatePoint(state,ray,a);
        for (unsigned j=0;j<3;++j) b[j]+=state[3+j];
        const Eigen::Matrix<T,3,1> av(a[0],a[1],a[2]),bv(b[0],b[1],b[2]);
        const auto q=link.query.cast<T>().eval();const auto n=bv.cross(av).eval();
        const auto da=q.cross(bv).eval(),dq=n.cross(q).eval();
        // Tangent-space Sampson normalization, with native camera angular scales.
        const T denominator=T(link.anchor_angle*link.anchor_angle)*(da.cross(av)).squaredNorm()+
            T(link.query_angle*link.query_angle)*dq.squaredNorm()+T(1e-24);
        residual[0]=q.dot(n)/sqrt(denominator);return true;
    }
};

inline bool structureless_pixel_support(const Scene& scene,Index target,const StructurelessLink& link,
    const Pose3D& pose,Vec3* point=nullptr,double* error=nullptr,double pixels=2) {
    const Vec3 v=pose.R.transpose()*link.query;const double c=link.ray.dot(v),d=1-c*c;
    if (d<std::pow(std::sin(3.141592653589793/180),2)) return false;
    const Vec3 delta=pose.C-link.origin;
    const double a=(link.ray.dot(delta)-c*v.dot(delta))/d;
    const double b=(c*link.ray.dot(delta)-v.dot(delta))/d;
    if (a<=0 || b<=0) return false;
    const Vec3 X=0.5*(link.origin+a*link.ray+pose.C+b*v);
    const auto& anchor=scene.images[link.anchor];const auto& query=scene.images[target];
    const auto& p=query.features.keypoints[link.feature];const auto& q=anchor.features.keypoints[link.observed];
    const double e1=scene.camera_of(anchor).checked_reprojection_error(anchor.pose.transform_world_to_camera(X),Vec2(q.x,q.y));
    const double e2=scene.camera_of(query).checked_reprojection_error(pose.transform_world_to_camera(X),Vec2(p.x,p.y));
    if (point) *point=X;if (error) *error=std::max(e1,e2);
    return std::isfinite(e1) && std::isfinite(e2) && std::max(e1,e2)<=pixels;
}

inline StructurelessReport estimate_structureless(const Scene& scene,Index target,unsigned iterations=4096) {
    StructurelessReport report;
    const auto reject=[&](const char* why) {report.reason=why;return report;};
    if (target>=scene.images.size() || scene.images[target].registered) return reject("invalid_target");
    auto links=structureless_links(scene,target,report);
    std::map<Index,unsigned> ordinal;
    for (const auto& l:links) ordinal.emplace(l.feature,0);
    unsigned count=0;for (auto& [f,n]:ordinal) n=count++;
    std::vector<StructurelessLink> fit,heldout;
    for (const auto& l:links) (ordinal.at(l.feature)%5==0?heldout:fit).push_back(l);
    report.fit=static_cast<unsigned>(fit.size());report.holdout=static_cast<unsigned>(heldout.size());
    std::set<Index> fa,ha;for (const auto& l:fit) fa.insert(l.anchor);for (const auto& l:heldout) ha.insert(l.anchor);
    report.fit_anchor_count=static_cast<unsigned>(fa.size());report.holdout_anchor_count=static_cast<unsigned>(ha.size());
    // Permit a diagnostic hypothesis with 30 fit constraints; certification below
    // still requires 60 fit inliers plus independent held-out feature support.
    if (fit.size()<30 || report.features<40 || heldout.size()<12) return reject("insufficient_unique_support");
    if (fa.size()<3 || ha.size()<3 || report.baseline<1e-8) return reject("insufficient_anchor_diversity");
    if (report.occupied_cells<6) return reject("poor_target_coverage");
    const Vec3 origin=fit.front().origin;const double scale=report.baseline;
    auto normalized=fit;for (auto& l:normalized) l.origin=(l.origin-origin)/scale;
    std::map<Index,std::vector<unsigned>> groups;
    for (unsigned i=0;i<fit.size();++i) groups[fit[i].anchor].push_back(i);
    std::vector<Index> eligible;for (const auto& [a,ids]:groups) if(ids.size()>=5) eligible.push_back(a);
    if (eligible.empty()) return reject("insufficient_five_plus_one_support");
    std::mt19937 random(8131);unsigned best=0;Pose3D best_pose;
    const auto world_pose=[&](const poselib::CameraPose& p) {
        Pose3D pose;pose.R=p.R();pose.C=origin+scale*p.center();return pose;
    };
    for (unsigned trial=0;trial<iterations;++trial) {
        const Index anchor=eligible[random()%eligible.size()];const auto& pool=groups.at(anchor);
        std::set<unsigned> chosen;while(chosen.size()<5) chosen.insert(pool[random()%pool.size()]);
        unsigned last=random()%fit.size();
        for (unsigned n=0;fit[last].anchor==anchor && n<fit.size();++n) last=(last+1)%fit.size();
        if (fit[last].anchor==anchor) continue;
        bool duplicate=false;for (unsigned i:chosen) duplicate|=fit[i].feature==fit[last].feature;
        if (duplicate) continue;
        std::vector<Vec3> p1,x1,p2,x2;
        const auto add=[&](unsigned i) {p1.push_back(normalized[i].origin);x1.push_back(normalized[i].ray);p2.push_back(Vec3::Zero());x2.push_back(normalized[i].query);};
        for (unsigned i:chosen) add(i);add(last);
        std::vector<poselib::CameraPose> solutions;poselib::gen_relpose_5p1pt(p1,x1,p2,x2,&solutions);
        for (const auto& solution:solutions) {
            const auto pose=world_pose(solution);if (!pose.C.allFinite() || !pose.R.allFinite()) continue;
            unsigned support=0;for (const auto& l:fit) support+=structureless_pixel_support(scene,target,l,pose,nullptr,nullptr,4);
            if (support>best) {best=support;best_pose=pose;}
        }
        if (best==fit.size() && trial>=127) break;
    }
    report.proposal_inliers=best;
    if(best)for(const auto& l:fit)report.fit_inliers+=structureless_pixel_support(scene,target,l,best_pose);
    if (best<30 || 4*best<fit.size()) return reject("generalized_pose_consensus_failed");
    double state[6];const Eigen::AngleAxisd angle(best_pose.R);
    const Vec3 axis=angle.angle()*angle.axis(),t=-best_pose.R*((best_pose.C-origin)/scale);
    for (unsigned j=0;j<3;++j) {state[j]=axis[j];state[j+3]=t[j];}
    ceres::Problem problem;std::vector<StructurelessEpipolarCost> costs;
    for (unsigned i=0;i<fit.size();++i) if(structureless_pixel_support(scene,target,fit[i],best_pose,nullptr,nullptr,4)) {
        costs.push_back({normalized[i]});
        problem.AddResidualBlock(new ceres::AutoDiffCostFunction<StructurelessEpipolarCost,1,6>(new StructurelessEpipolarCost{normalized[i]}),new ceres::HuberLoss(1),state);
    }
    ceres::Solver::Options options;options.max_num_iterations=60;options.num_threads=1;
    options.linear_solver_type=ceres::DENSE_QR;ceres::Solver::Summary summary;ceres::Solve(options,&problem,&summary);
    if (!summary.IsSolutionUsable()) return reject("epipolar_refinement_failed");
    double rotation[9];ceres::AngleAxisToRotationMatrix(state,rotation);
    Pose3D pose;pose.R=Eigen::Map<Mat3>(rotation);pose.C=origin-scale*pose.R.transpose()*Vec3(state[3],state[4],state[5]);
    Eigen::Matrix<double,6,6> information=Eigen::Matrix<double,6,6>::Zero();
    for (const auto& cost:costs) {
        auto original_link=cost.link;original_link.origin=origin+scale*original_link.origin;
        if(!structureless_pixel_support(scene,target,original_link,pose))continue;
        Eigen::Matrix<double,1,6> jac;
        for (unsigned j=0;j<6;++j) {
            const double saved=state[j];double plus,minus;
            state[j]=saved+1e-6;cost(state,&plus);state[j]=saved-1e-6;cost(state,&minus);state[j]=saved;
            jac[j]=(plus-minus)/2e-6;
        }
        information+=jac.transpose()*jac;
    }
    Eigen::SelfAdjointEigenSolver<Eigen::Matrix<double,6,6>> eig(information);
    report.condition=eig.eigenvalues()[0]/std::max(1e-20,eig.eigenvalues()[5]);
    if (eig.info()!=Eigen::Success || report.condition<1e-8) return reject("unobservable_pose");
    const auto score=[&](const auto& observations,unsigned& inliers,unsigned& features,std::map<Index,unsigned>& anchors,std::vector<Vec3>& points,double& sum) {
        std::map<Index,bool> good;
        for (const auto& l:observations) {
            Vec3 point;double error=1000;
            const bool pass=structureless_pixel_support(scene,target,l,pose,&point,&error);
            auto [it,added]=good.emplace(l.feature,pass);if (!added) it->second &=pass;
            inliers+=pass;if(pass) {++anchors[l.anchor];points.push_back(point);}
            sum+=std::min(1000.,std::isfinite(error)?error:1000.);
        }
        for (const auto& [f,pass]:good) features+=pass;
        return good.size();
    };
    std::map<Index,unsigned> fit_anchors,heldout_anchors;std::vector<Vec3> points;
    double fit_sum=0,holdout_sum=0;report.fit_inliers=0;
    const auto fit_count=score(fit,report.fit_inliers,report.fit_features,fit_anchors,points,fit_sum);
    const auto holdout_count=score(heldout,report.holdout_inliers,report.holdout_features,heldout_anchors,points,holdout_sum);
    report.heldout_mean=holdout_sum/heldout.size();
    if (report.fit_inliers<60 || 5*report.fit_inliers<4*fit.size() || 5*report.fit_features<4*fit_count)
        return reject("refined_fit_failed");
    if (report.holdout_features<12 || 5*report.holdout_features<4*holdout_count || 5*report.holdout_inliers<4*heldout.size())
        return reject("independent_holdout_failed");
    unsigned strong_fit=0,strong_holdout=0;
    for(const auto& [a,n]:fit_anchors) strong_fit+=n>=4;
    for(const auto& [a,n]:heldout_anchors) strong_holdout+=n>=2;
    if(strong_fit<3 || strong_holdout<3) return reject("validated_anchor_diversity_failed");
    Vec3 mean=Vec3::Zero();for(const auto& p:points) mean+=p;mean/=points.size();
    Mat3 covariance=Mat3::Zero();for(const auto& p:points) covariance+=(p-mean)*(p-mean).transpose();
    Eigen::SelfAdjointEigenSolver<Mat3> spatial(covariance);
    if(spatial.info()!=Eigen::Success || spatial.eigenvalues()[0]<1e-5*spatial.eigenvalues()[2]) return reject("planar_structure");
    report.pose=pose;report.accepted=true;report.reason="validated_pose_hypothesis";return report;
}

inline void probe_structureless_caches(const Scene& parent,const std::filesystem::path& cache_root,
    std::ostream& out,std::ostream& anchors_out,std::ostream* completion_out=nullptr,
    const std::filesystem::path& windows_root={},const std::filesystem::path& exports_root={}) {
    out<<"group,target,constraints,unique_features,ambiguous,anchors,multi_anchor_features,baseline,cells,fit,heldout,fit_anchors,heldout_anchors,proposal_inliers_4px,fit_inliers_2px,heldout_inliers,fit_features,heldout_features,condition,heldout_mean,accepted,reason\n";
    anchors_out<<"group,target,anchor,unique_features,fit_features,heldout_features\n";
    if(completion_out)*completion_out<<"group,target,before_2d3d,after_2d3d,added_observations,merged_tracks,retriangulated,conflicts,projection_rejected,before_accepted_local_resection,accepted_local_resection\n";
    std::map<std::filesystem::path,Index> global;for(Index i=0;i<parent.images.size();++i)global.emplace(parent.images[i].path,i);
    for(const auto& directory:std::filesystem::directory_iterator(cache_root)) {
        if(!directory.is_directory())continue;
        std::vector<RawPairMatches> raw;auto local=load_local_proposal_snapshot(directory.path(),raw);
        std::vector<std::uint8_t> stable;
        for(auto& image:local.images) {
            const auto& original=parent.images.at(global.at(image.path));
            image.camera_id=original.camera_id;image.registered=original.registered;
            image.pose=original.registered?original.pose:Pose3D{};stable.push_back(original.registered);
        }
        local.cameras=parent.cameras;for(auto& camera:local.cameras)camera.trust_intrinsics=true;
        verify_local_raw_proposals(local,stable,raw,false);
        if(!windows_root.empty())probe_boundary_windows(local,stable,windows_root/directory.path().filename());
        if(completion_out) {
            build_tracks(local);
            TriangulationOptions options;options.reproj_threshold_px=2;options.min_angle_deg=1;options.split_tracks=false;
            triangulate_tracks(local,false,options);
            std::map<Index,unsigned> before;
            for(Index id=0;id<local.images.size();++id)if(!stable[id])before[id]=stable_depth_support(local,id);
            auto control_trial=local;
            const auto control_recovered=recover_stable_resections(control_trial,stable);
            const std::set<Index> control_accepted(control_recovered.begin(),control_recovered.end());
            const auto completed=complete_stable_tracks(local);
            auto resection_trial=local;
            const auto recovered=recover_stable_resections(resection_trial,stable);
            const std::set<Index> accepted(recovered.begin(),recovered.end());
            for(const auto& [id,n]:before)*completion_out<<directory.path().filename().string()<<','<<std::quoted(local.images[id].path.filename().string())<<','
                <<n<<','<<stable_depth_support(local,id)<<','<<completed.added<<','<<completed.merged<<','<<completed.retriangulated<<','
                <<completed.conflicts<<','<<completed.projection_rejected<<','<<control_accepted.contains(id)<<','<<accepted.contains(id)<<'\n';
            completion_out->flush();
            const auto export_candidate=[&](const Scene& trial,const std::set<Index>& selected,const std::string& label) {
                if(exports_root.empty() || selected.empty())return;
                Scene snapshot;snapshot.cameras=parent.cameras;
                for(const auto& original:parent.images) {
                    Image image;image.id=original.id;image.camera_id=original.camera_id;image.path=original.path;
                    image.pose=original.pose;image.registered=original.registered;snapshot.images.push_back(std::move(image));
                }
                if(!std::filesystem::exists(exports_root/"parent-colmap")) {
                    save_colmap_text(snapshot,exports_root/"parent-colmap");
                    save_asfm(snapshot,exports_root/"parent-poses.asfm");
                }
                for(Index id:selected) {
                    auto& image=snapshot.images.at(global.at(trial.images[id].path));
                    image.pose=trial.images[id].pose;image.registered=true;
                }
                // Pose-only evaluation snapshot, not a complete reconstruction.
                save_asfm(snapshot,exports_root/(directory.path().filename().string()+"-"+label+"-poses.asfm"));
                save_colmap_text(snapshot,exports_root/(directory.path().filename().string()+"-"+label+"-colmap"));
            };
            export_candidate(control_trial,control_accepted,"control");
            export_candidate(resection_trial,accepted,"completion");
        }
        for(Index target=0;target<local.images.size();++target)if(!stable[target]) {
            const auto report=estimate_structureless(local,target);
            out<<directory.path().filename().string()<<','<<std::quoted(local.images[target].path.filename().string())<<','
                <<report.constraints<<','<<report.features<<','<<report.ambiguous<<','<<report.anchors<<','<<report.multi_anchor_features<<','
                <<report.baseline<<','<<report.occupied_cells<<','<<report.fit<<','<<report.holdout<<','<<report.fit_anchor_count<<','<<report.holdout_anchor_count<<','
                <<report.proposal_inliers<<','<<report.fit_inliers<<','<<report.holdout_inliers<<','<<report.fit_features<<','<<report.holdout_features<<','
                <<report.condition<<','<<report.heldout_mean<<','<<report.accepted<<','<<report.reason<<'\n';out.flush();
            StructurelessReport support;const auto links=structureless_links(local,target,support);
            std::map<Index,unsigned> ordinal;for(const auto& l:links)ordinal.emplace(l.feature,0);
            unsigned n=0;for(auto& [f,i]:ordinal)i=n++;
            std::map<Index,std::array<unsigned,3>> counts;
            for(const auto& l:links) {auto& c=counts[l.anchor];++c[0];++c[ordinal.at(l.feature)%5==0?2:1];}
            for(const auto& [a,c]:counts)anchors_out<<directory.path().filename().string()<<','<<std::quoted(local.images[target].path.filename().string())<<','
                <<std::quoted(local.images[a].path.filename().string())<<','<<c[0]<<','<<c[1]<<','<<c[2]<<'\n';
        }
    }
    if(!out || !anchors_out)throw std::runtime_error("Cannot write structure-less audit");
}
} // namespace photara::sfm::experiment
