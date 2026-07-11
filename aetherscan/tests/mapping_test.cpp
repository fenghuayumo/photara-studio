#include "aetherscan/sfm/mapping.hpp"

#include <cmath>
#include <iostream>

namespace {

const std::array<std::array<double,2>,6> centers{{{0,0},{0.35,0.05},{0.7,-0.04},{1.05,0.08},{1.4,-0.06},{1.75,0.03}}};

aetherscan::sfm::Scene make_scene() {
    using namespace aetherscan;
    sfm::Scene scene;
    constexpr std::size_t view_count=6,point_count=120;
    for(std::size_t i=0;i<view_count;++i){scene.cameras.push_back({static_cast<sfm::Id>(i),1280,720,900,900,640,360});
        sfm::View view;view.id=static_cast<sfm::Id>(i);view.camera_id=view.id;view.features.image_width=1280;view.features.image_height=720;scene.views.push_back(std::move(view));}
    for(std::size_t point=0;point<point_count;++point){const double x=-1.5+3.0*(point%12)/11.0,y=-1.0+2.0*((point/12)%10)/9.0,z=5.0+0.02*point;
        sfm::Track track;track.id=static_cast<sfm::Id>(point);
        for(std::size_t view=0;view<view_count;++view){const auto feature=static_cast<features::FeatureIndex>(scene.views[view].features.keypoints.size());
            scene.views[view].features.keypoints.push_back({static_cast<float>(900*(x-centers[view][0])/z+640),static_cast<float>(900*(y-centers[view][1])/z+360)});track.observations.push_back({static_cast<sfm::Id>(view),feature});}
        scene.tracks.push_back(std::move(track));}
    return scene;
}

std::vector<aetherscan::sfm::RelativePoseEdge> edges() {
    using namespace aetherscan::sfm;std::vector<RelativePoseEdge> result;
    const std::array<double,9> identity{1,0,0,0,1,0,0,0,1};
    const auto add=[&](Id first,Id second,double weight,std::size_t inliers){const double dx=centers[second][0]-centers[first][0],dy=centers[second][1]-centers[first][1];const double norm=std::sqrt(dx*dx+dy*dy);
        result.push_back({first,second,identity,{-dx/norm,-dy/norm,0},weight,inliers});};
    for(Id i=0;i<5;++i)add(i,i+1,1.0,100);
    for(Id i=0;i<4;++i)add(i,i+2,0.7,80);
    return result;
}

std::vector<aetherscan::sfm::RelativePoseEdge> rotated_edges() {
    using namespace aetherscan::sfm;std::vector<RelativePoseEdge> result;
    const auto rotation=[](double angle){const double c=std::cos(angle),s=std::sin(angle);return std::array<double,9>{c,0,s,0,1,0,-s,0,c};};
    const auto multiply=[](const std::array<double,9>& a,const std::array<double,9>& b){std::array<double,9> r{};for(int i=0;i<3;++i)for(int j=0;j<3;++j)for(int k=0;k<3;++k)r[i*3+j]+=a[i*3+k]*b[j*3+k];return r;};
    const auto add=[&](Id first,Id second,double weight){const auto relative=multiply(rotation(0.04*second),rotation(0.04*first));
        const double dx=centers[second][0]-centers[first][0],dy=centers[second][1]-centers[first][1],norm=std::sqrt(dx*dx+dy*dy);const auto target_rotation=rotation(0.04*second);
        const double tx=-(target_rotation[0]*dx+target_rotation[1]*dy)/norm,ty=-(target_rotation[3]*dx+target_rotation[4]*dy)/norm,tz=-(target_rotation[6]*dx+target_rotation[7]*dy)/norm;
        result.push_back({first,second,relative,{tx,ty,tz},weight,100});};
    for(Id i=0;i<5;++i)add(i,i+1,1.0);
    for(Id i=0;i<4;++i)add(i,i+2,0.7);
    return result;
}

}  // namespace

int main(){using namespace aetherscan;
    const auto rotated=sfm::average_global_poses(6,rotated_edges());
    if(!rotated.valid||std::abs(rotated.poses.back().qy-std::sin(0.1))>1e-4)return 5;
    const auto averaged=sfm::average_global_poses(6,edges());
    if(!averaged.valid||averaged.poses.size()!=6)return 1;
    for(std::size_t i=1;i<6;++i)if(!(averaged.poses[i].cx>averaged.poses[i-1].cx))return 2;
    auto incremental=make_scene();sfm::RelativePoseEdge seed=edges().front();
    sfm::IncrementalMapperOptions incremental_options;incremental_options.bundle.optimizer.maximum_iterations=3;incremental_options.bundle.maximum_local_views=6;
    const auto incremental_summary=sfm::run_incremental_mapping(incremental,seed,incremental_options);
    std::cout<<"incremental views="<<incremental_summary.registered_views<<" points="<<incremental_summary.landmarks<<'\n';
    if(!incremental_summary.valid||incremental_summary.registered_views!=6||incremental_summary.landmarks<100)return 3;
    auto global=make_scene();sfm::GlobalMapperOptions global_options;global_options.bundle.maximum_iterations=3;
    const auto global_summary=sfm::run_global_mapping(global,edges(),global_options);
    std::cout<<"global views="<<global_summary.registered_views<<" points="<<global_summary.landmarks<<'\n';
    if(!global_summary.valid||global_summary.registered_views!=6||global_summary.landmarks<100)return 4;
    return 0;
}
