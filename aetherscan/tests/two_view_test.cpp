#include "aetherscan/sfm/two_view.hpp"
#include "aetherscan/sfm/triangulation.hpp"

#include <cmath>
#include <iostream>
#include <random>

int main() {
    using namespace aetherscan;
    sfm::Camera camera0{0,1280,720,900.0,900.0,640.0,360.0};
    sfm::Camera camera1=camera0; camera1.id=1;
    features::FeatureSet first,second;
    first.image_width=second.image_width=1280; first.image_height=second.image_height=720;
    features::MatchSet matches;
    std::mt19937 random(9); std::uniform_real_distribution<double> xy(-2.0,2.0),z(4.0,10.0);
    std::normal_distribution<double> noise(0.0,0.2);
    for (std::uint32_t i=0;i<150;++i) {
        const double x=xy(random),y=xy(random),depth=z(random);
        first.keypoints.push_back({static_cast<float>(900*x/depth+640+noise(random)),
                                   static_cast<float>(900*y/depth+360+noise(random))});
        second.keypoints.push_back({static_cast<float>(900*(x-1.0)/depth+640+noise(random)),
                                    static_cast<float>(900*y/depth+360+noise(random))});
        matches.matches.push_back({i,i,1.0F});
    }
    for (std::uint32_t i=0;i<25;++i) matches.matches[i].train=(i*37+11)%150;
    const auto geometry=sfm::verify_two_view(camera0,camera1,first,second,matches);
    std::cout << "inliers=" << geometry.inliers.matches.size()
              << " homography=" << geometry.homography_inliers << '\n';
    if (!geometry.valid || geometry.inliers.matches.size()<100 ||
        std::abs(geometry.translation_direction[0])<0.9) return 1;
    std::vector<sfm::View> views(2);
    views[0].id=0; views[1].id=1; views[0].features=first; views[1].features=second;
    sfm::VerifiedPair pair{0,1,geometry.inliers};
    const auto tracks=sfm::build_tracks(views,{pair});
    if (tracks.size()<100) return 2;
    for (const auto& track : tracks)
        if (track.observations.size()!=2 ||
            track.observations[0].view_id==track.observations[1].view_id) return 3;
    sfm::Scene scene;
    scene.cameras={camera0,camera1}; scene.views=views;
    scene.views[0].camera_id=0; scene.views[1].camera_id=1;
    scene.views[0].registered=scene.views[1].registered=true;
    scene.views[1].pose.cx=1.0;
    const auto index0=static_cast<features::FeatureIndex>(scene.views[0].features.keypoints.size());
    const auto index1=static_cast<features::FeatureIndex>(scene.views[1].features.keypoints.size());
    scene.views[0].features.keypoints.push_back({676.0F,378.0F});
    scene.views[1].features.keypoints.push_back({496.0F,378.0F});
    const sfm::Track known_track{0,{{0,index0},{1,index1}}};
    const auto triangulated=sfm::triangulate_track(scene,known_track);
    if (!triangulated.valid || std::abs(triangulated.position[0]-0.2)>1e-6 ||
        std::abs(triangulated.position[1]-0.1)>1e-6 ||
        std::abs(triangulated.position[2]-5.0)>1e-6) return 4;
    return 0;
}
