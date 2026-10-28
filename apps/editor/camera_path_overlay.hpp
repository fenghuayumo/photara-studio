#pragma once

#include "camera_timeline.hpp"

namespace editor::camera_path_overlay {
using V2=std::array<float,2>;
using camera_recording::V3;
template <typename Key> V3 eye(const Key& key) {
    const auto f=camera_recording::forward(key);
    return {key.target[0]-f[0]*key.distance,key.target[1]-f[1]*key.distance,key.target[2]-f[2]*key.distance};
}
template <std::size_t N> float chord_error(const std::array<float,N>& p,const std::array<float,N>& a,const std::array<float,N>& b) {
    float dot=0.F,length=0.F;
    for (std::size_t i=0;i<N;++i) {dot+=(p[i]-a[i])*(b[i]-a[i]);length+=(b[i]-a[i])*(b[i]-a[i]);}
    const float u=length>1e-20F ? std::clamp(dot/length,0.F,1.F) : 0.F;
    float error=0.F;
    for (std::size_t i=0;i<N;++i) {const float d=p[i]-(a[i]+u*(b[i]-a[i]));error+=d*d;}
    return std::sqrt(error);
}

// Independent of keyframe count: refine curved segments until their projected
// chord error is below a pixel. Invisible/near-plane samples use world error.
template <typename Key,typename Project,typename Emit>
std::size_t tessellate(const Key& a,const Key& b,float world_tolerance,Project project,Emit emit) {
    if (a.interpolation==2) return 0;
    const auto point=[&](float t){return eye(camera_recording::interpolate(a,b,camera_timeline::interpolation_weight(t,a.interpolation)));};
    std::size_t count=0;
    const auto refine=[&](auto&& self,float start,float end,V3 p,V3 q,int depth)->void {
        const float mid=(start+end)*.5F;
        const V3 middle=point(mid),quarter=point((start+mid)*.5F),three_quarters=point((mid+end)*.5F);
        V2 ps,qs,ms,s1,s3;
        const bool visible=project(p,ps) && project(q,qs) && project(middle,ms) && project(quarter,s1) && project(three_quarters,s3);
        const float error=visible
            ? std::max({chord_error(ms,ps,qs),chord_error(s1,ps,qs),chord_error(s3,ps,qs)})
            : std::max({chord_error(middle,p,q),chord_error(quarter,p,q),chord_error(three_quarters,p,q)});
        if (depth<10 && error>(visible ? .75F : std::max(1e-5F,world_tolerance))) {
            self(self,start,mid,p,middle,depth+1); self(self,mid,end,middle,q,depth+1);
        } else {emit(p,q);++count;}
    };
    refine(refine,0.F,1.F,eye(a),eye(b),0);
    return count;
}
} // namespace editor::camera_path_overlay
