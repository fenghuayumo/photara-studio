#pragma once

#include "camera_path_recording.hpp"
#include <cctype>
#include <numeric>
#include <string_view>
#include <stdexcept>

namespace editor::camera_capture_path {
inline bool natural_less(std::string_view a, std::string_view b) {
    std::size_t i=0,j=0;
    while (i<a.size() && j<b.size()) {
        if (std::isdigit(static_cast<unsigned char>(a[i])) && std::isdigit(static_cast<unsigned char>(b[j]))) {
            auto ae=i,be=j;
            while (ae<a.size() && std::isdigit(static_cast<unsigned char>(a[ae]))) ++ae;
            while (be<b.size() && std::isdigit(static_cast<unsigned char>(b[be]))) ++be;
            auto az=i,bz=j;
            while (az<ae && a[az]=='0') ++az;
            while (bz<be && b[bz]=='0') ++bz;
            if (ae-az!=be-bz) return ae-az<be-bz;
            const auto cmp=a.substr(az,ae-az).compare(b.substr(bz,be-bz));
            if (cmp) return cmp<0;
            i=ae; j=be;
        } else {
            const int ac=std::tolower(static_cast<unsigned char>(a[i]));
            const int bc=std::tolower(static_cast<unsigned char>(b[j]));
            if (ac!=bc) return ac<bc;
            ++i; ++j;
        }
    }
    return i==a.size() && j!=b.size();
}
template <typename View> bool usable(const View& v) {
    if (!v.registered || !std::isfinite(v.centre.x) || !std::isfinite(v.centre.y) || !std::isfinite(v.centre.z)) return false;
    for (float x : v.rotation) if (!std::isfinite(x)) return false;
    const auto& r=v.rotation;
    const camera_recording::V3 right{r[0],r[1],r[2]}, down{r[3],r[4],r[5]}, forward{r[6],r[7],r[8]};
    return std::abs(camera_recording::dot(right,right)-1.F)<.02F &&
           std::abs(camera_recording::dot(down,down)-1.F)<.02F &&
           std::abs(camera_recording::dot(forward,forward)-1.F)<.02F &&
           std::abs(camera_recording::dot(right,forward))<.02F &&
           camera_recording::dot(camera_recording::cross(right,down),forward)>.98F;
}
template <typename View> std::vector<std::size_t> ordered(const std::vector<View>& views, bool by_name, bool reverse) {
    std::vector<std::size_t> indices;
    for (std::size_t i=0;i<views.size();++i) if (usable(views[i])) indices.push_back(i);
    if (by_name) std::stable_sort(indices.begin(),indices.end(),[&](auto a,auto b){return natural_less(views[a].name,views[b].name);});
    if (reverse) std::reverse(indices.begin(),indices.end());
    return indices;
}
template <typename Key,typename View> Key key_from_view(const View& v, int projection, float distance, float time) {
    Key key{};
    const auto& r=v.rotation;
    camera_recording::V3 look{r[6],r[7],r[8]}, right{r[0],r[1],r[2]};
    const float look_length=std::sqrt(camera_recording::dot(look,look));
    for (float& x : look) x/=look_length;
    const float tilt=camera_recording::dot(right,look);
    for (int j=0;j<3;++j) right[j]-=tilt*look[j];
    const float right_length=std::sqrt(camera_recording::dot(right,right));
    for (float& x : right) x/=right_length;
    key.distance=std::max(.01F,distance); key.time=time;
    key.target[0]=v.centre.x+look[0]*key.distance;
    key.target[1]=v.centre.y+look[1]*key.distance;
    key.target[2]=v.centre.z+look[2]*key.distance;
    key.pitch=std::asin(std::clamp(look[1],-1.F,1.F));
    key.yaw=std::hypot(look[0],look[2])>1e-6F ? std::atan2(-look[0],-look[2]) : 0.F;
    const camera_recording::V3 level{-std::cos(key.yaw),0.F,std::sin(key.yaw)};
    key.roll=std::atan2(camera_recording::dot(right,camera_recording::cross(level,look)),camera_recording::dot(right,level));
    key.projection=projection;
    key.interpolation=1;
    if (std::isfinite(v.fy) && v.fy>1e-4F && v.height>0)
        key.fov_degrees=std::clamp(2.F*std::atan(float(v.height)*.5F/v.fy)*57.2957795F,1.F,179.F);
    key.fisheye_k1=std::isfinite(v.k1) && projection==2 ? v.k1 : 0.F;
    return key;
}
template <typename Key,typename View,typename Projection>
std::vector<Key> generate(const std::vector<View>& views,const std::vector<std::size_t>& indices,
                         std::size_t first,std::size_t last,float duration,std::size_t step,float distance,Projection projection) {
    if (first>=last || last>=indices.size() || !step || !std::isfinite(duration) ||
        duration/float(last-first)<.01F) throw std::invalid_argument("Invalid capture path range or duration");
    const auto intervals=last-first;
    const auto count=intervals/step+1+(intervals%step!=0);
    if (count>10000) throw std::invalid_argument("Too many capture path samples");
    std::vector<Key> samples;
    for (std::size_t i=first;i<=last;++i) {
        if ((i-first)%step && i!=last) continue;
        const auto& view=views.at(indices[i]);
        samples.push_back(key_from_view<Key>(view,projection(view),distance,duration*float(i-first)/float(intervals)));
    }
    return samples;
}
} // namespace editor::camera_capture_path
