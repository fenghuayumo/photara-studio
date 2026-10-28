#include "camera_timeline.hpp"
#include "camera_path_recording.hpp"
#include "camera_capture_path.hpp"
#include "camera_path_overlay.hpp"
#include <string>
#include <array>

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
struct Key { float time; };
struct CaptureView {
    std::string name;
    bool registered{true};
    struct { float x{},y{},z{}; } centre;
    std::array<float,9> rotation{1,0,0,0,1,0,0,0,1};
    float fy{240.F},k1{};
    unsigned height{480};
};
struct OrbitKey { float time{}, yaw{}, roll{}; int interpolation{}; unsigned thumbnail_width{}, thumbnail_height{}; std::vector<unsigned char> thumbnail_rgb; };
struct RecordedKey {
    float time{}, target[3]{}, yaw{}, pitch{}, roll{}, distance{1.F};
    float fov_degrees{50.F}, ortho_height{4.F}, fisheye_k1{};
    int projection{}, interpolation{};
};
struct CopyProbe : RecordedKey {
    inline static int copies=0;
    CopyProbe()=default;
    CopyProbe(const CopyProbe& value):RecordedKey(value){++copies;}
    CopyProbe(CopyProbe&&)=default;
    CopyProbe& operator=(const CopyProbe&)=default;
};
void expect(bool ok, const char* message) {
    if (!ok) { std::cerr << message << '\n'; std::exit(1); }
}
bool near(float a, float b) { return std::abs(a - b) < 0.0001F; }
}
int main() {
    using namespace editor::camera_timeline;
    const std::vector<Key> keys{{0.F}, {2.F}, {5.F}};
    expect(near(span(0.F), 6.F) && near(span(20.F), 23.F), "timeline span");
    expect(near(tick_step(0.6F), 1.F) && near(tick_step(1.1F), 2.F) &&
           near(tick_step(2.2F), 5.F) && near(tick_step(7.F), 10.F), "tick spacing");
    expect(previous(keys, 0.F) == 0 && previous(keys, 2.F) == 0 && previous(keys, 3.F) == 1,
           "previous key from key and from between keys");
    expect(next(keys, 0.F) == 1 && next(keys, 3.F) == 2 && next(keys, 5.F) == 2,
           "next key from key and from between keys");
    expect(near(bounded_time(keys, 1, -1.F), 0.01F) && near(bounded_time(keys, 1, 8.F), 4.99F),
           "dragging cannot cross adjacent keys");
    expect(near(bounded_time(keys, 0, -1.F), 0.F) && near(bounded_time(keys, 2, 50000.F), 36000.F),
           "path endpoints stay within allowed time");
    expect(near(bounded_time(keys, 1, std::numeric_limits<float>::quiet_NaN()), 2.F),
           "non-finite time is ignored");
    expect(near(interpolation_weight(.25F, 0), .15625F), "smooth easing");
    expect(near(interpolation_weight(.25F, 1), .25F), "linear interpolation");
    expect(near(interpolation_weight(.99F, 2), 0.F) && near(interpolation_weight(1.F, 2), 1.F),
           "hold switches at segment endpoint");
    std::vector<OrbitKey> orbit;
    OrbitKey pose; pose.yaw = .3F; pose.roll = .2F;
    expect(append_orbit(orbit, pose, 12.F, false) == 0 && orbit.size() == 9, "orbit generates a full revolution");
    expect(near(orbit.back().time, 12.F) && near(orbit.back().yaw - orbit.front().yaw, 6.2831853F), "orbit duration and direction");
    expect(orbit[3].interpolation == 1 && near(orbit[3].roll, .2F), "orbit maintains pose and constant angular speed");
    expect(append_orbit(orbit, pose, 4.F, true) == 9 && near(orbit.back().time, 18.F), "orbit appends without replacing keys");
    expect(orbit.back().yaw < orbit[9].yaw, "clockwise reverses direction");
    const auto count = orbit.size();
    append_orbit(orbit, pose, std::numeric_limits<float>::infinity(), false);
    expect(orbit.size() == count, "invalid orbit duration cannot mutate path");
    std::vector<RecordedKey> recording(11);
    for (std::size_t i = 0; i < recording.size(); ++i) {
        recording[i].time = float(i) * .1F;
        recording[i].target[0] = float(i) * .1F;
    }
    auto reduced = editor::camera_recording::reduce(recording, {.01F,.01F,.2F});
    expect(reduced.size() == 2 && reduced.front().time == 0.F && near(reduced.back().time, 1.F) &&
           reduced.front().interpolation == 1, "straight constant-speed recording needs only endpoints");
    recording[4].target[1] = .25F;
    reduced = editor::camera_recording::reduce(recording, {.01F,.01F,.2F});
    expect(reduced.size() > 2, "turns outside the position tolerance are retained");
    for (std::size_t i = 0; i < recording.size(); ++i) {
        recording[i].target[1] = 0.F;
        recording[i].target[0] = i < 5 ? 0.F : float(i - 5) * .2F;
    }
    reduced = editor::camera_recording::reduce(recording, {.01F,.01F,.2F});
    expect(reduced.size() > 2 && near(reduced[1].time, .5F), "timing preserves a pause before motion");
    for (std::size_t i = 0; i < recording.size(); ++i) {
        recording[i].target[0] = 0.F;
        recording[i].yaw = i < 5 ? 3.13F : -3.13F;
        recording[i].projection = i < 5 ? 0 : 1;
    }
    reduced = editor::camera_recording::reduce(recording, {.01F,.01F,.2F});
    bool before_cut = false, after_cut = false;
    for (const auto& k : reduced) {
        before_cut |= near(k.time, .4F) && k.projection == 0;
        after_cut |= near(k.time, .5F) && k.projection == 1;
    }
    expect(before_cut && after_cut, "projection changes retain both sides of the transition");
    for (std::size_t i = 0; i < recording.size(); ++i) {
        recording[i].projection = 0;
        recording[i].yaw = 0.F;
        recording[i].fov_degrees = i == 5 ? 60.F : 50.F;
    }
    expect(editor::camera_recording::reduce(recording, {.01F,.01F,.2F}).size() > 2, "lens changes are retained");
    using editor::camera_recording::Recorder;
    const auto quality = editor::camera_recording::quality_for(1, 1.F);
    RecordedKey sample;
    Recorder<RecordedKey> take;
    take.begin(sample,quality,false);
    for (int i=1;i<10000;++i) take.observe(sample,float(i)*.01F);
    expect(take.samples.empty() && take.finish(sample,100.F).empty(), "stillness allocates no samples and produces no path");
    take.begin(sample,quality,false);
    for (int i=1;i<=60;++i) { sample.target[0]=float(i)/60.F; take.observe(sample,float(i)/60.F); }
    auto flight = take.finish(sample,1.F);
    expect(flight.size()==2 && flight.front().time==0.F, "constant-speed flight compresses to endpoints");
    for (bool preserve : {false,true}) {
        sample = {}; take.begin(sample,quality,preserve);
        take.observe(sample,5.F); // preparation must never pad the path
        sample.target[0]=1.F; take.observe(sample,5.1F);
        take.observe(sample,10.F); // idle observations do not grow samples
        const auto before=take.samples.size();
        take.observe(sample,12.F);
        expect(take.samples.size()==before, "idle does not allocate candidates");
        sample.target[0]=2.F; take.observe(sample,12.1F);
        flight=take.finish(sample,12.1F);
        expect(flight.front().time==0.F, "preparation is trimmed for both timing modes");
        if (preserve) expect(flight.back().time>7.F, "preserve mode retains internal pauses");
        else expect(flight.back().time<.3F, "trim mode removes internal idle time");
    }
    sample={}; take.begin(sample,quality,false);
    for(int i=1;i<100;++i) { sample.target[0]=(i%2 ? 1.F : -1.F)*1e-6F; take.observe(sample,float(i)*.02F); }
    expect(take.samples.empty(), "sub-tolerance jitter cannot create a path");
    std::vector<RecordedKey> turning(31);
    for(std::size_t i=0;i<turning.size();++i) { turning[i].time=float(i)*.1F; turning[i].yaw=float(i)*.20943951F; }
    auto curved=editor::camera_recording::reduce(turning,quality);
    expect(curved.size()>2, "a full rotation cannot collapse into identical endpoints");
    for(std::size_t i=1;i+1<turning.size();++i) {
        auto right=std::upper_bound(curved.begin(),curved.end(),turning[i].time,[](float t,const auto& k){return t<k.time;});
        const auto& b=*right; const auto& a=*(right-1);
        const auto prediction=editor::camera_recording::interpolate(a,b,(turning[i].time-a.time)/(b.time-a.time));
        expect(editor::camera_recording::error(turning[i],prediction,quality)<=1.001F,"reduced playback respects pose error bound");
    }
    const auto precise=editor::camera_recording::reduce(turning,editor::camera_recording::quality_for(0,1.F));
    const auto compact=editor::camera_recording::reduce(turning,editor::camera_recording::quality_for(2,1.F));
    expect(precise.size()>=compact.size(),"precision setting cannot reduce full-turn detail");
    using namespace editor::camera_capture_path;
    std::vector<CaptureView> views(4);
    views[0].name="frame10.png"; views[1].name="frame2.png"; views[2].name="frame1.png";
    views[3].registered=false;
    const auto sorted=ordered(views,true,false);
    expect(sorted==std::vector<std::size_t>({2,1,0}),"capture path uses numeric filename ordering and skips unregistered cameras");
    expect(ordered(views,false,true)==std::vector<std::size_t>({2,1,0}),"capture dataset order supports reversal");
    views[0].rotation[0]=std::numeric_limits<float>::quiet_NaN();
    expect(ordered(views,true,false).size()==2,"non-finite capture rotations are rejected");
    auto view=views[1]; view.centre={2.F,3.F,4.F};
    const auto key=key_from_view<RecordedKey>(view,0,2.F,3.F);
    const auto direction=editor::camera_recording::forward(key);
    expect(near(key.target[0]-direction[0]*key.distance,2.F) && near(key.target[1]-direction[1]*key.distance,3.F) &&
           near(key.target[2]-direction[2]*key.distance,4.F),"capture conversion preserves camera position");
    expect(near(key.fov_degrees,90.F) && key.interpolation==1 && near(key.time,3.F),"capture conversion keeps lens and assigns linear timeline time");
    view.rotation={0,-1,0,1,0,0,0,0,1};
    const auto rolled=key_from_view<RecordedKey>(view,0,2.F,0.F);
    const auto rolled_right=editor::camera_recording::right(rolled);
    expect(near(rolled_right[0],0.F) && near(rolled_right[1],-1.F),"capture path preserves camera roll");
    view.k1=.12F;
    expect(near(key_from_view<RecordedKey>(view,2,2.F,0.F).fisheye_k1,.12F),"capture fisheye lens coefficient survives conversion");
    views.resize(5);
    for(std::size_t i=0;i<views.size();++i) { views[i]={}; views[i].name="frame"+std::to_string(i); views[i].centre.x=float(i); }
    const auto path=generate<RecordedKey>(views,ordered(views,false,false),0,4,8.F,3,2.F,[](const auto&){return 0;});
    expect(path.size()==3 && near(path[0].time,0.F) && near(path[1].time,6.F) && near(path.back().time,8.F),
           "capture interval preserves source timing and includes the final camera");
    const auto subset=generate<RecordedKey>(views,ordered(views,false,false),1,3,4.F,1,2.F,[](const auto&){return 0;});
    expect(subset.size()==3 && near(subset.front().target[0],1.F) && near(subset.back().target[0],3.F),"capture range selects requested endpoints");
    bool invalid_range=false;
    try { (void)generate<RecordedKey>(views,ordered(views,false,false),2,2,4.F,1,2.F,[](const auto&){return 0;}); }
    catch(const std::invalid_argument&) { invalid_range=true; }
    expect(invalid_range,"single-camera paths are rejected");
    std::vector<RecordedKey> playback(2);
    playback[1].time=2.F; playback[1].target[0]=2.F;
    playback[0].interpolation=0;
    expect(near(editor::camera_timeline::evaluate(playback,.5F).target[0],.3125F),"shared path evaluation uses smooth interpolation");
    playback[0].interpolation=1;
    expect(near(editor::camera_timeline::evaluate(playback,.5F).target[0],.5F),"shared path evaluation uses linear interpolation");
    playback[0].interpolation=2;
    expect(near(editor::camera_timeline::evaluate(playback,1.9F).target[0],0.F) && near(editor::camera_timeline::evaluate(playback,2.F).target[0],2.F),"shared path evaluation preserves hold and cut endpoints");
    std::vector<CopyProbe> probes(2); probes[1].time=1.F;
    CopyProbe::copies=0;
    (void)editor::camera_recording::interpolate(probes[0],probes[1],.5F);
    (void)editor::camera_timeline::evaluate(probes,.5F);
    (void)editor::camera_timeline::evaluate(probes,0.F);
    (void)editor::camera_timeline::evaluate(probes,1.F);
    expect(CopyProbe::copies==0,"pose interpolation and endpoint evaluation never copy a full keyframe");
    RecordedKey arc_a,arc_b; arc_a.distance=arc_b.distance=1.F;
    arc_a.interpolation=1; arc_b.yaw=3.14159265F;
    std::vector<std::pair<editor::camera_recording::V3,editor::camera_recording::V3>> chords;
    const auto projection=[](const auto& p,editor::camera_path_overlay::V2& screen){screen={p[0]*100.F,p[2]*100.F};return true;};
    auto arc_segments=editor::camera_path_overlay::tessellate(arc_a,arc_b,.001F,projection,
        [&](auto a,auto b){chords.emplace_back(a,b);});
    expect(arc_segments>8,"half-turn trajectory is subdivided by curvature");
    for(int i=1;i<128;++i) {
        const auto eye=editor::camera_path_overlay::eye(editor::camera_recording::interpolate(arc_a,arc_b,float(i)/128.F));
        editor::camera_path_overlay::V2 screen; projection(eye,screen);
        float best=1e9F;
        for(const auto& chord:chords) {
            editor::camera_path_overlay::V2 a,b; projection(chord.first,a);projection(chord.second,b);
            best=std::min(best,editor::camera_path_overlay::chord_error(screen,a,b));
        }
        expect(best<=.751F,"rendered arc remains within the screen-space tolerance");
    }
    arc_b=arc_a; arc_b.target[0]=1.F;
    expect(editor::camera_path_overlay::tessellate(arc_a,arc_b,.001F,projection,[](auto,auto){})==1,"straight flight uses a single chord");
    arc_a.interpolation=2;
    expect(editor::camera_path_overlay::tessellate(arc_a,arc_b,.001F,projection,[](auto,auto){})==0,"hold/cut has no flight chord");
    editor::camera_timeline::View timeline_view;
    timeline_view.constrain(40.F);
    expect(near(timeline_view.duration,40.F) && near(timeline_view.start,0.F),"timeline initially fits the full path");
    const float anchor=timeline_view.time_at(.75F);
    timeline_view.zoom(40.F,4.F,.75F);
    expect(near(timeline_view.duration,10.F) && near(timeline_view.time_at(.75F),anchor),"zoom preserves the time under the pointer");
    timeline_view.pan(40.F,100.F);
    expect(near(timeline_view.start,30.F),"pan clamps to the end of the path");
    timeline_view.pan(40.F,-100.F);
    expect(near(timeline_view.start,0.F),"pan clamps to the start of the path");
    timeline_view.reveal(40.F,37.F);
    expect(timeline_view.fraction(37.F)>=0.F && timeline_view.fraction(37.F)<=1.F,"transport navigation reveals an offscreen playhead");
    expect(timeline_view.fraction(0.F)<0.F,"offscreen keys remain outside the range instead of clamping to its edge");
    const float timestamp=timeline_view.time_at(.33F);
    expect(near(timeline_view.fraction(timestamp),.33F),"zoomed timeline coordinates round-trip");
    timeline_view.zoom(40.F,1e9F,.5F);
    expect(near(timeline_view.duration,.1F),"timeline zoom stops at a useful minimum duration");
    timeline_view.zoom(40.F,1e-9F,.5F);
    expect(timeline_view.fit && near(timeline_view.duration,40.F),"zooming out restores fit mode");
    expect(near(editor::camera_timeline::jump_time(3.F,-5.F,0.F,33.F),0.F) &&
           near(editor::camera_timeline::jump_time(31.F,5.F,0.F,33.F),33.F),"transport jumps clamp to path endpoints");
    return 0;
}
