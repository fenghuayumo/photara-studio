#include "camera_timeline.hpp"
#include "camera_path_recording.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
struct Key { float time; };
struct OrbitKey { float time{}, yaw{}, roll{}; int interpolation{}; unsigned thumbnail_width{}, thumbnail_height{}; std::vector<unsigned char> thumbnail_rgb; };
struct RecordedKey {
    float time{}, target[3]{}, yaw{}, pitch{}, roll{}, distance{1.F};
    float fov_degrees{50.F}, ortho_height{4.F}, fisheye_k1{};
    int projection{}, interpolation{};
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
    return 0;
}
