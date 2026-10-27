#include "camera_timeline.hpp"

#include <cmath>
#include <iostream>
#include <limits>
#include <vector>

namespace {
struct Key { float time; };
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
    return 0;
}
