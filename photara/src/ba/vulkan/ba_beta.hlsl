#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> scalars;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(1, 1, 1)]
void main() {
    double previous = scalars[0];
    double next = scalars[3];
    scalars[4] = (abs_d(previous) > 1e-30 && isfinite_d(next)) ? next / previous : 0.0;
    scalars[0] = next;
}
