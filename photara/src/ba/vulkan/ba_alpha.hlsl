#include "ba_common.hlsli"

[[vk::binding(0, 0)]] RWStructuredBuffer<double> scalars;
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(1, 1, 1)]
void main() {
    double denominator = scalars[1];
    scalars[4] = (denominator > 1e-30 && isfinite_d(denominator)) ? scalars[0] / denominator : 0.0;
}
