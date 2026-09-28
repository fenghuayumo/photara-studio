#include "common.hlsli"

// Projects each Gaussian's three log scales into a band of
// maximum_log_ratio around their midpoint, matching the CUDA
// constrain_scale_ratio_kernel: only rows whose longest/shortest ratio
// exceeds the bound are touched, and the midpoint of the (min, max) pair
// is preserved.
struct PushConstants {
    uint count;
    float maximum_log_ratio;
};

[[vk::binding(0, 0)]] RWByteAddressBuffer log_scales;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint i = dtid.x;
    if (i >= pc.count) return;
    float3 v = float3(
        load_f32(log_scales, (3u*i)*4u),
        load_f32(log_scales, (3u*i+1u)*4u),
        load_f32(log_scales, (3u*i+2u)*4u));
    float minimum = min(v.x, min(v.y, v.z));
    float maximum = max(v.x, max(v.y, v.z));
    if (!(maximum - minimum > pc.maximum_log_ratio)) return;
    float midpoint = 0.5f * (minimum + maximum);
    float half_range = 0.5f * pc.maximum_log_ratio;
    float3 clamped = clamp(v, midpoint - half_range, midpoint + half_range);
    store_f32(log_scales, (3u*i)*4u, clamped.x);
    store_f32(log_scales, (3u*i+1u)*4u, clamped.y);
    store_f32(log_scales, (3u*i+2u)*4u, clamped.z);
}
