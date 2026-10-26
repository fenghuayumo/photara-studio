#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> key_lo;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> key_hi;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> values;
[[vk::binding(5, 0)]] StructuredBuffer<uint> sorted_ids;
[[vk::binding(6, 0)]] StructuredBuffer<uint> compact;
[[vk::binding(7, 0)]] StructuredBuffer<uint> control;

#include "splat_enum.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    if (control[2] != 0u) return;
    uint mode = pc.u1;
    uint gaussian_count = pc.u5;
    uint count = mode == 0u ? gaussian_count : control[1];
    uint index = dispatch_id.x;
    if (index >= count) return;
    uint gaussian_slots = pc.u6;

    // Depth-key compaction is launched over all Gaussians. The two remaining
    // modes are dispatched indirectly from the visible count.
    if (mode == 0u) {
        if (gauss_u[gaussian_count + index] == 0u) return;
        uint pos = gauss_u[5u * gaussian_count + index] - 1u;
        key_lo[pos] = gauss_u[2u * gaussian_count + index];
        values[pos] = index;
        return;
    }
    if (mode == 1u) {
        values[index] = gauss_u[sorted_ids[index]];
        return;
    }

    uint gaussian = sorted_ids[index];
    uint exclusive = index == 0u ? 0u : compact[index - 1u];
    float4 mean_depth = gauss_f[gaussian * gaussian_slots];
    float4 conic = gauss_f[gaussian * gaussian_slots + 1u];
    if (pc.u7 != 0u) {
        uint base=gaussian*gaussian_slots;
        float4 c0=gauss_f[base+7u],c1=gauss_f[base+8u],c2=gauss_f[base+9u];
        float4 info=gauss_f[base+10u];
        gut_support_enabled=info.w!=0.0f;
        gut_c0=c0.xyz;gut_c1=c1.xyz;gut_c2=c2.xyz;
        gut_origin=float3(c0.w,c1.w,c2.w);gut_support2=info.z;gut_sign=info.w;
        gut_intrinsics=float4(asfloat(pc.u10),asfloat(pc.u11),asfloat(pc.u12),asfloat(pc.u13));
        gut_extent=uint2(pc.u8,pc.u9);
        uint bx=uint(info.x),by=uint(info.y);
        gut_tile_bounds=int4(bx&4095u,by&4095u,bx>>12u,by>>12u);
    }
    uint depth = gauss_u[2u * gaussian_count + gaussian];
    enumerate_tiles(
        mean_depth.xy, conic, int(pc.u2), int(pc.u3), gaussian, exclusive,
        int(pc.u4), depth, 1u);
}
