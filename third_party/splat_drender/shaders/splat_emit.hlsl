#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(1, 0)]] StructuredBuffer<uint> gauss_u;
[[vk::binding(2, 0)]] RWStructuredBuffer<uint> key_lo;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> key_hi;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> values;
[[vk::binding(5, 0)]] StructuredBuffer<uint> sorted_ids;
[[vk::binding(6, 0)]] StructuredBuffer<uint> compact;

#include "splat_enum.hlsli"

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint count = pc.u0;
    uint index = dispatch_id.x;
    if (index >= count) return;
    uint mode = pc.u1;
    int grid_x = int(pc.u2);
    int grid_y = int(pc.u3);
    int wrap_width = int(pc.u4);
    uint gaussian_count = pc.u5;
    uint gaussian_slots = pc.u6;

    // Convert the per-tile [begin, end) instance ranges into bucket counts.
    // The caller scans values in-place afterwards, producing the inclusive
    // bucket prefix consumed by the backward snapshot layout. Keeping this on
    // the device avoids a queue flush, a full range download, and an upload on
    // every training forward.
    if (mode == 5) {
        uint begin = key_lo[2u * index];
        uint end = key_lo[2u * index + 1u];
        values[index] = (end - begin + 31u) >> 5;
        return;
    }

    // Compact the two scan totals into one host-visible control buffer. The
    // CPU needs both before sizing the instance sort, and reading them
    // separately from device-local gauss_u caused two extra queue submits.
    if (mode == 4) {
        values[0] = gauss_u[5u * gaussian_count - 1u];
        values[1] = gauss_u[6u * gaussian_count - 1u];
        return;
    }

    if (mode == 1) {
        if (gauss_u[gaussian_count + index] == 0) return;
        uint pos = gauss_u[5 * gaussian_count + index] - 1;
        key_lo[pos] = gauss_u[2 * gaussian_count + index];
        values[pos] = index;
        return;
    }
    if (mode == 2) {
        values[index] = gauss_u[sorted_ids[index]];
        return;
    }

    uint gaussian = mode == 0 ? index : sorted_ids[index];
    uint touched = gauss_u[gaussian];
    if (mode == 0 && touched == 0) return;
    uint exclusive = 0;
    if (mode == 0) {
        exclusive = index == 0 ? 0 : gauss_u[4 * gaussian_count + index - 1];
    } else {
        exclusive = index == 0 ? 0 : compact[index - 1];
    }
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
    uint depth = gauss_u[2 * gaussian_count + gaussian];
    enumerate_tiles(mean_depth.xy, conic, grid_x, grid_y, gaussian, exclusive, wrap_width, depth,
                    mode == 0 ? 2u : 1u);
}
