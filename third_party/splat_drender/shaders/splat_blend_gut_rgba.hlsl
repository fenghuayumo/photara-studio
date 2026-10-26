#include "splat_math.hlsli"

// GPU RGBA viewport variant: no geometry attachments or training statistics.

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] RWStructuredBuffer<uint> rgba;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_u;
// Same bindings as the existing forward blender; no backward state is used.
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> snap;

groupshared float4 rows0[64];
groupshared float4 rows1[64];
groupshared float4 rows2[64];
groupshared float4 colors[64];
groupshared uint alive;
groupshared float3 planes[4];
groupshared uint plane_count;

// A camera-origin halfspace containing every ray in this tile. Transform its
// normal by A^-T (up to a positive determinant scale) and test the Gaussian's
// finite alpha support sphere. This never replaces the per-pixel ray response.
bool outside_plane(float3 n, float3 c0, float3 c1, float3 c2,
                   float sign_det, float3 origin, float support2) {
    float3 local = sign_det*float3(dot(c0,n), dot(c1,n), dot(c2,n));
    float center = -dot(local,origin);
    // Expand for cancellation in the plane transform and dot product.
    center += 1e-3f*dot(abs(local),abs(origin));
    return center < 0.0f && center*center > support2*dot(local,local)*1.0001f;
}

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    uint width = pc.u0, height = pc.u1, pixels = pc.u18;
    uint2 xy = group.xy*16u + thread.xy;
    bool inside = xy.x < width && xy.y < height;
    bool rings = (pc.u3 & 64u) != 0u;
    uint mode = pc.u4;
    float fx = asfloat(pc.u6), fy = asfloat(pc.u7), cx = asfloat(pc.u8), cy = asfloat(pc.u9);
    float3 direction = mode == kModeOrtho ? float3(0.0f, 0.0f, 1.0f)
        : pixel_unit_ray(float(xy.x), float(xy.y), mode, fx, fy, cx, cy,
                         asfloat(pc.u10), asfloat(pc.u11), asfloat(pc.u12), asfloat(pc.u13));
    float3 offset = mode == kModeOrtho ? float3((float(xy.x)-cx)/fx, (float(xy.y)-cy)/fy, 0.0f) : 0.0f;
    uint tile = group.y*pc.u2 + group.x;
    uint begin = ranges[2u*tile], end = ranges[2u*tile+1u];
    float T = 1.0f;
    float3 rgb = 0.0f;
    bool done = !inside;
    if (rank == 0u) {
        plane_count = 0u;
        float2 lo = float2(group.xy*16u);
        float2 hi = min(lo+15.0f, float2(width-1u,height-1u));
        if (mode == kModePinhole && fx > 0.0f && fy > 0.0f) {
            planes[0] = float3(1,0,-(lo.x-cx)/fx);
            planes[1] = float3(-1,0,(hi.x-cx)/fx);
            planes[2] = float3(0,1,-(lo.y-cy)/fy);
            planes[3] = float3(0,-1,(hi.y-cy)/fy);
            plane_count = 4u;
        } else if (mode == kModeEquirect && fx > 0.0f && fy > 0.0f &&
                   (hi.x-lo.x)/fx < 3.1415926f &&
                   max(abs(-cy/fy),abs((float(height)-1.0f-cy)/fy)) <= 1.5707964f) {
            // Meridian planes are exact for the longitude interval, including
            // tiles at the panorama seam. No latitude-plane approximation.
            float left = (lo.x-cx)/fx, right = (hi.x-cx)/fx;
            planes[0] = float3(cos(left),0,-sin(left));
            planes[1] = float3(-cos(right),0,sin(right));
            plane_count = 2u;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    [loop] for (uint base = begin; base < end; base += 64u) {
        if (rank == 0u) alive = 0u;
        GroupMemoryBarrierWithGroupSync();
        // One shared atomic per live subgroup, instead of per live pixel.
        bool wave_alive = WaveActiveAnyTrue(!done);
        if (WaveIsFirstLane() && wave_alive) InterlockedAdd(alive, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (alive == 0u) break;
        uint batch = min(64u, end-base);
        if (rank < batch) {
            uint id = instances[base+rank];
            uint g = id*((pc.u3 & 512u)!=0u ? 11u : 7u);
            colors[rank] = gauss_f[g+2u];
            rows0[rank] = gauss_f[g+3u]; rows1[rank] = gauss_f[g+4u]; rows2[rank] = gauss_f[g+5u];
            float3 a = rows0[rank].xyz, b = rows1[rank].xyz, c = rows2[rank].xyz;
            float3 lengths = float3(dot(a,a),dot(b,b),dot(c,c));
            // Ill-conditioned transforms use the original pixel evaluation;
            // their floating-point response is not a reliable support oracle.
            if (plane_count != 0u && max(lengths.x,max(lengths.y,lengths.z)) <
                1e8f*min(lengths.x,min(lengths.y,lengths.z))) {
                float3 c0 = cross(b,c), c1 = cross(c,a), c2 = cross(a,b);
                float sign_det = dot(a,c0) >= 0.0f ? 1.0f : -1.0f;
                float3 o = float3(rows0[rank].w,rows1[rank].w,rows2[rank].w);
                float sigma = max(0.5f,asfloat(pc.u19));
                float support2 = rings ? sigma*sigma : -2.0f*log(kAlphaFloor/colors[rank].w);
                support2 = max(0.0f,support2)+1e-3f;
                [loop] for (uint plane = 0u; plane < plane_count; ++plane) {
                    if (outside_plane(planes[plane],c0,c1,c2,sign_det,o,support2)) {
                        colors[rank].w = 0.0f;
                        break;
                    }
                }
            }
        }
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint j = 0u; j < batch && !done; ++j) {
            if (colors[j].w == 0.0f) continue;
            float4 a = rows0[j], b = rows1[j], c = rows2[j];
            float3 o = float3(a.w, b.w, c.w);
            // Perspective/fisheye/panorama rays share one origin. Avoid three
            // redundant dot products per candidate outside orthographic mode.
            if (mode == kModeOrtho)
                o += float3(dot(a.xyz, offset), dot(b.xyz, offset), dot(c.xyz, offset));
            float3 d = float3(dot(a.xyz, direction), dot(b.xyz, direction), dot(c.xyz, direction));
            float dd = dot(d, d);
            float peak = -dot(o, d)/dd;
            if (!(dd > 0.0f) || !(peak > 0.0f)) continue;
            float3 closest = o + peak*d;
            float distance = dot(closest, closest);
            float alpha = min(kAlphaClip, colors[j].w*exp(-0.5f*distance));
            if (rings) {
                float sigma = max(0.5f, asfloat(pc.u19));
                float radius = distance/(sigma*sigma);
                if (radius > 1.0f) continue;
                float norm = (exp(-4.0f*radius)-exp(-4.0f))/(1.0f-exp(-4.0f));
                alpha = radius >= 0.96f ? 0.6f : max(0.05f, norm*colors[j].w);
            }
            if (alpha < kAlphaFloor) continue;
            float next = T*(1.0f-alpha);
            if (next < kTransmittanceFloor) { done = true; continue; }
            float weight = alpha*T;
            rgb += weight*colors[j].xyz;
            T = next;
        }
        // Do not overwrite shared Gaussian rows until all pixels finish.
        GroupMemoryBarrierWithGroupSync();
    }
    if (inside) {
        uint pix = xy.y*width+xy.x;
        uint3 packed = uint3(saturate(rgb + T*float3(asfloat(pc.u14), asfloat(pc.u15), asfloat(pc.u16)))*255.0f + 0.5f);
        rgba[pix] = packed.x | (packed.y << 8) | (packed.z << 16) | (255u << 24);
    }
}
