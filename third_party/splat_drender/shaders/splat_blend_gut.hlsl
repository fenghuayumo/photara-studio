#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> out_u;
// Same bindings as the existing forward blender; no backward state is used.
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] RWStructuredBuffer<float4> snap;

groupshared float4 rows0[256];
groupshared float4 rows1[256];
groupshared float4 rows2[256];
groupshared float4 colors[256];
groupshared float3 normals[256];
groupshared uint ids[256];
groupshared uint alive;

[numthreads(16, 16, 1)]
void main(uint3 group : SV_GroupID, uint3 thread : SV_GroupThreadID, uint rank : SV_GroupIndex) {
    uint width = pc.u0, height = pc.u1, pixels = pc.u18;
    uint2 xy = group.xy*16u + thread.xy;
    bool inside = xy.x < width && xy.y < height;
    bool geometry = (pc.u3 & 4u) != 0u;
    bool stats = (pc.u3 & 16u) != 0u;
    bool rings = (pc.u3 & 64u) != 0u;
    uint mode = pc.u4;
    float fx = asfloat(pc.u6), fy = asfloat(pc.u7), cx = asfloat(pc.u8), cy = asfloat(pc.u9);
    float3 direction = mode == kModeOrtho ? float3(0.0f, 0.0f, 1.0f)
        : pixel_unit_ray(float(xy.x), float(xy.y), mode, fx, fy, cx, cy,
                         asfloat(pc.u10), asfloat(pc.u11), asfloat(pc.u12), asfloat(pc.u13));
    float3 offset = mode == kModeOrtho ? float3((float(xy.x)-cx)/fx, (float(xy.y)-cy)/fy, 0.0f) : 0.0f;
    uint tile = group.y*pc.u2 + group.x;
    uint begin = ranges[2u*tile], end = ranges[2u*tile+1u];
    float T = 1.0f, depth = 0.0f;
    float3 rgb = 0.0f, normal = 0.0f;
    uint last = 0u;
    bool done = !inside;
    [loop] for (uint base = begin; base < end; base += 256u) {
        if (rank == 0u) alive = 0u;
        GroupMemoryBarrierWithGroupSync();
        if (!done) InterlockedAdd(alive, 1u);
        GroupMemoryBarrierWithGroupSync();
        if (alive == 0u) break;
        uint batch = min(256u, end-base);
        if (rank < batch) {
            uint id = instances[base+rank];
            ids[rank] = id;
            uint g = id*((pc.u3 & 512u)!=0u ? 11u : 7u);
            colors[rank] = gauss_f[g+2u];
            rows0[rank] = gauss_f[g+3u]; rows1[rank] = gauss_f[g+4u]; rows2[rank] = gauss_f[g+5u];
            normals[rank] = gauss_f[g+6u].xyz;
        }
        GroupMemoryBarrierWithGroupSync();
        [loop] for (uint j = 0u; j < batch && !done; ++j) {
            float4 a = rows0[j], b = rows1[j], c = rows2[j];
            float3 o = float3(a.w, b.w, c.w) + float3(dot(a.xyz, offset), dot(b.xyz, offset), dot(c.xyz, offset));
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
            if (geometry) {
                float3 n = normals[j];
                n = dot(n, direction) > 0.0f ? -n : n;
                normal += weight*n;
                if (T >= 0.5f && next < 0.5f) depth = peak*(mode == kModeEquirect ? 1.0f : direction.z);
            }
            if (stats) InterlockedOr(out_u[ids[j]], 1u);
            T = next;
            last = base+j-begin+1u;
        }
        // Do not overwrite shared Gaussian rows until all pixels finish.
        GroupMemoryBarrierWithGroupSync();
    }
    if (inside) {
        uint pix = xy.y*width+xy.x;
        out_f[pix] = rgb.x+T*asfloat(pc.u14);
        out_f[pixels+pix] = rgb.y+T*asfloat(pc.u15);
        out_f[2u*pixels+pix] = rgb.z+T*asfloat(pc.u16);
        out_f[3u*pixels+pix] = 1.0f-T;
        if (geometry) {
            normal /= max(1.0f-T, 1.0e-8f);
            out_f[4u*pixels+pix] = normal.x;
            out_f[5u*pixels+pix] = normal.y;
            out_f[6u*pixels+pix] = normal.z;
            out_f[7u*pixels+pix] = depth;
        }
        if (stats) {
            out_u[pc.u17+pix] = last;
            uint output_base = (geometry ? 8u : 4u)*pixels;
            out_f[output_base+pix] = rgb.x;
            out_f[output_base+pixels+pix] = rgb.y;
            out_f[output_base+2u*pixels+pix] = rgb.z;
        }
    }
}
