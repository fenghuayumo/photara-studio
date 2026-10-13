#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> camera;
[[vk::binding(4, 0)]] StructuredBuffer<float> points;
[[vk::binding(5, 0)]] StructuredBuffer<float> sample_f;
[[vk::binding(6, 0)]] StructuredBuffer<uint> sample_u;
[[vk::binding(7, 0)]] StructuredBuffer<float> loss_points;
#ifdef SPLAT_SAMPLE_NATIVE_ATOMIC
[[vk::binding(8, 0)]] RWStructuredBuffer<float> blend_grad;
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_capability(6033)]]
[[vk::ext_instruction(6035)]]
float op_atomic_f_add([[vk::ext_reference]] float destination,
uint scope, uint semantics, float value);
#else
[[vk::binding(8, 0)]] RWStructuredBuffer<uint> blend_grad;
#endif
[[vk::binding(9, 0)]] RWStructuredBuffer<float> point_grad;

void atomic_add_f32(uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    #ifdef SPLAT_SAMPLE_NATIVE_ATOMIC
    op_atomic_f_add(blend_grad[index], 1u, 0u, value);
    #else
    uint expected = blend_grad[index];
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint observed;
        InterlockedCompareExchange(blend_grad[index], expected, desired, observed);
        if (observed == expected) return;
        expected = observed;
    }
    #endif
}

[[vk::binding(10, 0)]] StructuredBuffer<uint> counts;
[[vk::binding(11, 0)]] StructuredBuffer<uint> order;
groupshared float2 gs_mean[128];
groupshared float4 gs_conic[128];
groupshared float4 gs_plane[128];
groupshared uint gs_id[128];
groupshared uint gs_last;

// All lanes, including invalid/tail points, participate in shared-memory
// barriers and wave reductions. A wave commits one sum per Gaussian.
void commit(uint index, float value) {
    value = (isnan(value) || isinf(value)) ? 0.0f : value;
    float sum = WaveActiveSum(value);
    if (WaveIsFirstLane()) atomic_add_f32(index, sum);
}
void load_gaussian(uint lane, uint p, uint limit, uint begin) {
    if (p < limit) {
        uint g = instances[begin + p];
        gs_id[lane] = g;
        gs_mean[lane] = gauss_f[7u*g].xy;
        gs_conic[lane] = gauss_f[7u*g+1u];
        gs_plane[lane] = gauss_f[7u*g+3u];
    }
}

[numthreads(128, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint lane : SV_GroupIndex) {
    uint tile = group_id.y * pc.u3 + group_id.x;
    uint point_begin = tile == 0u ? 0u : counts[tile-1u];
    uint point_end = counts[tile];
    uint begin = ranges[2u*tile], end = ranges[2u*tile+1u];
    uint point_count = pc.u0, count = pc.u5;
    uint width = pc.u1, height = pc.u2, mode = pc.u4;
    float fx = asfloat(pc.u6), fy = asfloat(pc.u7);
    float cx = asfloat(pc.u8), cy = asfloat(pc.u9);
    float k1 = asfloat(pc.u10), k2 = asfloat(pc.u11);
    float k3 = asfloat(pc.u12), k4 = asfloat(pc.u13);
    for (uint batch = point_begin; batch < point_end; batch += 128u) {
        bool owned = batch + lane < point_end;
        uint i = owned ? order[batch+lane] : 0u;
        uint last = owned && sample_u[point_count+i] != 0u
        ? min(sample_u[i],end-begin) : 0u;
        float3 world = float3(points[3u*i],points[3u*i+1u],points[3u*i+2u]);
        float3 t = xform_point(world, camera[0],camera[1],camera[2],
        camera[4],camera[5],camera[6], camera[8],camera[9],camera[10],
        camera[12],camera[13],camera[14]);
        Projection pr = project_camera(t,mode,int(width),int(height),fx,fy,cx,cy,k1,k2,k3,k4);
        if (!pr.valid) last = 0u;
        if (lane == 0u) gs_last = 0u;
        GroupMemoryBarrierWithGroupSync();
        InterlockedMax(gs_last,last);
        GroupMemoryBarrierWithGroupSync();
        uint limit = gs_last;
        float median = sample_f[3u*point_count+i];
        float3 dl = float3(loss_points[3u*i],loss_points[3u*i+1u],loss_points[3u*i+2u]);
        float3 ray = pixel_unit_ray(pr.pixel_x,pr.pixel_y,mode,fx,fy,cx,cy,k1,k2,k3,k4);
        float d_depth = dot(ray,dl);
        float2 dxy = 0.0f;
        if (last != 0u && (mode == kModePinhole || mode == kModeOrtho)) {
            float2 nf = float2((pr.pixel_x-cx)/fx,(pr.pixel_y-cy)/fy);
            float rln = rsqrt(dot(nf,nf)+1.0f);
            float depth = median*rln;
            float rln2 = 1.0f/(dot(nf,nf)+1.0f);
            float aux = (dl.x*nf.x+dl.y*nf.y+dl.z)*rln2;
            dxy = float2((dl.x-aux*nf.x)*depth/fx,(dl.y-aux*nf.y)*depth/fy);
        }
        float dT = 0.0f;
        for (uint base = 0u; base < limit; base += 128u) {
            load_gaussian(lane,base+lane,limit,begin);
            GroupMemoryBarrierWithGroupSync();
            for (uint slot = 0u; slot < min(128u,limit-base); ++slot) {
                uint p = base+slot;
                if (p >= last) continue;
                float2 d = gs_mean[slot] - float2(pr.pixel_x, pr.pixel_y);
                float4 co = gs_conic[slot];
                float power = gaussian_power(co, d.x, d.y);
                if (power > 0.0f) continue;
                float alpha = min(kAlphaClip, co.w * exp(power));
                if (alpha < kAlphaFloor) continue;
                float4 rp = gs_plane[slot];
                float peak = rp.x * d.x + rp.y * d.y + rp.z;
                float td = (median - peak) * rp.w;
                float gt = alpha * exp(-0.5f * td * td);
                dT += -0.25f * gt / (1.0f - gt) * abs(td) * rp.w;
            }
            GroupMemoryBarrierWithGroupSync();
        }
        float scale = last == 0u ? 0.0f : d_depth/max(-dT,1.0e-7f);
        for (uint top = limit; top > 0u;) {
            uint base = top > 128u ? top-128u : 0u;
            load_gaussian(lane,base+lane,top,begin);
            GroupMemoryBarrierWithGroupSync();
            for (uint rev = top-base; rev > 0u; --rev) {
                uint slot = rev-1u, p = base+slot;
                uint g = gs_id[slot];
                float2 gm = 0.0f;
                float4 gc = 0.0f, gp = 0.0f;
                if (p < last) {
                    float2 d = gs_mean[slot] - float2(pr.pixel_x, pr.pixel_y);
                    float4 co = gs_conic[slot];
                    float power = gaussian_power(co, d.x, d.y);
                    if (power <= 0.0f) {
                        float G = exp(power);
                        float raw = co.w * G;
                        if (raw >= kAlphaFloor) {
                            float alpha = min(kAlphaClip, raw);
                            float4 rp = gs_plane[slot];
                            float peak = rp.x * d.x + rp.y * d.y + rp.z;
                            float td = (median - peak) * rp.w;
                            float ge = exp(-0.5f * td * td);
                            float gt = alpha * ge;
                            float dgt = scale * 0.25f / (1.0f - gt);
                            dgt = median > peak ? dgt : -dgt;
                            dgt = rp.w > 0.0f ? dgt : 0.0f;
                            float dopa = dgt * ge - scale * (td > 0.0f ? 0.5f / (1.0f - alpha) : 0.0f);
                            float ddelta = -dgt * gt * td;
                            float dpeak = -ddelta * rp.w;
                            if (mode == kModeFisheye && raw >= kAlphaClip) dopa = 0.0f;
                            float dG = co.w * dopa;
                            float gdx = G * d.x;
                            float gdy = G * d.y;
                            float ddx = dG * (-gdx * co.x - gdy * co.y) + dpeak * rp.x;
                            float ddy = dG * (-gdy * co.z - gdx * co.y) + dpeak * rp.y;
                            dxy -= float2(ddx, ddy);
                            gm = float2(ddx,ddy);
                            gc = float4(-0.5f*gdx*d.x*dG,-0.5f*gdx*d.y*dG,
                            -0.5f*gdy*d.y*dG,G*dopa);
                            gp = float4(dpeak*d.x,dpeak*d.y,dpeak,ddelta*(median-peak));
                        }
                    }
                }
                // Most tile primitives affect only a few query points. Avoid
                // ten wave reductions when this whole wave has no contribution.
                if (WaveActiveAnyTrue(any(gm != 0.0f) || any(gc != 0.0f) || any(gp != 0.0f))) {
                    commit(3u*g,gm.x); commit(3u*g+1u,gm.y);
                    for (uint c = 0u; c < 4u; ++c) {
                        commit(3u*count+4u*g+c,gc[c]);
                        commit(10u*count+4u*g+c,gp[c]);
                    }
                }
            }
            GroupMemoryBarrierWithGroupSync();
            top = base;
        }
        if (owned && last != 0u) {
            float3 gc = float3(
            pr.du0 * dxy.x + pr.dv0 * dxy.y,
            pr.du1 * dxy.x + pr.dv1 * dxy.y,
            pr.du2 * dxy.x + pr.dv2 * dxy.y);
            float3 gw = float3(
            camera[0] * gc.x + camera[1] * gc.y + camera[2] * gc.z,
            camera[4] * gc.x + camera[5] * gc.y + camera[6] * gc.z,
            camera[8] * gc.x + camera[9] * gc.y + camera[10] * gc.z);
            point_grad[3u * i] = gw.x;
            point_grad[3u * i + 1u] = gw.y;
            point_grad[3u * i + 2u] = gw.z;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
