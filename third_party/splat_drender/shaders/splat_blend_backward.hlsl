#include "splat_math.hlsli"

#ifdef SPLAT_BLEND_BACKWARD_NO_GEOMETRY
#define SPLAT_BLEND_BACKWARD_GEOMETRY 0
#else
#define SPLAT_BLEND_BACKWARD_GEOMETRY 1
#endif

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] StructuredBuffer<uint> out_u;
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] StructuredBuffer<float4> snap;
[[vk::binding(7, 0)]] StructuredBuffer<float> loss_color;
[[vk::binding(8, 0)]] StructuredBuffer<float> loss_alpha;
[[vk::binding(9, 0)]] StructuredBuffer<float> loss_normal;
[[vk::binding(10, 0)]] StructuredBuffer<float2> median_state;
// IEEE-754 bits are used so Vulkan 1.2 does not require float atomics.
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> grad_values;

void atomic_add_f32(uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    uint expected = grad_values[index];
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint observed;
        InterlockedCompareExchange(grad_values[index], expected, desired, observed);
        if (observed == expected) return;
        expected = observed;
    }
}

void commit_gradient(uint gaussian, uint count, float3 acc_mean,
                     float4 acc_conic, float3 acc_color, float4 acc_plane,
                     float3 acc_normal, float acc_refine) {
    uint mean_base = gaussian * 3u;
    uint conic_base = count * 3u + gaussian * 4u;
    uint color_base = count * 7u + gaussian * 3u;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
    uint plane_base = count * 10u + gaussian * 4u;
    uint normal_base = count * 14u + gaussian * 3u;
#endif
    uint refine_base = count * 17u + gaussian;
    atomic_add_f32(mean_base, acc_mean.x);
    atomic_add_f32(mean_base + 1u, acc_mean.y);
    atomic_add_f32(mean_base + 2u, acc_mean.z);
    [unroll] for (uint i = 0u; i < 4u; ++i) atomic_add_f32(conic_base + i, acc_conic[i]);
    [unroll] for (uint c = 0u; c < 3u; ++c) atomic_add_f32(color_base + c, acc_color[c]);
#if SPLAT_BLEND_BACKWARD_GEOMETRY
    [unroll] for (uint p = 0u; p < 4u; ++p) atomic_add_f32(plane_base + p, acc_plane[p]);
    [unroll] for (uint n = 0u; n < 3u; ++n) atomic_add_f32(normal_base + n, acc_normal[n]);
#endif
    atomic_add_f32(refine_base, acc_refine);
}

// One logical bucket lane owns one Gaussian and replays at most the preceding
// 31 entries for each pixel. This portable form follows the CUDA state-after
// equations exactly and works for any Vulkan subgroup width.
[numthreads(256, 1, 1)]
void main(uint3 group_id : SV_GroupID, uint group_thread : SV_GroupIndex) {
    uint lane = group_thread & 31u;
    uint bucket_idx = group_id.x * 8u + (group_thread >> 5u);
    uint bucket_limit = pc.u8;
    if (bucket_idx >= bucket_limit) return;

    uint width = pc.u0, height = pc.u1, tiles_x = pc.u2, mode = pc.u3;
    int wrap_width = int(pc.u4);
    uint count = pc.u5, pixel_count = pc.u6, tile_count = pc.u7;
    float3 background = float3(asfloat(pc.u9), asfloat(pc.u10), asfloat(pc.u11));
#if SPLAT_BLEND_BACKWARD_GEOMETRY
    bool geometry = pc.u12 != 0u;
#endif

    uint tile_id = out_u[count + pixel_count + tile_count + bucket_idx];
    if (tile_id >= tile_count) return;
    uint range_begin = ranges[tile_id * 2u], range_end = ranges[tile_id * 2u + 1u];
    uint tile_first = tile_id == 0u ? 0u : bucket_offset[tile_id - 1u];
    if (bucket_idx < tile_first) return;
    uint tile_bucket = bucket_idx - tile_first;
    uint tile_max = out_u[count + pixel_count + tile_id];
    if (tile_bucket * 32u >= tile_max) return;
    uint tile_n = range_end - range_begin;
    uint pos = tile_bucket * 32u + lane;
    if (pos >= tile_n) return;

    uint gaussian = instances[range_begin + pos];
    float2 mean = gauss_f[gaussian * 8u].xy;
    float4 conic = gauss_f[gaussian * 8u + 1u];
    float3 color = gauss_f[gaussian * 8u + 2u].xyz;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
    float4 ray_plane = geometry ? gauss_f[gaussian * 8u + 3u] : 0.0f;
    float3 gaussian_normal = geometry ? gauss_f[gaussian * 8u + 4u].xyz : 0.0f;
#endif
    uint4 bounds = asuint(gauss_f[gaussian * 8u + 5u]);
    uint pix_min_x = (tile_id % tiles_x) * 16u;
    uint pix_min_y = (tile_id / tiles_x) * 16u;

    float3 acc_color = 0.0f, acc_mean = 0.0f;
    float4 acc_conic = 0.0f;
    float acc_refine = 0.0f;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
    float3 acc_normal = 0.0f;
    float4 acc_plane = 0.0f;
    uint normal_snap_base = bucket_limit * 256u;
#else
    float3 acc_normal = 0.0f;
    float4 acc_plane = 0.0f;
#endif
    [loop] for (uint local = 0u; local < 256u; ++local) {
        uint px = pix_min_x + (local & 15u), py = pix_min_y + (local >> 4u);
        if (px >= width || py >= height) continue;
        uint pixel = py * width + px;
        uint last = out_u[count + pixel];
        if (pos >= last) continue;
        if (wrap_width == 0 && (px < bounds.x || px >= bounds.y || py < bounds.z || py >= bounds.w)) continue;

        uint snap_index = bucket_idx * 256u + local;
        float4 state = snap[snap_index];
        float3 color_after = float3(out_f[8u * pixel_count + pixel],
            out_f[9u * pixel_count + pixel], out_f[10u * pixel_count + pixel]) - state.xyz;
        float alpha_final = out_f[3u * pixel_count + pixel], final_t = 1.0f - alpha_final;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
        float3 normal_out = geometry ? float3(out_f[4u * pixel_count + pixel],
            out_f[5u * pixel_count + pixel], out_f[6u * pixel_count + pixel]) : 0.0f;
        float3 normal_after = geometry
            ? normal_out * alpha_final - snap[normal_snap_base + snap_index].xyz : 0.0f;
#endif
        float transmittance = state.w, G = 0.0f, alpha = 0.0f;
        float2 delta = 0.0f;
        bool accepted = false;

        [loop] for (uint k = 0u; k <= lane; ++k) {
            uint qpos = tile_bucket * 32u + k;
            if (qpos >= tile_n || qpos >= last) break;
            uint q = instances[range_begin + qpos];
            float2 qmean = gauss_f[q * 8u].xy;
            float4 qconic = gauss_f[q * 8u + 1u];
            float2 qdelta = float2(wrap_dx(qmean.x - float(px), wrap_width, mode), qmean.y - float(py));
            float power = gaussian_power(qconic, qdelta.x, qdelta.y);
            if (power > 0.0f) continue;
            float qG = exp(power), qalpha = min(kAlphaClip, qconic.w * qG);
            if (qalpha < kAlphaFloor) continue;
            float weight = qalpha * transmittance;
            color_after -= weight * gauss_f[q * 8u + 2u].xyz;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
            if (geometry) normal_after -= weight * gauss_f[q * 8u + 4u].xyz;
#endif
            if (k == lane) { G = qG; alpha = qalpha; delta = qdelta; accepted = true; break; }
            transmittance *= 1.0f - qalpha;
        }
        if (!accepted) continue;

        float3 pixel_grad = float3(loss_color[pixel], loss_color[pixel_count + pixel],
                                   loss_color[2u * pixel_count + pixel]);
        float d_final_t_render = dot(background, pixel_grad) -
            (pc.u13 != 0u ? loss_alpha[pixel] : 0.0f);
        float d_final_t = d_final_t_render;
        float3 normal_grad = 0.0f;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
        if (geometry && alpha_final > 0.0f) {
            normal_grad = float3(loss_normal[pixel], loss_normal[pixel_count + pixel],
                                 loss_normal[2u * pixel_count + pixel]) / alpha_final;
            d_final_t += dot(normal_grad, normal_out);
        }
#endif
        float inv_1ma = 1.0f / (1.0f - alpha);
        float d_opacity_render = transmittance * dot(color, pixel_grad) - dot(color_after, pixel_grad) * inv_1ma;
        float d_opacity = d_opacity_render, d_peak = 0.0f;
#if SPLAT_BLEND_BACKWARD_GEOMETRY
        if (geometry) {
            d_opacity += transmittance * dot(gaussian_normal, normal_grad) - dot(normal_after, normal_grad) * inv_1ma;
            acc_normal += alpha * transmittance * normal_grad;
            float2 ms = median_state[pixel];
            float peak = ray_plane.x * delta.x + ray_plane.y * delta.y + ray_plane.z;
            float td = (ms.x - peak) * ray_plane.w;
            float ge = exp(-0.5f * td * td), gt = alpha * ge;
            float d_gt = ms.y * 0.25f / (1.0f - gt);
            d_gt = ms.x > peak ? d_gt : -d_gt;
            d_gt = ray_plane.w > 0.0f ? d_gt : 0.0f;
            d_opacity += d_gt * ge - ms.y * (td > 0.0f ? 0.5f * inv_1ma : 0.0f);
            float d_delta = -d_gt * gt * td;
            acc_plane.w += d_delta * (ms.x - peak);
            d_peak = -d_delta * ray_plane.w;
            acc_plane.xyz += float3(d_peak * delta.x, d_peak * delta.y, d_peak);
        }
#endif
        d_opacity_render -= final_t * inv_1ma * d_final_t_render;
        d_opacity -= final_t * inv_1ma * d_final_t;
        if (mode == kModeFisheye && conic.w * G >= kAlphaClip) { d_opacity = 0.0f; d_opacity_render = 0.0f; }

        float blend_weight = alpha * transmittance;
        acc_color += blend_weight * pixel_grad;
        float dG = conic.w * d_opacity;
        float dG_render = conic.w * d_opacity_render;
        float gdx = G * delta.x, gdy = G * delta.y;
        float d_del_x = dG * (-gdx * conic.x - gdy * conic.y);
        float d_del_y = dG * (-gdy * conic.z - gdx * conic.y);
#if SPLAT_BLEND_BACKWARD_GEOMETRY
        d_del_x += d_peak * ray_plane.x;
        d_del_y += d_peak * ray_plane.y;
#endif
        acc_mean += float3(d_del_x, d_del_y, abs(d_del_x) + abs(d_del_y));
        float refine_x = dG_render * (-gdx * conic.x - gdy * conic.y) * float(width);
        float refine_y = dG_render * (-gdy * conic.z - gdx * conic.y) * float(height);
        acc_refine += length(float2(refine_x, refine_y)) /
                      max(1.0f - final_t, 1.0e-5f);
        acc_conic += float4(-0.5f * gdx * delta.x * dG, -0.5f * gdx * delta.y * dG,
                            -0.5f * gdy * delta.y * dG, G * d_opacity);
    }
    commit_gradient(gaussian, count, acc_mean, acc_conic, acc_color, acc_plane,
                    acc_normal, acc_refine);
}
