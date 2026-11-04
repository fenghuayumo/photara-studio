// CUDA-compatible 32-Gaussian compositing pipeline. Forwarding pixel state
// preserves CUDA's compositing order instead of parallel prefix reassociation.
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint rank : SV_GroupIndex) {
    uint lane = WaveGetLaneIndex(), bucket = group.x * 8u + (rank >> 5u);
    if (bucket >= pc.u8)
        return;
    uint width = pc.u0, height = pc.u1, count = pc.u5, pixels = pc.u6;
    uint tile = out_u[count + pixels + pc.u7 + bucket];
    if (tile >= pc.u7)
        return;
    uint first = tile == 0u ? 0u : bucket_offset[tile - 1u];
    if (bucket < first)
        return;
    uint start = (bucket - first) * 32u, position = start + lane;
    if (start >= out_u[count + pixels + tile])
        return;
    uint begin = ranges[2u * tile], end = ranges[2u * tile + 1u];
    bool valid = begin + position < end;
    uint id = valid ? instances[begin + position] : 0u, base = id * pc.u19;
    float4 a = 0.0f, b = 0.0f, c = 0.0f, rgb = 0.0f, refine = 0.0f;
    if (valid) {
        a = gauss_f[base + 3u];
        b = gauss_f[base + 4u];
        c = gauss_f[base + 5u];
        rgb = gauss_f[base + 2u];
        refine = gauss_f[base + 6u];
    }
    float3 bu, bv;
    fisheye_refine_basis(refine.xyz, bu, bv);
    float3 acc_o = 0.0f, acc_a = 0.0f, acc_b = 0.0f, acc_c = 0.0f, acc_rgb = 0.0f;
    float acc_opacity = 0.0f, acc_refine = 0.0f;
    float4 staged_state = 0.0f, staged_gradient = 0.0f, staged_direction = 0.0f;
    float3 staged_offset = 0.0f;
    uint staged_limit = 0u;
    float T = 0.0f, dfinal = 0.0f, final_T = 0.0f;
    float3 remaining = 0.0f, dl = 0.0f, direction = 0.0f, offset = 0.0f;
    uint limit = 0u;
    [loop] for (uint step = 0u; step < 287u; ++step) {
        if ((step & 31u) == 0u && step < 256u) {
            uint local = step + lane;
            uint x = (tile % pc.u2) * 16u + (local & 15u), y = (tile / pc.u2) * 16u + (local >> 4u);
            uint pixel = y * width + x;
            staged_state = 0.0f;
            staged_gradient = 0.0f;
            staged_direction = float4(0, 0, 0, 1);
            staged_offset = 0.0f;
            staged_limit = (x < width && y < height) ? out_u[count + pixel] : 0u;
            if (staged_limit > start) {
                float4 prefix = snap[bucket * 256u + local];
                staged_state = float4(float3(out_f[4u * pixels + pixel], out_f[5u * pixels + pixel],
                                             out_f[6u * pixels + pixel]) -
                                          prefix.xyz,
                                      prefix.w);
                staged_gradient.xyz = float3(loss_color[pixel], loss_color[pixels + pixel],
                                             loss_color[2u * pixels + pixel]);
                staged_gradient.w =
                    asfloat(pc.u4) * staged_gradient.x + asfloat(pc.u18) * staged_gradient.y +
                    asfloat(pc.u23) * staged_gradient.z - (pc.u13 != 0u ? loss_alpha[pixel] : 0.0f);
                staged_direction =
                    float4(out_f[7u * pixels + pixel], out_f[8u * pixels + pixel],
                           out_f[9u * pixels + pixel], 1.0f - out_f[3u * pixels + pixel]);
                if (pc.u3 == kModeOrtho)
                    staged_offset = gut_world_direction(
                        float3((float(x) - asfloat(pc.u11)) / asfloat(pc.u9),
                               (float(y) - asfloat(pc.u12)) / asfloat(pc.u10), 0));
            }
        }
        if (step > 0u) {
            uint previous = lane > 0u ? lane - 1u : 0u;
            T = WaveReadLaneAt(T, previous);
            remaining = WaveReadLaneAt(remaining, previous);
            dl = WaveReadLaneAt(dl, previous);
            dfinal = WaveReadLaneAt(dfinal, previous);
            final_T = WaveReadLaneAt(final_T, previous);
            direction = WaveReadLaneAt(direction, previous);
            offset = WaveReadLaneAt(offset, previous);
            limit = WaveReadLaneAt(limit, previous);
        }
        float4 initial = WaveReadLaneAt(staged_state, step & 31u);
        float4 gradient = WaveReadLaneAt(staged_gradient, step & 31u);
        float4 ray = WaveReadLaneAt(staged_direction, step & 31u);
        float3 origin_offset = WaveReadLaneAt(staged_offset, step & 31u);
        uint last = WaveReadLaneAt(staged_limit, step & 31u);
        if (lane == 0u && step < 256u) {
            T = initial.w;
            remaining = initial.xyz;
            dl = gradient.xyz;
            dfinal = gradient.w;
            direction = ray.xyz;
            final_T = ray.w;
            offset = origin_offset;
            limit = last;
        }
        int pixel = int(step) - int(lane);
        if (valid && pixel >= 0 && pixel < 256 && position < limit) {
            float3 o = float3(a.w, b.w, c.w);
            if (pc.u3 == kModeOrtho)
                o += float3(gut_dot(a.xyz, offset), gut_dot(b.xyz, offset), gut_dot(c.xyz, offset));
            float3 d = float3(gut_dot(a.xyz, direction), gut_dot(b.xyz, direction),
                              gut_dot(c.xyz, direction));
            float dd = gut_dot(d, d), peak = gut_div(-gut_dot(o, d), dd);
            if (dd > 0.0f && peak > 0.0f) {
                float3 q = float3(gut_fma(peak, d.x, o.x), gut_fma(peak, d.y, o.y),
                                  gut_fma(peak, d.z, o.z));
                float G = gut_gaussian(q);
                float raw = rgb.w * G, alpha = min(kAlphaClip, raw);
                if (alpha >= kAlphaFloor) {
                    float weight = T * alpha;
                    remaining -= weight * rgb.xyz;
                    float da = T * (dl.x * rgb.x + dl.y * rgb.y + dl.z * rgb.z) -
                               (dl.x * remaining.x + dl.y * remaining.y + dl.z * remaining.z +
                                final_T * dfinal) /
                                   (1.0f - alpha);
                    float factor = raw < kAlphaClip ? -da * raw : 0.0f;
                    float h = gut_div(gut_mul(factor, gut_dot(q, d)), dd);
                    float3 go = factor * q - h * d, v = offset + peak * direction;
                    acc_o += go;
                    acc_a += go.x * v - h * q.x * direction;
                    acc_b += go.y * v - h * q.y * direction;
                    acc_c += go.z * v - h * q.z * direction;
                    acc_rgb += weight * dl;
                    if (raw < kAlphaClip)
                        acc_opacity += da * G;
                    float3 gm = -(a.xyz * go.x + b.xyz * go.y + c.xyz * go.z);
                    if (pc.u3 != kModeEquirect)
                        gm = gut_camera_direction(gm);
                    float score = pc.u3 == kModeEquirect
                                      ? refine.w * length(cross(refine.xyz, gm))
                                      : length(gm.xy * refine.xy * float2(width, height));
                    if (pc.u3 == kModeFisheye)
                        score = length(float2(gut_dot(bu, gm) * width, gut_dot(bv, gm) * height));
                    acc_refine += score / max(1.0f - final_T, 1.0e-6f);
                    T *= 1.0f - alpha;
                }
            }
        }
    }
    if (!valid)
        return;
    [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
        atomic_add(3u * id + axis, acc_o[axis]);
        atomic_add(3u * count + 4u * id + axis, acc_a[axis]);
        atomic_add(7u * count + 3u * id + axis, acc_rgb[axis]);
        atomic_add(10u * count + 4u * id + axis, acc_b[axis]);
        atomic_add(14u * count + 3u * id + axis, acc_c[axis]);
    }
    atomic_add(3u * count + 4u * id + 3u, acc_opacity);
    atomic_add(17u * count + id, acc_refine);
}
