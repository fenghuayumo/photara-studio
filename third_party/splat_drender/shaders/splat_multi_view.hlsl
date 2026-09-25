#include "splat_math.hlsli"

StructuredBuffer<float> reference_depth : register(t0);
StructuredBuffer<float> reference_normal : register(t1);
StructuredBuffer<float> reference_gray : register(t2);
StructuredBuffer<float> sampled_points : register(t3);
StructuredBuffer<uint> sampled_inside : register(t4);
StructuredBuffer<float> neighbour_gray : register(t5);
StructuredBuffer<float> transform_buffer : register(t6);
StructuredBuffer<float> reference_mask : register(t7);
StructuredBuffer<float> neighbour_mask : register(t8);
RWByteAddressBuffer output_buffer : register(u9);

float push_float(uint bits) { return asfloat(bits); }

void store_float(uint index, float value) {
    output_buffer.Store(4u * index, asuint(value));
}

float load_float(uint index) {
    return asfloat(output_buffer.Load(4u * index));
}

void atomic_add_float(uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    const uint address = 4u * index;
    uint expected = output_buffer.Load(address);
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        uint observed;
        output_buffer.InterlockedCompareExchange(address, expected, desired, observed);
        if (observed == expected) return;
        expected = observed;
    }
}

struct GraySample {
    float value;
    float du;
    float dv;
};

GraySample sample_gray_gradient(float u, float v, uint width, uint height) {
    float floor_u = floor(u);
    float floor_v = floor(v);
    int x0 = clamp((int)floor_u, 0, (int)width - 1);
    int y0 = clamp((int)floor_v, 0, (int)height - 1);
    int x1 = min(x0 + 1, (int)width - 1);
    int y1 = min(y0 + 1, (int)height - 1);
    float tx = u - floor_u;
    float ty = v - floor_v;
    float c00 = neighbour_gray[(uint)y0 * width + (uint)x0];
    float c01 = neighbour_gray[(uint)y0 * width + (uint)x1];
    float c10 = neighbour_gray[(uint)y1 * width + (uint)x0];
    float c11 = neighbour_gray[(uint)y1 * width + (uint)x1];
    GraySample result;
    result.value = lerp(lerp(c00, c01, tx), lerp(c10, c11, tx), ty);
    result.du = lerp(c01 - c00, c11 - c10, ty);
    result.dv = lerp(c10 - c00, c11 - c01, tx);
    return result;
}

float sample_reference(float u, float v, uint width, uint height) {
    float floor_u = floor(u);
    float floor_v = floor(v);
    int x0 = clamp((int)floor_u, 0, (int)width - 1);
    int y0 = clamp((int)floor_v, 0, (int)height - 1);
    int x1 = min(x0 + 1, (int)width - 1);
    int y1 = min(y0 + 1, (int)height - 1);
    float tx = u - floor_u;
    float ty = v - floor_v;
    float c00 = reference_gray[(uint)y0 * width + (uint)x0];
    float c01 = reference_gray[(uint)y0 * width + (uint)x1];
    float c10 = reference_gray[(uint)y1 * width + (uint)x0];
    float c11 = reference_gray[(uint)y1 * width + (uint)x1];
    return lerp(lerp(c00, c01, tx), lerp(c10, c11, tx), ty);
}

bool plane_warp_ncc(
    float depth, float3 input_normal, int center_x, int center_y,
    uint reference_width, uint reference_height,
    uint neighbour_width, uint neighbour_height,
    float reference_fx, float reference_fy, float reference_cx, float reference_cy,
    float neighbour_fx, float neighbour_fy, float neighbour_cx, float neighbour_cy,
    out float ncc, out float grad_depth, out float3 grad_normal) {
    static const int radius = 3;
    static const float radius_scaled = 1.5f;
    static const float inverse_samples = 1.0f / 49.0f;
    ncc = 0.0f;
    grad_depth = 0.0f;
    grad_normal = 0.0f;
    if ((float)center_x - radius_scaled <= 0.0f ||
        (float)center_x + radius_scaled >= (float)reference_width - 1.0f ||
        (float)center_y - radius_scaled <= 0.0f ||
        (float)center_y + radius_scaled >= (float)reference_height - 1.0f)
        return false;
    float normal_length = length(input_normal);
    if (!(normal_length > 1.0e-8f)) return false;
    float3 normal = input_normal / normal_length;
    float qcx = ((float)center_x - reference_cx) / reference_fx;
    float qcy = ((float)center_y - reference_cy) / reference_fy;
    float distance = -(qcx * normal.x + qcy * normal.y + normal.z) * depth;
    if (!(abs(distance) > 1.0e-7f)) return false;

    float h[9];
    [unroll] for (int row = 0; row < 3; ++row) {
        float translation = transform_buffer[9 + row];
        h[3 * row] = transform_buffer[3 * row] - translation * normal.x / distance;
        h[3 * row + 1] = transform_buffer[3 * row + 1] - translation * normal.y / distance;
        h[3 * row + 2] = transform_buffer[3 * row + 2] - translation * normal.z / distance;
    }
    float sum_r = 0.0f, sum_n = 0.0f, sum_r2 = 0.0f, sum_n2 = 0.0f, sum_rn = 0.0f;
    float3 derivative_sum = 0.0f;
    float3 derivative_sum2 = 0.0f;
    float3 derivative_cross = 0.0f;
    float3 aux = float3(transform_buffer[9], transform_buffer[10], transform_buffer[11]) / distance;
    [loop] for (int dv = -radius; dv <= radius; ++dv) {
        float vr = (float)center_y + 0.5f * (float)dv;
        [loop] for (int du = -radius; du <= radius; ++du) {
            float ur = (float)center_x + 0.5f * (float)du;
            float qx = (ur - reference_cx) / reference_fx;
            float qy = (vr - reference_cy) / reference_fy;
            float hx = h[0] * qx + h[1] * qy + h[2];
            float hy = h[3] * qx + h[4] * qy + h[5];
            float hz = h[6] * qx + h[7] * qy + h[8];
            if (!(hz > 1.0e-7f)) return false;
            float un = neighbour_fx * hx / hz + neighbour_cx;
            float vn = neighbour_fy * hy / hz + neighbour_cy;
            if (!(un - radius_scaled > 0.0f &&
                  un + radius_scaled < (float)neighbour_width - 1.0f &&
                  vn - radius_scaled > 0.0f &&
                  vn + radius_scaled < (float)neighbour_height - 1.0f))
                return false;
            float cr = sample_reference(ur, vr, reference_width, reference_height);
            GraySample neighbour_sample = sample_gray_gradient(
                un, vn, neighbour_width, neighbour_height);
            float cn = neighbour_sample.value;
            float dc_dhx = neighbour_sample.du * neighbour_fx / hz;
            float dc_dhy = neighbour_sample.dv * neighbour_fy / hz;
            float dc_dhz = -(neighbour_sample.du * (un - neighbour_cx) +
                              neighbour_sample.dv * (vn - neighbour_cy)) / hz;
            float factor = dc_dhx * aux.x + dc_dhy * aux.y + dc_dhz * aux.z;
            float3 derivative = float3(qx, qy, 1.0f) * factor;
            derivative_sum += derivative;
            derivative_sum2 += 2.0f * cn * derivative;
            derivative_cross += cr * derivative;
            sum_r += cr;
            sum_n += cn;
            sum_r2 += cr * cr;
            sum_n2 += cn * cn;
            sum_rn += cr * cn;
        }
    }
    float cross_value = sum_rn - sum_r * sum_n * inverse_samples;
    float variance_r = sum_r2 - sum_r * sum_r * inverse_samples;
    float variance_n = sum_n2 - sum_n * sum_n * inverse_samples;
    if (!(variance_r > 5.0e-6f && variance_n > 5.0e-6f)) return false;
    float denominator = variance_r * variance_n + 1.0e-8f;
    ncc = cross_value * cross_value / denominator;
    float grad_cross = 2.0f * cross_value / denominator;
    float grad_variance_n = -ncc / (variance_n + 1.0e-8f);
    float coefficient_sum =
        (-grad_cross * sum_r - 2.0f * grad_variance_n * sum_n) * inverse_samples;
    float3 derivative = coefficient_sum * derivative_sum +
                        grad_variance_n * derivative_sum2 +
                        grad_cross * derivative_cross;
    grad_normal = -derivative;
    float grad_distance = dot(derivative, normal) / distance;
    grad_normal -= depth * grad_distance * float3(qcx, qcy, 1.0f);
    grad_depth = -(qcx * normal.x + qcy * normal.y + normal.z) * grad_distance;
    grad_normal = (grad_normal - normal * dot(normal, grad_normal)) / normal_length;
    return !isnan(ncc) && !isinf(ncc) &&
           !isnan(grad_depth) && !isinf(grad_depth) &&
           all(!isnan(grad_normal)) && all(!isinf(grad_normal));
}

[numthreads(16, 16, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint reference_width = pc.u0;
    uint reference_height = pc.u1;
    uint neighbour_width = pc.u2;
    uint neighbour_height = pc.u3;
    uint x = dispatch_id.x;
    uint y = dispatch_id.y;
    if (x >= reference_width || y >= reference_height) return;
    uint pixels = reference_width * reference_height;
    uint pixel = y * reference_width + x;
    uint normal_base = pixels;
    uint sampled_base = 4u * pixels;
    uint terms_base = 7u * pixels;

    if ((pc.u13 & 0x80000000u) != 0u) {
        float geometry_scale = load_float(terms_base + 1u) > 0.0f
            ? push_float(pc.u14) / load_float(terms_base + 1u) : 0.0f;
        float ncc_scale = load_float(terms_base + 3u) > 0.0f
            ? push_float(pc.u15) / load_float(terms_base + 3u) : 0.0f;
        store_float(pixel, ncc_scale * load_float(pixel));
        [unroll] for (uint axis = 0u; axis < 3u; ++axis) {
            uint normal_index = normal_base + axis * pixels + pixel;
            uint sampled_index = sampled_base + 3u * pixel + axis;
            store_float(normal_index, ncc_scale * load_float(normal_index));
            store_float(sampled_index, geometry_scale * load_float(sampled_index));
        }
        return;
    }

    float depth = reference_depth[pixel];
    bool reference_has_mask = (pc.u13 & 1u) != 0u;
    bool neighbour_has_mask = (pc.u13 & 2u) != 0u;
    if (sampled_inside[pixel] == 0u || !(depth > 0.0f) ||
        (reference_has_mask && reference_mask[pixel] <= 0.5f))
        return;

    float reference_fx = push_float(pc.u4);
    float reference_fy = push_float(pc.u5);
    float reference_cx = push_float(pc.u6);
    float reference_cy = push_float(pc.u7);
    float neighbour_fx = push_float(pc.u8);
    float neighbour_fy = push_float(pc.u9);
    float neighbour_cx = push_float(pc.u10);
    float neighbour_cy = push_float(pc.u11);
    if (neighbour_has_mask) {
        float3 query = float3(
            ((float)x - reference_cx) / reference_fx * depth,
            ((float)y - reference_cy) / reference_fy * depth,
            depth);
        float3 neighbour_query;
        neighbour_query.x = dot(float3(transform_buffer[0], transform_buffer[1], transform_buffer[2]), query) + transform_buffer[9];
        neighbour_query.y = dot(float3(transform_buffer[3], transform_buffer[4], transform_buffer[5]), query) + transform_buffer[10];
        neighbour_query.z = dot(float3(transform_buffer[6], transform_buffer[7], transform_buffer[8]), query) + transform_buffer[11];
        if (!(neighbour_query.z > 0.2f)) return;
        int neighbour_x = (int)round(neighbour_fx * neighbour_query.x / neighbour_query.z + neighbour_cx);
        int neighbour_y = (int)round(neighbour_fy * neighbour_query.y / neighbour_query.z + neighbour_cy);
        if (neighbour_x < 0 || neighbour_y < 0 ||
            neighbour_x >= (int)neighbour_width || neighbour_y >= (int)neighbour_height ||
            neighbour_mask[(uint)neighbour_y * neighbour_width + (uint)neighbour_x] <= 0.5f)
            return;
    }

    float3 sampled = float3(
        sampled_points[3u * pixel], sampled_points[3u * pixel + 1u],
        sampled_points[3u * pixel + 2u]);
    if (!(sampled.z > 0.2f)) return;
    float3 delta = sampled - float3(
        transform_buffer[9], transform_buffer[10], transform_buffer[11]);
    float3 round_trip;
    round_trip.x = transform_buffer[0] * delta.x + transform_buffer[3] * delta.y + transform_buffer[6] * delta.z;
    round_trip.y = transform_buffer[1] * delta.x + transform_buffer[4] * delta.y + transform_buffer[7] * delta.z;
    round_trip.z = transform_buffer[2] * delta.x + transform_buffer[5] * delta.y + transform_buffer[8] * delta.z;
    if (!(round_trip.z > 0.2f)) return;
    atomic_add_float(terms_base + 4u, 1.0f);
    float projected_x = reference_fx * round_trip.x / round_trip.z + reference_cx;
    float projected_y = reference_fy * round_trip.y / round_trip.z + reference_cy;
    float du = projected_x - (float)x;
    float dv = projected_y - (float)y;
    float noise = sqrt(du * du + dv * dv + 1.0e-12f);
    if (!(noise < push_float(pc.u12))) return;
    float geometry_confidence = exp(-noise);
    float inverse_noise = 1.0f / noise;
    float3 grad_round_trip = float3(
        du * inverse_noise * reference_fx / round_trip.z,
        dv * inverse_noise * reference_fy / round_trip.z,
        -(du * reference_fx * round_trip.x + dv * reference_fy * round_trip.y) *
            inverse_noise / (round_trip.z * round_trip.z));
    float3 grad_sampled = geometry_confidence * float3(
        transform_buffer[0] * grad_round_trip.x + transform_buffer[1] * grad_round_trip.y + transform_buffer[2] * grad_round_trip.z,
        transform_buffer[3] * grad_round_trip.x + transform_buffer[4] * grad_round_trip.y + transform_buffer[5] * grad_round_trip.z,
        transform_buffer[6] * grad_round_trip.x + transform_buffer[7] * grad_round_trip.y + transform_buffer[8] * grad_round_trip.z);
    store_float(sampled_base + 3u * pixel, grad_sampled.x);
    store_float(sampled_base + 3u * pixel + 1u, grad_sampled.y);
    store_float(sampled_base + 3u * pixel + 2u, grad_sampled.z);
    atomic_add_float(terms_base, geometry_confidence * noise);
    atomic_add_float(terms_base + 1u, 1.0f);

    if ((pc.u13 & 8u) == 0u) return;
    float3 normal = float3(
        reference_normal[pixel], reference_normal[pixels + pixel],
        reference_normal[2u * pixels + pixel]);
    float ncc, ncc_grad_depth;
    float3 ncc_grad_normal;
    if (!plane_warp_ncc(
            depth, normal, (int)x, (int)y, reference_width, reference_height,
            neighbour_width, neighbour_height, reference_fx, reference_fy,
            reference_cx, reference_cy, neighbour_fx, neighbour_fy,
            neighbour_cx, neighbour_cy, ncc, ncc_grad_depth, ncc_grad_normal))
        return;
    float error = clamp(1.0f - ncc, 0.0f, 2.0f);
    bool robust_ncc = (pc.u13 & 4u) != 0u;
    if (!robust_ncc && error >= 0.9f) return;
    float confidence = robust_ncc
        ? 1.0f / (1.0f + exp(-(push_float(pc.u16) - error) * push_float(pc.u17)))
        : 1.0f;
    float minimum_weight = push_float(pc.u18);
    confidence = confidence * (1.0f - minimum_weight) + minimum_weight;
    float factor = -geometry_confidence * confidence;
    store_float(pixel, factor * ncc_grad_depth);
    store_float(normal_base + pixel, factor * ncc_grad_normal.x);
    store_float(normal_base + pixels + pixel, factor * ncc_grad_normal.y);
    store_float(normal_base + 2u * pixels + pixel, factor * ncc_grad_normal.z);
    atomic_add_float(terms_base + 2u, geometry_confidence * confidence * error);
    atomic_add_float(terms_base + 3u, 1.0f);
}
