// Per-pixel oriented-box ray test matching photara/src/splat/focus_mask.cu.
struct Push {
    uint width;
    uint height;
    uint model;
    float fx;
    float fy;
    float cx;
    float cy;
    float k1;
    float k2;
    float k3;
    float k4;
    float origin_x;
    float origin_y;
    float origin_z;
    float m00;
    float m01;
    float m02;
    float m10;
    float m11;
    float m12;
    float m20;
    float m21;
    float m22;
    float half_x;
    float half_y;
    float half_z;
};

[[vk::binding(0, 0)]] RWStructuredBuffer<float> mask;
[[vk::push_constant]] ConstantBuffer<Push> pc;

static const uint kModelFisheye = 1u;
static const uint kModelEquirect = 3u;
static const float kPi = 3.14159265358979323846f;

bool focus_ray_axis(inout float t_near, inout float t_far, float origin,
                    float direction, float half_extent) {
    if (abs(direction) < 1e-9f)
        return abs(origin) <= half_extent;
    float enter = (-half_extent - origin) / direction;
    float exit = (half_extent - origin) / direction;
    if (enter > exit) {
        float swap = enter;
        enter = exit;
        exit = swap;
    }
    t_near = max(t_near, enter);
    t_far = min(t_far, exit);
    return t_far >= t_near;
}

bool focus_ray_hits(float3 origin, float3 direction, float3 half_extent) {
    float t_near = 0.0f;
    float t_far = 3.402823466e+38f;
    return focus_ray_axis(t_near, t_far, origin.x, direction.x, half_extent.x) &&
        focus_ray_axis(t_near, t_far, origin.y, direction.y, half_extent.y) &&
        focus_ray_axis(t_near, t_far, origin.z, direction.z, half_extent.z) &&
        t_far >= 0.0f;
}

float fisheye_distorted(float theta, float k1, float k2, float k3, float k4) {
    float t2 = theta * theta;
    return theta * (1.0f + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
}

bool unproject_fisheye(float u, float v, out float3 ray) {
    float xn = (u - pc.cx) / pc.fx;
    float yn = (v - pc.cy) / pc.fy;
    float radius = sqrt(xn * xn + yn * yn);
    if (radius < 1e-12f) {
        ray = float3(0.0f, 0.0f, 1.0f);
        return true;
    }
    float hi = 1.5707963267948966f - 1e-8f;
    if (radius >= fisheye_distorted(hi, pc.k1, pc.k2, pc.k3, pc.k4)) {
        ray = 0.0f;
        return false;
    }
    float theta = radius < hi ? radius : 0.5f * hi;
    [loop] for (uint i = 0u; i < 50u; ++i) {
        float t2 = theta * theta;
        float poly = 1.0f + t2 * (pc.k1 + t2 * (pc.k2 + t2 * (pc.k3 + t2 * pc.k4)));
        float derivative = 1.0f + t2 * (3.0f * pc.k1 + t2 * (5.0f * pc.k2 +
            t2 * (7.0f * pc.k3 + t2 * 9.0f * pc.k4)));
        float error = theta * poly - radius;
        if (abs(error) < 1e-12f) break;
        theta -= error / max(derivative, 1e-12f);
        theta = clamp(theta, 0.0f, hi);
    }
    float scale = sin(theta) / radius;
    ray = float3(scale * xn, scale * yn, cos(theta));
    return true;
}

bool unproject_equirect(float u, float v, out float3 ray) {
    if (pc.width == 0u || pc.height == 0u) {
        ray = 0.0f;
        return false;
    }
    float azimuth = 2.0f * kPi * (u / float(pc.width) - 0.5f);
    float elevation = kPi * (v / float(pc.height) - 0.5f);
    float cos_el = cos(elevation);
    ray = float3(cos_el * sin(azimuth), sin(elevation), cos_el * cos(azimuth));
    return true;
}

[numthreads(256, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID) {
    uint pixels = pc.width * pc.height;
    uint pixel = dtid.x;
    if (pixel >= pixels) return;
    uint x = pixel % pc.width;
    uint y = pixel / pc.width;
    float3 ray;
    bool valid = false;
    if (pc.model == kModelFisheye)
        valid = unproject_fisheye(float(x), float(y), ray);
    else if (pc.model == kModelEquirect)
        valid = unproject_equirect(float(x), float(y), ray);
    else {
        ray = float3((float(x) - pc.cx) / pc.fx, (float(y) - pc.cy) / pc.fy, 1.0f);
        valid = true;
    }
    if (!valid) {
        mask[pixel] = 0.0f;
        return;
    }
    float3 direction = float3(
        pc.m00 * ray.x + pc.m01 * ray.y + pc.m02 * ray.z,
        pc.m10 * ray.x + pc.m11 * ray.y + pc.m12 * ray.z,
        pc.m20 * ray.x + pc.m21 * ray.y + pc.m22 * ray.z);
    float3 origin = float3(pc.origin_x, pc.origin_y, pc.origin_z);
    float3 half_extent = float3(pc.half_x, pc.half_y, pc.half_z);
    mask[pixel] = focus_ray_hits(origin, direction, half_extent) ? 1.0f : 0.0f;
}
