// Backend-neutral geometry operations. CAS float accumulation also works on
// devices without VK_EXT_shader_atomic_float.
StructuredBuffer<float> input0 : register(t0);
StructuredBuffer<float> input1 : register(t1);
StructuredBuffer<float> camera_data : register(t2);
RWByteAddressBuffer output0 : register(u3);
RWByteAddressBuffer output1 : register(u4);
RWByteAddressBuffer terms : register(u5);
struct Push { uint mode, count, width, height; float fx, fy, cx, cy;
              float weight, focal; uint camera_count, euclidean; };
[[vk::push_constant]] Push pc;

void add_float(RWByteAddressBuffer buffer, uint index, float value) {
    if (value == 0.0f || isnan(value) || isinf(value)) return;
    uint expected = buffer.Load(4u * index);
    for (;;) {
        uint observed;
        buffer.InterlockedCompareExchange(4u * index, expected,
            asuint(asfloat(expected) + value), observed);
        if (observed == expected) return;
        expected = observed;
    }
}
float3 ray(uint x, uint y) {
    return float3(((float)x - pc.cx) / pc.fx, ((float)y - pc.cy) / pc.fy, 1.0f);
}
float3 depth_point(uint x, uint y) { return ray(x, y) * input0[y * pc.width + x]; }
float3 camera_to_world(float3 p) {
    return float3(dot(float3(camera_data[0], camera_data[1], camera_data[2]), p),
                  dot(float3(camera_data[4], camera_data[5], camera_data[6]), p),
                  dot(float3(camera_data[8], camera_data[9], camera_data[10]), p));
}

[numthreads(256, 1, 1)]
void main(uint3 tid : SV_DispatchThreadID) {
    uint i = tid.x + tid.y * 65535u * 256u;
    if (i >= pc.count) return;
    if (pc.mode == 0u) {
        uint x = i % pc.width, y = i / pc.width;
        if (x == 0u || y == 0u || x + 1u >= pc.width || y + 1u >= pc.height) return;
        uint top = i - pc.width, bottom = i + pc.width, left = i - 1u, right = i + 1u;
        if (!(input0[i] > 0.0f && input0[top] > 0.0f && input0[bottom] > 0.0f &&
              input0[left] > 0.0f && input0[right] > 0.0f)) return;
        float3 dy = depth_point(x, y + 1u) - depth_point(x, y - 1u);
        float3 dx = depth_point(x + 1u, y) - depth_point(x - 1u, y);
        float3 c = cross(dy, dx);
        float length2 = dot(c, c);
        if (!(length2 > 1.0e-20f) || isinf(length2)) return;
        float inv_length = rsqrt(length2);
        float3 dn = c * inv_length;
        float3 n = float3(input1[i], input1[pc.count + i], input1[2u * pc.count + i]);
        float scale = pc.weight / (float)pc.count;
        add_float(terms, 0u, scale * (1.0f - dot(n, dn)));
        add_float(output1, i, -scale * dn.x);
        add_float(output1, pc.count + i, -scale * dn.y);
        add_float(output1, 2u * pc.count + i, -scale * dn.z);
        float3 g = -scale * n;
        float3 gc = (g - dn * dot(dn, g)) * inv_length;
        float3 gy = cross(dx, gc), gx = cross(gc, dy);
        add_float(output0, bottom, dot(gy, ray(x, y + 1u)));
        add_float(output0, top, -dot(gy, ray(x, y - 1u)));
        add_float(output0, right, dot(gx, ray(x + 1u, y)));
        add_float(output0, left, -dot(gx, ray(x - 1u, y)));
    } else if (pc.mode == 1u) {
        float3 world = float3(input0[3u*i], input0[3u*i+1u], input0[3u*i+2u]);
        float distance = asfloat(0x7f7fffffu);
        for (uint v = 0u; v < pc.camera_count; ++v) {
            uint b = 27u * v;
            float3 p = float3(
                camera_data[b]*world.x + camera_data[b+4u]*world.y + camera_data[b+8u]*world.z + camera_data[b+12u],
                camera_data[b+1u]*world.x + camera_data[b+5u]*world.y + camera_data[b+9u]*world.z + camera_data[b+13u],
                camera_data[b+2u]*world.x + camera_data[b+6u]*world.y + camera_data[b+10u]*world.z + camera_data[b+14u]);
            if (pc.euclidean != 0u) { distance = min(distance, length(p)); continue; }
            float fx = camera_data[b+16u], fy = camera_data[b+17u];
            float w = camera_data[b+18u], h = camera_data[b+19u];
            if ((uint)camera_data[b+20u] == 1u) {
                if (!(p.z > 1.0e-6f)) continue;
                float2 q = p.xy / p.z;
                float r = length(q), theta = atan(r), t2 = theta * theta;
                float k1 = camera_data[b+23u], k2 = camera_data[b+24u];
                float k3 = camera_data[b+25u], k4 = camera_data[b+26u];
                float poly = 1.0f + t2*(k1+t2*(k2+t2*(k3+t2*k4)));
                float a = r > 1.0e-8f ? theta * poly / r : 1.0f;
                float2 pixel = float2(fx,fy) * a * q + float2(camera_data[b+21u],camera_data[b+22u]);
                if (pixel.x < -0.15f*w || pixel.x > 1.15f*w || pixel.y < -0.15f*h || pixel.y > 1.15f*h) continue;
                float derivative = (1.0f+t2*(3.0f*k1+t2*(5.0f*k2+t2*(7.0f*k3+9.0f*k4*t2)))) / (1.0f+r*r);
                float bq = r > 1.0e-8f ? (derivative-a)/(r*r) : 2.0f*(k1-1.0f/3.0f);
                float j00 = a+bq*q.x*q.x, j01 = bq*q.x*q.y, j11 = a+bq*q.y*q.y;
                float3 j0 = fx / p.z * float3(j00,j01,-dot(float2(j00,j01),q));
                float3 j1 = fy / p.z * float3(j01,j11,-dot(float2(j01,j11),q));
                float aa = dot(j0,j0), bb = dot(j0,j1), cc = dot(j1,j1);
                float sigma = sqrt(0.5f*(aa+cc+sqrt((aa-cc)*(aa-cc)+4.0f*bb*bb)));
                if (sigma > 0.0f && !isinf(sigma)) distance = min(distance, pc.focal/sigma);
            } else {
                if (!(p.z > 0.2f)) continue;
                if (abs(p.x/p.z) > w/fx*0.575f || abs(p.y/p.z) > h/fy*0.575f) continue;
                distance = min(distance, p.z);
            }
        }
        output0.Store(4u*i, asuint(distance));
        if (distance < asfloat(0x7f7fffffu)) {
            uint ignored;
            terms.InterlockedMax(0u, asuint(distance), ignored);
        }
    } else if (pc.mode == 2u) {
        float fallback = asfloat(terms.Load(0u));
        if (!(fallback > 0.0f) || isinf(fallback)) fallback = 1.0f;
        float distance = asfloat(output0.Load(4u*i));
        if (!(distance < asfloat(0x7f7fffffu))) distance = fallback;
        output0.Store(4u*i, asuint(distance * pc.weight / pc.focal));
    } else if (pc.mode == 3u) {
        float d = input0[i];
        float3 p = float3(asfloat(0x7fc00000u),asfloat(0x7fc00000u),asfloat(0x7fc00000u));
        if (d > 0.0f && !isinf(d))
            p = camera_to_world(ray(i%pc.width,i/pc.width)*d - float3(camera_data[12],camera_data[13],camera_data[14]));
        output0.Store3(12u*i, asuint(p));
    } else if (pc.mode == 4u) {
        float3 g = float3(input0[3u*i],input0[3u*i+1u],input0[3u*i+2u]);
        float value = asfloat(output0.Load(4u*i)) + dot(g, camera_to_world(ray(i%pc.width,i/pc.width)));
        output0.Store(4u*i, asuint(value));
    } else if (pc.mode == 5u) {
        if (i == 0u) {
            add_float(output0,0u,input0[1]);
            add_float(output0,1u,input0[4]);
            add_float(output0,2u,1.0f);
        }
    } else if (pc.mode == 6u) {
        float opacity = 1.0f/(1.0f+exp(-input1[i]));
        float3 s = float3(input0[3u*i],input0[3u*i+1u],input0[3u*i+2u]);
        float mean_s = (s.x+s.y+s.z)/3.0f;
        float anisotropy = max(s.x,max(s.y,s.z))-min(s.x,min(s.y,s.z));
        add_float(terms,0u,opacity); add_float(terms,1u,opacity*opacity);
        add_float(terms,2u,mean_s); add_float(terms,3u,mean_s*mean_s);
        add_float(terms,4u,anisotropy); add_float(terms,5u,anisotropy*anisotropy);
    }
}
