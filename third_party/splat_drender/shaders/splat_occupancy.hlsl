#include "splat_math.hlsli"
[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> camera;
[[vk::binding(4, 0)]] StructuredBuffer<float> points;
[[vk::binding(5, 0)]] RWStructuredBuffer<float> occupancy;
[[vk::binding(6, 0)]] RWStructuredBuffer<uint> inside;
[numthreads(64, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint i = id.x + id.y * 65535u * 64u;
    if (i >= pc.u0) return;
    occupancy[i] = 0.0f; inside[i] = 0u;
    float3 world = float3(points[3u*i], points[3u*i+1u], points[3u*i+2u]);
    float3 t = xform_point(world, camera[0],camera[1],camera[2],
        camera[4],camera[5],camera[6],camera[8],camera[9],camera[10],
        camera[12],camera[13],camera[14]);
    uint mode = pc.u4;
    if (mode == kModeEquirect) { if (!(dot(t,t)>1.0e-12f)) return; }
    else if (!(t.z > (mode == kModeFisheye ? 1.0e-6f : asfloat(pc.u21)))) return;
    Projection pr = project_camera(t, mode, int(pc.u1), int(pc.u2),
        asfloat(pc.u6),asfloat(pc.u7),asfloat(pc.u8),asfloat(pc.u9),
        asfloat(pc.u10),asfloat(pc.u11),asfloat(pc.u12),asfloat(pc.u13));
    if (!pr.valid || pr.pixel_x<0.0f || pr.pixel_y<0.0f ||
        pr.pixel_x>float(pc.u1-1u) || pr.pixel_y>float(pc.u2-1u)) return;
    inside[i] = 1u;
    uint tx=min(pc.u3-1u,uint(floor((pr.pixel_x+0.5f)/16.0f)));
    uint ty=min((pc.u2+15u)/16u-1u,uint(floor((pr.pixel_y+0.5f)/16.0f)));
    uint begin=ranges[2u*(ty*pc.u3+tx)], end=ranges[2u*(ty*pc.u3+tx)+1u];
    float T=1.0f, Tp=1.0f, distance=length(t);
    [loop] for(uint at=begin;at<end;++at) {
        uint g=instances[at];
        float2 d=gauss_f[7u*g].xy-float2(pr.pixel_x,pr.pixel_y);
        float4 co=gauss_f[7u*g+1u];
        float power=gaussian_power(co,d.x,d.y);
        if(power>0.0f) continue;
        float alpha=min(kAlphaClip,co.w*exp(power));
        if(alpha<kAlphaFloor) continue;
        float next=T*(1.0f-alpha);
        if(next<kTransmittanceFloor) break;
        float4 rp=gauss_f[7u*g+3u];
        float peak=rp.x*d.x+rp.y*d.y+rp.z;
        float delta=(peak-distance)*rp.w;
        float value=rp.w>0.0f?exp(-0.5f*delta*delta):0.0f;
        float om=1.0f-alpha*value;
        Tp *= (distance>peak?(1.0f-alpha):om)*rsqrt(om);
        T=next;
    }
    occupancy[i]=1.0f-Tp;
}
