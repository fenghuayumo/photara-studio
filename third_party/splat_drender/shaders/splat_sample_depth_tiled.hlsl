#include "splat_math.hlsli"

[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float2> pixels;
[[vk::binding(4, 0)]] StructuredBuffer<uint> counts;
[[vk::binding(5, 0)]] StructuredBuffer<uint> order;
[[vk::binding(6, 0)]] RWStructuredBuffer<float> sample_f;
[[vk::binding(7, 0)]] RWStructuredBuffer<uint> sample_u;

// One group processes query rounds of the same neighbour tile. Each Gaussian
// batch is fetched once for all queries; no subgroup-size assumptions needed.
static const uint kQueryThreads = 128u;
groupshared float2 gs_mean[128];
groupshared float4 gs_conic[128];
groupshared float4 gs_plane[128];
groupshared uint gs_live;
groupshared uint gs_max;

void load_batch(uint begin, uint end, uint lane) {
    if (begin+lane < end) {
        uint g = instances[begin+lane];
        gs_mean[lane] = gauss_f[7u*g].xy;
        gs_conic[lane] = gauss_f[7u*g+1u];
        gs_plane[lane] = gauss_f[7u*g+3u];
    }
}

[numthreads(128, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint tile = group.y * pc.u3 + group.x, n = pc.u0;
    uint pbegin = tile == 0u ? 0u : counts[tile-1u];
    uint pend = counts[tile];
    if (pbegin == pend) return; // Uniform for the entire group.
    uint begin = ranges[2u*tile], end = ranges[2u*tile+1u];
    [loop] for (uint batch = pbegin; batch < pend; batch += kQueryThreads) {
        bool present = batch+lane < pend;
        uint i = present ? order[batch+lane] : 0u;
        float2 xy = present ? pixels[i] : 0.0f;
        float T = 1.0f, seed = 0.0f;
        uint last = 0u;
        bool stopped = !present;
        [loop] for (uint at = begin; at < end; at += kQueryThreads) {
            // Every lane, including empty/tail and finished queries, must
            // reach both barriers around each shared Gaussian batch.
            GroupMemoryBarrierWithGroupSync();
            load_batch(at,end,lane);
            GroupMemoryBarrierWithGroupSync();
            uint size = min(kQueryThreads,end-at);
            if (!stopped) {
                [loop] for (uint j=0u; j<size; ++j) {
                    float2 d = gs_mean[j]-xy;
                    float4 co = gs_conic[j];
                    float power = gaussian_power(co,d.x,d.y);
                    if (power > 0.0f) continue;
                    float alpha = min(kAlphaClip,co.w*exp(power));
                    if (alpha < kAlphaFloor) continue;
                    float test_t = T*(1.0f-alpha);
                    if (test_t < kTransmittanceFloor) { stopped=true; break; }
                    float4 rp = gs_plane[j];
                    if (T > 0.5f) seed = rp.x*d.x+rp.y*d.y+rp.z;
                    T = test_t;
                    last = at-begin+j+1u;
                }
            }
            if (lane == 0u) gs_live=0u;
            GroupMemoryBarrierWithGroupSync();
            if (!stopped) InterlockedAdd(gs_live,1u);
            GroupMemoryBarrierWithGroupSync();
            if (gs_live == 0u) break;
        }
        bool in_range = present && T <= kDepthMinTransmittance;
        bool refining = in_range;
        if (present) sample_u[i]=last;
        float window = asfloat(pc.u14)>0.0f ? asfloat(pc.u14) : 200.0f;
        float tolerance = asfloat(pc.u15);
        float lo=max(seed-window,0.0f), hi=max(seed+window,0.0f);
        float probes[9];
        [unroll] for (uint s=0u;s<9u;++s) probes[s]=1.0f;
        [loop] for (uint refine=0u;refine<8u;++refine) {
            // Bound all shared walks uniformly by the largest live query's
            // last contributor. Individual tolerances never skip barriers.
            if (lane == 0u) gs_max=0u;
            GroupMemoryBarrierWithGroupSync();
            if (refining) InterlockedMax(gs_max,last);
            GroupMemoryBarrierWithGroupSync();
            uint max_last=gs_max;
            if (max_last == 0u) break;
            bool first=refine==0u;
            uint sb=first?0u:1u, se=first?9u:8u;
            [unroll] for (uint sr=0u;sr<9u;++sr)
                if (refining && sr>=sb && sr<se) probes[sr]=1.0f;
            float interval=(hi-lo)/8.0f;
            [loop] for (uint at2=0u;at2<max_last;at2+=kQueryThreads) {
                GroupMemoryBarrierWithGroupSync();
                load_batch(begin+at2,min(end,begin+max_last),lane);
                GroupMemoryBarrierWithGroupSync();
                if (refining) {
                    uint size=min(kQueryThreads,last-at2);
                    if (at2>=last) size=0u;
                    [loop] for (uint j2=0u;j2<size;++j2) {
                        float2 d=gs_mean[j2]-xy;
                        float4 co=gs_conic[j2];
                        float power=gaussian_power(co,d.x,d.y);
                        if (power>0.0f) continue;
                        float alpha=min(kAlphaClip,co.w*exp(power));
                        if (alpha<kAlphaFloor) continue;
                        float4 rp=gs_plane[j2];
                        float peak=rp.x*d.x+rp.y*d.y+rp.z;
                        int band=0;
                        if (rp.w>0.0f) {
                            float half_band=8.0f/rp.w;
                            if (peak-half_band>lo+interval*float(se-1u)) band=1;
                            else if (peak+half_band<lo+interval*float(sb)) band=-1;
                        }
                        if (band==1) continue;
                        [unroll] for (uint s2=0u;s2<9u;++s2) {
                            if (s2<sb || s2>=se) continue;
                            if (band==-1) { probes[s2]*=1.0f-alpha; continue; }
                            float ts=lo+interval*float(s2);
                            float td=(ts-peak)*rp.w;
                            float ge=rp.w>0.0f ? exp(-0.5f*td*td) : 0.0f;
                            float om=1.0f-alpha*ge;
                            probes[s2]*=(ts>peak ? 1.0f-alpha : om)*rsqrt(om);
                        }
                    }
                }
            }
            if (refining) {
                if (first && !(probes[0]>=0.5f && probes[8]<=0.5f)) {
                    in_range=false; refining=false;
                } else {
                    uint sid=0u;
                    [unroll] for (uint s3=1u;s3<8u;++s3)
                        if (probes[s3]>=0.5f) sid=s3;
                    float nlo=lo+float(sid)*interval, nhi=lo+float(sid+1u)*interval;
                    probes[0]=probes[sid]; probes[8]=probes[sid+1u];
                    lo=nlo; hi=nhi;
                    if (tolerance>0.0f && hi-lo<=tolerance) refining=false;
                }
            }
        }
        if (present && in_range) {
            float weight=saturate((probes[0]-0.5f)/(probes[0]-probes[8]));
            float depth=weight*hi+(1.0f-weight)*lo;
            float3 ray=pixel_unit_ray(xy.x,xy.y,pc.u4,asfloat(pc.u6),asfloat(pc.u7),
                asfloat(pc.u8),asfloat(pc.u9),asfloat(pc.u10),asfloat(pc.u11),
                asfloat(pc.u12),asfloat(pc.u13));
            sample_f[3u*i]=ray.x*depth; sample_f[3u*i+1u]=ray.y*depth;
            sample_f[3u*i+2u]=ray.z*depth; sample_f[3u*n+i]=depth;
            sample_u[n+i]=1u;
        }
        GroupMemoryBarrierWithGroupSync();
    }
}
