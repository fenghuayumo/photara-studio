// Forward-only diagnostic conversion and approximate 1st/99th percentiles.
struct Push {
    uint pixels, stage, channel, grayscale, world;
    float near, far;
    float r0, r1, r2, r3, r4, r5, r6, r7, r8;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;
[[vk::binding(0, 0)]] StructuredBuffer<float> depth;
[[vk::binding(1, 0)]] StructuredBuffer<float> alpha;
[[vk::binding(2, 0)]] StructuredBuffer<float> normal;
[[vk::binding(3, 0)]] RWStructuredBuffer<float> color;
[[vk::binding(4, 0)]] RWStructuredBuffer<uint> stats;

[numthreads(256, 1, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    uint p = id.x;
    if (pc.stage == 3) {
        if (p != 0) return;
        uint count = 0;
        for (uint b=0; b<256; ++b) count += stats[4+b];
        float lo=0, hi=1;
        if (count > 0) {
            float minimum=asfloat(stats[1]), span=asfloat(stats[0])-minimum;
            uint low=(count-1)/100, high=(count-1)*99/100, cumulative=0;
            bool found=false;
            for (uint b=0; b<256; ++b) {
                cumulative += stats[4+b];
                if (!found && cumulative > low) { lo=minimum+span*b/256; found=true; }
                if (cumulative > high) { hi=minimum+span*(b+1)/256; break; }
            }
            hi=max(hi,lo+max(0.0001,lo*0.0001));
        }
        stats[2]=asuint(lo); stats[3]=asuint(hi);
        return;
    }
    if (p >= pc.pixels) return;
    float d=depth[p];
    bool valid=isfinite(d) && d>0 && isfinite(alpha[p]) && alpha[p]>.05;
    if (pc.stage == 1) {
        if (valid) { InterlockedMax(stats[0],asuint(d)); InterlockedMin(stats[1],asuint(d)); }
        return;
    }
    if (pc.stage == 2) {
        if (valid) {
            float minimum=asfloat(stats[1]), span=max(asfloat(stats[0])-minimum,1e-8);
            uint bin=min(255u,(uint)(saturate((d-minimum)/span)*256));
            InterlockedAdd(stats[4+bin],1);
        }
        return;
    }
    float3 rgb=0;
    if (valid) {
        if (pc.channel == 1) {
            float t=saturate((d-asfloat(stats[2]))/max(asfloat(stats[3])-asfloat(stats[2]),1e-8));
            rgb=pc.grayscale ? (1-t).xxx : (t<.5
                ? lerp(float3(.90,.80,.53),float3(.51,.73,.65),t*2)
                : lerp(float3(.51,.73,.65),float3(.16,.24,.40),(t-.5)*2));
        } else {
            float3 n=float3(normal[p],normal[pc.pixels+p],normal[2*pc.pixels+p]);
            if (pc.world) n=float3(dot(float3(pc.r0,pc.r1,pc.r2),n),
                                  dot(float3(pc.r3,pc.r4,pc.r5),n),
                                  dot(float3(pc.r6,pc.r7,pc.r8),n));
            float len=length(n);
            if (all(isfinite(n)) && len>1e-8) rgb=saturate(n/len*.5+.5);
        }
    }
    color[p]=rgb.x; color[pc.pixels+p]=rgb.y; color[2*pc.pixels+p]=rgb.z;
}
