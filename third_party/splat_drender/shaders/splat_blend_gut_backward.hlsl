// Appearance-only 3DGUT backward. Each lane owns one tile instance and
// replays at most 32 responses from the forward bucket snapshot, as in CUDA.
#include "splat_math.hlsli"
[[vk::binding(0, 0)]] StructuredBuffer<uint> ranges;
[[vk::binding(1, 0)]] StructuredBuffer<uint> instances;
[[vk::binding(2, 0)]] StructuredBuffer<float4> gauss_f;
[[vk::binding(3, 0)]] StructuredBuffer<float> out_f;
[[vk::binding(4, 0)]] StructuredBuffer<uint> out_u;
[[vk::binding(5, 0)]] StructuredBuffer<uint> bucket_offset;
[[vk::binding(6, 0)]] StructuredBuffer<float4> snap;
[[vk::binding(7, 0)]] StructuredBuffer<float> loss_color;
[[vk::binding(8, 0)]] StructuredBuffer<float> loss_alpha;
[[vk::binding(9, 0)]] StructuredBuffer<float> unused_normal;
[[vk::binding(10, 0)]] StructuredBuffer<float2> unused_median;
#ifdef GUT_NATIVE_ATOMIC
[[vk::binding(11, 0)]] RWStructuredBuffer<float> gradients;
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_capability(6033)]]
[[vk::ext_instruction(6035)]]
float op_atomic_f_add([[vk::ext_reference]] float destination, uint scope, uint semantics, float value);
#else
[[vk::binding(11, 0)]] RWStructuredBuffer<uint> gradients;
#endif

void atomic_add(uint index, float value) {
    if (value == 0.0f || !finite1(value)) return;
#ifdef GUT_NATIVE_ATOMIC
    op_atomic_f_add(gradients[index],1u,0u,value);
#else
    uint expected = gradients[index];
    [loop] for (;;) {
        uint observed;
        InterlockedCompareExchange(gradients[index], expected,
            asuint(asfloat(expected)+value), observed);
        if (observed == expected) return;
        expected = observed;
    }
#endif
}

bool response(uint id, float3 direction, float3 offset,
              out float3 origin, out float3 ray, out float3 closest,
              out float peak, out float dd, out float weight, out float alpha) {
    uint base = id*pc.u19;
    float4 a=gauss_f[base+3u], b=gauss_f[base+4u], c=gauss_f[base+5u];
    origin=float3(a.w,b.w,c.w)+float3(dot(a.xyz,offset),dot(b.xyz,offset),dot(c.xyz,offset));
    ray=float3(dot(a.xyz,direction),dot(b.xyz,direction),dot(c.xyz,direction));
    dd=dot(ray,ray);
    peak=-dot(origin,ray)/dd;
    closest=origin+peak*ray;
    weight=exp(-0.5f*dot(closest,closest));
    alpha=min(kAlphaClip,gauss_f[base+2u].w*weight);
    return dd>0.0f && peak>0.0f && alpha>=kAlphaFloor;
}

#ifdef GUT_SUBGROUP
#include "splat_blend_gut_backward_subgroup.hlsli"
#else
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint rank : SV_GroupIndex) {
    uint lane=rank&31u, bucket=group.x*8u+(rank>>5u);
    if (bucket>=pc.u8) return;
    uint width=pc.u0,height=pc.u1,count=pc.u5,pixels=pc.u6;
    uint tile=out_u[count+pixels+pc.u7+bucket];
    if (tile>=pc.u7) return;
    uint first=tile==0u ? 0u : bucket_offset[tile-1u];
    if (bucket<first) return;
    uint local_bucket=bucket-first,position=local_bucket*32u+lane;
    if (local_bucket*32u>=out_u[count+pixels+tile]) return;
    uint begin=ranges[2u*tile],end=ranges[2u*tile+1u];
    if (begin+position>=end) return;
    uint id=instances[begin+position],base=id*pc.u19;
    float4 a=gauss_f[base+3u],b=gauss_f[base+4u],c=gauss_f[base+5u];
    float4 rgb=gauss_f[base+2u];
    float2 refine=gauss_f[base+6u].xy;
    float3 acc_o=0.0f,acc_a=0.0f,acc_b=0.0f,acc_c=0.0f,acc_rgb=0.0f;
    float acc_opacity=0.0f,acc_refine=0.0f;
    [loop] for (uint local=0u;local<256u;++local) {
        uint x=(tile%pc.u2)*16u+(local&15u),y=(tile/pc.u2)*16u+(local>>4u);
        if (x>=width || y>=height) continue;
        uint pixel=y*width+x,last=out_u[count+pixel];
        if (position>=last) continue;
        float fx=asfloat(pc.u9),fy=asfloat(pc.u10),cx=asfloat(pc.u11),cy=asfloat(pc.u12);
        float3 direction=pc.u3==kModeOrtho ? float3(0,0,1) : pixel_unit_ray(float(x),float(y),pc.u3,
            fx,fy,cx,cy,asfloat(pc.u14),asfloat(pc.u15),asfloat(pc.u16),asfloat(pc.u17));
        float3 offset=pc.u3==kModeOrtho ? float3((float(x)-cx)/fx,(float(y)-cy)/fy,0) : 0.0f;
        float4 state=snap[bucket*256u+local];
        // Include the background in the remaining colour. This is equivalent
        // to CUDA's separate final_T * dot(background, dL_rgb) term.
        float3 remaining=float3(out_f[pixel],out_f[pixels+pixel],out_f[2u*pixels+pixel])-state.xyz;
        float T=state.w,alpha=0.0f,G=0.0f,peak=0.0f,dd=0.0f;
        float3 o=0.0f,d=0.0f,q=0.0f;
        bool accepted=false;
        [loop] for (uint k=0u;k<=lane;++k) {
            uint pos=local_bucket*32u+k;
            if (pos>=last || begin+pos>=end) break;
            uint source=instances[begin+pos];
            float3 qo,qd,qq;float qt,qdd,qG,qa;
            if (!response(source,direction,offset,qo,qd,qq,qt,qdd,qG,qa)) continue;
            remaining-=T*qa*gauss_f[source*pc.u19+2u].xyz;
            if (k==lane) {alpha=qa;G=qG;peak=qt;dd=qdd;o=qo;d=qd;q=qq;accepted=true;break;}
            T*=1.0f-qa;
        }
        if (!accepted) continue;
        float3 dl=float3(loss_color[pixel],loss_color[pixels+pixel],loss_color[2u*pixels+pixel]);
        float final_T=1.0f-out_f[3u*pixels+pixel];
        float da=T*dot(rgb.xyz,dl)-dot(remaining,dl)/(1.0f-alpha);
        if (pc.u13!=0u) da+=loss_alpha[pixel]*final_T/(1.0f-alpha);
        float raw=rgb.w*G;
        float factor=raw<kAlphaClip ? -da*raw : 0.0f;
        // Retain the orthogonality residual for very thin Gaussians.
        float h=factor*dot(q,d)/dd;
        float3 go=factor*q-h*d,v=offset+peak*direction;
        acc_o+=go;
        acc_a+=go.x*v-h*q.x*direction;
        acc_b+=go.y*v-h*q.y*direction;
        acc_c+=go.z*v-h*q.z*direction;
        acc_rgb+=T*alpha*dl;
        if (raw<kAlphaClip) acc_opacity+=da*G;
        float3 gm=-(a.xyz*go.x+b.xyz*go.y+c.xyz*go.z);
        acc_refine+=length(gm.xy*refine*float2(width,height))/max(1.0f-final_T,1.0e-6f);
    }
    [unroll] for(uint axis=0u;axis<3u;++axis) {
        atomic_add(3u*id+axis,acc_o[axis]);
        atomic_add(3u*count+4u*id+axis,acc_a[axis]);
        atomic_add(7u*count+3u*id+axis,acc_rgb[axis]);
        atomic_add(10u*count+4u*id+axis,acc_b[axis]);
        atomic_add(14u*count+3u*id+axis,acc_c[axis]);
    }
    atomic_add(3u*count+4u*id+3u,acc_opacity);
    atomic_add(17u*count+id,acc_refine);
}
#endif
