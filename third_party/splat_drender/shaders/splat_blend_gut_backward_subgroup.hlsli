// One 32-lane subgroup owns a bucket. Prefix operations composite all its
// Gaussians in parallel, eliminating the portable path's per-lane replay.
[numthreads(256,1,1)]
void main(uint3 group : SV_GroupID, uint rank : SV_GroupIndex) {
    uint lane=WaveGetLaneIndex(),bucket=group.x*8u+(rank>>5u);
    if(bucket>=pc.u8) return;
    uint width=pc.u0,height=pc.u1,count=pc.u5,pixels=pc.u6;
    uint tile=out_u[count+pixels+pc.u7+bucket];
    if(tile>=pc.u7) return;
    uint first=tile==0u ? 0u : bucket_offset[tile-1u];
    if(bucket<first) return;
    uint start=(bucket-first)*32u,position=start+lane;
    if(start>=out_u[count+pixels+tile]) return;
    uint begin=ranges[2u*tile],end=ranges[2u*tile+1u];
    bool valid=begin+position<end;
    uint id=valid ? instances[begin+position] : 0u,base=id*pc.u19;
    float4 a=0.0f,b=0.0f,c=0.0f,rgb=0.0f;
    float2 refine=0.0f;
    if(valid) {
        a=gauss_f[base+3u];b=gauss_f[base+4u];c=gauss_f[base+5u];
        rgb=gauss_f[base+2u];refine=gauss_f[base+6u].xy;
    }
    float fx=asfloat(pc.u9),fy=asfloat(pc.u10),cx=asfloat(pc.u11),cy=asfloat(pc.u12);
    float3 acc_o=0.0f,acc_a=0.0f,acc_b=0.0f,acc_c=0.0f,acc_rgb=0.0f;
    float acc_opacity=0.0f,acc_refine=0.0f;
    [loop] for(uint pixel_base=0u;pixel_base<256u;pixel_base+=32u) {
        uint local=pixel_base+lane;
        uint x=(tile%pc.u2)*16u+(local&15u),y=(tile/pc.u2)*16u+(local>>4u);
        uint pixel=y*width+x,last=0u;
        float4 staged_state=0.0f,staged_grad=0.0f;
        float3 staged_direction=0.0f,staged_offset=0.0f;
        float staged_after=0.0f;
        if(x<width && y<height) {
            last=out_u[count+pixel];
            if(last>start) {
                float4 state=snap[bucket*256u+local];
                float3 dl=float3(loss_color[pixel],loss_color[pixels+pixel],loss_color[2u*pixels+pixel]);
                float3 remaining=float3(out_f[pixel],out_f[pixels+pixel],out_f[2u*pixels+pixel])-state.xyz;
                staged_after=dot(remaining,dl);
                staged_state=float4(0,0,0,state.w);
                staged_grad=float4(dl,(pc.u13!=0u ? loss_alpha[pixel] : 0.0f)*(1.0f-out_f[3u*pixels+pixel]));
                staged_state.x=out_f[3u*pixels+pixel];
                staged_direction=pc.u3==kModeOrtho ? float3(0,0,1) : pixel_unit_ray(float(x),float(y),pc.u3,
                    fx,fy,cx,cy,asfloat(pc.u14),asfloat(pc.u15),asfloat(pc.u16),asfloat(pc.u17));
                if(pc.u3==kModeOrtho) staged_offset=float3((float(x)-cx)/fx,(float(y)-cy)/fy,0);
            }
        }
        [loop] for(uint j=0u;j<32u;++j) {
            uint limit=WaveReadLaneAt(last,j);
            float4 state=WaveReadLaneAt(staged_state,j),dl=WaveReadLaneAt(staged_grad,j);
            float after=WaveReadLaneAt(staged_after,j);
            float3 direction=WaveReadLaneAt(staged_direction,j),offset=WaveReadLaneAt(staged_offset,j);
            // Uniform across the subgroup: inactive pixels need no response
            // or prefix operations, including out-of-image partial tiles.
            if(limit<=start) continue;
            float3 o=float3(a.w,b.w,c.w)+float3(dot(a.xyz,offset),dot(b.xyz,offset),dot(c.xyz,offset));
            float3 d=float3(dot(a.xyz,direction),dot(b.xyz,direction),dot(c.xyz,direction));
            float dd=dot(d,d),peak=-dot(o,d)/max(dd,1.0e-30f);
            float3 q=o+peak*d;
            bool active=valid && position<limit && dd>0.0f && peak>0.0f;
            float G=active ? exp(-.5f*dot(q,q)) : 0.0f;
            float raw=rgb.w*G,alpha=min(kAlphaClip,raw);
            alpha=alpha>=kAlphaFloor ? alpha : 0.0f;
            float T=state.w*WavePrefixProduct(1.0f-alpha);
            float weight=T*alpha,contribution=weight*dot(rgb.xyz,dl.xyz);
            float remaining=after-WavePrefixSum(contribution)-contribution;
            if(!active || alpha==0.0f) continue;
            float da=T*dot(rgb.xyz,dl.xyz)+(dl.w-remaining)/(1.0f-alpha);
            float factor=raw<kAlphaClip ? -da*raw : 0.0f;
            float h=factor*dot(q,d)/dd;
            float3 go=factor*q-h*d,v=offset+peak*direction;
            acc_o+=go;
            acc_a+=go.x*v-h*q.x*direction;
            acc_b+=go.y*v-h*q.y*direction;
            acc_c+=go.z*v-h*q.z*direction;
            acc_rgb+=weight*dl.xyz;
            if(raw<kAlphaClip) acc_opacity+=da*G;
            float3 gm=-(a.xyz*go.x+b.xyz*go.y+c.xyz*go.z);
            acc_refine+=length(gm.xy*refine*float2(width,height))/max(state.x,1.0e-6f);
        }
    }
    if(!valid) return;
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
