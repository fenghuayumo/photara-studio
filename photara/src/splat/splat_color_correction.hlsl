#include "splat_math.hlsli"

// PPISP modes: 0 forward, 2 reduce, 3 regularize, 4 constrain.
// Grid 5..10: forward, atomic backward, TV, constrain, input backward,
// cell-gather backward. Mode 11 precomputes PPISP homography derivatives;
// mode 12 processes thirty-two pixels per thread for the nine-parameter PPISP;
// mode 13 uses analytic vignetting and CRF derivatives for larger layouts;
// mode 14 computes per-channel grid means for the parallel projection;
// mode 15 fuses PPISP regularization, Adam and gauge projection for small tables.
StructuredBuffer<float> source : register(t0);
RWStructuredBuffer<float> parameters : register(u1);
StructuredBuffer<float> upstream : register(t2);
RWStructuredBuffer<float> output : register(u3);
#ifdef SPLAT_COLOR_ATOMIC_FLOAT
RWStructuredBuffer<float> gradient_bits : register(u4);
[[vk::ext_extension("SPV_EXT_shader_atomic_float_add")]]
[[vk::ext_capability(6033)]]
[[vk::ext_instruction(6035)]]
float op_atomic_f_add([[vk::ext_reference]] float destination,
                      uint scope, uint semantics, float value);
#else
RWStructuredBuffer<uint> gradient_bits : register(u4);
#endif
RWStructuredBuffer<float> auxiliary : register(u5);

groupshared float ppisp_partials[36][64];
groupshared float ppisp_group_h[9];
groupshared float ppisp_group_params[36];
groupshared float ppisp_update_grads[256];
groupshared float ppisp_update_params[256];

float pf(uint value) { return asfloat(value); }
void atomic_add(uint index, float value) {
    if (value == 0.0f) return;
#ifdef SPLAT_COLOR_ATOMIC_FLOAT
    op_atomic_f_add(gradient_bits[index], 1u, 0u, value);
#else
    uint expected = gradient_bits[index], observed;
    [loop] for (;;) {
        uint desired = asuint(asfloat(expected) + value);
        InterlockedCompareExchange(gradient_bits[index], expected, desired, observed);
        if (observed == expected) break;
        expected = observed;
    }
#endif
}
void write_partial(uint index, float value) {
#ifdef SPLAT_COLOR_ATOMIC_FLOAT
    gradient_bits[index]=value;
#else
    gradient_bits[index]=asuint(value);
#endif
}
float read_partial(uint index) {
#ifdef SPLAT_COLOR_ATOMIC_FLOAT
    return gradient_bits[index];
#else
    return asfloat(gradient_bits[index]);
#endif
}

float3 cross3(float3 a, float3 b) {
    return float3(a.y*b.z-a.z*b.y, a.z*b.x-a.x*b.z, a.x*b.y-a.y*b.x);
}
void matrix_product(in float a[9], in float b[9], out float o[9]) {
    [unroll] for (uint i=0; i<3; ++i)
        [unroll] for (uint j=0; j<3; ++j)
            o[3*i+j] = a[3*i]*b[j]+a[3*i+1]*b[3+j]+a[3*i+2]*b[6+j];
}
void homography(in float c[8], out float h[9]) {
    float bx=0.0480542f*c[0]-0.0043631f*c[1];
    float by=-0.0043631f*c[0]+0.0481283f*c[1];
    float rx=0.0580570f*c[2]-0.0179872f*c[3];
    float ry=-0.0179872f*c[2]+0.0431061f*c[3];
    float gx=0.0433336f*c[4]-0.0180537f*c[5];
    float gy=-0.0180537f*c[4]+0.0580500f*c[5];
    float nx=0.0128369f*c[6]-0.0034654f*c[7];
    float ny=-0.0034654f*c[6]+0.0128158f*c[7];
    float t[9]={bx,1.0f+rx,gx,by,ry,1.0f+gy,1.0f,1.0f,1.0f};
    float xx=1.0f/3.0f+nx, yy=1.0f/3.0f+ny;
    float skew[9]={0,-1,yy,1,0,-xx,-yy,xx,0};
    float m[9]; matrix_product(skew,t,m);
    float3 r0=float3(m[0],m[1],m[2]);
    float3 r1=float3(m[3],m[4],m[5]);
    float3 r2=float3(m[6],m[7],m[8]);
    float3 lambda=cross3(r0,r1);
    if (dot(lambda,lambda)<1.0e-20f) lambda=cross3(r0,r2);
    if (dot(lambda,lambda)<1.0e-20f) lambda=cross3(r1,r2);
    float td[9];
    [unroll] for(uint i=0;i<3;++i)
        [unroll] for(uint j=0;j<3;++j) td[3*i+j]=t[3*i+j]*lambda[j];
    float inv[9]={-1,-1,1,1,0,0,0,1,0};
    matrix_product(td,inv,h);
    float divisor=h[8];
    if(abs(divisor)>1.0e-20f)
        [unroll] for(uint k=0;k<9;++k) h[k]/=divisor;
}
float softplus(float x) { return x>20.0f ? x : log(1.0f+exp(x)); }
float softplus_gradient(float x) {
    return x>20.0f ? 1.0f : 1.0f/(1.0f+exp(-x));
}
float crf_channel(float x, in float p[36], uint offset) {
    x=saturate(x);
    float toe=0.3f+softplus(p[offset]);
    float shoulder=0.3f+softplus(p[offset+1]);
    float gamma=0.1f+softplus(p[offset+2]);
    float center=clamp(1.0f/(1.0f+exp(-p[offset+3])),1.0e-4f,1.0f-1.0e-4f);
    float a=shoulder*center/max((1.0f-center)*toe+center*shoulder,1.0e-8f);
    float y=x<=center ? a*pow(x/center,toe) :
        1.0f-(1.0f-a)*pow((1.0f-x)/(1.0f-center),shoulder);
    return pow(max(y,0.0f),gamma);
}
float3 apply_homography(float3 rgb,in float h[9]) {
    float intensity=rgb.x+rgb.y+rgb.z;
    float3 rgi=float3(rgb.x,rgb.y,intensity);
    float3 mapped=float3(dot(float3(h[0],h[1],h[2]),rgi),
                         dot(float3(h[3],h[4],h[5]),rgi),
                         dot(float3(h[6],h[7],h[8]),rgi));
    float z=max(mapped.z,1.0e-4f*abs(intensity)+1.0e-8f);
    float norm=intensity/z;
    return float3(mapped.x*norm,mapped.y*norm,
                  intensity-(mapped.x+mapped.y)*norm);
}
void homography_vjp(float3 rgb,in float h[9],float3 go,
                    out float3 din,out float pullback[9]) {
    float intensity=rgb.x+rgb.y+rgb.z;
    float3 rgi=float3(rgb.x,rgb.y,intensity);
    float3 mapped=float3(dot(float3(h[0],h[1],h[2]),rgi),
                         dot(float3(h[3],h[4],h[5]),rgi),
                         dot(float3(h[6],h[7],h[8]),rgi));
    float zmin=1.0e-4f*abs(intensity)+1.0e-8f;
    float z=max(mapped.z,zmin);
    float norm=intensity/z;
    float dr=go.x-go.z,dg=go.y-go.z;
    float dint=go.z;
    float dmx=dr*norm,dmy=dg*norm;
    float dnorm=dr*mapped.x+dg*mapped.y;
    dint+=dnorm/z;
    float dz=-dnorm*intensity/(z*z);
    float dmz=0.0f;
    if(mapped.z>zmin)dmz+=dz;
    else dint+=dz*(intensity>=0.0f?1.0e-4f:-1.0e-4f);
    pullback[0]=dmx*rgi.x;pullback[1]=dmx*rgi.y;pullback[2]=dmx*rgi.z;
    pullback[3]=dmy*rgi.x;pullback[4]=dmy*rgi.y;pullback[5]=dmy*rgi.z;
    pullback[6]=dmz*rgi.x;pullback[7]=dmz*rgi.y;pullback[8]=dmz*rgi.z;
    float ir=h[0]*dmx+h[3]*dmy+h[6]*dmz;
    float ig=h[1]*dmx+h[4]*dmy+h[7]*dmz;
    dint+=h[2]*dmx+h[5]*dmy+h[8]*dmz;
    din=float3(ir+dint,ig+dint,dint);
}
float3 vignette_pixel(float3 rgb,uint x,uint y,uint width,uint height,
                      float cx,float cy,in float p[36]) {
    float scale=(float)max(width,height);
    float2 uv=(float2((float)x,(float)y)-float2(cx,cy))/scale;
    [unroll] for(uint c=0;c<3;++c) {
        uint q=1+5*c;
        float2 d=uv-float2(p[q],p[q+1]);
        float r2=dot(d,d),r4=r2*r2;
        rgb[c]*=saturate(1.0f+p[q+2]*r2+p[q+3]*r4+p[q+4]*r4*r2);
    }
    return rgb;
}
void vignette_vjp(float3 exposed,uint x,uint y,uint width,uint height,
                  float cx,float cy,in float p[36],float3 go,
                  out float3 din,inout float local[36]) {
    float scale=(float)max(width,height);
    float2 uv=(float2((float)x,(float)y)-float2(cx,cy))/scale;
    din=0.0f;
    [unroll] for(uint c=0;c<3;++c) {
        uint q=1+5*c;
        float2 d=uv-float2(p[q],p[q+1]);
        float r2=dot(d,d),r4=r2*r2,r6=r4*r2;
        float raw=1.0f+p[q+2]*r2+p[q+3]*r4+p[q+4]*r6;
        din[c]=go[c]*saturate(raw);
        if(raw>0.0f && raw<1.0f) {
            float draw=go[c]*exposed[c];
            local[q+2]+=draw*r2;
            local[q+3]+=draw*r4;
            local[q+4]+=draw*r6;
            float dr2=draw*(p[q+2]+2.0f*p[q+3]*r2+3.0f*p[q+4]*r4);
            local[q]+=dr2*(-2.0f*d.x);
            local[q+1]+=dr2*(-2.0f*d.y);
        }
    }
}
void crf_channel_grad(float xraw,float toe_raw,float shoulder_raw,
                      float gamma_raw,float center_raw,
                      out float value,out float dx,out float dp[4]) {
    float toe=0.3f+softplus(toe_raw);
    float shoulder=0.3f+softplus(shoulder_raw);
    float gamma=0.1f+softplus(gamma_raw);
    float center_unclamped=1.0f/(1.0f+exp(-center_raw));
    bool center_clamped=center_unclamped<1.0e-4f ||
                        center_unclamped>1.0f-1.0e-4f;
    float center=clamp(center_unclamped,1.0e-4f,1.0f-1.0e-4f);
    float denominator=max((1.0f-center)*toe+center*shoulder,1.0e-8f);
    float a=shoulder*center/denominator,b=1.0f-a;
    float inverse_squared=1.0f/(denominator*denominator);
    float da_toe=-shoulder*center*(1.0f-center)*inverse_squared;
    float da_shoulder=center*(denominator-shoulder*center)*inverse_squared;
    float da_center=(shoulder*denominator-
                    shoulder*center*(shoulder-toe))*inverse_squared;
    float x=saturate(xraw),y,dy_dx,dy_toe,dy_shoulder,dy_center;
    if(x<=center) {
        float u=x/center, power=pow(u,toe);
        y=a*power;
        dy_dx=a*toe*pow(u,toe-1.0f)/center;
        dy_toe=da_toe*power+a*power*(u>0.0f?log(u):0.0f);
        dy_shoulder=da_shoulder*power;
        dy_center=da_center*power-a*toe*power/center;
    } else {
        float v=(1.0f-x)/(1.0f-center),power=pow(v,shoulder);
        y=1.0f-b*power;
        dy_dx=b*shoulder*pow(v,shoulder-1.0f)/(1.0f-center);
        dy_toe=da_toe*power;
        dy_shoulder=da_shoulder*power-b*power*(v>0.0f?log(v):0.0f);
        dy_center=da_center*power-b*power*shoulder/(1.0f-center);
    }
    float base=max(y,0.0f);
    value=pow(base,gamma);
    float dvalue_y=base>0.0f?gamma*value/base:0.0f;
    float dvalue_gamma=base>0.0f?value*log(base):0.0f;
    dx=xraw>=0.0f && xraw<=1.0f?dvalue_y*dy_dx:0.0f;
    dp[0]=dvalue_y*dy_toe*softplus_gradient(toe_raw);
    dp[1]=dvalue_y*dy_shoulder*softplus_gradient(shoulder_raw);
    dp[2]=dvalue_gamma*softplus_gradient(gamma_raw);
    dp[3]=dvalue_y*dy_center*(center_clamped?0.0f:center*(1.0f-center));
}
float3 image_pixel(uint pixel,uint pixels) {
    return float3(source[pixel],source[pixels+pixel],source[2*pixels+pixel]);
}
float3 gradient_pixel(uint pixel,uint pixels) {
    return float3(upstream[pixel],upstream[pixels+pixel],upstream[2*pixels+pixel]);
}
void store_pixel(uint pixel,uint pixels,float3 value) {
    output[pixel]=value.x; output[pixels+pixel]=value.y;
    output[2*pixels+pixel]=value.z;
}
void load_params(uint view,uint count,out float p[36]) {
    [unroll] for(uint k=0;k<36;++k)
        p[k]=k<count ? parameters[view*count+k] : 0.0f;
}

void grid_sample(uint pixel,uint width,uint height,uint view,
                 uint gl,uint gh,uint gw,bool wrap,
                 out float3 rgb,out uint offsets[8],out float weights[8],
                 out float xy_weights[8],out float luma_factor,
                 out bool luma_inside) {
    uint pixels=width*height;
    rgb=image_pixel(pixel,pixels);
    float luma=dot(rgb,float3(0.299f,0.587f,0.114f));
    luma_inside=luma>0.0f && luma<1.0f;
    luma_factor=(float)(gl-1);
    uint x=pixel%width, y=pixel/width;
    float fx=width>1 ? (float)x/(float)(wrap?width:width-1)*(float)(wrap?gw:gw-1) : 0.0f;
    float fy=height>1 ? (float)y/(float)(height-1)*(float)(gh-1) : 0.0f;
    float fz=saturate(luma)*luma_factor;
    uint ix=(uint)floor(fx),iy=(uint)floor(fy),iz=(uint)floor(fz);
    uint x0=min(ix,gw-1),x1=wrap?(ix+1)%gw:min(ix+1,gw-1);
    uint y0=min(iy,gh-1),y1=min(iy+1,gh-1);
    uint z0=min(iz,gl-1),z1=min(iz+1,gl-1);
    float wx=fx-(float)ix,wy=fy-(float)iy,wz=fz-(float)iz;
    [unroll] for(uint dz=0;dz<2;++dz)
      [unroll] for(uint dy=0;dy<2;++dy)
        [unroll] for(uint dx=0;dx<2;++dx) {
            uint n=dz*4+dy*2+dx;
            uint xx=dx?x1:x0,yy=dy?y1:y0,zz=dz?z1:z0;
            offsets[n]=(((view*gl+zz)*gh+yy)*gw+xx)*12;
            xy_weights[n]=(dx?wx:1.0f-wx)*(dy?wy:1.0f-wy);
            weights[n]=xy_weights[n]*(dz?wz:1.0f-wz);
        }
}
float3 grid_pixel(float3 rgb,in uint offsets[8],in float weights[8]) {
    float3 value=0.0f;
    [unroll] for(uint n=0;n<8;++n)
      [unroll] for(uint c=0;c<3;++c) {
        uint base=offsets[n]+4*c;
        value[c]+=weights[n]*(parameters[base]*rgb.x+
            parameters[base+1]*rgb.y+parameters[base+2]*rgb.z+
            parameters[base+3]);
      }
    return value;
}

float ppisp_regularization_extra(uint pixel,uint count,uint views) {
    uint v=pixel/count,k=pixel%count;
    uint color=count==9?1:16;
    float extra=0.0f;
    if(k==0) {
        float mean=0.0f;
        [loop] for(uint n=0;n<views;++n) mean+=source[n*count]/(float)views;
        extra=pf(pc.u10)*clamp(mean/0.1f,-1.0f,1.0f);
    } else if(k>=color && k<color+8) {
        uint pair=(k-color)/2,part=(k-color)%2;
        static const float diagonal[8]={0.0480542f,0.0481283f,
            0.0580570f,0.0431061f,0.0433336f,0.0580500f,
            0.0128369f,0.0128158f};
        static const float offdiag[4]={-0.0043631f,-0.0179872f,
            -0.0180537f,-0.0034654f};
        float a=0.0f,b=0.0f;
        [loop] for(uint n=0;n<views;++n) {
            float x=source[n*count+color+2*pair];
            float y=source[n*count+color+2*pair+1];
            a+=(diagonal[2*pair]*x+offdiag[pair]*y)/(float)views;
            b+=(offdiag[pair]*x+diagonal[2*pair+1]*y)/(float)views;
        }
        float ga=pf(pc.u11)/8.0f*clamp(a/0.005f,-1.0f,1.0f);
        float gb=pf(pc.u11)/8.0f*clamp(b/0.005f,-1.0f,1.0f);
        extra=part==0?diagonal[2*pair]*ga+offdiag[pair]*gb:
            offdiag[pair]*ga+diagonal[2*pair+1]*gb;
    } else if(count>=24 && k>=1 && k<16) {
        uint axis=(k-1)%5,channel=(k-1)/5;
        float value=source[pixel],mean=0.0f;
        [unroll] for(uint c=0;c<3;++c)
            mean+=source[v*count+1+5*c+axis]/3.0f;
        extra=(2.0f/3.0f)*(value-mean)*pf(pc.u14)/(5.0f*(float)views);
        if(axis<2)extra+=2.0f*value*pf(pc.u12)/(3.0f*(float)views);
        else if(value>0.0f)extra+=pf(pc.u13)/(9.0f*(float)views);
    } else if(count==36 && k>=24) {
        uint axis=(k-24)%4;
        float mean=0.0f;
        [unroll] for(uint c=0;c<3;++c)
            mean+=source[v*count+24+4*c+axis]/3.0f;
        extra=(2.0f/3.0f)*(source[pixel]-mean)*
            pf(pc.u15)/(4.0f*(float)views);
    }
    return extra;
}

[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID,uint3 group_id:SV_GroupID,
          uint3 lane_id:SV_GroupThreadID) {
    uint mode=pc.u7, width=pc.u0,height=pc.u1,view=pc.u2;
    uint pixel=id.x,pixels=width*height;
    if(mode==0) {
        uint count=pc.u3,lane=lane_id.x;
        if(lane<count)
            ppisp_group_params[lane]=parameters[view*count+lane];
        GroupMemoryBarrierWithGroupSync();
        if(lane==0) {
            uint color=count==9?1:16;
            float latents[8],h[9];
            [unroll] for(uint k=0;k<8;++k)
                latents[k]=ppisp_group_params[color+k];
            homography(latents,h);
            [unroll] for(uint k=0;k<9;++k)ppisp_group_h[k]=h[k];
        }
        GroupMemoryBarrierWithGroupSync();
        if(pixel>=pixels) return;
        float3 rgb=image_pixel(pixel,pixels)*exp2(ppisp_group_params[0]);
        if(count>=24)
            rgb=vignette_pixel(rgb,pixel%width,pixel/width,width,height,
                               pf(pc.u10),pf(pc.u11),ppisp_group_params);
        rgb=apply_homography(rgb,ppisp_group_h);
        if(count==36)
            rgb=float3(crf_channel(rgb.x,ppisp_group_params,24),
                       crf_channel(rgb.y,ppisp_group_params,28),
                       crf_channel(rgb.z,ppisp_group_params,32));
        store_pixel(pixel,pixels,pc.u9!=0?saturate(rgb):rgb);
        return;
    }
    if(mode==12) {
        uint lane=lane_id.x;
        uint first=group_id.x*2048+lane;
        float h[9],pullback_sum[9],local[9];
        [unroll] for(uint k=0;k<9;++k) {
            h[k]=auxiliary[k];
            pullback_sum[k]=0.0f;
            local[k]=0.0f;
        }
        float gain=exp2(parameters[view*9]);
        [unroll] for(uint slot=0;slot<32;++slot) {
            uint p=first+slot*64;
            if(p>=pixels) break;
            float3 exposed=image_pixel(p,pixels)*gain;
            float3 go=gradient_pixel(p,pixels);
            if(pc.u9!=0) {
                float3 before=apply_homography(exposed,h);
                [unroll] for(uint c=0;c<3;++c)
                    if(before[c]<=0.0f || before[c]>=1.0f)go[c]=0.0f;
            }
            float3 dexposed;float pullback[9];
            homography_vjp(exposed,h,go,dexposed,pullback);
            store_pixel(p,pixels,dexposed*gain);
            local[0]+=dot(dexposed,exposed)*0.69314718056f;
            [unroll] for(uint k=0;k<9;++k)
                pullback_sum[k]+=pullback[k];
        }
        [unroll] for(uint c=0;c<8;++c) {
            float total=0.0f;
            [unroll] for(uint k=0;k<9;++k)
                total+=pullback_sum[k]*auxiliary[9+c*9+k];
            local[1+c]=total;
        }
        [unroll] for(uint k=0;k<9;++k)ppisp_partials[k][lane]=local[k];
        GroupMemoryBarrierWithGroupSync();
        if(lane<9) {
            float sum=0.0f;
            [unroll] for(uint i=0;i<64;++i)sum+=ppisp_partials[lane][i];
            write_partial(group_id.x*9+lane,sum);
        }
        return;
    }
    if(mode==13) {
        uint count=pc.u3,lane=lane_id.x;
        uint first=group_id.x*2048+lane;
        float params[36],h[9],pullback_sum[9],local[36];
        load_params(view,count,params);
        [unroll] for(uint k=0;k<9;++k) {
            h[k]=auxiliary[k];
            pullback_sum[k]=0.0f;
        }
        [unroll] for(uint k=0;k<36;++k)local[k]=0.0f;
        float gain=exp2(params[0]);
        [unroll] for(uint slot=0;slot<32;++slot) {
            uint p=first+slot*64;
            if(p>=pixels) break;
            uint x=p%width,y=p/width;
            float3 exposed=image_pixel(p,pixels)*gain;
            float3 vig=vignette_pixel(exposed,x,y,width,height,
                                     pf(pc.u10),pf(pc.u11),params);
            float3 coloured=apply_homography(vig,h);
            float3 go=gradient_pixel(p,pixels);
            float3 preclamp=coloured;
            if(count==36) {
                [unroll] for(uint c=0;c<3;++c)
                    preclamp[c]=crf_channel(coloured[c],params,24+4*c);
            }
            if(pc.u9!=0) {
                [unroll] for(uint c=0;c<3;++c)
                    if(preclamp[c]<=0.0f || preclamp[c]>=1.0f)go[c]=0.0f;
            }
            float3 dcoloured=go;
            if(count==36) {
                [unroll] for(uint c=0;c<3;++c) {
                    if(go[c]==0.0f) {
                        dcoloured[c]=0.0f;
                        continue;
                    }
                    float value,dx,dp[4];
                    uint q=24+4*c;
                    crf_channel_grad(coloured[c],params[q],params[q+1],
                                     params[q+2],params[q+3],value,dx,dp);
                    dcoloured[c]=go[c]*dx;
                    [unroll] for(uint k=0;k<4;++k)local[q+k]+=go[c]*dp[k];
                }
            }
            float3 dvig;float pullback[9];
            homography_vjp(vig,h,dcoloured,dvig,pullback);
            float3 dexposed;
            vignette_vjp(exposed,x,y,width,height,pf(pc.u10),pf(pc.u11),
                         params,dvig,dexposed,local);
            store_pixel(p,pixels,dexposed*gain);
            local[0]+=dot(dexposed,exposed)*0.69314718056f;
            [unroll] for(uint k=0;k<9;++k)
                pullback_sum[k]+=pullback[k];
        }
        [unroll] for(uint c=0;c<8;++c) {
            float total=0.0f;
            [unroll] for(uint k=0;k<9;++k)
                total+=pullback_sum[k]*auxiliary[9+c*9+k];
            local[16+c]=total;
        }
        [unroll] for(uint k=0;k<36;++k)ppisp_partials[k][lane]=local[k];
        GroupMemoryBarrierWithGroupSync();
        if(lane<count) {
            float sum=0.0f;
            [unroll] for(uint i=0;i<64;++i)sum+=ppisp_partials[lane][i];
            write_partial(group_id.x*count+lane,sum);
        }
        return;
    }
    if(mode==11) {
        uint count=pc.u3;
        if(pixel>=81) return;
        uint color=count==9?1:16;
        float latents[8];
        [unroll] for(uint k=0;k<8;++k)
            latents[k]=source[view*count+color+k];
        float h[9];
        if(pixel<9) {
            homography(latents,h);
            output[pixel]=h[pixel];
        } else {
            uint param=(pixel-9)/9,element=(pixel-9)%9;
            latents[param]+=1.0e-2f;
            homography(latents,h);
            float plus=h[element];
            latents[param]-=2.0e-2f;
            homography(latents,h);
            output[pixel]=(plus-h[element])/2.0e-2f;
        }
        return;
    }
    if(mode==2) {
        uint count=pc.u3,groups=pc.u4;
        if(pixel>=count) return;
        float sum=0.0f;
        [loop] for(uint g=0;g<groups;++g)
            sum+=read_partial(g*count+pixel);
        auxiliary[view*count+pixel]=sum;
        return;
    }
    if(mode==3) {
        uint count=pc.u3,views=pc.u4;
        if(pixel>=count*views) return;
        output[pixel]+=ppisp_regularization_extra(pixel,count,views);
        return;
    }
    if(mode==4) {
        uint count=pc.u3,views=pc.u4;
        if(pixel>=9) return;
        uint color=count==9?1:16;
        uint index=pixel==0?0:color+pixel-1;
        float mean=0.0f;
        if(pc.u12!=0)
            [loop] for(uint v=0;v<views;++v)
                mean+=source[v*count+index]/(float)views;
        float limit=pixel==0?pf(pc.u13):pf(pc.u14);
        [loop] for(uint v=0;v<views;++v) {
            uint offset=v*count+index;
            float value=source[offset]-mean;
            output[offset]=limit>0.0f?clamp(value,-limit,limit):value;
        }
        return;
    }
    if(mode==15) {
        uint count=pc.u3,views=pc.u4,total=count*views,lane=lane_id.x;
        if(total>256) return;
        [loop] for(uint index=lane;index<total;index+=64)
            ppisp_update_grads[index]=parameters[index]+
                ppisp_regularization_extra(index,count,views);
        GroupMemoryBarrierWithGroupSync();
        [loop] for(uint index=lane;index<total;index+=64) {
            float previous=source[index],grad=ppisp_update_grads[index];
            float m=0.0f,v=0.0f,updated=previous;
            if(!isfinite(previous) || !isfinite(grad))
                updated=isfinite(previous)?previous:0.0f;
            else {
                m=pf(pc.u1)*upstream[index]+(1.0f-pf(pc.u1))*grad;
                v=pf(pc.u2)*read_partial(index)+
                    (1.0f-pf(pc.u2))*grad*grad;
                if(isfinite(m) && isfinite(v)) {
                    float candidate=previous-pf(pc.u0)*(m/pf(pc.u5))/
                        (sqrt(v/pf(pc.u6))+pf(pc.u8));
                    updated=isfinite(candidate)?candidate:previous;
                } else {
                    m=0.0f;
                    v=0.0f;
                }
            }
            parameters[index]=grad;
            auxiliary[index]=m;
            write_partial(index,v);
            output[index]=updated;
            ppisp_update_params[index]=updated;
        }
        GroupMemoryBarrierWithGroupSync();
        if(lane<9) {
            uint color=count==9?1:16;
            uint index=lane==0?0:color+lane-1;
            float mean=0.0f;
            if(pc.u9!=0)
                [loop] for(uint v=0;v<views;++v)
                    mean+=ppisp_update_params[v*count+index]/(float)views;
            float limit=lane==0?pf(pc.u16):pf(pc.u17);
            [loop] for(uint v=0;v<views;++v) {
                uint offset=v*count+index;
                float value=ppisp_update_params[offset]-mean;
                output[offset]=limit>0.0f?clamp(value,-limit,limit):value;
            }
        }
        return;
    }
    if(mode==5 || mode==6 || mode==9) {
        if(pixel>=pixels) return;
        uint gl=pc.u3,gw=pc.u4,gh=pc.u5;
        float3 rgb;uint offsets[8];float weights[8],xy[8],lf;bool inside;
        grid_sample(pixel,width,height,view,gl,gh,gw,pc.u8!=0,
                    rgb,offsets,weights,xy,lf,inside);
        if(mode==5) {
            store_pixel(pixel,pixels,grid_pixel(rgb,offsets,weights));
            return;
        }
        float3 go=gradient_pixel(pixel,pixels),din=0.0f;
        float dluma=0.0f;
        [unroll] for(uint n=0;n<8;++n)
          [unroll] for(uint c=0;c<3;++c) {
            uint base=offsets[n]+4*c;
            float affine=dot(float3(parameters[base],parameters[base+1],
                                    parameters[base+2]),rgb)+parameters[base+3];
            din+=go[c]*weights[n]*float3(parameters[base],parameters[base+1],
                                         parameters[base+2]);
            if(mode==6) {
                atomic_add(base,go[c]*weights[n]*rgb.x);
                atomic_add(base+1,go[c]*weights[n]*rgb.y);
                atomic_add(base+2,go[c]*weights[n]*rgb.z);
                atomic_add(base+3,go[c]*weights[n]);
            }
            dluma+=go[c]*affine*xy[n]*((n&4)?1.0f:-1.0f);
          }
        if(inside)din+=dluma*lf*float3(0.299f,0.587f,0.114f);
        store_pixel(pixel,pixels,din);
        return;
    }
    if(mode==10) {
        uint gl=pc.u3,gw=pc.u4,gh=pc.u5,rows=pc.u6;
        uint cells=rows*gl*gh*gw;
        uint cell=group_id.x,lane=lane_id.x;
        if(cell>=cells) return;
        if(cell/(gl*gh*gw)!=view) {
            if(lane<12) output[cell*12+lane]=0.0f;
            return;
        }
        uint cx=cell%gw,cy=(cell/gw)%gh,cz=(cell/(gw*gh))%gl;
        uint scale_x=pc.u8!=0?gw:gw-1;
        uint denom_x=pc.u8!=0?width:width-1;
        float xfactor=width>1?(float)scale_x/(float)denom_x:0.0f;
        float yfactor=height>1?(float)(gh-1)/(float)(height-1):0.0f;
        uint xbegin=0,xend=width;
        if(pc.u8==0 && gw>1 && width>1) {
            xbegin=(uint)max(0.0f,ceil(((float)cx-1.0f)/xfactor));
            xend=min(width,(uint)floor(((float)cx+1.0f)/xfactor)+1);
        }
        uint ybegin=gh>1 && height>1
            ? (uint)max(0.0f,ceil(((float)cy-1.0f)/yfactor)) : 0;
        uint yend=gh>1 && height>1
            ? min(height,(uint)floor(((float)cy+1.0f)/yfactor)+1) : height;
        float sums[12];
        [unroll] for(uint i=0;i<12;++i)sums[i]=0.0f;
        uint span_x=xend-xbegin;
        uint span_y=yend-ybegin;
        if(span_x==0 || span_y==0) {
            if(lane<12)output[cell*12+lane]=0.0f;
            return;
        }
        [loop] for(uint i=lane;i<span_x*span_y;i+=64) {
            uint x=xbegin+i%span_x;
            uint y=ybegin+i/span_x;
            float wy=gh==1?1.0f:max(1.0f-abs((float)y*yfactor-(float)cy),0.0f);
            if(wy==0.0f)continue;
            float distance=abs((float)x*xfactor-(float)cx);
            if(pc.u8!=0 && gw>1)distance=min(distance,(float)gw-distance);
            float wx=gw==1?1.0f:max(1.0f-distance,0.0f);
            if(wx==0.0f)continue;
            uint p=y*width+x;
            float3 rgb=image_pixel(p,pixels);
            float luma=saturate(dot(rgb,float3(0.299f,0.587f,0.114f)));
            float wz=gl==1?1.0f:max(1.0f-abs(luma*(float)(gl-1)-(float)cz),0.0f);
            float weight=wx*wy*wz;
            if(weight==0.0f)continue;
            float3 go=gradient_pixel(p,pixels)*weight;
            [unroll] for(uint c=0;c<3;++c) {
                sums[4*c]+=go[c]*rgb.x;
                sums[4*c+1]+=go[c]*rgb.y;
                sums[4*c+2]+=go[c]*rgb.z;
                sums[4*c+3]+=go[c];
            }
        }
        uint wave_size=WaveGetLaneCount();
        [unroll] for(uint k=0;k<12;++k) {
            float wave_sum=WaveActiveSum(sums[k]);
            if(WaveIsFirstLane())
                ppisp_partials[k][lane/wave_size]=wave_sum;
        }
        GroupMemoryBarrierWithGroupSync();
        if(lane<12) {
            float total=0.0f;
            [loop] for(uint i=0;i<64/wave_size;++i)
                total+=ppisp_partials[lane][i];
            output[cell*12+lane]=total;
        }
        return;
    }
    if(mode==7) {
        uint gl=pc.u3,gw=pc.u4,gh=pc.u5,rows=pc.u6;
        uint total=rows*gl*gh*gw*12;
        if(pixel>=total) return;
        uint c=pixel%12,cell=pixel/12;
        uint x=cell%gw,y=(cell/gw)%gh,z=(cell/(gw*gh))%gl;
        float value=source[pixel],g=0.0f;
        float nx=gw>1?pf(pc.u10)*2.0f/(float)(rows*gl*gh*(pc.u8!=0?gw:gw-1)*12):0.0f;
        float ny=gh>1?pf(pc.u10)*2.0f/(float)(rows*gl*(gh-1)*gw*12):0.0f;
        float nz=gl>1?pf(pc.u10)*2.0f/(float)(rows*(gl-1)*gh*gw*12):0.0f;
        if(gw>1 && (pc.u8!=0 || x>0))
            g+=(value-source[pixel+(x>0?-12:(int)(gw-1)*12)])*nx;
        if(gw>1 && (pc.u8!=0 || x+1<gw))
            g+=(value-source[pixel+(x+1<gw?12:-(int)(gw-1)*12)])*nx;
        if(gh>1 && y>0)g+=(value-source[pixel-gw*12])*ny;
        if(gh>1 && y+1<gh)g+=(value-source[pixel+gw*12])*ny;
        if(gl>1 && z>0)g+=(value-source[pixel-gh*gw*12])*nz;
        if(gl>1 && z+1<gl)g+=(value-source[pixel+gh*gw*12])*nz;
        output[pixel]+=g;
        return;
    }
    if(mode==8) {
        uint gl=pc.u3,gw=pc.u4,gh=pc.u5,rows=pc.u6;
        if(pixel>=rows*gl*gw*gh*12) return;
        uint cells=gl*gw*gh,row=pixel/(cells*12),c=pixel%12;
        float identity=(c==0 || c==5 || c==10)?1.0f:0.0f;
        float mean=identity;
        if(pc.u12!=0)mean=parameters[row*12+c];
        float value=source[pixel]-(mean-identity);
        float limit=pf(pc.u13);
        output[pixel]=limit>0.0f
            ? clamp(value,identity-limit,identity+limit):value;
        return;
    }
    if(mode==14) {
        uint gl=pc.u3,gw=pc.u4,gh=pc.u5,rows=pc.u6;
        uint row_channel=group_id.x,lane=lane_id.x,cells=gl*gw*gh;
        if(row_channel>=rows*12) return;
        uint row=row_channel/12,c=row_channel%12;
        float sum=0.0f;
        [loop] for(uint i=lane;i<cells;i+=64)
            sum+=source[(row*cells+i)*12+c];
        ppisp_partials[0][lane]=sum;
        GroupMemoryBarrierWithGroupSync();
        if(lane==0) {
            float total=0.0f;
            [unroll] for(uint i=0;i<64;++i)total+=ppisp_partials[0][i];
            output[row_channel]=total/(float)cells;
        }
        return;
    }
}
