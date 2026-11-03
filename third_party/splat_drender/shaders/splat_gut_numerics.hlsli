#ifndef SPLAT_GUT_NUMERICS_HLSLI
#define SPLAT_GUT_NUMERICS_HLSLI
// Vulkan implementation of CUDA-compatible float operations.
#define GUT_INLINE
float gut_fma(float a,float b,float c) { precise float r=mad(a,b,c); return r; }
float gut_mul(float a,float b) { precise float r=a*b; return r; }
float gut_add(float a,float b) { precise float r=a+b; return r; }
float gut_sqrt(float a) { precise float r=sqrt(a); return r; }
float gut_rsqrt(float a) { precise float r=rsqrt(a); return r; }
float gut_exp2(float a) { precise float r=exp2(a); return r; }
float gut_rcp(float a) { precise float r=rcp(a); return r; }
GUT_INLINE float gut_exp(float a) { return gut_exp2(gut_mul(a,1.4426950408889634074f)); }
GUT_INLINE float gut_dot(float3 a,float3 b) {
    return gut_fma(a.z,b.z,gut_fma(a.x,b.x,gut_mul(a.y,b.y)));
}
GUT_INLINE float gut_dot4(float w,float x,float y,float z) {
    return gut_fma(z,z,gut_fma(y,y,gut_fma(w,w,gut_mul(x,x))));
}
GUT_INLINE float gut_div(float a,float b) { return gut_mul(a,gut_rcp(b)); }
// CUDA norm3df/rnorm3df scale by an exponent-derived power of two.
float gut_scaled_norm_square(float3 v,out float scale,out float largest,out float total,out uint exponent) {
    float3 a=abs(v);
    total=gut_add(gut_add(a.x,a.y),a.z);
    float lo=min(a.x,a.y),ab=max(a.x,a.y),mid=min(ab,a.z);
    largest=max(ab,a.z);
    exponent=asuint(largest)&0xfe000000u;
    scale=asfloat(exponent^0x7e800000u);
    float x=gut_mul(mid,scale),y=gut_mul(lo,scale),z=gut_mul(largest,scale);
    return gut_fma(z,z,gut_fma(y,y,gut_mul(x,x)));
}
float gut_rnorm(float3 v) {
    float scale,largest,total;uint exponent;
    float square=gut_scaled_norm_square(v,scale,largest,total,exponent);
    if(isnan(total)) return total;
    if(isinf(largest)) return 0.0f;
    return gut_mul(gut_rsqrt(square),scale);
}
GUT_INLINE float gut_norm(float3 v) {
    float scale,largest,total;uint exponent;
    float square=gut_scaled_norm_square(v,scale,largest,total,exponent);
    if(isinf(largest)) return largest;
    return total>largest ? gut_mul(gut_sqrt(square),asfloat(exponent|0x00800000u)) : total;
}
GUT_INLINE float gut_gaussian(float3 q) { return gut_exp(gut_mul(-0.5f,gut_dot(q,q))); }
GUT_INLINE float gut_affine(float3 v,float a,float b,float c,float t) {
    return gut_add(gut_fma(c,v.z,gut_fma(a,v.x,gut_mul(b,v.y))),t);
}
struct GutRotation { float m[3][3]; };
GUT_INLINE GutRotation gut_rotation(float w,float x,float y,float z) {
    GutRotation r;
    float yy=gut_mul(y,y),zz=gut_mul(z,z);
    float xy=gut_mul(x,y),xz=gut_mul(x,z),yz=gut_mul(y,z);
    float wx=gut_mul(w,x),wy=gut_mul(w,y),wz=gut_mul(w,z);
    float a=gut_add(yy,zz),b=gut_fma(x,x,zz),c=gut_fma(x,x,yy);
    r.m[0][0]=gut_add(1.0f,-gut_add(a,a));
    r.m[0][1]=gut_mul(2.0f,gut_add(xy,wz));
    r.m[0][2]=gut_mul(2.0f,gut_add(xz,-wy));
    r.m[1][0]=gut_mul(2.0f,gut_add(xy,-wz));
    r.m[1][1]=gut_add(1.0f,-gut_add(b,b));
    r.m[1][2]=gut_mul(2.0f,gut_add(wx,yz));
    r.m[2][0]=gut_mul(2.0f,gut_add(wy,xz));
    r.m[2][1]=gut_mul(2.0f,gut_add(yz,-wx));
    r.m[2][2]=gut_add(1.0f,-gut_add(c,c));
    return r;
}
#define GUT_VEC3 float3
#define GUT_VEC4 float4
GUT_INLINE float3 gut_xyz(float4 v) { return GUT_VEC3(v.x,v.y,v.z); }
GUT_INLINE float3 gut_unit_direction(float3 v) {
    float inv=gut_rnorm(v);
    return GUT_VEC3(gut_mul(v.x,inv),gut_mul(v.y,inv),gut_mul(v.z,inv));
}
GUT_INLINE float3 gut_unit_direction_vjp(float3 original,float3 gradient) {
    float inv=gut_rnorm(original);
    float3 d=gut_unit_direction(original);
    float pull=gut_dot(d,gradient);
    return GUT_VEC3(gut_mul(inv,gut_fma(-d.x,pull,gradient.x)),
                    gut_mul(inv,gut_fma(-d.y,pull,gradient.y)),
                    gut_mul(inv,gut_fma(-d.z,pull,gradient.z)));
}
GUT_INLINE float4 gut_unit_quaternion(float4 q) {
    float q2=gut_dot4(q.x,q.y,q.z,q.w);
    float inv=gut_rsqrt(q2>1.0e-20f ? q2 : 1.0e-20f);
    return GUT_VEC4(gut_mul(q.x,inv),gut_mul(q.y,inv),gut_mul(q.z,inv),gut_mul(q.w,inv));
}
GUT_INLINE float4 gut_quaternion_vjp(float4 raw,float4 gradient) {
    float n=gut_dot4(raw.x,raw.y,raw.z,raw.w);
    float inv=gut_rsqrt(n>1.0e-20f ? n : 1.0e-20f);
    float4 q=gut_unit_quaternion(raw);
    float pull=gut_fma(q.x,gradient.x,gut_fma(q.y,gradient.y,gut_fma(q.z,gradient.z,gut_mul(q.w,gradient.w))));
    return GUT_VEC4(gut_mul(inv,gut_fma(-q.x,pull,gradient.x)),
                    gut_mul(inv,gut_fma(-q.y,pull,gradient.y)),
                    gut_mul(inv,gut_fma(-q.z,pull,gradient.z)),
                    gut_mul(inv,gut_fma(-q.w,pull,gradient.w)));
}
struct GutGeometryGradient { float3 mean; float3 scale; float4 rotation; };
GUT_INLINE GutGeometryGradient gut_geometry_vjp(float3 b0,float3 b1,float3 b2,
    float3 go,float3 row0,float3 row1,float3 row2,float3 offset,
    float3 scale,float modifier,float4 q) {
    float rows[3][3];
    rows[0][0]=gut_fma(go.x,offset.x,row0.x);rows[0][1]=gut_fma(go.x,offset.y,row0.y);rows[0][2]=gut_fma(go.x,offset.z,row0.z);
    rows[1][0]=gut_fma(go.y,offset.x,row1.x);rows[1][1]=gut_fma(go.y,offset.y,row1.y);rows[1][2]=gut_fma(go.y,offset.z,row1.z);
    rows[2][0]=gut_fma(go.z,offset.x,row2.x);rows[2][1]=gut_fma(go.z,offset.y,row2.y);rows[2][2]=gut_fma(go.z,offset.z,row2.z);
    GutGeometryGradient result;
    result.mean=GUT_VEC3(-gut_dot(GUT_VEC3(b0.x,b1.x,b2.x),go),
                         -gut_dot(GUT_VEC3(b0.y,b1.y,b2.y),go),
                         -gut_dot(GUT_VEC3(b0.z,b1.z,b2.z),go));
    result.scale=GUT_VEC3(gut_div(-gut_dot(GUT_VEC3(rows[0][0],rows[0][1],rows[0][2]),b0),scale.x),
                          gut_div(-gut_dot(GUT_VEC3(rows[1][0],rows[1][1],rows[1][2]),b1),scale.y),
                          gut_div(-gut_dot(GUT_VEC3(rows[2][0],rows[2][1],rows[2][2]),b2),scale.z));
    float inverse[3]={gut_rcp(gut_mul(scale.x,modifier)),gut_rcp(gut_mul(scale.y,modifier)),gut_rcp(gut_mul(scale.z,modifier))};
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) rows[r][c]=gut_mul(rows[r][c],inverse[r]);
    float w=q.x,x=q.y,y=q.z,z=q.w;
    float rw=gut_mul(2.0f,gut_fma(z,gut_add(rows[0][1],-rows[1][0]),
        gut_fma(y,gut_add(rows[2][0],-rows[0][2]),gut_mul(x,gut_add(rows[1][2],-rows[2][1])))));
    float rx=gut_mul(2.0f,gut_fma(y,gut_add(rows[0][1],rows[1][0]),
        gut_fma(z,gut_add(rows[0][2],rows[2][0]),gut_mul(w,gut_add(rows[1][2],-rows[2][1])))));
    float ry=gut_mul(2.0f,gut_fma(x,gut_add(rows[0][1],rows[1][0]),
        gut_fma(w,gut_add(rows[2][0],-rows[0][2]),gut_mul(z,gut_add(rows[1][2],rows[2][1])))));
    float rz=gut_mul(2.0f,gut_fma(w,gut_add(rows[0][1],-rows[1][0]),
        gut_fma(x,gut_add(rows[0][2],rows[2][0]),gut_mul(y,gut_add(rows[1][2],rows[2][1])))));
    result.rotation=GUT_VEC4(rw,
        gut_add(rx,-gut_mul(gut_mul(4.0f,x),gut_add(rows[1][1],rows[2][2]))),
        gut_add(ry,-gut_mul(gut_mul(4.0f,y),gut_add(rows[0][0],rows[2][2]))),
        gut_add(rz,-gut_mul(gut_mul(4.0f,z),gut_add(rows[0][0],rows[1][1]))));
    return result;
}
#undef GUT_VEC3
#undef GUT_VEC4
#undef GUT_INLINE
#endif
