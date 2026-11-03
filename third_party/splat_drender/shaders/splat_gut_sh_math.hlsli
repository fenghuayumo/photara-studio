#ifndef SPLAT_DRENDER_GUT_SH_H
#define SPLAT_DRENDER_GUT_SH_H
#include "splat_gut_numerics.hlsli"
#define GUT_SH_INLINE
#define GUT_SH_UNROLL [unroll]
struct GutShBasis { float value[16]; float dx[16]; float dy[16]; float dz[16]; };
GUT_SH_INLINE GutShBasis gut_sh_basis(int degree,float x,float y,float z) {
    GutShBasis b;
    GUT_SH_UNROLL for(int k=0;k<16;++k) { b.value[k]=0.0f;b.dx[k]=0.0f;b.dy[k]=0.0f;b.dz[k]=0.0f; }
    b.value[0]=0.28209479177387814f;
    const float c1=0.4886025119029199f;
    if(degree>0) {
        b.value[1]=gut_mul(-c1,y);b.dy[1]=-c1;
        b.value[2]=gut_mul(c1,z);b.dz[2]=c1;
        b.value[3]=gut_mul(-c1,x);b.dx[3]=-c1;
    }
    if(degree>1) {
        float xx=gut_mul(x,x),yy=gut_mul(y,y),zz=gut_mul(z,z);
        float xy=gut_mul(x,y),yz=gut_mul(y,z),xz=gut_mul(x,z);
        const float c2[5]={1.0925484305920792f,-1.0925484305920792f,0.31539156525252005f,-1.0925484305920792f,0.5462742152960396f};
        float q2[5]={xy,yz,gut_add(gut_add(gut_mul(2.0f,zz),-xx),-yy),xz,gut_add(xx,-yy)};
        float ux[5]={y,0,-gut_mul(2.0f,x),z,gut_mul(2.0f,x)};
        float uy[5]={x,z,-gut_mul(2.0f,y),0,-gut_mul(2.0f,y)};
        float uz[5]={0,y,gut_mul(4.0f,z),x,0};
        GUT_SH_UNROLL for(int k=0;k<5;++k) {
            b.value[4+k]=gut_mul(c2[k],q2[k]);b.dx[4+k]=gut_mul(c2[k],ux[k]);
            b.dy[4+k]=gut_mul(c2[k],uy[k]);b.dz[4+k]=gut_mul(c2[k],uz[k]);
        }
        if(degree>2) {
            const float c3[7]={-0.5900435899266435f,2.890611442640554f,-0.4570457994644658f,
                0.3731763325901154f,-0.4570457994644658f,1.445305721320277f,-0.5900435899266435f};
            float q3[7]={gut_mul(y,gut_add(gut_mul(3.0f,xx),-yy)),gut_mul(xy,z),
                gut_mul(y,gut_add(gut_add(gut_mul(4.0f,zz),-xx),-yy)),
                gut_mul(z,gut_add(gut_add(gut_mul(2.0f,zz),-gut_mul(3.0f,xx)),-gut_mul(3.0f,yy))),
                gut_mul(x,gut_add(gut_add(gut_mul(4.0f,zz),-xx),-yy)),
                gut_mul(z,gut_add(xx,-yy)),gut_mul(x,gut_add(xx,-gut_mul(3.0f,yy)))};
            float vx[7]={gut_mul(6.0f,xy),yz,-gut_mul(2.0f,xy),-gut_mul(6.0f,xz),
                gut_add(gut_add(-gut_mul(3.0f,xx),gut_mul(4.0f,zz)),-yy),gut_mul(2.0f,xz),gut_mul(3.0f,gut_add(xx,-yy))};
            float vy[7]={gut_mul(3.0f,gut_add(xx,-yy)),xz,
                gut_add(gut_add(-gut_mul(3.0f,yy),gut_mul(4.0f,zz)),-xx),-gut_mul(6.0f,yz),
                -gut_mul(2.0f,xy),-gut_mul(2.0f,yz),-gut_mul(6.0f,xy)};
            float vz[7]={0,xy,gut_mul(8.0f,yz),gut_mul(3.0f,gut_add(gut_add(gut_mul(2.0f,zz),-xx),-yy)),
                gut_mul(8.0f,xz),gut_add(xx,-yy),0};
            GUT_SH_UNROLL for(int k=0;k<7;++k) {
                b.value[9+k]=gut_mul(c3[k],q3[k]);b.dx[9+k]=gut_mul(c3[k],vx[k]);
                b.dy[9+k]=gut_mul(c3[k],vy[k]);b.dz[9+k]=gut_mul(c3[k],vz[k]);
            }
        }
    }
    return b;
}
#undef GUT_SH_INLINE
#undef GUT_SH_UNROLL
#endif
