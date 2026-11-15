StructuredBuffer<float> means : register(t0);
StructuredBuffer<float> sh : register(t1);
StructuredBuffer<float> camera : register(t2);
RWByteAddressBuffer accumulation : register(u3);
RWStructuredBuffer<float> output : register(u4);
struct Push {
    uint count, width, height, bases, degree, stage, model;
    float size, near_clip, br, bg, bb;
    uint o0, o1, o2, o3, o4;
};
[[vk::push_constant]] Push pc;
void add(uint index, float value) {
    uint expected = accumulation.Load(4 * index), observed;
    for (;;) {
        accumulation.InterlockedCompareExchange(4 * index, expected, asuint(asfloat(expected) + value), observed);
        if (observed == expected) return;
        expected = observed;
    }
}
float3 rgb(uint i, float3 mean) {
    float3 direction = mean - float3(camera[pc.o2+16],camera[pc.o2+17],camera[pc.o2+18]);
    direction *= rsqrt(max(dot(direction,direction),1e-12));
    float x=direction.x,y=direction.y,z=direction.z,xx=x*x,yy=y*y,zz=z*z;
    float b[16];
    b[0]=0.28209479177387814;
    b[1]=-0.4886025119029199*y;b[2]=0.4886025119029199*z;b[3]=-0.4886025119029199*x;
    b[4]=1.0925484305920792*x*y;b[5]=-1.0925484305920792*y*z;
    b[6]=0.31539156525252005*(2*zz-xx-yy);b[7]=-1.0925484305920792*x*z;b[8]=0.5462742152960396*(xx-yy);
    b[9]=-0.5900435899266435*y*(3*xx-yy);b[10]=2.890611442640554*x*y*z;
    b[11]=-0.4570457994644658*y*(4*zz-xx-yy);b[12]=0.3731763325901154*z*(2*zz-3*xx-3*yy);
    b[13]=-0.4570457994644658*x*(4*zz-xx-yy);b[14]=1.445305721320277*z*(xx-yy);b[15]=-0.5900435899266435*x*(xx-3*yy);
    float3 value=0.5;
    for(uint d=0;d<min(pc.bases,(pc.degree+1)*(pc.degree+1));++d) {
        uint at=pc.o1+(i*pc.bases+d)*3;
        value+=b[d]*float3(sh[at],sh[at+1],sh[at+2]);
    }
    return max(value,0);
}
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    uint i=(group.y*65535+group.x)*256+lane,pixels=pc.width*pc.height;
    if(pc.stage==1) {
        if(i>=pixels) return;
        float weight=max(asfloat(accumulation.Load(4*(pc.o3+3*pixels+i))),0);
        float coverage=min(weight,1),scale=weight>1e-6?coverage/weight:0;
        float3 bg=float3(pc.br,pc.bg,pc.bb);
        for(uint c=0;c<3;++c) output[pc.o4+c*pixels+i]=min(asfloat(accumulation.Load(4*(pc.o3+c*pixels+i)))*scale+bg[c]*(1-coverage),1);
        return;
    }
    if(i>=pc.count) return;
    uint at=pc.o0+3*i;
    float3 m=float3(means[at],means[at+1],means[at+2]);
    float3 p;
    for(uint c=0;c<3;++c) p[c]=camera[pc.o2+c]*m.x+camera[pc.o2+4+c]*m.y+camera[pc.o2+8+c]*m.z+camera[pc.o2+12+c];
    if(any(isnan(p))||any(isinf(p))) return;
    float fx=camera[pc.o2+19],fy=camera[pc.o2+20],cx=camera[pc.o2+21],cy=camera[pc.o2+22];
    float2 uv;
    if(pc.model==3) {
        float r=length(p); if(r<=1e-12) return;
        uv=float2((atan2(p.x,p.z)/6.28318530718+0.5)*pc.width,(asin(clamp(p.y/r,-1,1))/3.14159265359+0.5)*pc.height);
    } else if(pc.model==1) {
        float r=length(p.xy),theta=atan2(r,p.z),t2=theta*theta;
        if(p.z<=1e-8) return;
        float td=theta*(1+t2*(camera[pc.o2+23]+t2*(camera[pc.o2+24]+t2*(camera[pc.o2+25]+t2*camera[pc.o2+26]))));
        if(td<0) return;
        uv=float2(fx,fy)*p.xy*(r>1e-12?td/r:0)+float2(cx,cy);
    } else {
        if(p.z<=pc.near_clip) return;
        uv=float2(fx,fy)*p.xy*(pc.model==4?1:1/p.z)+float2(cx,cy);
    }
    if(any(isnan(uv))||any(isinf(uv)) || uv.x < -pc.size-1 || uv.y < -pc.size-1 || uv.x > pc.width+pc.size || uv.y > pc.height+pc.size) return;
    int2 center=int2(floor(uv));
    int radius=int(ceil(pc.size));float3 color=rgb(i,m);
    int2 lo=max(center-radius,int2(0,0)),hi=min(center+radius,int2(pc.width-1,pc.height-1));
    for(int y=lo.y;y<=hi.y;++y) for(int x=lo.x;x<=hi.x;++x) {
        float2 delta=float2(x-center.x,y-center.y);
        if(dot(delta,delta)>pc.size*pc.size) continue;
        uint pix=y*pc.width+x;
        for(uint c=0;c<3;++c) add(pc.o3+c*pixels+pix,color[c]);
        add(pc.o3+3*pixels+pix,1);
    }
}
