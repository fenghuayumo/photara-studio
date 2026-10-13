[[vk::binding(0,0)]] StructuredBuffer<float> values;
[[vk::binding(1,0)]] StructuredBuffer<int> offsets;
[[vk::binding(2,0)]] StructuredBuffer<int> indices;
[[vk::binding(3,0)]] RWStructuredBuffer<float> output;
struct Push { uint mode,count,first,nx,ny,nz; float ox,oy,oz,size,threshold; uint padding; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

// Same symmetric Jacobi solver and eigenvalue floor as native simplification.
void eigen(float3x3 input, out float3 eval, out float3x3 vectors) {
    float3x3 a=input;
    vectors=float3x3(1,0,0,0,1,0,0,0,1);
    [loop] for(int iter=0;iter<24;++iter) {
        int p=0,q=1; float v=abs(a[0][1]);
        if(abs(a[0][2])>v) {p=0;q=2;v=abs(a[0][2]);}
        if(abs(a[1][2])>v) {p=1;q=2;v=abs(a[1][2]);}
        if(v<1e-12f) break;
        float tau=(a[q][q]-a[p][p])/(2*a[p][q]);
        float t=(tau<0?-1:1)/(abs(tau)+sqrt(1+tau*tau));
        float c=rsqrt(1+t*t),s=t*c;
        [unroll] for(int k=0;k<3;++k) if(k!=p && k!=q) {
            float ap=a[k][p],aq=a[k][q];
            a[k][p]=a[p][k]=c*ap-s*aq; a[k][q]=a[q][k]=s*ap+c*aq;
        }
        float app=a[p][p],aqq=a[q][q],apq=a[p][q];
        a[p][p]=c*c*app-2*s*c*apq+s*s*aqq;
        a[q][q]=s*s*app+2*s*c*apq+c*c*aqq;
        a[p][q]=a[q][p]=0;
        [unroll] for(int k=0;k<3;++k) {
            float vp=vectors[k][p],vq=vectors[k][q];
            vectors[k][p]=c*vp-s*vq; vectors[k][q]=s*vp+c*vq;
        }
    }
    eval=max(float3(a[0][0],a[1][1],a[2][2]),1e-8f);
}
float3x3 cov(uint i) {
    return float3x3(values[i+4],values[i+7],values[i+8],values[i+7],values[i+5],values[i+9],values[i+8],values[i+9],values[i+6]);
}
float3 vec(uint i) {return float3(values[i],values[i+1],values[i+2]);}
float3x3 outer(float3 d) {return float3x3(d.x*d,d.y*d,d.z*d);}
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint i=id.x+id.y*65535u*64u;
    if(i>=pc.count) return;
    if(pc.mode==0) {
        uint a=16u*uint(offsets[2*i]),b=16u*uint(offsets[2*i+1]);
        float ma=values[a+3],mb=values[b+3],m=ma+mb;
        float3 pa=vec(a),pb=vec(b),mu=(ma*pa+mb*pb)/m;
        float3 da=pa-mu,db=pb-mu;
        float3x3 ca=cov(a),cb=cov(b),c=(ma*(ca+outer(da))+mb*(cb+outer(db)))/m;
        float3 ev; float3x3 r; eigen(c,ev,r);
        float3x3 inv=mul(mul(r,float3x3(1/ev.x,0,0,0,1/ev.y,0,0,0,1/ev.z)),transpose(r));
        float logdet=log(ev.x*ev.y*ev.z);
        float ta=0,tb=0;
        [unroll] for(int row=0;row<3;++row) [unroll] for(int col=0;col<3;++col) {
            ta+=inv[row][col]*ca[row][col]; tb+=inv[row][col]*cb[row][col];
        }
        float ka=0.5f*(ta+dot(da,mul(inv,da))-3+logdet-values[a+10]);
        float kb=0.5f*(tb+dot(db,mul(inv,db))-3+logdet-values[b+10]);
        float3 color=vec(a+11)-vec(b+11);
        output[i]=ma*ka+mb*kb+8*dot(color,color);
    } else { // Gaussian density occupancy over spatially binned candidates.
        uint voxel=pc.first+i;
        uint x=voxel%pc.nx,y=(voxel/pc.nx)%pc.ny,z=voxel/(pc.nx*pc.ny);
        uint bx=(pc.nx+7)/8,by=(pc.ny+7)/8;
        uint block=x/8+bx*(y/8+by*(z/8));
        float3 p=float3(pc.ox,pc.oy,pc.oz)+(float3(x,y,z)+0.5f)*pc.size;
        float transmittance=1;
        [loop] for(int j=offsets[block];j<offsets[block+1];++j) {
            uint g=16u*uint(indices[j]);
            float3 d=p-vec(g);
            float3 local=float3(dot(vec(g+4),d),dot(vec(g+8),d),dot(vec(g+12),d));
            float d2=dot(local,local);
            if(d2>9) continue;
            float alpha=values[g+3]*exp(-0.5f*d2);
            if(alpha<1.0f/255.0f) continue;
            transmittance*=1-alpha;
            if(1-transmittance>=pc.threshold) break;
        }
        output[i]=1-transmittance;
    }
}
