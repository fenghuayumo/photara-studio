struct Push {
    uint sw, sh, width, height;
    uint start, count;
    float shift, metric;
    float fx, fy, cx, cy, threshold;
};
[[vk::push_constant]] Push pc;
[[vk::binding(0,0)]] StructuredBuffer<float> points_in;
[[vk::binding(1,0)]] StructuredBuffer<float> normals_in;
[[vk::binding(2,0)]] StructuredBuffer<float> mask_in;
// Eight floats/pixel: depth, xyz, normal xyz, validity.
[[vk::binding(3,0)]] RWStructuredBuffer<float> result;
float sample_map(uint which, uint channels, uint c, uint2 pixel) {
    if(pc.sw==pc.width && pc.sh==pc.height) {
        uint index=(pixel.y*pc.sw+pixel.x)*channels+c;
        return which==0?points_in[index]:(which==1?normals_in[index]:mask_in[index]);
    }
    precise float2 pos = max(0.0, (float2(pixel)+0.5)*float2(pc.sw,pc.sh)/float2(pc.width,pc.height)-0.5);
    uint2 lo = uint2(pos), hi = min(lo+1, uint2(pc.sw-1,pc.sh-1));
    float2 t = pos-float2(lo);
    uint a=(lo.y*pc.sw+lo.x)*channels+c, b=(lo.y*pc.sw+hi.x)*channels+c;
    uint d=(hi.y*pc.sw+lo.x)*channels+c, e=(hi.y*pc.sw+hi.x)*channels+c;
    float va=which==0?points_in[a]:(which==1?normals_in[a]:mask_in[a]);
    float vb=which==0?points_in[b]:(which==1?normals_in[b]:mask_in[b]);
    float vd=which==0?points_in[d]:(which==1?normals_in[d]:mask_in[d]);
    float ve=which==0?points_in[e]:(which==1?normals_in[e]:mask_in[e]);
    precise float top=va*(1-t.x)+vb*t.x, bottom=vd*(1-t.x)+ve*t.x;
    precise float v=top*(1-t.y)+bottom*t.y;
    return v;
}
[numthreads(256,1,1)]
void main(uint3 tid:SV_DispatchThreadID) {
    uint i=tid.x; if(i>=pc.count)return;
    uint index=pc.start+i; uint2 xy=uint2(index%pc.width,index/pc.width);
    float3 p,n;
    for(uint c=0;c<3;++c) { p[c]=sample_map(0,3,c,xy); n[c]=sample_map(1,3,c,xy); }
    float logit=sample_map(2,1,0,xy);
    float z=exp(p.z), e=exp(-abs(logit));
    float probability=logit>=0?1/(1+e):e/(1+e);
    // Scaled normalization avoids overflow/underflow of dot(n,n).
    float scale=max(abs(n.x),max(abs(n.y),abs(n.z)));
    float norm=scale>0?length(n/scale):0;
    float depth=(z+pc.shift)*pc.metric;
    float3 xyz_out=float3((float2(xy)-float2(pc.cx,pc.cy))/float2(pc.fx,pc.fy)*depth,depth);
    bool valid=all(isfinite(float3(p.xy*z,z))) && z>0 && all(isfinite(n)) && scale>0 && norm>0 &&
        scale*norm>1e-12 && isfinite(logit) && probability>pc.threshold && depth>0 && all(isfinite(xyz_out));
    for(uint c=0;c<8;++c)result[i*8+c]=0;
    if(!valid)return;
    result[i*8]=depth;
    for(uint c=0;c<3;++c) { result[i*8+1+c]=xyz_out[c]; result[i*8+4+c]=(n[c]/scale)/norm; }
    result[i*8+7]=1;
}
