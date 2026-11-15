RWStructuredBuffer<float> means : register(u0);
StructuredBuffer<float> scales : register(t1);
StructuredBuffer<float> rotations : register(t2);
StructuredBuffer<float> opacity : register(t3);
StructuredBuffer<float> visibility : register(t4);
StructuredBuffer<int> radii : register(t5);
struct Push { uint count, seed; float scaler; uint has_radii; uint o0,o1,o2,o3,o4,o5; };
[[vk::push_constant]] Push pc;
uint hash(uint x) { x^=x>>16; x*=0x7feb352d; x^=x>>15; x*=0x846ca68b; x^=x>>16; return x; }
float normal(uint i,uint axis) {
    float u1=(float(hash(i*3+axis+pc.seed*17))+1.0)/4294967297.0;
    float u2=(float(hash(i*7+axis+pc.seed*29))+1.0)/4294967297.0;
    return sqrt(-2*log(max(u1,1e-12)))*cos(6.283185307179586*u2);
}
[numthreads(256,1,1)]
void main(uint3 group:SV_GroupID,uint lane:SV_GroupIndex) {
    uint i=(group.y*65535+group.x)*256+lane;
    if(i>=pc.count || !(visibility[pc.o4+i]>0) || (pc.has_radii && radii[pc.o5+i]<=0)) return;
    float alpha=1/(1+exp(-opacity[pc.o3+i]));
    float rate=min(.05*pc.scaler*sqrt(2*log(max(255*alpha,1)))*pow(1-alpha,150),1);
    if(!(rate>0)) return;
    float3 local;
    for(uint d=0;d<3;++d) local[d]=exp(scales[pc.o1+3*i+d])*rate*normal(i,d);
    uint at=pc.o2+4*i;
    float4 q=float4(rotations[at],rotations[at+1],rotations[at+2],rotations[at+3]);
    q*=rsqrt(max(dot(q,q),1e-20));
    float3 t=2*cross(q.yzw,local);
    float3 world=local+q.x*t+cross(q.yzw,t);
    for(uint d=0;d<3;++d) means[pc.o0+3*i+d]+=world[d];
}
