[[vk::binding(0,0)]] StructuredBuffer<int> nodes;
[[vk::binding(1,0)]] StructuredBuffer<float> values;
[[vk::binding(2,0)]] StructuredBuffer<float> bounds;
[[vk::binding(3,0)]] RWStructuredBuffer<float> points;
[[vk::binding(4,0)]] StructuredBuffer<float> occupancy;
[[vk::binding(5,0)]] RWStructuredBuffer<float> gradients;
struct Push { int root; uint count; uint neighbors; uint refine; float iso; float min_norm; float step; uint padding; };
[[vk::push_constant]] ConstantBuffer<Push> pc;
bool greater(float a,int ai,float b,int bi) { return a>b || (a==b && ai>bi); }
void sift(inout float ds[64],inout int ids[64],int count) {
    int p=0;
    [loop] while(2*p+1<count) {
        int c=2*p+1;
        if(c+1<count && greater(ds[c+1],ids[c+1],ds[c],ids[c])) ++c;
        if(!greater(ds[c],ids[c],ds[p],ids[p])) break;
        float d=ds[p]; ds[p]=ds[c]; ds[c]=d;
        int n=ids[p]; ids[p]=ids[c]; ids[c]=n; p=c;
    }
}
[numthreads(64,1,1)]
void main(uint3 id:SV_DispatchThreadID) {
    uint i=id.x+id.y*65535u*64u;
    if(i>=pc.count) return;
    float3 q=float3(points[3*i],points[3*i+1],points[3*i+2]);
    float ds[64]; int ids[64]; int pending[64]; float planes[64];
    int count=0, size=1;
    pending[0]=pc.root; planes[0]=-1.0f;
    [loop] while(size>0) {
        --size; int node=pending[size];
        if(node<0 || (count==int(pc.neighbors) && planes[size]>ds[0])) continue;
        if(count==int(pc.neighbors)) {
            uint b=6u*uint(node);
            float3 low=float3(bounds[b],bounds[b+1],bounds[b+2]);
            float3 high=float3(bounds[b+3],bounds[b+4],bounds[b+5]);
            float3 d=max(max(low-q,q-high),0.0f);
            float lower=d.x*d.x+(d.y*d.y+d.z*d.z);
            if(lower>ds[0]+16.0f*1.192092896e-7f*(1.0f+ds[0])) continue;
        }
        int n=nodes[4*node]; uint g=19u*uint(n);
        float3 d=q-float3(values[g],values[g+1],values[g+2]);
        precise float dist=d.x*d.x+(d.y*d.y+d.z*d.z);
        if(count<int(pc.neighbors)) {
            int p=count++; ds[p]=dist; ids[p]=n;
            [loop] while(p>0) {
                int parent=(p-1)/2;
                if(!greater(ds[p],ids[p],ds[parent],ids[parent])) break;
                float v=ds[parent]; ds[parent]=ds[p]; ds[p]=v;
                int j=ids[parent]; ids[parent]=ids[p]; ids[p]=j; p=parent;
            }
        } else if(greater(ds[0],ids[0],dist,n)) { ds[0]=dist; ids[0]=n; sift(ds,ids,count); }
        int axis=nodes[4*node+3]; float delta=d[axis];
        int near_child=nodes[4*node+(delta<0?1:2)];
        int far_child=nodes[4*node+(delta<0?2:1)];
        if(far_child>=0) { pending[size]=far_child; planes[size++]=delta*delta; }
        if(near_child>=0) { pending[size]=near_child; planes[size++]=-1.0f; }
    }
    int ordered[64]; int found=count;
    [loop] while(count>0) {
        ordered[count-1]=ids[0]; --count;
        if(count>0) { ds[0]=ds[count]; ids[0]=ids[count]; sift(ds,ids,count); }
    }
    float3 result=0.0f;
    [loop] for(int j=0;j<found;++j) {
        uint g=19u*uint(ordered[j]);
        float3 d=q-float3(values[g],values[g+1],values[g+2]);
        // Match CPU/CUDA's explicit three-component reduction at the
        // half-space boundary. FMA contraction can turn an exact zero into
        // a negative value and incorrectly drop the Gaussian contribution.
        precise float sidedness=values[g+3]*d.x+(values[g+4]*d.y+values[g+5]*d.z);
        if(sidedness<0.0f) continue;
        float3 local=float3(values[g+9]*d.x+(values[g+10]*d.y+values[g+11]*d.z),
            values[g+12]*d.x+(values[g+13]*d.y+values[g+14]*d.z),
            values[g+15]*d.x+(values[g+16]*d.y+values[g+17]*d.z));
        float3 inv=float3(values[g+6],values[g+7],values[g+8]);
        float exponent=-0.5f*(local.x*local.x*inv.x+(local.y*local.y*inv.y+local.z*local.z*inv.z));
        float value=clamp(values[g+18]*exp(exponent),0.0f,1.0f-1.0e-7f);
        float weight=value/(1.0f-value+1.0e-8f); float3 v=local*inv;
        result+=weight*float3(values[g+9]*v.x+(values[g+12]*v.y+values[g+15]*v.z),
            values[g+10]*v.x+(values[g+13]*v.y+values[g+16]*v.z),
            values[g+11]*v.x+(values[g+14]*v.y+values[g+17]*v.z));
    }
    if(pc.refine==1u) {
        float norm=result.x*result.x+(result.y*result.y+result.z*result.z);
        if(norm>pc.min_norm) q+=pc.step*clamp((occupancy[i]-pc.iso)/norm,-1.0f,1.0f)*result;
        points[3*i]=q.x; points[3*i+1]=q.y; points[3*i+2]=q.z;
    } else { gradients[3*i]=result.x; gradients[3*i+1]=result.y; gradients[3*i+2]=result.z; }
}
