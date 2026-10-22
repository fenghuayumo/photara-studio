#include "geometry_cuda.hpp"
#include "core/camera_projection.hpp"
#include <cuda_runtime.h>
#include <cmath>
#include <mutex>
#include <stdexcept>
#include <string>

namespace photara::sfm::gpu {
namespace {
void check(cudaError_t status) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string("SfM CUDA geometry: ") + cudaGetErrorString(status));
}
unsigned grid_blocks(const std::size_t work,const unsigned per_block) {
    const std::size_t blocks=(work+per_block-1)/per_block;
    if(blocks>0xffffffffull) throw std::runtime_error("SfM CUDA geometry: batch too large");
    return static_cast<unsigned>(blocks);
}

// Persistent device storage. The SfM stages are called repeatedly with
// similarly sized batches; allocating and destroying hundreds of megabytes per
// call dominated the measured runtime on Berlin/London. Buffers only grow.
struct Device {
    struct BufferBase {
        void* pointer{};
        std::size_t bytes{};
    };
    template<class T> struct Buffer : BufferBase {
        T* get(const std::size_t count) {
            const std::size_t needed = count * sizeof(T);
            if (needed > bytes) {
                if (pointer) check(cudaFree(pointer));
                pointer = nullptr;
                bytes = 0;
                check(cudaMalloc(&pointer, needed));
                bytes = needed;
            }
            return static_cast<T*>(pointer);
        }
    };
    std::mutex mutex;
    cudaStream_t stream{};
    Buffer<RayPair> pair_rays;
    Buffer<PairJob> pair_jobs;
    Buffer<PairResult> pair_results;
    Buffer<View> views;
    Buffer<Observation> observations;
    Buffer<TrackJob> jobs;
    Buffer<TrackResult> results;
    Buffer<unsigned char> inliers;

    Device() { check(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking)); }
};

Device& device() {
    // Intentionally leaked: CUDA teardown may run before static destructors in
    // host processes, and freeing context-owned allocations then is unsafe.
    static Device* instance = new Device();
    return *instance;
}

template<class T>
void upload(T* destination, const std::vector<T>& source, cudaStream_t stream) {
    if (!source.empty())
        check(cudaMemcpyAsync(destination, source.data(), source.size() * sizeof(T),
            cudaMemcpyHostToDevice, stream));
}
template<class T>
void download(std::vector<T>& destination, const T* source, cudaStream_t stream) {
    if (!destination.empty())
        check(cudaMemcpyAsync(destination.data(), source,
            destination.size() * sizeof(T), cudaMemcpyDeviceToHost, stream));
}

__device__ V3 add(V3 a,V3 b) { return {a.x+b.x,a.y+b.y,a.z+b.z}; }
__device__ V3 sub(V3 a,V3 b) { return {a.x-b.x,a.y-b.y,a.z-b.z}; }
__device__ V3 mul(V3 a,double s) { return {a.x*s,a.y*s,a.z*s}; }
__device__ double dot(V3 a,V3 b) { return a.x*b.x+a.y*b.y+a.z*b.z; }
__device__ double norm(V3 a) { return sqrt(dot(a,a)); }
__device__ bool finite(V3 a) { return isfinite(a.x)&&isfinite(a.y)&&isfinite(a.z); }
__device__ V3 unit(V3 a) { return mul(a,1/norm(a)); }
__device__ V3 center(const View& v) { return {v.C[0],v.C[1],v.C[2]}; }
__device__ V3 rotate(const View& v,V3 a) {
    return {v.R[0]*a.x+v.R[1]*a.y+v.R[2]*a.z,
            v.R[3]*a.x+v.R[4]*a.y+v.R[5]*a.z,
            v.R[6]*a.x+v.R[7]*a.y+v.R[8]*a.z};
}
__device__ V3 transpose_rotate(const View& v,V3 a) {
    return {v.R[0]*a.x+v.R[3]*a.y+v.R[6]*a.z,
            v.R[1]*a.x+v.R[4]*a.y+v.R[7]*a.z,
            v.R[2]*a.x+v.R[5]*a.y+v.R[8]*a.z};
}
__device__ void outer_add(double* a,V3 b) {
    double v[3]{b.x,b.y,b.z};
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) a[r*3+c]+=v[r]*v[c];
}
// Guarded elimination. Ill-conditioned cases retain the CPU LDLT/SVD path.
__device__ bool solve(double* matrix,V3 rhs,V3& result) {
    double a[3][4]; double b[3]{rhs.x,rhs.y,rhs.z}; double scale=0;
    for(int r=0;r<3;++r) for(int c=0;c<3;++c) { a[r][c]=matrix[r*3+c]; scale=fmax(scale,fabs(a[r][c])); }
    for(int r=0;r<3;++r) a[r][3]=b[r];
    if (!(scale>0) || !isfinite(scale)) return false;
    for(int c=0;c<3;++c) {
        int pivot=c;
        for(int r=c+1;r<3;++r) if(fabs(a[r][c])>fabs(a[pivot][c])) pivot=r;
        if(fabs(a[pivot][c])<scale*1e-10) return false;
        for(int k=c;k<4;++k) { double tmp=a[c][k]; a[c][k]=a[pivot][k]; a[pivot][k]=tmp; }
        for(int r=c+1;r<3;++r) {
            double factor=a[r][c]/a[c][c];
            for(int k=c;k<4;++k) a[r][k]-=factor*a[c][k];
        }
    }
    double x[3]{};
    for(int r=2;r>=0;--r) { double t=a[r][3]; for(int c=r+1;c<3;++c) t-=a[r][c]*x[c]; x[r]=t/a[r][r]; }
    result={x[0],x[1],x[2]}; return finite(result);
}
__device__ bool near(double a,double b) {
    return fabs(a-b)<=1e-10*(1+fabs(b));
}

__global__ void pair_kernel(const RayPair* rays,const PairJob* jobs,PairResult* out,std::size_t n) {
    std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    const auto& ray=rays[i]; const auto& job=jobs[ray.job]; PairResult result{};
    V3 b1=ray.first,b2=ray.second,C=center(job.view);
    if(!finite(b1)||!finite(b2)) { result.uncertain=1; out[i]=result; return; }
    double ep1=dot(b1,job.epipole1),ep2=dot(b2,job.epipole2);
    if(job.epipole && (near(ep1,job.cos_epipole)||near(ep2,job.cos_epipole))) result.uncertain=1;
    if(job.epipole && (ep1>job.cos_epipole||ep2>job.cos_epipole)) { out[i]=result; return; }
    V3 d1=unit(b1),d2=unit(transpose_rotate(job.view,b2));
    double matrix[9]{1,0,0,0,1,0,0,0,1};
    for(int k=0;k<9;++k) matrix[k]*=2;
    double products[9]{}; outer_add(products,d1); outer_add(products,d2);
    for(int k=0;k<9;++k) matrix[k]-=products[k];
    V3 X;
    if(!solve(matrix,sub(C,mul(d2,dot(d2,C))),X)) { result.uncertain=1; out[i]=result; return; }
    V3 X2=rotate(job.view,sub(X,C)); double n1=norm(X),n2=norm(X2);
    if(n1<1e-12||n2<1e-12) { result.uncertain=1; out[i]=result; return; }
    double e1=dot(b1,mul(X,1/n1)),e2=dot(b2,mul(X2,1/n2));
    double angle=dot(unit(X),unit(sub(X,C)));
    if(near(e1,0)||near(e2,0)||(job.reprojection&&(near(e1,job.cos_reproj)||near(e2,job.cos_reproj)))||
        (job.angle&&near(angle,job.min_cos_angle))) result.uncertain=1;
    if(!isfinite(e1)||!isfinite(e2)||!isfinite(angle)) result.uncertain=1;
    result.accepted=e1>0&&e2>0&&(!job.reprojection||(e1>=job.cos_reproj&&e2>=job.cos_reproj))&&(!job.angle||angle<=job.min_cos_angle);
    float delta=static_cast<float>(1-fmin(e1,e2));
    result.weight=1.f/fmaxf(delta,1e-10f); result.cosine=angle;
    out[i]=result;
}

__device__ double error(const View& view,const Observation& o,V3 position,bool checked) {
    V3 x=rotate(view,sub(position,center(view)));
    if(checked&&x.z<=1e-8) return INFINITY;
    auto p=project_camera_plane(static_cast<CameraModel>(view.model),x.x/x.z,x.y/x.z,
        view.k1,view.k2,view.p1,view.p2);
    double u=view.fx*p.x+view.cx-o.u,v=view.fy*p.y+view.cy-o.v;
    return sqrt(u*u+v*v);
}

// 256-bit support for the 255-observation batch limit, matching the CPU's
// ordered inlier index vectors.
struct Mask { unsigned long long word[4]{}; };
__device__ bool test(const Mask& m,unsigned i) { return (m.word[i>>6]>>(i&63))&1ull; }
__device__ void set(Mask& m,unsigned i) { m.word[i>>6]|=1ull<<(i&63); }
__device__ unsigned count(const Mask& m) {
    return static_cast<unsigned>(__popcll(m.word[0])+__popcll(m.word[1])+
        __popcll(m.word[2])+__popcll(m.word[3]));
}
__device__ Mask classify(const View* views,const Observation* obs,unsigned n,V3 p,
    double threshold,unsigned& uncertain) {
    Mask mask;
    for(unsigned i=0;i<n;++i) {
        double e=error(views[obs[i].view],obs[i],p,true);
        if(near(e,threshold)) uncertain=1;
        if(isfinite(e)&&e<=threshold) set(mask,i);
    }
    return mask;
}
__device__ float angle(const View* views,const Observation* obs,unsigned n,const Mask& mask,V3 p) {
    double cosine=1;
    for(unsigned i=0;i<n;++i) if(test(mask,i)) {
        V3 ci=center(views[obs[i].view]);
        for(unsigned j=i+1;j<n;++j) if(test(mask,j)) {
            V3 cj=center(views[obs[j].view]);
            cosine=fmin(cosine,dot(unit(sub(p,ci)),unit(sub(p,cj))));
        }
    }
    return static_cast<float>(acos(fmax(-1.,fmin(1.,cosine)))*57.2957795130823208768);
}
__device__ bool linear(const View* views,const Observation* obs,unsigned n,const Mask& mask,V3& p) {
    double H[9]{}; V3 rhs{};
    for(unsigned i=0;i<n;++i) if(test(mask,i)) {
        const auto& o=obs[i]; const auto& view=views[o.view]; V3 b=o.bearing;
        V3 rows[2]{transpose_rotate(view,{0,-b.z,b.y}),transpose_rotate(view,{b.z,0,-b.x})};
        for(auto a:rows) { outer_add(H,a); rhs=add(rhs,mul(a,dot(a,center(view)))); }
    }
    return solve(H,rhs,p);
}
__device__ bool refine(const View* views,const Observation* obs,unsigned n,const Mask& mask,
    V3& p,unsigned iterations) {
    double lambda=1e-3;
    for(unsigned it=0;it<iterations;++it) {
        double H[9]{}; V3 g{}; double cost=0;
        for(unsigned i=0;i<n;++i) if(test(mask,i)) {
            const auto& o=obs[i]; const auto& v=views[o.view];
            V3 x=rotate(v,sub(p,center(v)));
            if(x.z<=1e-8) return false;
            auto q=project_camera_plane(static_cast<CameraModel>(v.model),x.x/x.z,x.y/x.z,
                v.k1,v.k2,v.p1,v.p2);
            double u=v.fx*q.x+v.cx-o.u,w=v.fy*q.y+v.cy-o.v;
            cost+=u*u+w*w;
            V3 ju=transpose_rotate(v,{v.fx/x.z,0,-v.fx*x.x/(x.z*x.z)});
            V3 jv=transpose_rotate(v,{0,v.fy/x.z,-v.fy*x.y/(x.z*x.z)});
            outer_add(H,ju); outer_add(H,jv); g=add(g,add(mul(ju,u),mul(jv,w)));
        }
        H[0]+=lambda; H[4]+=lambda; H[8]+=lambda;
        V3 step; if(!solve(H,mul(g,-1),step)) return false;
        V3 next=add(p,step); double next_cost=0;
        for(unsigned i=0;i<n;++i) if(test(mask,i)) {
            double e=error(views[obs[i].view],obs[i],next,true); next_cost+=e*e;
        }
        if(!isfinite(next_cost)||!(next_cost<cost)) { lambda*=10; if(lambda>1e8) break; continue; }
        p=next; lambda=fmax(1e-8,lambda*.3); if(norm(step)<1e-8) break;
    }
    return finite(p);
}
// 0 = success, 1 = rejected (support dropped below min_inliers),
// 2 = guarded linear solve failed and the CPU path must decide.
__device__ int local_optimize(const View* views,const Observation* obs,unsigned n,
    Mask& mask,V3& p,const TriangulationConfig& cfg,unsigned& uncertain) {
    for(int pass=0;pass<2;++pass) {
        if(!linear(views,obs,n,mask,p)) return 2;
        if(cfg.refine) refine(views,obs,n,mask,p,cfg.refine_iterations);
        mask=classify(views,obs,n,p,cfg.threshold,uncertain);
        if(count(mask)<cfg.min_inliers) return 1;
    }
    return 0;
}

struct Candidate {
    V3 position{};
    Mask mask;
};

__device__ bool linear_consensus(const View* views,const Observation* obs,unsigned n,
    const TriangulationConfig& cfg,Candidate& out,unsigned& uncertain) {
    Mask all;
    for(unsigned i=0;i<n;++i) set(all,i);
    if(!linear(views,obs,n,all,out.position)) return false;
    out.mask=classify(views,obs,n,out.position,cfg.threshold,uncertain);
    if(count(out.mask)<cfg.min_inliers) return false;
    const int status=local_optimize(views,obs,n,out.mask,out.position,cfg,uncertain);
    if(status==2) { uncertain=1; return false; } // numeric guard: host falls back to CPU
    if(status!=0||count(out.mask)<cfg.min_inliers) return false;
    // triangulate_linear_consensus rejects below the angle threshold before
    // apply_consensus_to_track refines once more and checks it again.
    float a=angle(views,obs,n,out.mask,out.position);
    if(near(a,cfg.min_angle)) uncertain=1;
    if(a<cfg.min_angle) return false;
    if(cfg.refine) {
        if(!refine(views,obs,n,out.mask,out.position,cfg.refine_iterations)) {
            uncertain=1; return false;
        }
        a=angle(views,obs,n,out.mask,out.position);
        if(near(a,cfg.min_angle)) uncertain=1;
        if(a<cfg.min_angle) return false;
    }
    return true;
}

__global__ void triangulate_kernel(const View* views,const Observation* observations,
    const TrackJob* jobs,TriangulationConfig cfg,TrackResult* output,
    unsigned char* inliers,std::size_t n) {
    std::size_t i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    const auto job=jobs[i]; const auto* obs=observations+job.begin; TrackResult r{};
    Candidate result; unsigned uncertain=0;
    const bool ok=linear_consensus(views,obs,job.count,cfg,result,uncertain);
    if(!ok||uncertain||count(result.mask)<cfg.min_inliers) {
        // Leave the track to the CPU implementation, including its historical
        // handling of partial LO-RANSAC failures and angle rejection.
        r.fallback=1; output[i]=r; return;
    }
    r.position=result.position; r.inliers=count(result.mask);
    for(unsigned j=0;j<job.count;++j) inliers[job.begin+j]=test(result.mask,j)?1:0;
    output[i]=r;
}

constexpr int kFilterWarps=4;
constexpr int kFilterThreads=kFilterWarps*32;
constexpr unsigned kMaxObservations=255;
__device__ double warp_sum(double value) {
    for(int offset=16;offset>0;offset>>=1) value+=__shfl_down_sync(0xffffffffu,value,offset);
    return value;
}
__device__ double warp_min(double value) {
    for(int offset=16;offset>0;offset>>=1) value=fmin(value,__shfl_down_sync(0xffffffffu,value,offset));
    return value;
}
__device__ unsigned warp_sum(unsigned value) {
    for(int offset=16;offset>0;offset>>=1) value+=__shfl_down_sync(0xffffffffu,value,offset);
    return value;
}
__global__ void filter_kernel(const View* views,const Observation* observations,
    const TrackJob* jobs,double threshold,double min_angle,TrackResult* output,
    unsigned char* inliers,std::size_t n) {
    // Four warps per block, one warp per track. Phase one classifies one
    // observation per lane; phase two computes the pair-wise parallax with the
    // ray directions cached in shared memory instead of renormalizing per pair.
    __shared__ unsigned char kept[kFilterWarps][kMaxObservations+1];
    __shared__ double rays[kFilterWarps][(kMaxObservations+1)*3];
    const int warp=threadIdx.x>>5, lane=threadIdx.x&31;
    const std::size_t i=static_cast<std::size_t>(blockIdx.x)*kFilterWarps+warp;
    if(i>=n) return;
    const auto job=jobs[i]; const auto* obs=observations+job.begin;
    auto* local_kept=kept[warp]; auto* local_rays=rays[warp];
    unsigned local_inliers=0, local_uncertain=0;
    double sum_error=0,sum_cosine=0,sum_distance=0;
    for(unsigned j=lane;j<job.count;j+=32) {
        const auto& o=obs[j]; const auto& view=views[o.view];
        V3 x=rotate(view,sub(job.position,center(view)));
        const double length=norm(x);
        const double cosine=dot(o.bearing,x)/length;
        const double e=error(view,o,job.position,false);
        if(near(cosine,o.minimum_cosine)||near(e,threshold)) local_uncertain=1;
        const bool keep=length>1e-12&&!(cosine<o.minimum_cosine)&&
            isfinite(e)&&e<=threshold;
        local_kept[j]=keep?1:0; inliers[job.begin+j]=keep?1:0;
        if(!keep) continue;
        ++local_inliers; sum_error+=e; sum_cosine+=cosine; sum_distance+=length;
        const V3 ray=unit(sub(job.position,center(view)));
        local_rays[j*3+0]=ray.x; local_rays[j*3+1]=ray.y; local_rays[j*3+2]=ray.z;
    }
    __syncwarp();
    double best_cos=1;
    for(unsigned j=lane;j<job.count;j+=32) {
        if(!local_kept[j]) continue;
        for(unsigned k=j+1;k<job.count;++k) {
            if(!local_kept[k]) continue;
            best_cos=fmin(best_cos,
                local_rays[j*3+0]*local_rays[k*3+0]+
                local_rays[j*3+1]*local_rays[k*3+1]+
                local_rays[j*3+2]*local_rays[k*3+2]);
        }
    }
    const unsigned inlier_count=warp_sum(local_inliers);
    sum_error=warp_sum(sum_error); sum_cosine=warp_sum(sum_cosine);
    sum_distance=warp_sum(sum_distance); best_cos=warp_min(best_cos);
    const bool uncertain=__any_sync(0xffffffffu,local_uncertain!=0);
    if(lane==0) {
        TrackResult r{}; r.position=job.position;
        const float a=static_cast<float>(acos(fmax(-1.,fmin(1.,best_cos)))*
            57.2957795130823208768);
        if(fabs(a-min_angle)<1e-5) r.fallback=1;
        if(uncertain) r.fallback=1;
        r.inliers=inlier_count;
        if(inlier_count<2||a<min_angle) r.inliers=0;
        if(r.inliers) { r.sum_error=sum_error; r.sum_cosine=sum_cosine; r.distance=sum_distance; }
        output[i]=r;
    }
}
} // namespace

bool cuda_available() noexcept { int count=0; return cudaGetDeviceCount(&count)==cudaSuccess&&count>0; }

void cuda_filter_pairs(const std::vector<RayPair>& rays,const std::vector<PairJob>& jobs,
    std::vector<PairResult>& results) {
    results.resize(rays.size()); if(rays.empty()) return;
    auto& d=device(); std::lock_guard lock(d.mutex);
    RayPair* device_rays=d.pair_rays.get(rays.size());
    PairJob* device_jobs=d.pair_jobs.get(jobs.size());
    PairResult* device_results=d.pair_results.get(results.size());
    upload(device_rays,rays,d.stream); upload(device_jobs,jobs,d.stream);
    pair_kernel<<<grid_blocks(rays.size(),256),256,0,d.stream>>>(
        device_rays,device_jobs,device_results,rays.size());
    check(cudaGetLastError());
    download(results,device_results,d.stream);
    check(cudaStreamSynchronize(d.stream));
}

void cuda_triangulate(const std::vector<View>& views,
    const std::vector<Observation>& observations,const std::vector<TrackJob>& jobs,
    const TriangulationConfig& config,std::vector<TrackResult>& results,
    std::vector<unsigned char>& inliers) {
    results.resize(jobs.size()); inliers.resize(observations.size()); if(jobs.empty()) return;
    auto& d=device(); std::lock_guard lock(d.mutex);
    View* device_views=d.views.get(views.size());
    Observation* device_observations=d.observations.get(observations.size());
    TrackJob* device_jobs=d.jobs.get(jobs.size());
    TrackResult* device_results=d.results.get(results.size());
    unsigned char* device_inliers=d.inliers.get(inliers.size());
    upload(device_views,views,d.stream); upload(device_observations,observations,d.stream);
    upload(device_jobs,jobs,d.stream);
    triangulate_kernel<<<grid_blocks(jobs.size(),128),128,0,d.stream>>>(
        device_views,device_observations,device_jobs,config,
        device_results,device_inliers,jobs.size());
    check(cudaGetLastError());
    download(results,device_results,d.stream); download(inliers,device_inliers,d.stream);
    check(cudaStreamSynchronize(d.stream));
}

void cuda_filter_tracks(const std::vector<View>& views,
    const std::vector<Observation>& observations,const std::vector<TrackJob>& jobs,
    double threshold,double min_angle,std::vector<TrackResult>& results,
    std::vector<unsigned char>& inliers) {
    results.resize(jobs.size()); inliers.resize(observations.size()); if(jobs.empty()) return;
    auto& d=device(); std::lock_guard lock(d.mutex);
    View* device_views=d.views.get(views.size());
    Observation* device_observations=d.observations.get(observations.size());
    TrackJob* device_jobs=d.jobs.get(jobs.size());
    TrackResult* device_results=d.results.get(results.size());
    unsigned char* device_inliers=d.inliers.get(inliers.size());
    upload(device_views,views,d.stream); upload(device_observations,observations,d.stream);
    upload(device_jobs,jobs,d.stream);
    filter_kernel<<<grid_blocks(jobs.size(),kFilterWarps),kFilterThreads,0,d.stream>>>(
        device_views,device_observations,device_jobs,threshold,min_angle,
        device_results,device_inliers,jobs.size());
    check(cudaGetLastError());
    download(results,device_results,d.stream); download(inliers,device_inliers,d.stream);
    check(cudaStreamSynchronize(d.stream));
}
} // namespace photara::sfm::gpu
