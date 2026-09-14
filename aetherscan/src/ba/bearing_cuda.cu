#include "ba/bearing_cuda.hpp"
#include <cuda_runtime.h>
#include <cusolverDn.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <limits>

namespace aetherscan::ba {
namespace {
void check(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
void check(cusolverStatus_t e) { if(e!=CUSOLVER_STATUS_SUCCESS) throw std::runtime_error("Bearing cuSolver error "+std::to_string(int(e))); }
template<class T> struct Buffer {
    T* p{}; std::size_t n{};
    Buffer()=default;
    Buffer(const Buffer&)=delete;
    Buffer& operator=(const Buffer&)=delete;
    ~Buffer(){if(p) cudaFree(p);}
    void resize(std::size_t size){if(size==n)return; if(p)check(cudaFree(p));p=nullptr;n=0;
        if(size){check(cudaMalloc(reinterpret_cast<void**>(&p),size*sizeof(T)));n=size;}}
    void upload(const T* src,std::size_t size){resize(size);if(size)check(cudaMemcpy(p,src,size*sizeof(T),cudaMemcpyHostToDevice));}
    void download(T* dst)const{if(n)check(cudaMemcpy(dst,p,n*sizeof(T),cudaMemcpyDeviceToHost));}
};
struct Obs {unsigned camera,point; double d[3];};
struct Linear {double h[9],b[3];};
unsigned blocks(std::size_t n){return static_cast<unsigned>((n+255)/256);}

__global__ void linearize(const double* cameras,const double* points,const Obs* obs,
    const double* weights,int n,double huber,Linear* lin,double* costs) {
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    const Obs o=obs[i]; double u[3],s=0;
    for(int k=0;k<3;++k){u[k]=points[3*o.point+k]-cameras[3*o.camera+k];s+=u[k]*u[k];}
    const double length=sqrt(s);
    if(!(length>1e-10)||!isfinite(length)){costs[i]=INFINITY;return;}
    double r2=0,residual[3];
    for(int k=0;k<3;++k){u[k]/=length;residual[k]=o.d[k]-u[k];r2+=residual[k]*residual[k];}
    const double r=sqrt(r2),w=weights[i]*(r>huber ? huber/r:1.0);
    costs[i]=weights[i]*(r>huber ? huber*(r-.5*huber):.5*r2);
    if(!lin)return;
    // Do not simplify P^T P to P or P(d-u) to d-u(u.d): those identities
    // lose relative precision in nearly unobservable directions.
    double projection[9];
    for(int r=0;r<3;++r)for(int c=0;c<3;++c)
        projection[r*3+c]=((r==c?1.0:0.0)-u[r]*u[c])/length;
    for(int r=0;r<3;++r){double b=0;
        for(int k=0;k<3;++k)b+=projection[k*3+r]*residual[k];lin[i].b[r]=w*b;
        for(int c=0;c<3;++c){double h=0;
            for(int k=0;k<3;++k)h+=projection[k*3+r]*projection[k*3+c];lin[i].h[r*3+c]=w*h;}}
}
__global__ void sum_cost(const double* costs,int n,const double* cameras,unsigned first,
    unsigned second,double baseline,double* out) {
    __shared__ double part[256]; double v=0;
    for(int i=threadIdx.x;i<n;i+=256)v+=costs[i];
    part[threadIdx.x]=v;__syncthreads();
    for(int s=128;s;s/=2){if(threadIdx.x<s)part[threadIdx.x]+=part[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0){double d=0;for(int k=0;k<3;++k){double x=cameras[3*first+k]-cameras[3*second+k];d+=x*x;}
        double r=sqrt(d)-baseline;*out=part[0]+.5*r*r;}
}
__device__ bool inverse3(const double* a,double* out){
    const double x=a[4]*a[8]-a[5]*a[7],y=a[2]*a[7]-a[1]*a[8],z=a[1]*a[5]-a[2]*a[4];
    const double det=a[0]*x+a[3]*y+a[6]*z;
    if(!(det>0)||!isfinite(det))return false;
    out[0]=x/det;out[1]=y/det;out[2]=z/det;
    out[3]=y/det;out[4]=(a[0]*a[8]-a[2]*a[6])/det;out[5]=(a[2]*a[3]-a[0]*a[5])/det;
    out[6]=z/det;out[7]=out[5];out[8]=(a[0]*a[4]-a[1]*a[3])/det;return true;
}
__global__ void points_inverse(const Linear* lin,const unsigned* offsets,const unsigned* indices,
    int n,double damping,const double* scale,double* inverse,double* rhs,int* failure) {
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;
    double h[9]={},b[3]={};
    for(unsigned j=offsets[p];j<offsets[p+1];++j){const Linear l=lin[indices[j]];
        for(int k=0;k<9;++k)h[k]+=l.h[k];for(int k=0;k<3;++k)b[k]+=l.b[k];}
    for(int k=0;k<3;++k)h[k*3+k]+=damping*fmin(fmax(h[k*3+k],1e-6*scale[3*p+k]),1e32*scale[3*p+k]);
    double inv[9]={};if(!inverse3(h,inv))atomicExch(failure,1);
    for(int k=0;k<9;++k)inverse[p*9+k]=inv[k];
    for(int k=0;k<3;++k)rhs[p*3+k]=b[k];
}
__global__ void eliminate(const Linear* lin,const Obs* obs,const double* inverse,int n,double* e){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    for(int r=0;r<3;++r)for(int c=0;c<3;++c){double v=0;
        for(int k=0;k<3;++k)v+=lin[i].h[r*3+k]*inverse[obs[i].point*9+k*3+c];e[i*9+r*3+c]=v;}
}
__global__ void camera_blocks(const Linear* lin,const Obs* obs,const double* e,const double* pb,
    const unsigned* offsets,const unsigned* indices,int n,double damping,const double* scale,
    const double* cameras,unsigned first,unsigned second,double* matrix,double* rhs){
    int cam=blockIdx.x*blockDim.x+threadIdx.x;if(cam>=n)return;
    double h[9]={},b[3]={};
    for(unsigned j=offsets[cam];j<offsets[cam+1];++j){const unsigned i=indices[j];
        for(int k=0;k<9;++k)h[k]+=lin[i].h[k];
        for(int r=0;r<3;++r){b[r]-=lin[i].b[r];for(int k=0;k<3;++k)b[r]+=e[i*9+r*3+k]*pb[obs[i].point*3+k];}}
    double u[3],length2=0;
    for(int k=0;k<3;++k){u[k]=cameras[second*3+k]-cameras[first*3+k];length2+=u[k]*u[k];}
    for(int r=0;r<3;++r){rhs[cam*3+r]=b[r];
        const double diagonal=h[r*3+r]+((cam==first||cam==second)?u[r]*u[r]/fmax(length2,1e-20):0);
        h[r*3+r]+=damping*fmin(fmax(diagonal,1e-6*scale[3*cam+r]),1e32*scale[3*cam+r]);
        for(int c=0;c<3;++c)matrix[(cam*3+r)*(n*3)+cam*3+c]=h[r*3+c];}
}
__global__ void schur(const Linear* lin,const Obs* obs,const double* e,
    const unsigned* offsets,const unsigned* indices,int nc,double* matrix){
    unsigned p=blockIdx.x,begin=offsets[p],count=offsets[p+1]-begin;
    for(std::size_t v=threadIdx.x;v<std::size_t(count)*count*9;v+=blockDim.x){
        unsigned a=indices[begin+v/(count*9)],b=indices[begin+(v/9)%count];int r=(v%9)/3,c=v%3;
        double value=0;for(int k=0;k<3;++k)value+=e[a*9+r*3+k]*lin[b].h[k*3+c];
        atomicAdd(matrix+(obs[a].camera*3+r)*(nc*3)+obs[b].camera*3+c,-value);
    }
}
__global__ void baseline_gauge(const double* cameras,int nc,unsigned anchor,unsigned first,unsigned second,
    double baseline,double* matrix,double* rhs){
    if(threadIdx.x||blockIdx.x)return;
    double u[3],length=0;for(int k=0;k<3;++k){u[k]=cameras[second*3+k]-cameras[first*3+k];length+=u[k]*u[k];}
    length=sqrt(length);for(int k=0;k<3;++k)u[k]/=fmax(length,1e-10);
    const int n=nc*3;
    for(int r=0;r<3;++r){rhs[first*3+r]+=u[r]*(length-baseline);rhs[second*3+r]-=u[r]*(length-baseline);
        for(int c=0;c<3;++c){double h=u[r]*u[c];
            matrix[(first*3+r)*n+first*3+c]+=h;matrix[(second*3+r)*n+second*3+c]+=h;
            matrix[(first*3+r)*n+second*3+c]-=h;matrix[(second*3+r)*n+first*3+c]-=h;}
    }
    for(int r=0;r<3;++r){int row=anchor*3+r;rhs[row]=0;
        for(int c=0;c<n;++c){matrix[row*n+c]=0;matrix[c*n+row]=0;}matrix[row*n+row]=1;}
}
// Ceres scales each column by 1/(1+sqrt(initial diagonal)) before clamping
// the LM diagonal. Express the same clamp in original parameter units.
__global__ void diagonal_scale(const Linear* lin,const unsigned* offsets,const unsigned* indices,
    int n,const double* cameras,unsigned first,unsigned second,bool camera_blocks,double* scale){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double diagonal[3]={};
    for(unsigned j=offsets[i];j<offsets[i+1];++j)for(int k=0;k<3;++k)diagonal[k]+=lin[indices[j]].h[k*3+k];
    if(camera_blocks&&(i==first||i==second)){
        double u[3],length2=0;
        for(int k=0;k<3;++k){u[k]=cameras[second*3+k]-cameras[first*3+k];length2+=u[k]*u[k];}
        for(int k=0;k<3;++k)diagonal[k]+=u[k]*u[k]/fmax(length2,1e-20);
    }
    for(int k=0;k<3;++k){const double s=1+sqrt(diagonal[k]);scale[3*i+k]=s*s;}
}
__global__ void update_points(const Linear* lin,const Obs* obs,const unsigned* offsets,const unsigned* indices,
    const double* inverse,const double* rhs,const double* step,const double* points,int n,double* candidate){
    int p=blockIdx.x*blockDim.x+threadIdx.x;if(p>=n)return;double b[3];
    for(int k=0;k<3;++k)b[k]=rhs[p*3+k];
    for(unsigned j=offsets[p];j<offsets[p+1];++j){unsigned i=indices[j];for(int r=0;r<3;++r)
        for(int k=0;k<3;++k)b[r]+=lin[i].h[r*3+k]*step[obs[i].camera*3+k];}
    for(int r=0;r<3;++r){double v=points[p*3+r];for(int k=0;k<3;++k)v+=inverse[p*9+r*3+k]*b[k];candidate[p*3+r]=v;}
}
__global__ void update_cameras(const double* camera,const double* step,int n,double* out){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i<n)out[i]=camera[i]+step[i];
}
// Predicted decrease of the undamped quadratic model, including both camera
// and point steps. LM must compare this with the actual robust cost decrease.
__global__ void model_decrease(const Linear* lin,const Obs* obs,int n,
    const double* step,const double* points,const double* candidate,double* values){
    int i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    double v[3],value=0;
    for(int k=0;k<3;++k)v[k]=candidate[3*obs[i].point+k]-points[3*obs[i].point+k]-step[3*obs[i].camera+k];
    for(int r=0;r<3;++r){value+=lin[i].b[r]*v[r];
        for(int c=0;c<3;++c)value-=.5*v[r]*lin[i].h[3*r+c]*v[c];}
    values[i]=value;
}
__global__ void sum_model(const double* values,int n,const double* cameras,
    const double* step,unsigned first,unsigned second,double baseline,double* out){
    __shared__ double part[256];double value=0;
    for(int i=threadIdx.x;i<n;i+=256)value+=values[i];
    part[threadIdx.x]=value;__syncthreads();
    for(int s=128;s;s/=2){if(threadIdx.x<s)part[threadIdx.x]+=part[threadIdx.x+s];__syncthreads();}
    if(threadIdx.x==0){double u[3],length=0,delta=0;
        for(int k=0;k<3;++k){u[k]=cameras[3*second+k]-cameras[3*first+k];length+=u[k]*u[k];}
        length=sqrt(length);
        for(int k=0;k<3;++k)delta+=u[k]/fmax(length,1e-10)*(step[3*second+k]-step[3*first+k]);
        *out=part[0]-(length-baseline)*delta-.5*delta*delta;
    }
}
}
class CudaBearingOptimizer::Impl {
public:
    int nc{},np{},no{},workspace_size{};unsigned anchor{},first{},second{};double baseline{},huber{};
    cusolverDnHandle_t solver{};
    Buffer<double> cameras,points,candidate_cameras,candidate_points,weights,costs,total,inverse,pb,e,matrix,rhs,workspace,camera_scale,point_scale;
    Buffer<Obs> obs; Buffer<Linear> lin;Buffer<unsigned> po,pi,co,ci;Buffer<int> info;
    ~Impl(){if(solver)cusolverDnDestroy(solver);}
    double cost(bool candidate,bool derivatives){
        const auto* c=candidate?candidate_cameras.p:cameras.p;const auto* p=candidate?candidate_points.p:points.p;
        linearize<<<blocks(no),256>>>(c,p,obs.p,weights.p,no,huber,derivatives?lin.p:nullptr,costs.p);
        sum_cost<<<1,256>>>(costs.p,no,c,first,second,baseline,total.p);
        check(cudaGetLastError());double result{};total.download(&result);return result;
    }
};
CudaBearingOptimizer::CudaBearingOptimizer(const BearingProblem& p):impl_(std::make_unique<Impl>()){
    auto& d=*impl_;
    if(p.cameras.empty()||p.cameras.size()>2000||p.points.empty()||p.observations.empty()||
        p.points.size()>std::numeric_limits<int>::max()/9||p.observations.size()>std::numeric_limits<int>::max()/9||
        p.anchor>=p.cameras.size()||p.baseline_first>=p.cameras.size()||p.baseline_second>=p.cameras.size()||
        p.baseline_first==p.baseline_second||!(p.baseline>1e-10)||!std::isfinite(p.baseline)||!(p.huber>0)||!std::isfinite(p.huber))
        throw std::invalid_argument("Unsupported CUDA bearing problem");
    for(const auto& c:p.cameras)for(double v:c)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite camera");
    for(const auto& x:p.points)for(double v:x)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite point");
    d.nc=static_cast<int>(p.cameras.size());d.np=static_cast<int>(p.points.size());d.no=static_cast<int>(p.observations.size());
    d.anchor=p.anchor;d.first=p.baseline_first;d.second=p.baseline_second;d.baseline=p.baseline;d.huber=p.huber;
    std::vector<Obs> obs;std::vector<unsigned> po(d.np+1),co(d.nc+1),pi(d.no),ci(d.no);
    for(const auto& o:p.observations){if(o.camera>=p.cameras.size()||o.point>=p.points.size())throw std::invalid_argument("Invalid bearing observation");
        for(double v:o.direction)if(!std::isfinite(v))throw std::invalid_argument("Nonfinite bearing");
        obs.push_back({o.camera,o.point,{o.direction[0],o.direction[1],o.direction[2]}});++po[o.point+1];++co[o.camera+1];}
    std::partial_sum(po.begin(),po.end(),po.begin());std::partial_sum(co.begin(),co.end(),co.begin());auto pc=po,cc=co;
    for(unsigned i=0;i<obs.size();++i){pi[pc[obs[i].point]++]=i;ci[cc[obs[i].camera]++]=i;}
    d.po.upload(po.data(),po.size());d.pi.upload(pi.data(),pi.size());d.co.upload(co.data(),co.size());d.ci.upload(ci.data(),ci.size());
    d.obs.upload(obs.data(),obs.size());d.cameras.upload(p.cameras.front().data(),d.nc*3);d.points.upload(p.points.front().data(),d.np*3);
    d.candidate_cameras.resize(d.nc*3);d.candidate_points.resize(d.np*3);d.lin.resize(d.no);d.costs.resize(d.no);d.total.resize(1);
    d.camera_scale.resize(d.nc*3);d.point_scale.resize(d.np*3);
    d.inverse.resize(d.np*9);d.pb.resize(d.np*3);d.e.resize(d.no*9);d.rhs.resize(d.nc*3);d.info.resize(2);
    d.matrix.resize(std::size_t(d.nc*3)*(d.nc*3));check(cusolverDnCreate(&d.solver));
    check(cusolverDnDpotrf_bufferSize(d.solver,CUBLAS_FILL_MODE_LOWER,d.nc*3,d.matrix.p,d.nc*3,&d.workspace_size));
    d.workspace.resize(d.workspace_size);
}
CudaBearingOptimizer::~CudaBearingOptimizer()=default;
void CudaBearingOptimizer::download(BearingProblem& p)const{
    auto& d=*impl_;if(p.cameras.size()!=d.nc||p.points.size()!=d.np)throw std::invalid_argument("Bearing download shape changed");
    d.cameras.download(p.cameras.front().data());d.points.download(p.points.front().data());
}
BearingSolveSummary CudaBearingOptimizer::solve(const std::vector<double>& weights,unsigned iterations,double tolerance,double max_seconds){
    auto& d=*impl_;if(weights.size()!=d.no)throw std::invalid_argument("Bearing weight count mismatch");
    if(!std::isfinite(tolerance)||tolerance<0||!std::isfinite(max_seconds)||max_seconds<0)
        throw std::invalid_argument("Invalid bearing solve tolerance or time limit");
    for(double w:weights)if(!std::isfinite(w)||w<0)throw std::invalid_argument("Invalid bearing weight");
    d.weights.upload(weights.data(),weights.size());BearingSolveSummary result;
    auto start=std::chrono::steady_clock::now();double current=d.cost(false,true),lambda=1e-4,rejection_factor=2;
    result.initial_cost=current;if(!std::isfinite(current))return result;result.usable=true;
    diagonal_scale<<<blocks(d.nc),256>>>(d.lin.p,d.co.p,d.ci.p,d.nc,d.cameras.p,d.first,d.second,true,d.camera_scale.p);
    diagonal_scale<<<blocks(d.np),256>>>(d.lin.p,d.po.p,d.pi.p,d.np,d.cameras.p,d.first,d.second,false,d.point_scale.p);
    for(unsigned it=0;it<iterations;++it){
        if(max_seconds>0&&std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()>max_seconds)break;
        check(cudaMemset(d.info.p,0,2*sizeof(int)));
        points_inverse<<<blocks(d.np),256>>>(d.lin.p,d.po.p,d.pi.p,d.np,lambda,d.point_scale.p,d.inverse.p,d.pb.p,d.info.p+1);
        eliminate<<<blocks(d.no),256>>>(d.lin.p,d.obs.p,d.inverse.p,d.no,d.e.p);
        check(cudaMemset(d.matrix.p,0,d.matrix.n*sizeof(double)));
        camera_blocks<<<blocks(d.nc),256>>>(d.lin.p,d.obs.p,d.e.p,d.pb.p,d.co.p,d.ci.p,d.nc,lambda,d.camera_scale.p,d.cameras.p,d.first,d.second,d.matrix.p,d.rhs.p);
        schur<<<d.np,256>>>(d.lin.p,d.obs.p,d.e.p,d.po.p,d.pi.p,d.nc,d.matrix.p);
        baseline_gauge<<<1,1>>>(d.cameras.p,d.nc,d.anchor,d.first,d.second,d.baseline,d.matrix.p,d.rhs.p);
        check(cudaGetLastError());
        check(cusolverDnDpotrf(d.solver,CUBLAS_FILL_MODE_LOWER,d.nc*3,d.matrix.p,d.nc*3,d.workspace.p,d.workspace_size,d.info.p));
        int info[2]{};d.info.download(info);++result.iterations;
        if(info[0]<0)throw std::runtime_error("CUDA bearing factorization argument error");
        if(info[0]||info[1]){lambda*=rejection_factor;rejection_factor*=2;if(lambda>1e16){result.usable=false;break;}continue;}
        check(cusolverDnDpotrs(d.solver,CUBLAS_FILL_MODE_LOWER,d.nc*3,1,d.matrix.p,d.nc*3,d.rhs.p,d.nc*3,d.info.p));
        d.info.download(info);if(info[0])throw std::runtime_error("CUDA bearing triangular solve failed");
        update_cameras<<<blocks(d.nc*3),256>>>(d.cameras.p,d.rhs.p,d.nc*3,d.candidate_cameras.p);
        update_points<<<blocks(d.np),256>>>(d.lin.p,d.obs.p,d.po.p,d.pi.p,d.inverse.p,d.pb.p,d.rhs.p,d.points.p,d.np,d.candidate_points.p);
        model_decrease<<<blocks(d.no),256>>>(d.lin.p,d.obs.p,d.no,d.rhs.p,d.points.p,d.candidate_points.p,d.costs.p);
        sum_model<<<1,256>>>(d.costs.p,d.no,d.cameras.p,d.rhs.p,d.first,d.second,d.baseline,d.total.p);
        check(cudaGetLastError());double predicted{};d.total.download(&predicted);
        double candidate=d.cost(true,false);
        if(std::isfinite(candidate)&&std::abs(current-candidate)<=tolerance*current)break;
        const double quality=predicted>0 ? (current-candidate)/predicted : -1;
        if(std::isfinite(candidate)&&quality>1e-3){current=candidate;
            std::swap(d.cameras.p,d.candidate_cameras.p);std::swap(d.points.p,d.candidate_points.p);
            lambda=std::max(1e-16,lambda*std::max(1.0/3.0,1-std::pow(2*quality-1,3)));rejection_factor=2;
            current=d.cost(false,true);
        }else{lambda*=rejection_factor;rejection_factor*=2;if(lambda>1e16)break;}
    }
    result.final_cost=current;result.seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();return result;
}
}
