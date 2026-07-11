#include "ba/optimizer.hpp"

#include "ba/linearizer.hpp"
#include "reprojection_detail.cuh"

#include <cuda_runtime.h>
#include <cooperative_groups.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <utility>
#include <vector>

namespace aetherscan::ba {
namespace {

constexpr unsigned threads = 256;
constexpr std::size_t cross_n = 18;

void check(const cudaError_t status, const char* operation) {
    if (status == cudaSuccess) return;
    std::ostringstream message;
    message << operation << " failed: " << cudaGetErrorString(status);
    throw std::runtime_error(message.str());
}

unsigned blocks(const std::size_t count) {
    return static_cast<unsigned>(std::min<std::size_t>((count + threads - 1) / threads, 65535));
}

template <class T>
class Buffer {
public:
    Buffer() = default;
    ~Buffer() { if (data_) cudaFree(data_); }
    Buffer(const Buffer&) = delete;
    Buffer& operator=(const Buffer&) = delete;
    void resize(const std::size_t size) {
        if (size == size_) return;
        if (data_) check(cudaFree(data_), "cudaFree");
        data_ = nullptr; size_ = 0;
        if (size) {
            check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)), "cudaMalloc");
            size_ = size;
        }
    }
    void upload(const T* source, const std::size_t size) {
        resize(size);
        if (size) check(cudaMemcpy(data_, source, size * sizeof(T), cudaMemcpyHostToDevice), "upload");
    }
    void download(T* destination, const std::size_t size) const {
        if (size > size_) throw std::out_of_range("CUDA buffer download exceeds allocation");
        if (size) check(cudaMemcpy(destination, data_, size * sizeof(T), cudaMemcpyDeviceToHost), "download");
    }
    void zero() { if (size_) check(cudaMemset(data_, 0, size_ * sizeof(T)), "cudaMemset"); }
    T* data() { return data_; }
    const T* data() const { return data_; }
    std::size_t size() const { return size_; }
private:
    T* data_{};
    std::size_t size_{};
};

__device__ void mul3(const double* matrix, const double* vector, double* output) {
#pragma unroll
    for (int row = 0; row < 3; ++row)
        output[row] = matrix[row * 3] * vector[0] + matrix[row * 3 + 1] * vector[1] +
                      matrix[row * 3 + 2] * vector[2];
}

__global__ void linearize_kernel(
    const Pose* poses, const PinholeIntrinsics* intrinsics, const Point3* points,
    const Index* cameras, const Index* point_ids, const double* x, const double* y,
    const double* weights, const std::size_t count, const LinearizerOptions options,
    LinearizedObservation* output) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        detail::linearize_observation(poses[cameras[i]], intrinsics[cameras[i]], points[point_ids[i]],
                                      x[i], y[i], weights[i], options, output[i]);
}

__global__ void assemble_kernel(
    const LinearizedObservation* values, const Index* cameras, const Index* points,
    const std::size_t count, double* camera_h, double* camera_b,
    double* point_h, double* point_b, double* cross) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        const auto& value = values[i];
        if (!value.valid) continue;
        const std::size_t camera = cameras[i], point = points[i];
        double* w = cross + i * cross_n;
#pragma unroll
        for (int row = 0; row < 6; ++row) {
            const double jc0 = value.pose_jacobian[row];
            const double jc1 = value.pose_jacobian[6 + row];
            atomicAdd(camera_b + camera * 6 + row,
                      -(jc0 * value.residual[0] + jc1 * value.residual[1]));
#pragma unroll
            for (int column = 0; column < 6; ++column)
                atomicAdd(camera_h + camera * 36 + row * 6 + column,
                          jc0 * value.pose_jacobian[column] +
                          jc1 * value.pose_jacobian[6 + column]);
#pragma unroll
            for (int column = 0; column < 3; ++column)
                w[row * 3 + column] = jc0 * value.point_jacobian[column] +
                                      jc1 * value.point_jacobian[3 + column];
        }
#pragma unroll
        for (int row = 0; row < 3; ++row) {
            const double jp0 = value.point_jacobian[row];
            const double jp1 = value.point_jacobian[3 + row];
            atomicAdd(point_b + point * 3 + row,
                      -(jp0 * value.residual[0] + jp1 * value.residual[1]));
#pragma unroll
            for (int column = 0; column < 3; ++column)
                atomicAdd(point_h + point * 9 + row * 3 + column,
                          jp0 * value.point_jacobian[column] +
                          jp1 * value.point_jacobian[3 + column]);
        }
    }
}

__global__ void damp_cameras(double* hessian, const std::size_t count, const double damping) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
#pragma unroll
        for (int d = 0; d < 6; ++d) {
            double& diagonal = hessian[i * 36 + d * 6 + d];
            diagonal += damping * (diagonal + 1.0);
        }
}

__global__ void invert_points(
    double* hessian, double* rhs, const std::size_t count,
    const double damping, const bool fix_first) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double* m = hessian + i * 9;
        if (fix_first && i == 0) {
            for (int k = 0; k < 9; ++k) m[k] = 0.0;
            for (int k = 0; k < 3; ++k) rhs[i * 3 + k] = 0.0;
            continue;
        }
#pragma unroll
        for (int d = 0; d < 3; ++d) m[d * 3 + d] += damping * (m[d * 3 + d] + 1.0);
        const double a=m[0], b=m[1], c=m[2], d=m[4], e=m[5], f=m[8];
        const double c00=d*f-e*e, c01=c*e-b*f, c02=b*e-c*d;
        const double c11=a*f-c*c, c12=b*c-a*e, c22=a*d-b*b;
        const double determinant=a*c00+b*c01+c*c02;
        if (!(determinant > 1e-30) || !isfinite(determinant)) {
            for (int k = 0; k < 9; ++k) m[k] = 0.0;
            continue;
        }
        const double s=1.0/determinant;
        m[0]=c00*s; m[1]=c01*s; m[2]=c02*s; m[3]=m[1]; m[4]=c11*s;
        m[5]=c12*s; m[6]=m[2]; m[7]=m[5]; m[8]=c22*s;
    }
}

__global__ void reduced_point_rhs_kernel(
    const double* inverse, const double* rhs, const std::size_t count, double* reduced) {
    for (std::size_t i = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         i < count; i += static_cast<std::size_t>(blockDim.x) * gridDim.x)
        mul3(inverse + i * 9, rhs + i * 3, reduced + i * 3);
}

__global__ void schur_rhs_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* points,
    const double* camera_rhs, const double* reduced_points, const double* cross,
    const std::size_t camera_count, const bool fix_first, double* output) {
    for (std::size_t camera = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         camera < camera_count; camera += static_cast<std::size_t>(blockDim.x) * gridDim.x) {
        double result[6];
        for (int row=0; row<6; ++row) result[row]=camera_rhs[camera*6+row];
        if (fix_first && camera == 0) for (double& value : result) value=0.0;
        else for (std::size_t cursor=offsets[camera]; cursor<offsets[camera+1]; ++cursor) {
            const std::size_t obs=observations[cursor];
            const double* w=cross+obs*18;
            const double* p=reduced_points+static_cast<std::size_t>(points[obs])*3;
            for (int row=0; row<6; ++row)
                result[row]-=w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2];
        }
        for (int row=0; row<6; ++row) output[camera*6+row]=result[row];
    }
}

__global__ void point_product_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* cameras,
    const double* cross, const double* inverse, const double* input,
    const std::size_t point_count, const bool fix_first_camera, double* temporary) {
    for (std::size_t point=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         point<point_count; point+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double sum[3]{};
        for (std::size_t cursor=offsets[point]; cursor<offsets[point+1]; ++cursor) {
            const std::size_t obs=observations[cursor], camera=cameras[obs];
            if (fix_first_camera && camera==0) continue;
            const double* w=cross+obs*18; const double* x=input+camera*6;
            for (int column=0; column<3; ++column)
                for (int row=0; row<6; ++row) sum[column]+=w[row*3+column]*x[row];
        }
        mul3(inverse+point*9, sum, temporary+point*3);
    }
}

__global__ void camera_product_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* points,
    const double* hessian, const double* cross, const double* input,
    const double* point_temporary, const std::size_t camera_count,
    const bool fix_first, double* output) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (fix_first && camera==0) {
            for (int row=0; row<6; ++row) output[row]=input[row];
            continue;
        }
        double result[6]{}; const double* h=hessian+camera*36; const double* x=input+camera*6;
        for (int row=0; row<6; ++row)
            for (int column=0; column<6; ++column) result[row]+=h[row*6+column]*x[column];
        for (std::size_t cursor=offsets[camera]; cursor<offsets[camera+1]; ++cursor) {
            const std::size_t obs=observations[cursor]; const double* w=cross+obs*18;
            const double* p=point_temporary+static_cast<std::size_t>(points[obs])*3;
            for (int row=0; row<6; ++row)
                result[row]-=w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2];
        }
        for (int row=0; row<6; ++row) output[camera*6+row]=result[row];
    }
}

__global__ void precondition_kernel(
    const double* hessian, const double* residual, const std::size_t camera_count,
    const bool fix_first, double* output) {
    for (std::size_t camera=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         camera<camera_count; camera+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double* solution=output+camera*6;
        if (fix_first && camera==0) { for (int i=0;i<6;++i) solution[i]=0.0; continue; }
        const double* matrix=hessian+camera*36; double lower[36]{}; bool valid=true;
        for (int row=0;row<6 && valid;++row) for (int column=0;column<=row;++column) {
            double value=matrix[row*6+column];
            for (int k=0;k<column;++k) value-=lower[row*6+k]*lower[column*6+k];
            if (row==column) { if (!(value>1e-24)||!isfinite(value)) { valid=false; break; }
                lower[row*6+column]=sqrt(value); }
            else lower[row*6+column]=value/lower[column*6+column];
        }
        if (!valid) { for (int i=0;i<6;++i) solution[i]=residual[camera*6+i]/fmax(matrix[i*6+i],1e-12); continue; }
        double temporary[6]{};
        for (int row=0;row<6;++row) { double value=residual[camera*6+row];
            for (int k=0;k<row;++k) value-=lower[row*6+k]*temporary[k];
            temporary[row]=value/lower[row*6+row]; }
        for (int row=5;row>=0;--row) { double value=temporary[row];
            for (int k=row+1;k<6;++k) value-=lower[k*6+row]*solution[k];
            solution[row]=value/lower[row*6+row]; }
    }
}

__global__ void dot_kernel(const double* a, const double* b, const std::size_t count, double* result) {
    __shared__ double shared[threads];
    double sum=0.0;
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) sum+=a[i]*b[i];
    shared[threadIdx.x]=sum; __syncthreads();
    for (unsigned stride=blockDim.x/2; stride; stride/=2) {
        if (threadIdx.x<stride) shared[threadIdx.x]+=shared[threadIdx.x+stride];
        __syncthreads();
    }
    if (threadIdx.x==0) atomicAdd(result, shared[0]);
}

__global__ void update_pcg_kernel(
    double* solution, double* residual, const double* direction, const double* product,
    const std::size_t count, const double* alpha) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        solution[i]+=*alpha*direction[i]; residual[i]-=*alpha*product[i];
    }
}

__global__ void direction_kernel(
    double* direction, const double* z, const std::size_t count, const double* beta) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) direction[i]=z[i]+*beta*direction[i];
}

__global__ void alpha_kernel(double* values) {
    if (threadIdx.x==0 && blockIdx.x==0)
        values[4]=(values[1]>1e-30 && isfinite(values[1])) ? values[0]/values[1] : 0.0;
}

__global__ void beta_kernel(double* values) {
    if (threadIdx.x==0 && blockIdx.x==0) {
        values[4]=(fabs(values[0])>1e-30 && isfinite(values[3])) ? values[3]/values[0] : 0.0;
        values[0]=values[3];
    }
}

__global__ void persistent_pcg_kernel(
    const std::size_t* point_offsets, const std::size_t* point_observations,
    const std::size_t* camera_offsets, const std::size_t* camera_observations,
    const Index* cameras, const Index* points, const double* cross,
    const double* point_inverse, const double* camera_hessian, const double* rhs,
    const std::size_t point_count, const std::size_t camera_count, const bool fix_first,
    const std::size_t iteration_count, double* solution, double* residual, double* z,
    double* direction, double* product, double* point_temporary, double* scalars) {
    namespace cg = cooperative_groups;
    const cg::grid_group grid=cg::this_grid();
    const std::size_t rank=grid.thread_rank(), stride=grid.size();
    const std::size_t camera_values=camera_count*6;
    for (std::size_t i=rank;i<camera_values;i+=stride) {
        solution[i]=0.0; residual[i]=rhs[i];
        const std::size_t camera=i/6, local=i%6;
        z[i]=(fix_first&&camera==0)?0.0:rhs[i]/fmax(camera_hessian[camera*36+local*6+local],1e-12);
        direction[i]=z[i];
    }
    if (rank==0) scalars[0]=0.0;
    grid.sync();
    double local_dot=0.0;
    for (std::size_t i=rank;i<camera_values;i+=stride) local_dot+=residual[i]*z[i];
    atomicAdd(scalars,local_dot); grid.sync();
    for (std::size_t iteration=0;iteration<iteration_count;++iteration) {
        for (std::size_t point=rank;point<point_count;point+=stride) {
            double sum[3]{};
            for (std::size_t cursor=point_offsets[point];cursor<point_offsets[point+1];++cursor) {
                const std::size_t obs=point_observations[cursor],camera=cameras[obs];
                if (fix_first&&camera==0) continue;
                const double* w=cross+obs*18; const double* x=direction+camera*6;
                for (int column=0;column<3;++column)
                    for (int row=0;row<6;++row) sum[column]+=w[row*3+column]*x[row];
            }
            mul3(point_inverse+point*9,sum,point_temporary+point*3);
        }
        grid.sync();
        for (std::size_t camera=rank;camera<camera_count;camera+=stride) {
            double* out=product+camera*6; const double* x=direction+camera*6;
            if (fix_first&&camera==0) { for (int row=0;row<6;++row) out[row]=x[row]; continue; }
            double value[6]{}; const double* h=camera_hessian+camera*36;
            for (int row=0;row<6;++row) for (int column=0;column<6;++column)
                value[row]+=h[row*6+column]*x[column];
            for (std::size_t cursor=camera_offsets[camera];cursor<camera_offsets[camera+1];++cursor) {
                const std::size_t obs=camera_observations[cursor]; const double* w=cross+obs*18;
                const double* p=point_temporary+static_cast<std::size_t>(points[obs])*3;
                for (int row=0;row<6;++row) value[row]-=w[row*3]*p[0]+w[row*3+1]*p[1]+w[row*3+2]*p[2];
            }
            for (int row=0;row<6;++row) out[row]=value[row];
        }
        if (rank==0) scalars[1]=0.0; grid.sync(); local_dot=0.0;
        for (std::size_t i=rank;i<camera_values;i+=stride) local_dot+=direction[i]*product[i];
        atomicAdd(scalars+1,local_dot); grid.sync();
        if (rank==0) scalars[4]=(scalars[1]>1e-30&&isfinite(scalars[1]))?scalars[0]/scalars[1]:0.0;
        grid.sync();
        for (std::size_t i=rank;i<camera_values;i+=stride) {
            solution[i]+=scalars[4]*direction[i]; residual[i]-=scalars[4]*product[i];
            const std::size_t camera=i/6,local=i%6;
            z[i]=(fix_first&&camera==0)?0.0:residual[i]/fmax(camera_hessian[camera*36+local*6+local],1e-12);
        }
        if (rank==0) scalars[3]=0.0; grid.sync(); local_dot=0.0;
        for (std::size_t i=rank;i<camera_values;i+=stride) local_dot+=residual[i]*z[i];
        atomicAdd(scalars+3,local_dot); grid.sync();
        if (rank==0) { scalars[4]=(fabs(scalars[0])>1e-30&&isfinite(scalars[3]))?scalars[3]/scalars[0]:0.0; scalars[0]=scalars[3]; }
        grid.sync();
        for (std::size_t i=rank;i<camera_values;i+=stride) direction[i]=z[i]+scalars[4]*direction[i];
        grid.sync();
    }
}

__global__ void recover_points_kernel(
    const std::size_t* offsets, const std::size_t* observations, const Index* cameras,
    const double* cross, const double* inverse, const double* rhs, const double* camera_step,
    const std::size_t point_count, double* point_step) {
    for (std::size_t point=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         point<point_count; point+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        double value[3]={rhs[point*3],rhs[point*3+1],rhs[point*3+2]};
        for (std::size_t cursor=offsets[point]; cursor<offsets[point+1]; ++cursor) {
            const std::size_t obs=observations[cursor], camera=cameras[obs];
            const double* w=cross+obs*18; const double* step=camera_step+camera*6;
            for (int column=0; column<3; ++column)
                for (int row=0; row<6; ++row) value[column]-=w[row*3+column]*step[row];
        }
        mul3(inverse+point*9, value, point_step+point*3);
    }
}

__global__ void update_poses_kernel(Pose* poses, const double* step, const std::size_t count, const bool fix_first) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        if (fix_first && i==0) continue;
        Pose& p=poses[i]; const double* s=step+i*6;
        const double angle=sqrt(s[0]*s[0]+s[1]*s[1]+s[2]*s[2]);
        double dw=1.0, dx=0.5*s[0], dy=0.5*s[1], dz=0.5*s[2];
        if (angle>1e-12) { dw=cos(0.5*angle); const double k=sin(0.5*angle)/angle;
            dx=k*s[0]; dy=k*s[1]; dz=k*s[2]; }
        const double qw=dw*p.qw-dx*p.qx-dy*p.qy-dz*p.qz;
        const double qx=dw*p.qx+dx*p.qw+dy*p.qz-dz*p.qy;
        const double qy=dw*p.qy-dx*p.qz+dy*p.qw+dz*p.qx;
        const double qz=dw*p.qz+dx*p.qy-dy*p.qx+dz*p.qw;
        const double inv=rsqrt(qw*qw+qx*qx+qy*qy+qz*qz);
        p.qw=qw*inv; p.qx=qx*inv; p.qy=qy*inv; p.qz=qz*inv;
        p.cx+=s[3]; p.cy+=s[4]; p.cz+=s[5];
    }
}

__global__ void update_points_kernel(Point3* points, const double* step, const std::size_t count) {
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        points[i].x+=step[i*3]; points[i].y+=step[i*3+1]; points[i].z+=step[i*3+2];
    }
}

__global__ void cost_kernel(
    const Pose* poses, const PinholeIntrinsics* intrinsics, const Point3* points,
    const Index* cameras, const Index* point_ids, const double* ox, const double* oy,
    const double* weights, const std::size_t count, const double huber,
    const double minimum_depth, double* total) {
    double local=0.0;
    for (std::size_t i=static_cast<std::size_t>(blockIdx.x)*blockDim.x+threadIdx.x;
         i<count; i+=static_cast<std::size_t>(blockDim.x)*gridDim.x) {
        const Pose p=poses[cameras[i]]; const Point3 point=points[point_ids[i]];
        const auto k=intrinsics[cameras[i]]; const double dx=point.x-p.cx,dy=point.y-p.cy,dz=point.z-p.cz;
        const double px=(1-2*(p.qy*p.qy+p.qz*p.qz))*dx+2*(p.qx*p.qy-p.qw*p.qz)*dy+2*(p.qx*p.qz+p.qw*p.qy)*dz;
        const double py=2*(p.qx*p.qy+p.qw*p.qz)*dx+(1-2*(p.qx*p.qx+p.qz*p.qz))*dy+2*(p.qy*p.qz-p.qw*p.qx)*dz;
        const double pz=2*(p.qx*p.qz-p.qw*p.qy)*dx+2*(p.qy*p.qz+p.qw*p.qx)*dy+(1-2*(p.qx*p.qx+p.qy*p.qy))*dz;
        if (!(pz>minimum_depth) || !isfinite(pz)) { local+=weights[i]*1e12; continue; }
        const double x=px/pz,y=py/pz,r2=x*x+y*y,radial=1+k.k1*r2+k.k2*r2*r2;
        const double xd=x*radial+2*k.p1*x*y+k.p2*(r2+2*x*x);
        const double yd=y*radial+k.p1*(r2+2*y*y)+2*k.p2*x*y;
        const double rx=k.fx*xd+k.cx-ox[i],ry=k.fy*yd+k.cy-oy[i],norm=sqrt(rx*rx+ry*ry);
        if (!isfinite(norm)) { local+=weights[i]*1e12; continue; }
        local+=weights[i]*((huber>0 && norm>huber)?huber*(norm-0.5*huber):0.5*norm*norm);
    }
    atomicAdd(total,local);
}

}  // namespace

class CudaOptimizer::Impl {
public:
    explicit Impl(OptimizerOptions value) : options(value) {}
    OptimizerOptions options;
    std::size_t camera_count{}, point_count{}, observation_count{};
    Buffer<Pose> poses, pose_backup;
    Buffer<PinholeIntrinsics> intrinsics;
    Buffer<Point3> points, point_backup;
    Buffer<Index> cameras, point_ids;
    Buffer<double> ox, oy, weights;
    Buffer<std::size_t> point_offsets, point_observations, camera_offsets, camera_observations;
    Buffer<LinearizedObservation> linearized;
    Buffer<double> camera_h, camera_b, point_inverse, point_b, cross, reduced_point;
    Buffer<double> rhs, solution, residual, z, direction, product, point_temporary, point_step;
    Buffer<double> scalar, pcg_scalars;

    double dot(const Buffer<double>& a, const Buffer<double>& b, const std::size_t count) {
        scalar.zero(); dot_kernel<<<blocks(count),threads>>>(a.data(),b.data(),count,scalar.data());
        check(cudaGetLastError(),"dot_kernel"); double result{}; scalar.download(&result,1); return result;
    }
    void dot_to(const Buffer<double>& a, const Buffer<double>& b, const std::size_t count, const std::size_t index) {
        check(cudaMemset(pcg_scalars.data()+index,0,sizeof(double)),"PCG scalar reset");
        dot_kernel<<<blocks(count),threads>>>(a.data(),b.data(),count,pcg_scalars.data()+index);
        check(cudaGetLastError(),"PCG dot_kernel");
    }
    double cost() {
        scalar.zero(); cost_kernel<<<blocks(observation_count),threads>>>(poses.data(),intrinsics.data(),points.data(),
            cameras.data(),point_ids.data(),ox.data(),oy.data(),weights.data(),observation_count,
            options.huber_delta,options.minimum_depth,scalar.data());
        check(cudaGetLastError(),"cost_kernel"); double result{}; scalar.download(&result,1); return result;
    }
    void multiply() {
        point_product_kernel<<<blocks(point_count),threads>>>(point_offsets.data(),point_observations.data(),
            cameras.data(),cross.data(),point_inverse.data(),direction.data(),point_count,
            options.fix_first_pose,point_temporary.data());
        camera_product_kernel<<<blocks(camera_count),threads>>>(camera_offsets.data(),camera_observations.data(),
            point_ids.data(),camera_h.data(),cross.data(),direction.data(),point_temporary.data(),
            camera_count,options.fix_first_pose,product.data());
        check(cudaGetLastError(),"Schur multiply");
    }
    bool launch_persistent_pcg() {
        int device=0; check(cudaGetDevice(&device),"cudaGetDevice"); cudaDeviceProp properties{};
        check(cudaGetDeviceProperties(&properties,device),"cudaGetDeviceProperties");
        if (!properties.cooperativeLaunch) return false;
        constexpr int block_size=128; int active_blocks=0;
        check(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &active_blocks,persistent_pcg_kernel,block_size,0),"persistent PCG occupancy");
        const int grid_size=std::max(1,active_blocks*properties.multiProcessorCount);
        auto* po=point_offsets.data(); auto* pobs=point_observations.data();
        auto* co=camera_offsets.data(); auto* cobs=camera_observations.data();
        auto* camera_ids=cameras.data(); auto* point_indices=point_ids.data(); auto* w=cross.data();
        auto* inverse=point_inverse.data(); auto* h=camera_h.data(); auto* b=rhs.data();
        const std::size_t pc=point_count,cc=camera_count,iterations=options.maximum_pcg_iterations;
        const bool fixed=options.fix_first_pose; auto* x=solution.data(); auto* r=residual.data();
        auto* zv=z.data(); auto* p=direction.data(); auto* ap=product.data();
        auto* temporary=point_temporary.data(); auto* values=pcg_scalars.data();
        void* arguments[]={&po,&pobs,&co,&cobs,&camera_ids,&point_indices,&w,&inverse,&h,&b,
            const_cast<std::size_t*>(&pc),const_cast<std::size_t*>(&cc),const_cast<bool*>(&fixed),
            const_cast<std::size_t*>(&iterations),&x,&r,&zv,&p,&ap,&temporary,&values};
        check(cudaLaunchCooperativeKernel(reinterpret_cast<void*>(persistent_pcg_kernel),
            dim3(grid_size),dim3(block_size),arguments,0,nullptr),"persistent_pcg_kernel");
        return true;
    }
};

CudaOptimizer::CudaOptimizer(OptimizerOptions options) : impl_(std::make_unique<Impl>(options)) {}
CudaOptimizer::~CudaOptimizer()=default;
CudaOptimizer::CudaOptimizer(CudaOptimizer&&) noexcept=default;
CudaOptimizer& CudaOptimizer::operator=(CudaOptimizer&&) noexcept=default;
bool CudaOptimizer::is_available() noexcept { int count=0; return cudaGetDeviceCount(&count)==cudaSuccess && count>0; }
std::string CudaOptimizer::device_name() { int device=0; check(cudaGetDevice(&device),"cudaGetDevice");
    cudaDeviceProp p{}; check(cudaGetDeviceProperties(&p,device),"cudaGetDeviceProperties"); return p.name; }

void CudaOptimizer::upload(const Problem& problem) {
    problem.validate(); auto& d=*impl_; d.camera_count=problem.poses.size(); d.point_count=problem.points.size();
    d.observation_count=problem.observations.size();
    d.poses.upload(problem.poses.data(),d.camera_count); d.pose_backup.resize(d.camera_count);
    d.intrinsics.upload(problem.intrinsics.data(),d.camera_count);
    d.points.upload(problem.points.data(),d.point_count); d.point_backup.resize(d.point_count);
    d.cameras.upload(problem.observations.camera.data(),d.observation_count);
    d.point_ids.upload(problem.observations.point.data(),d.observation_count);
    d.ox.upload(problem.observations.x.data(),d.observation_count); d.oy.upload(problem.observations.y.data(),d.observation_count);
    d.weights.upload(problem.observations.weight.data(),d.observation_count);
    std::vector<std::size_t> po(d.point_count+1),co(d.camera_count+1),pi(d.observation_count),ci(d.observation_count);
    for (std::size_t i=0;i<d.observation_count;++i) { ++po[problem.observations.point[i]+1]; ++co[problem.observations.camera[i]+1]; }
    std::partial_sum(po.begin(),po.end(),po.begin()); std::partial_sum(co.begin(),co.end(),co.begin());
    auto pc=po,cc=co; for (std::size_t i=0;i<d.observation_count;++i) {
        pi[pc[problem.observations.point[i]]++]=i; ci[cc[problem.observations.camera[i]]++]=i; }
    d.point_offsets.upload(po.data(),po.size()); d.point_observations.upload(pi.data(),pi.size());
    d.camera_offsets.upload(co.data(),co.size()); d.camera_observations.upload(ci.data(),ci.size());
    d.linearized.resize(d.observation_count); d.camera_h.resize(d.camera_count*36); d.camera_b.resize(d.camera_count*6);
    d.point_inverse.resize(d.point_count*9); d.point_b.resize(d.point_count*3); d.cross.resize(d.observation_count*18);
    d.reduced_point.resize(d.point_count*3); const std::size_t cn=d.camera_count*6;
    d.rhs.resize(cn); d.solution.resize(cn); d.residual.resize(cn); d.z.resize(cn); d.direction.resize(cn);
    d.product.resize(cn); d.point_temporary.resize(d.point_count*3); d.point_step.resize(d.point_count*3);
    d.scalar.resize(1); d.pcg_scalars.resize(5);
}

OptimizerSummary CudaOptimizer::optimize() {
    auto& d=*impl_; if (!d.observation_count) throw std::logic_error("Upload a BA problem before optimization");
    const auto started=std::chrono::steady_clock::now(); OptimizerSummary summary;
    summary.initial_cost=d.cost(); summary.final_cost=summary.initial_cost; double damping=d.options.initial_damping;
    const std::size_t camera_values=d.camera_count*6;
    for (std::size_t iteration=0;iteration<d.options.maximum_iterations;++iteration) {
        linearize_kernel<<<blocks(d.observation_count),threads>>>(d.poses.data(),d.intrinsics.data(),d.points.data(),
            d.cameras.data(),d.point_ids.data(),d.ox.data(),d.oy.data(),d.weights.data(),d.observation_count,
            LinearizerOptions{d.options.huber_delta,d.options.minimum_depth},d.linearized.data());
        d.camera_h.zero(); d.camera_b.zero(); d.point_inverse.zero(); d.point_b.zero(); d.cross.zero();
        assemble_kernel<<<blocks(d.observation_count),threads>>>(d.linearized.data(),d.cameras.data(),d.point_ids.data(),
            d.observation_count,d.camera_h.data(),d.camera_b.data(),d.point_inverse.data(),d.point_b.data(),d.cross.data());
        damp_cameras<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.camera_count,damping);
        invert_points<<<blocks(d.point_count),threads>>>(d.point_inverse.data(),d.point_b.data(),d.point_count,damping,d.options.fix_first_point);
        reduced_point_rhs_kernel<<<blocks(d.point_count),threads>>>(d.point_inverse.data(),d.point_b.data(),d.point_count,d.reduced_point.data());
        schur_rhs_kernel<<<blocks(d.camera_count),threads>>>(d.camera_offsets.data(),d.camera_observations.data(),d.point_ids.data(),
            d.camera_b.data(),d.reduced_point.data(),d.cross.data(),d.camera_count,d.options.fix_first_pose,d.rhs.data());
        check(cudaGetLastError(),"system assembly");
        std::size_t pcg=0;
        if (d.launch_persistent_pcg()) {
            pcg=d.options.maximum_pcg_iterations;
        } else {
          d.solution.zero(); check(cudaMemcpy(d.residual.data(),d.rhs.data(),camera_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG init");
          precondition_kernel<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.residual.data(),d.camera_count,d.options.fix_first_pose,d.z.data());
          check(cudaMemcpy(d.direction.data(),d.z.data(),camera_values*sizeof(double),cudaMemcpyDeviceToDevice),"PCG direction");
          d.dot_to(d.residual,d.z,camera_values,0); d.dot_to(d.rhs,d.rhs,camera_values,1);
          double rhs_norm{}; check(cudaMemcpy(&rhs_norm,d.pcg_scalars.data()+1,sizeof(double),cudaMemcpyDeviceToHost),"PCG rhs norm");
          const double target=d.options.pcg_tolerance*d.options.pcg_tolerance*std::max(rhs_norm,1e-30);
          for (;pcg<d.options.maximum_pcg_iterations;++pcg) {
            d.multiply(); d.dot_to(d.direction,d.product,camera_values,1); alpha_kernel<<<1,1>>>(d.pcg_scalars.data());
            update_pcg_kernel<<<blocks(camera_values),threads>>>(d.solution.data(),d.residual.data(),d.direction.data(),d.product.data(),camera_values,d.pcg_scalars.data()+4);
            if ((pcg+1)%10==0 || pcg+1==d.options.maximum_pcg_iterations) {
                d.dot_to(d.residual,d.residual,camera_values,2); double residual_norm{};
                check(cudaMemcpy(&residual_norm,d.pcg_scalars.data()+2,sizeof(double),cudaMemcpyDeviceToHost),"PCG residual norm");
                if (!std::isfinite(residual_norm) || residual_norm<=target) { ++pcg; break; }
            }
            precondition_kernel<<<blocks(d.camera_count),threads>>>(d.camera_h.data(),d.residual.data(),d.camera_count,d.options.fix_first_pose,d.z.data());
            d.dot_to(d.residual,d.z,camera_values,3); beta_kernel<<<1,1>>>(d.pcg_scalars.data());
            direction_kernel<<<blocks(camera_values),threads>>>(d.direction.data(),d.z.data(),camera_values,d.pcg_scalars.data()+4);
          }
        }
        recover_points_kernel<<<blocks(d.point_count),threads>>>(d.point_offsets.data(),d.point_observations.data(),d.cameras.data(),
            d.cross.data(),d.point_inverse.data(),d.point_b.data(),d.solution.data(),d.point_count,d.point_step.data());
        const double norm=std::sqrt(d.dot(d.solution,d.solution,camera_values)+d.dot(d.point_step,d.point_step,d.point_count*3));
        check(cudaMemcpy(d.pose_backup.data(),d.poses.data(),d.camera_count*sizeof(Pose),cudaMemcpyDeviceToDevice),"pose backup");
        check(cudaMemcpy(d.point_backup.data(),d.points.data(),d.point_count*sizeof(Point3),cudaMemcpyDeviceToDevice),"point backup");
        update_poses_kernel<<<blocks(d.camera_count),threads>>>(d.poses.data(),d.solution.data(),d.camera_count,d.options.fix_first_pose);
        update_points_kernel<<<blocks(d.point_count),threads>>>(d.points.data(),d.point_step.data(),d.point_count);
        const double candidate=d.cost(); const bool accepted=std::isfinite(candidate)&&candidate<summary.final_cost;
        summary.iterations.push_back({iteration,accepted?candidate:summary.final_cost,damping,norm,pcg,accepted});
        if (accepted) { const double previous=summary.final_cost; summary.final_cost=candidate; ++summary.successful_steps;
            damping=std::max(d.options.minimum_damping,damping/3.0);
            if (norm<=d.options.step_tolerance || previous-candidate<=d.options.function_tolerance*std::max(1.0,previous)) {
                summary.termination=TerminationReason::converged; break; }
        } else { check(cudaMemcpy(d.poses.data(),d.pose_backup.data(),d.camera_count*sizeof(Pose),cudaMemcpyDeviceToDevice),"pose rollback");
            check(cudaMemcpy(d.points.data(),d.point_backup.data(),d.point_count*sizeof(Point3),cudaMemcpyDeviceToDevice),"point rollback");
            ++summary.unsuccessful_steps; damping=std::min(d.options.maximum_damping,damping*10.0);
            if (damping>=d.options.maximum_damping) { summary.termination=TerminationReason::numerical_failure; break; } }
    }
    check(cudaDeviceSynchronize(),"CUDA optimizer synchronize");
    summary.total_time_ms=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-started).count(); return summary;
}

void CudaOptimizer::download(Problem& problem) const {
    auto& d=*impl_; if (problem.poses.size()!=d.camera_count || problem.points.size()!=d.point_count)
        throw std::invalid_argument("Download target topology differs from uploaded BA problem");
    d.poses.download(problem.poses.data(),d.camera_count); d.points.download(problem.points.data(),d.point_count);
}

OptimizerSummary optimize_cuda(Problem& problem,const OptimizerOptions& options) {
    CudaOptimizer optimizer(options); optimizer.upload(problem); auto summary=optimizer.optimize(); optimizer.download(problem); return summary;
}

}  // namespace aetherscan::ba
