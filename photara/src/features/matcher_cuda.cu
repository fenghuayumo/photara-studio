#include "matcher_cuda.hpp"
#include <cuda_runtime.h>
#include <mma.h>
#include <algorithm>
#include <array>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>

namespace photara::features {
namespace {
void check(cudaError_t error) {
    if (error != cudaSuccess) throw std::runtime_error(
        std::string("CUDA descriptor matcher: ") + cudaGetErrorString(error));
}
std::size_t padded(std::size_t n) { return (n+31)/32*32; }
template<class T> struct Buffer {
    T* data{};
    std::size_t capacity{};
    ~Buffer() { if (data) cudaFree(data); }
    void reserve(std::size_t n) {
        if (n <= capacity) return;
        T* next{};
        check(cudaMalloc(reinterpret_cast<void**>(&next), n * sizeof(T)));
        if (data) cudaFree(data);
        data = next; capacity = n;
    }
};

// One dot-product tile feeds both directional top-two reductions. No full
// N*M distance matrix is stored. Packed integer arithmetic is exact for u8.
__device__ void insert(int value, int index, int3& best) {
    if (value > best.x) { best.z = best.x; best.x = value; best.y = index; }
    else best.z = max(best.z, value);
}
__global__ void tiles(const unsigned* query, const unsigned* train,
                     int nq, int nt, int3* rows, int3* cols, bool mutual) {
#if __CUDA_ARCH__ >= 720
    // Keep the entire Tensor Core tile in shared memory, then immediately
    // reduce it. Unlike GEMM + reduction this never writes N*M int32 scores
    // to VRAM. Each 16-byte K slice starts at a 32-byte aligned address.
    __shared__ __align__(32) unsigned qs[8][32][4], ts[8][32][4];
#else
    __shared__ unsigned q[32][33], t[32][33];
#endif
    __shared__ __align__(32) int dots[32][36];
    const int lane = threadIdx.x;
    for (int e = lane; e < 1024; e += 256) {
        const int r = e / 32, k = e % 32;
#if __CUDA_ARCH__ >= 720
        qs[k/4][r][k%4] = blockIdx.y*32+r<nq ? query[(blockIdx.y*32+r)*32+k] : 0;
        ts[k/4][r][k%4] = blockIdx.x*32+r<nt ? train[(blockIdx.x*32+r)*32+k] : 0;
#else
        q[r][k] = blockIdx.y * 32 + r < nq ? query[(blockIdx.y * 32 + r)*32+k] : 0;
        t[r][k] = blockIdx.x * 32 + r < nt ? train[(blockIdx.x * 32 + r)*32+k] : 0;
#endif
    }
    __syncthreads();
#if __CUDA_ARCH__ >= 720
    if (lane < 128) {
        using namespace nvcuda;
        const int warp=lane/32, row=(warp/2)*16, col=(warp%2)*16;
        wmma::fragment<wmma::matrix_a,16,16,16,unsigned char,wmma::row_major> a;
        wmma::fragment<wmma::matrix_b,16,16,16,unsigned char,wmma::col_major> b;
        wmma::fragment<wmma::accumulator,16,16,16,int> acc;
        wmma::fill_fragment(acc,0);
#pragma unroll
        for (int k=0;k<8;++k) {
            wmma::load_matrix_sync(a,reinterpret_cast<const unsigned char*>(&qs[k][row][0]),16);
            wmma::load_matrix_sync(b,reinterpret_cast<const unsigned char*>(&ts[k][col][0]),16);
            wmma::mma_sync(acc,a,b,acc);
        }
        wmma::store_matrix_sync(&dots[row][col],acc,36,wmma::mem_row_major);
    }
#else
    for (int e = lane; e < 1024; e += 256) {
        const int r = e / 32, c = e % 32;
        unsigned dot = 0;
#pragma unroll
        for (int k = 0; k < 32; ++k) {
#if __CUDA_ARCH__ >= 610
            dot = __dp4a(q[r][k], t[c][k], dot);
#else
            const unsigned a = q[r][k], b = t[c][k];
#pragma unroll
            for (int s = 0; s < 32; s += 8) dot += ((a>>s)&255)*((b>>s)&255);
#endif
        }
        dots[r][c] = dot;
    }
#endif
    __syncthreads();
    if (lane < 32 && blockIdx.y*32+lane < nq) {
        int3 best = make_int3(0, -1, 0);
        for (int c = 0; c < 32 && blockIdx.x*32+c < nt; ++c)
            insert(dots[lane][c], blockIdx.x*32+c, best);
        rows[blockIdx.x*nq+blockIdx.y*32+lane] = best;
    }
    if (mutual && lane < 32 && blockIdx.x*32+lane < nt) {
        int3 best = make_int3(0, -1, 0);
        for (int r = 0; r < 32 && blockIdx.y*32+r < nq; ++r)
            insert(dots[r][lane], blockIdx.y*32+r, best);
        cols[blockIdx.y*nt+blockIdx.x*32+lane] = best;
    }
}
__global__ void reduce(const int3* partial, int n, int chunks, float ratio, int* result) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i >= n) return;
    int3 best = make_int3(0, -1, 0);
    for (int j = 0; j < chunks; ++j) {
        const int3 candidate = partial[j*n+i];
        insert(candidate.x, candidate.y, best);
        best.z = max(best.z, candidate.z);
    }
    // Keep SiftGPU's angular distance and ratio test in BOTH directions.
    // SiftGPU promotes the clamped dot to double for acos, then rounds to
    // float. acosf can differ by one ULP and flip an exact 0.8 boundary.
    const float d = static_cast<float>(acos(fmin(best.x * (1.0/262144.0), 1.0)));
    const float second = static_cast<float>(acos(fmin(best.z * (1.0/262144.0), 1.0)));
    result[i] = d < .7f && d < ratio*second ? best.y : -1;
}
__global__ void cross_check(int* rows, const int* cols, int n) {
    const int i = blockIdx.x*blockDim.x+threadIdx.x;
    if (i < n && rows[i] >= 0 && cols[rows[i]] != i) rows[i] = -1;
}

class NativeMatcher final : public FeatureMatcher {
    struct Entry { std::size_t offset, bytes; std::uint64_t generation; };
    SiftGpuMatcherOptions options_;
    std::thread::id owner_{std::this_thread::get_id()};
    mutable Buffer<unsigned> descriptors_;
    mutable Buffer<int3> row_partial_, col_partial_;
    mutable Buffer<int> output_, reverse_;
    mutable std::unordered_map<std::uint64_t, Entry> cache_;
    mutable std::size_t used_{};
    std::size_t budget_{};
    cudaStream_t stream_{};
    mutable int* host_{};
    mutable std::size_t host_capacity_{};

    const unsigned* upload(const FeatureSet& features) const {
        const auto found = cache_.find(features.descriptor_identity);
        if (found != cache_.end() && found->second.generation == features.descriptor_generation &&
            found->second.bytes == features.descriptors_u8.size())
            return descriptors_.data + found->second.offset/4;
        const std::size_t bytes = padded(features.keypoints.size())*128;
        if (used_ + bytes > budget_) throw std::runtime_error("CUDA descriptor cache budget exceeded");
        const Entry entry{used_, features.descriptors_u8.size(), features.descriptor_generation};
        check(cudaMemcpyAsync(descriptors_.data + used_/4, features.descriptors_u8.data(),
                              features.descriptors_u8.size(), cudaMemcpyHostToDevice, stream_));
        if (bytes>features.descriptors_u8.size())
            check(cudaMemsetAsync(reinterpret_cast<char*>(descriptors_.data)+used_+features.descriptors_u8.size(),
                0,bytes-features.descriptors_u8.size(),stream_));
        cache_[features.descriptor_identity] = entry;
        used_ += bytes;
        return descriptors_.data + entry.offset/4;
    }
public:
    explicit NativeMatcher(SiftGpuMatcherOptions options) : options_(options) {
        check(cudaSetDevice(options.device_index));
        std::size_t free{}, total{};
        check(cudaMemGetInfo(&free, &total));
        budget_ = std::min<std::size_t>(1536ull<<20, free/4);
        if (budget_ < 2*padded(options.maximum_features)*128)
            throw std::runtime_error("Insufficient CUDA descriptor cache memory");
        descriptors_.reserve(budget_/4);
        check(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking));
    }
    ~NativeMatcher() override {
        if (stream_) { cudaStreamSynchronize(stream_); cudaStreamDestroy(stream_); }
        if (host_) cudaFreeHost(host_);
    }
    std::string_view name() const override { return "native_cuda_mutual_ratio"; }
    bool requires_owner_thread() const override { return true; }
    std::unique_ptr<FeatureMatcher> clone() const override {
        return std::make_unique<NativeMatcher>(options_);
    }
    void clear_prepared() override { cache_.clear(); used_ = 0; }
    MatchSet match(const FeatureSet& q, const FeatureSet& t) const override {
        const Pair pair{&q, &t};
        return std::move(match_batch(std::span(&pair, 1))[0]);
    }
    std::vector<MatchSet> match_batch(std::span<const Pair> pairs) const override {
        if (owner_ != std::this_thread::get_id())
            throw std::runtime_error("CUDA matcher called outside owner thread");
        std::vector<MatchSet> result(pairs.size());
        // Bound pinned output and cached uploads independently of caller batch size.
        for (std::size_t begin = 0; begin < pairs.size();) {
            std::size_t batch_count=0, batch_bytes=0;
            while (batch_count<8 && begin+batch_count<pairs.size()) {
                const auto& [q,t]=pairs[begin+batch_count];
                const auto bytes=(padded(q->keypoints.size())+padded(t->keypoints.size()))*128;
                if (bytes>budget_) throw std::invalid_argument("CUDA matching pair exceeds descriptor budget");
                if (batch_count && batch_bytes+bytes>budget_) break;
                batch_bytes+=bytes; ++batch_count;
            }
            const auto batch = pairs.subspan(begin, batch_count);
            std::size_t outputs=0, row_size=0, col_size=0, max_train=0, upload_bytes=0;
            for (const auto& [q,t] : batch) {
                for (const auto* f : {q,t}) {
                    f->validate();
                    if (f->storage != DescriptorStorage::uint8 || f->descriptor_dimension != 128 ||
                        f->keypoints.size() > options_.maximum_features)
                        throw std::invalid_argument("Native CUDA matcher expects bounded 128D u8 descriptors");
                    upload_bytes += padded(f->keypoints.size())*128;
                }
                const auto nq=q->keypoints.size(), nt=t->keypoints.size();
                outputs += nq;
                row_size = std::max(row_size, nq*((nt+31)/32));
                col_size = std::max(col_size, nt*((nq+31)/32));
                max_train = std::max(max_train, nt);
            }
            // Recycle only at a completed submission boundary; queued kernels
            // never observe overwritten descriptor storage.
            if (used_+upload_bytes > budget_) { cache_.clear(); used_=0; }
            row_partial_.reserve(row_size);
            if (options_.mutual_check) { col_partial_.reserve(col_size); reverse_.reserve(max_train); }
            output_.reserve(outputs);
            if (outputs > host_capacity_) {
                if (host_) { check(cudaFreeHost(host_)); host_=nullptr; host_capacity_=0; }
                check(cudaMallocHost(reinterpret_cast<void**>(&host_), outputs*sizeof(int)));
                host_capacity_=outputs;
            }
            std::size_t offset=0;
            try {
                for (const auto& [q,t] : batch) {
                    const int nq=static_cast<int>(q->keypoints.size()), nt=static_cast<int>(t->keypoints.size());
                    if (nq && nt) {
                        const auto* dq=upload(*q); const auto* dt=upload(*t);
                        tiles<<<dim3((nt+31)/32,(nq+31)/32),256,0,stream_>>>(
                            dq,dt,nq,nt,row_partial_.data,col_partial_.data,options_.mutual_check);
                        reduce<<<(nq+255)/256,256,0,stream_>>>(row_partial_.data,nq,(nt+31)/32,
                            options_.ratio_threshold,output_.data+offset);
                        if (options_.mutual_check) {
                            reduce<<<(nt+255)/256,256,0,stream_>>>(col_partial_.data,nt,(nq+31)/32,
                                options_.ratio_threshold,reverse_.data);
                            cross_check<<<(nq+255)/256,256,0,stream_>>>(output_.data+offset,reverse_.data,nq);
                        }
                    } else if (nq) check(cudaMemsetAsync(output_.data+offset,255,nq*sizeof(int),stream_));
                    offset+=nq;
                }
                check(cudaGetLastError());
                if (outputs) check(cudaMemcpyAsync(host_,output_.data,outputs*sizeof(int),cudaMemcpyDeviceToHost,stream_));
                check(cudaStreamSynchronize(stream_));
            } catch (...) {
                cudaStreamSynchronize(stream_); cache_.clear(); used_=0; throw;
            }
            offset=0;
            for (std::size_t p=0;p<batch.size();++p) {
                const auto nq=batch[p].first->keypoints.size();
                auto& matches=result[begin+p].matches;
                for (std::size_t i=0;i<nq;++i)
                    if (host_[offset+i]>=0) matches.push_back({static_cast<FeatureIndex>(i),
                        static_cast<FeatureIndex>(host_[offset+i]),1.f});
                offset+=nq;
            }
            begin+=batch_count;
        }
        return result;
    }
};
}
std::unique_ptr<FeatureMatcher> make_native_cuda_matcher(SiftGpuMatcherOptions options) {
    return std::make_unique<NativeMatcher>(options);
}
}
