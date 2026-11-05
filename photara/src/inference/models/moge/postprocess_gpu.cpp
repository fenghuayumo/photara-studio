#include "postprocess_gpu.hpp"
#if defined(PHOTARA_MOGE_GPU_GEOMETRY)
#include "photara_vk/photara_vk.hpp"
#include "moge_geometry.hlsl.embedded.hpp"
#include <algorithm>
#include <cmath>
#include <array>
#include <stdexcept>
#include <limits>
namespace photara::inference::detail {
namespace {
struct Push {
    std::uint32_t sw, sh, width, height, start, count;
    float shift, metric, fx, fy, cx, cy, threshold;
};
static_assert(sizeof(Push)==52);
class GeometryGpu final : public MogeGeometryGpu {
    vk::Device device_;
    vk::ComputePipeline pipeline_;
public:
    explicit GeometryGpu(const DeviceInfo& info) {
        vk::DeviceRequest request;
        const auto devices=vk::Device::enumerate(request);
        const auto match=std::find_if(devices.begin(),devices.end(),[&](const auto& d){return d.name==info.name;});
        if(match==devices.end()) throw std::runtime_error("Cannot match MoGe Vulkan geometry device");
        request.device_index=std::uint32_t(match-devices.begin());
        device_=vk::Device::create(request);
        pipeline_=device_.create_compute(std::as_bytes(std::span{moge_geometry_hlsl_spv}),4,sizeof(Push));
    }
    void maps(const MogeRawGeometry& raw, const MogeGeometryOptions& options,
        const MogeRecoveryInfo& recovery, GeometryPrior& prior) override {
        const auto count=std::size_t(options.width)*options.height;
        if(count>std::numeric_limits<std::uint32_t>::max())
            throw std::invalid_argument("MoGe Vulkan geometry output exceeds 32-bit indexing");
        const auto upload=[&](const std::vector<float>& values) {
            if(values.size()*sizeof(float)>device_.caps().max_storage_buffer_range)
                throw std::invalid_argument("MoGe Vulkan raw map exceeds storage buffer range");
            auto buffer=device_.create_buffer(values.size()*sizeof(float),vk::MemoryKind::host_visible);
            buffer.upload(values.data(),values.size()*sizeof(float)); return buffer;
        };
        auto points=upload(raw.points), normals=upload(raw.normals), mask=upload(raw.mask_logits);
        const std::size_t tile=std::min<std::size_t>({count, 1024*1024,
            device_.caps().max_storage_buffer_range/(8*sizeof(float))});
        if(!tile) throw std::runtime_error("MoGe Vulkan geometry storage range is too small");
        auto output=device_.create_buffer(tile*8*sizeof(float),vk::MemoryKind::device_local);
        auto readback=device_.create_buffer(tile*8*sizeof(float),vk::MemoryKind::host_cached);
        std::vector<float> values(tile*8);
        const auto& k=*prior.intrinsics;
        for(std::size_t start=0;start<count;start+=tile) {
            const auto size=std::min(tile,count-start);
            Push push{raw.width,raw.height,options.width,options.height,std::uint32_t(start),std::uint32_t(size),
                float(recovery.z_shift),float(std::exp(double(raw.log_metric_scale))),k.fx,k.fy,k.cx,k.cy,options.mask_threshold};
            const std::array bindings{vk::binding(points),vk::binding(normals),vk::binding(mask),vk::binding(output)};
            auto encoder=device_.encoder();
            encoder.dispatch(pipeline_,bindings,&push,sizeof(push),std::uint32_t((size+255)/256));
            encoder.barrier(VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,
                VK_ACCESS_SHADER_WRITE_BIT,VK_ACCESS_TRANSFER_READ_BIT);
            encoder.copy(readback,0,output,0,size*8*sizeof(float));
            encoder.submit_wait();
            readback.download(values.data(),size*8*sizeof(float));
            for(std::size_t i=0;i<size;++i) {
                const auto j=start+i;
                prior.depth[j]=values[i*8]; prior.valid[j]=std::uint8_t(values[i*8+7]);
                for(int c=0;c<3;++c) { prior.points[j][c]=values[i*8+1+c]; prior.normals[j][c]=values[i*8+4+c]; }
            }
        }
    }
};
}
std::unique_ptr<MogeGeometryGpu> create_moge_geometry_gpu(const DeviceInfo& info) {
    return std::make_unique<GeometryGpu>(info);
}
}
#else
namespace photara::inference::detail {
std::unique_ptr<MogeGeometryGpu> create_moge_geometry_gpu(const DeviceInfo&) { return {}; }
}
#endif
