#pragma once
#include "inference/models/moge.hpp"
namespace photara::inference::detail {
class MogeGeometryGpu {
public:
    virtual ~MogeGeometryGpu() = default;
    virtual void maps(const MogeRawGeometry&, const MogeGeometryOptions&,
        const MogeRecoveryInfo&, GeometryPrior&) = 0;
};
// Returns null only when GPU geometry support was not compiled in.
std::unique_ptr<MogeGeometryGpu> create_moge_geometry_gpu(const DeviceInfo&);
MogeGeometryResult recover_moge_geometry_gpu(const MogeRawGeometry&,
    const MogeGeometryOptions&, const ModelInfo&, const DeviceInfo&, MogeGeometryGpu&);
}
