#pragma once
#include "splat_drender/api.h"
#include "splat_drender/buffers.h"
namespace splat_drender::gut {
void preprocess(const Gaussians &, const CameraView &, const RenderSettings &,
                CameraIntrinsics, int, int, ws::GaussianState, int *);
void forward(const Gaussians &, const CameraView &, const RenderSettings &,
             CameraIntrinsics, ws::GaussianState, ws::TileState,
             const unsigned *, ws::PixelState, const RenderOutputs &);
void backward(const Gaussians &, const CameraView &, const RenderSettings &,
              CameraIntrinsics, ws::GaussianState, ws::TileState,
              const unsigned *, ws::PixelState, const ForwardOutputsView &,
              const LossGradients &, ws::GradState, const ModelGradients &);
} // namespace splat_drender::gut
