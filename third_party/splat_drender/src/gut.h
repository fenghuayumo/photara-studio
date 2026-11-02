#pragma once
#include "splat_drender/api.h"
#include "splat_drender/buffers.h"
namespace splat_drender::gut {
bool use_rgb_snapshots(const RenderSettings &);
void count_contributors(const Gaussians &, const CameraView &,
                        const RenderSettings &, CameraIntrinsics,
                        ws::GaussianState, ws::TileState, const unsigned *);
void preprocess(const Gaussians &, const CameraView &, const RenderSettings &,
                CameraIntrinsics, int, int, ws::GaussianState, int *);
void forward(const Gaussians &, const CameraView &, const RenderSettings &,
             CameraIntrinsics, ws::GaussianState, ws::TileState,
             const unsigned *, ws::PixelState, const RenderOutputs &,
             int bucket_size = 32);
void backward(const Gaussians &, const CameraView &, const RenderSettings &,
              CameraIntrinsics, ws::GaussianState, ws::TileState,
              const unsigned *, ws::PixelState, const ForwardOutputsView &,
              const LossGradients &, ws::GradState, const ModelGradients &,
              int buckets, int bucket_size = 32);
} // namespace splat_drender::gut
