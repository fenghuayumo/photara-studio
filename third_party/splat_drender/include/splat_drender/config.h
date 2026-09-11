// splat_drender: differentiable 3D Gaussian rasterizer backend.
//
// Compile-time tuning. Values that influence training numerics are fixed
// here so that a saved workspace layout stays valid across rebuilds.
#pragma once

namespace splat_drender::cfg {

// Tile size. One CTA renders one tile; one thread renders one pixel.
constexpr int kTileWidth  = 16;
constexpr int kTileHeight = 16;
constexpr int kTileThreads = kTileWidth * kTileHeight;

// Point queries (sample_depth / occupancy) process two points per thread
// per tile round, matching the batching used by the sorted-instance walk.
constexpr int kPointsPerRound = 2;

// Median-depth bisection: 8 interior probes, refined 5 times, seeded with
// a +/-0.4 ray-length window around the transmittance-median initializer.
constexpr int kDepthSplit = 8;
constexpr int kDepthRefinements = 5;
constexpr float kDepthSeedWindow = 0.4f;
// World-space point queries (sample_depth) bracket much wider: the seed
// window is SAMPLE_RANGE_TESTING*2 and one more bisection level is used,
// matching the reference evaluateSDFCUDA numerics exactly.
constexpr int kDepthRefinementsTesting = 8;
constexpr float kDepthSeedWindowTesting = 200.0f;
constexpr float kDepthMinTransmittance = 0.45f;

// Alpha-blending thresholds (identical to the reference semantics).
constexpr float kAlphaClip = 0.99f;
constexpr float kAlphaFloor = 1.0f / 255.0f;
constexpr float kTransmittanceFloor = 1.0e-4f;

// Frustum / numeric guards.
constexpr float kNearPlane = 0.2f;
constexpr float kCovDetFloor = 1.0e-6f;
constexpr float kNormalEpsilon = 1.0e-12f;

// Points per thread block in per-Gaussian kernels.
constexpr int kGaussianBlock = 256;

}  // namespace splat_drender::cfg
