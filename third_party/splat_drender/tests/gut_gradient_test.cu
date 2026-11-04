#include "splat_drender/api.h"
#include "splat_drender/buffers.h"
#include <array>
#include <cmath>
#include <cstdlib>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <vector>

using namespace splat_drender;
void check(cudaError_t e) {
  if (e != cudaSuccess)
    throw std::runtime_error(cudaGetErrorString(e));
}
struct Buffer {
  void *p = nullptr;
  std::size_t capacity = 0;
  ~Buffer() {
    if (p)
      cudaFree(p);
  }
  char *resize(std::size_t n) {
    if (n > capacity) {
      if (p)
        check(cudaFree(p));
      check(cudaMalloc(&p, n));
      capacity = n;
    }
    return static_cast<char *>(p);
  }
  float *floats() { return static_cast<float *>(p); }
  void put(const std::vector<float> &v) {
    resize(v.size() * 4);
    check(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
  }
  std::vector<float> get(int n) {
    std::vector<float> v(n);
    check(cudaMemcpy(v.data(), p, n * 4, cudaMemcpyDeviceToHost));
    return v;
  }
};
bool sanitizer_smoke = false;
void run(CameraMode mode, bool geometry, bool compact, bool sh_colors,
         bool transformed = false, bool tiled = false, bool many = false,
         bool thin = false, bool occluded = false, int boundary = 0,
         bool auxiliary = false) {
  const int N = many ? 300 : 2;
  const int W = boundary ? 193 : tiled ? 35 : 16;
  const int H = boundary ? 129 : tiled ? 29 : 12, P = W * H;
  std::array<Buffer, 6> workspace;
  Buffer compact_pool;
  WorkspacePools pools;
  pools.gaussian = [&](auto n) { return workspace[0].resize(n); };
  pools.grad = [&](auto n) { return workspace[1].resize(n); };
  pools.instance = [&](auto n) { return workspace[2].resize(n); };
  pools.pixel = [&](auto n) {
    if (compact && compact_pool.p && workspace[2].p)
      throw std::runtime_error("GUT sorting storage retained during snapshot allocation");
    return workspace[3].resize(n);
  };
  pools.tile = [&](auto n) { return workspace[4].resize(n); };
  if (compact) {
    pools.compact_instance = [&](auto n) { return compact_pool.resize(n); };
    pools.release_instance = [&] {
      check(cudaFree(workspace[2].p));
      workspace[2].p = nullptr;
      workspace[2].capacity = 0;
    };
  }
  std::vector<float> means = {-.12f, .08f, 2.f, .2f, -.1f, 2.6f};
  if (mode == CameraMode::equirectangular) {
    means[2] = -2.f;
    means[5] = -2.6f;
  }
  std::vector<float> scales = {.36f, .28f, .20f, .4f, .3f, .25f};
  if (thin)
    scales[2] = 1e-8f;
  std::vector<float> rotations = {.98f, .08f, -.12f, .09f, 1.f, 0.f, 0.f, 0.f};
  const float norm =
      std::sqrt(.98f * .98f + .08f * .08f + .12f * .12f + .09f * .09f);
  for (int k = 0; k < 4; ++k)
    rotations[k] /= norm;
  std::vector<float> opacity = {.88f, .65f},
                     colors = {.8f, .2f, .4f, .1f, .6f, .3f};
  if (boundary) {
    means = {2.f, .1f, .08f, 2.3f, -.1f, .12f};
    scales.assign(6, boundary == 2 ? 3.f : .2f);
  }
  if (many) {
    means.resize(3 * N);
    scales.resize(3 * N);
    rotations.resize(4 * N);
    opacity.assign(N, occluded ? .95f : .01f);
    colors.resize(3 * N);
    const float q[4] = {rotations[0], rotations[1], rotations[2], rotations[3]};
    for (int i = 0; i < N; ++i) {
      means[3 * i] = -.12f;
      means[3 * i + 1] = .08f;
      means[3 * i + 2] = 2.f + .002f * i;
      scales[3 * i] = occluded ? 10.f : .36f;
      scales[3 * i + 1] = occluded ? 10.f : .28f;
      scales[3 * i + 2] = .2f;
      for (int a = 0; a < 4; ++a)
        rotations[4 * i + a] = q[a];
      colors[3 * i] = .8f;
      colors[3 * i + 1] = .2f;
      colors[3 * i + 2] = .4f;
    }
  }
  if (sh_colors) {
    colors.resize(N * 12);
    for (int k = 0; k < N * 12; ++k)
      colors[k] = .2f * std::sin(float(k) * .53f);
  }
  std::array<Buffer, 5> input;
  input[0].put(means);
  input[1].put(scales);
  input[2].put(rotations);
  input[3].put(opacity);
  input[4].put(colors);
  Buffer view, center;
  view.put({1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
  center.put({0, 0, 0});
  if (transformed) {
    const float c = std::cos(.2f), s = std::sin(.2f);
    center.put({.15f, -.05f, .1f});
    view.put({c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, -.15f * c - .1f * s, .05f,
              .15f * s - .1f * c, 1});
  }
  Gaussians g;
  g.count = N;
  g.means = input[0].floats();
  g.scales = input[1].floats();
  g.rotations = input[2].floats();
  g.opacities = input[3].floats();
  g.colors = input[4].floats();
  g.primitive = Primitive::gaussian_3dgut;
  if (sh_colors) {
    g.colors = nullptr;
    g.sh = input[4].floats();
    g.sh_bases = 4;
    g.sh_degree = 1;
  }
  CameraView cam;
  cam.width = W;
  cam.height = H;
  cam.fx = cam.fy = boundary ? 55.f : 12.f;
  cam.cx = boundary ? W*.5f : tiled ? 17.5f : 7.5f;
  cam.cy = boundary ? H*.5f : tiled ? 14.5f : 5.5f;
  cam.mode = mode;
  cam.k1 = .05f;
  cam.world_to_camera = view.floats();
  cam.center = center.floats();
  if (mode == CameraMode::equirectangular) {
    cam.fx = W / (2 * 3.14159265359f);
    cam.fy = H / 3.14159265359f;
  }
  RenderSettings s;
  s.need_depth = geometry;
  // Exercise compaction even on the small finite-difference fixtures.
  s.snapshot_compaction_threshold = 0;
  s.background[0] = .13f;
  s.background[1] = .07f;
  s.background[2] = .11f;
  std::array<Buffer, 6> output;
  const int sizes[6] = {3 * P, P, P, 3 * P, N, N};
  for (int k = 0; k < 6; ++k) {
    output[k].resize(sizes[k] * 4);
    check(cudaMemset(output[k].p, 0, sizes[k] * 4));
  }
  RenderOutputs out;
  out.color = output[0].floats();
  out.alpha = output[1].floats();
  out.median_depth = output[2].floats();
  out.normal = output[3].floats();
  out.visibility = output[4].floats();
  out.radii = static_cast<int *>(output[5].p);
  std::array<Buffer, 4> lossbuf;
  std::array<std::vector<float>, 4> dl;
  const int ls[4] = {3 * P, P, P, 3 * P};
  for (int k = 0; k < 4; ++k) {
    dl[k].resize(ls[k]);
    for (int j = 0; j < ls[k]; ++j) {
      const int pixel = j % P, x = pixel % W, y = pixel / W;
      // Sample a smooth patch inside the boundary Gaussian's support. RGB
      // and alpha exercise both attachment paths without median-depth jumps.
      const bool outside_boundary_patch = boundary &&
          (x < W-8 || x > W-5 || y < H/2 || y > H/2+2);
      // Keep the transformed-camera objective inside the smooth central
      // support, away from opacity-floor and median-selection boundaries.
      dl[k][j] =
          (boundary && k >= 2) || outside_boundary_patch ||
              ((transformed || many) &&
               (x < int(cam.cx) || x > int(cam.cx) + 1 ||
                y < int(cam.cy) || y > int(cam.cy) + 1))
              ? 0.f
              : .01f * std::sin(float(j + 7 * k) * .37f);
    }
    lossbuf[k].put(dl[k]);
  }
  LossGradients loss;
  loss.color = lossbuf[0].floats();
  loss.alpha = lossbuf[1].floats();
  loss.median_depth = lossbuf[2].floats();
  loss.normal = lossbuf[3].floats();
  std::array<Buffer, 5> grads;
  const int gs[5] = {3 * N, 3 * N, 4 * N, N, static_cast<int>(colors.size())};
  for (int k = 0; k < 5; ++k) {
    grads[k].resize(gs[k] * 4);
    check(cudaMemset(grads[k].p, 0, gs[k] * 4));
  }
  ModelGradients grad;
  grad.means = grads[0].floats();
  grad.scales = grads[1].floats();
  grad.rotations = grads[2].floats();
  grad.opacities = grads[3].floats();
  if (sh_colors)
    grad.sh = grads[4].floats();
  else
    grad.colors = grads[4].floats();
  Buffer densify_map;
  std::array<Buffer, 3> auxiliary_grads;
  if (auxiliary) {
    std::vector<float> map(P);
    for (int pixel = 0; pixel < P; ++pixel)
      map[pixel] = .2f + .6f * float(pixel % 17) / 16.f;
    densify_map.put(map);
    loss.densify_map = densify_map.floats();
    for (auto &buffer : auxiliary_grads)
      buffer.resize(N * sizeof(float));
    grad.refine_weight = auxiliary_grads[0].floats();
    grad.densify_weight = auxiliary_grads[1].floats();
    grad.densify_weight_den = auxiliary_grads[2].floats();
  }
  auto zero_auxiliary = [&] {
    if (auxiliary)
      for (auto &buffer : auxiliary_grads)
        check(cudaMemset(buffer.p, 0, N * sizeof(float)));
  };
  std::array<std::vector<float>, 3> analytic_auxiliary;
  auto compare_auxiliary = [&](float absolute, float relative) {
    if (auxiliary)
      for (int k = 0; k < 3; ++k) {
        const auto values = auxiliary_grads[k].get(N);
        for (int i = 0; i < N; ++i)
          if (!std::isfinite(values[i]) ||
              std::abs(values[i] - analytic_auxiliary[k][i]) >
                  absolute + relative * std::abs(analytic_auxiliary[k][i]))
            throw std::runtime_error("GUT ADC/densify output mismatch");
      }
  };
  auto fwd = Rasterizer::forward(pools, g, cam, s, out);
  if (fwd.instance_count == 0)
    throw std::runtime_error("3DGUT produced no draw list");
  if (boundary == 1 && fwd.instance_count >= N*((W+15)/16)*((H+15)/16)/2)
    throw std::runtime_error("Fisheye boundary support did not reduce draw instances");
  if (boundary == 2 && fwd.instance_count != N*((W+15)/16)*((H+15)/16))
    throw std::runtime_error("Camera-inside support lost its full-image fallback");
  // Independent double-precision CPU oracle, traversing every Gaussian
  // without projected bounds or tile lists. Also checks UT culling coverage.
  if (mode == CameraMode::pinhole && !sh_colors && !thin) {
    const auto actual_color = output[0].get(3 * P),
               actual_alpha = output[1].get(P);
    for (int py = 0; py < H; ++py)
      for (int px = 0; px < W; ++px) {
        double direction[3] = {(px - cam.cx) / cam.fx, (py - cam.cy) / cam.fy,
                               1};
        const double norm = std::sqrt(direction[0] * direction[0] +
                                      direction[1] * direction[1] + 1);
        for (auto &d : direction)
          d /= norm;
        double T = 1, rgb[3] = {};
        for (int i = 0; i < N; ++i) {
          const double w = rotations[4 * i], x = rotations[4 * i + 1],
                       y = rotations[4 * i + 2], z = rotations[4 * i + 3];
          const double R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y + w * z),
                                   2 * (x * z - w * y)},
                                  {2 * (x * y - w * z), 1 - 2 * (x * x + z * z),
                                   2 * (y * z + w * x)},
                                  {2 * (x * z + w * y), 2 * (y * z - w * x),
                                   1 - 2 * (x * x + y * y)}};
          double o[3] = {}, d[3] = {}, od = 0, dd = 0;
          for (int a = 0; a < 3; ++a) {
            for (int b = 0; b < 3; ++b) {
              o[a] -= R[a][b] * means[3 * i + b] / scales[3 * i + a];
              d[a] += R[a][b] * direction[b] / scales[3 * i + a];
            }
            od += o[a] * d[a];
            dd += d[a] * d[a];
          }
          const double t = -od / dd;
          double distance = 0;
          for (int a = 0; a < 3; ++a)
            distance += (o[a] + t * d[a]) * (o[a] + t * d[a]);
          const double alpha =
              std::min(.99, double(opacity[i]) * std::exp(-.5 * distance));
          if (t <= 0 || alpha < 1. / 255)
            continue;
          if (T * (1 - alpha) < 1e-4)
            break;
          for (int c = 0; c < 3; ++c)
            rgb[c] += T * alpha * colors[3 * i + c];
          T *= 1 - alpha;
        }
        const int pixel = py * W + px;
        if (std::abs(actual_alpha[pixel] - (1 - T)) > 2e-5)
          throw std::runtime_error("3DGUT alpha differs from CPU ray oracle");
        for (int c = 0; c < 3; ++c)
          if (std::abs(actual_color[c * P + pixel] -
                       (rgb[c] + T * s.background[c])) > 2e-5)
            throw std::runtime_error("3DGUT color differs from CPU ray oracle");
      }
  }
  ForwardOutputsView fo;
  fo.alpha = out.alpha;
  fo.median_depth = out.median_depth;
  fo.normal = out.normal;
  fo.radii = out.radii;
  zero_auxiliary();
  Rasterizer::backward(pools, g, cam, s, fwd, fo, loss, grad);
  check(cudaDeviceSynchronize());
  if (auxiliary) {
    for (int k = 0; k < 3; ++k)
      analytic_auxiliary[k] = auxiliary_grads[k].get(N);
    bool contributed = false;
    for (int i = 0; i < N; ++i) {
      const float refine = analytic_auxiliary[0][i];
      const float weight = analytic_auxiliary[1][i];
      const float denominator = analytic_auxiliary[2][i];
      if (!std::isfinite(refine) || !std::isfinite(weight) ||
          !std::isfinite(denominator) || refine < 0.f || denominator < 0.f ||
          weight < .2f * denominator - 1e-6f ||
          weight > .8f * denominator + 1e-6f)
        throw std::runtime_error("Invalid GUT ADC/densify output");
      contributed |= refine > 0.f && denominator > 0.f;
    }
    if (!contributed)
      throw std::runtime_error("GUT auxiliary fixture produced no contributions");
  }
  std::array<std::vector<float>, 5> analytic;
  for (int k = 0; k < 5; ++k)
    analytic[k] = grads[k].get(gs[k]);
  for (const auto &values : analytic)
    for (float value : values)
      if (!std::isfinite(value))
        throw std::runtime_error("Nonfinite GUT gradient");
  if (!geometry) {
    const auto compact_color = output[0].get(3 * P);
    s.compact_pixel_snapshots = false;
    for (int k = 0; k < 5; ++k)
      check(cudaMemset(grads[k].p, 0, gs[k] * sizeof(float)));
    const auto full = Rasterizer::forward(pools, g, cam, s, out);
    zero_auxiliary();
    Rasterizer::backward(pools, g, cam, s, full, fo, loss, grad);
    compare_auxiliary(1e-6f, 1e-5f);
    if (fwd.snapshot_buckets > full.snapshot_buckets)
      throw std::runtime_error("GUT compaction increased snapshot storage");
    if (occluded) {
      if (fwd.snapshot_buckets * 2 >= full.snapshot_buckets)
        throw std::runtime_error("GUT occlusion did not reduce snapshot storage");
      std::cout << "GUT occluded snapshot bytes "
                << full.snapshot_buckets * 4096ull << " -> "
                << fwd.snapshot_buckets * 4096ull << '\n';
    }
    if (compact_color != output[0].get(3 * P))
      throw std::runtime_error("GUT compact color mismatch");
    for (int k = 0; k < 5; ++k) {
      const auto full_gradient = grads[k].get(gs[k]);
      for (int j = 0; j < gs[k]; ++j) {
        if (!std::isfinite(full_gradient[j]))
          throw std::runtime_error("Nonfinite uncompressed GUT gradient");
        // The existing 1e-8-scale case checks finiteness only, because its
        // ray transforms are ill-conditioned in float (as with the oracle below).
        if (!thin && std::abs(full_gradient[j] - analytic[k][j]) >
            1e-6f + 1e-5f * std::abs(full_gradient[j])) {
          std::cerr << "compact gradient thin=" << thin << " group=" << k
                    << " index=" << j << " full=" << full_gradient[j]
                    << " compact=" << analytic[k][j] << '\n';
          throw std::runtime_error("GUT compact gradient mismatch");
        }
      }
    }
    s.compact_pixel_snapshots = true;
    s.snapshot_compaction_threshold = ws::PixelState::bytes(
        P, full.snapshot_buckets, false, true) / 2;
    for (int k = 0; k < 5; ++k)
      check(cudaMemset(grads[k].p, 0, gs[k] * sizeof(float)));
    const auto grouped = Rasterizer::forward(pools, g, cam, s, out);
    if (grouped.snapshot_bucket_size != 128)
      throw std::runtime_error("GUT grouped snapshot path was not selected");
    zero_auxiliary();
    Rasterizer::backward(pools, g, cam, s, grouped, fo, loss, grad);
    compare_auxiliary(1e-6f, 1e-5f);
    if (compact_color != output[0].get(3 * P))
      throw std::runtime_error("GUT grouped color mismatch");
    for (int k = 0; k < 5; ++k) {
      const auto values = grads[k].get(gs[k]);
      for (int j = 0; j < gs[k]; ++j) {
        if (!std::isfinite(values[j]) ||
            (!thin && std::abs(values[j] - analytic[k][j]) >
                1e-6f + 1e-5f * std::abs(analytic[k][j])))
          throw std::runtime_error("GUT grouped gradient mismatch");
      }
    }
    s.snapshot_compaction_threshold = 0;
  }
  // Extreme 1e-8 scales are a finiteness stress case: float ray transforms
  // suffer cancellation and cannot use the old float AD path as an oracle.
  if (!geometry && !thin &&
      !(std::getenv("PHOTARA_GUT_REFERENCE") &&
        std::getenv("PHOTARA_GUT_REFERENCE")[0] == '1')) {
    const auto fast_color = output[0].get(3 * P);
    const char* candidate_env = std::getenv("PHOTARA_GUT_CANDIDATE_REFERENCE");
    const std::string previous_candidates = candidate_env ? candidate_env : "";
#ifdef _WIN32
    _putenv_s("PHOTARA_GUT_REFERENCE", "1");
    _putenv_s("PHOTARA_GUT_CANDIDATE_REFERENCE", "1");
#else
    setenv("PHOTARA_GUT_REFERENCE", "1", 1);
    setenv("PHOTARA_GUT_CANDIDATE_REFERENCE", "1", 1);
#endif
    for (int k = 0; k < 5; ++k)
      check(cudaMemset(grads[k].p, 0, gs[k] * sizeof(float)));
    const auto reference_forward = Rasterizer::forward(pools, g, cam, s, out);
    zero_auxiliary();
    Rasterizer::backward(pools, g, cam, s, reference_forward, fo, loss, grad);
    compare_auxiliary(3e-5f, .01f);
    const auto reference_color = output[0].get(3 * P);
    double color_error = 0, color_norm = 0;
    for (int k = 0; k < 3 * P; ++k) {
      const double d = fast_color[k] - reference_color[k];
      color_error += d * d;
      color_norm += double(reference_color[k]) * reference_color[k];
    }
    if (std::sqrt(color_error / std::max(color_norm, 1e-20)) > 2e-5)
      throw std::runtime_error("GUT optimized/reference color mismatch");
    for (int k = 0; k < 5; ++k) {
      const auto reference = grads[k].get(gs[k]);
      for (int j = 0; j < gs[k]; ++j)
        if (std::abs(reference[j] - analytic[k][j]) >
            3e-5f + .01f * std::abs(reference[j])) {
          std::cerr << "Reference gradient mismatch thin=" << thin
                    << " group=" << k << " index=" << j
                    << " analytic=" << analytic[k][j]
                    << " reference=" << reference[j] << '\n';
          throw std::runtime_error("GUT optimized/reference gradient mismatch");
        }
    }
#ifdef _WIN32
    _putenv_s("PHOTARA_GUT_REFERENCE", "");
    _putenv_s("PHOTARA_GUT_CANDIDATE_REFERENCE", previous_candidates.c_str());
#else
    unsetenv("PHOTARA_GUT_REFERENCE");
    if (previous_candidates.empty()) unsetenv("PHOTARA_GUT_CANDIDATE_REFERENCE");
    else setenv("PHOTARA_GUT_CANDIDATE_REFERENCE", previous_candidates.c_str(), 1);
#endif
  }
  auto objective = [&] {
    Rasterizer::forward(pools, g, cam, s, out);
    double l = 0;
    for (int k = 0; k < (geometry ? 4 : 2); ++k) {
      auto v = output[k].get(ls[k]);
      for (int j = 0; j < ls[k]; ++j)
        l += double(v[j]) * dl[k][j];
    }
    return l;
  };
  std::array<std::vector<float> *, 5> parameters = {&means, &scales, &rotations,
                                                    &opacity, &colors};
  for (int group = 0; group < (thin || sanitizer_smoke ? 0 : 5); ++group)
    for (int k = 0; k < gs[group]; ++k) {
      if (many && k > 7 && k < gs[group] - 4)
        continue;
      auto &v = *parameters[group];
      const float original = v[k], eps = 2e-4f;
      v[k] = original + eps;
      input[group].put(v);
      const double plus = objective();
      v[k] = original - eps;
      input[group].put(v);
      const double minus = objective();
      v[k] = original;
      input[group].put(v);
      const double numeric = (plus - minus) / (2 * eps),
                   actual = analytic[group][k];
      if (!std::isfinite(actual) ||
          std::abs(actual - numeric) > 3e-4 + std::abs(numeric) * .025) {
        std::cerr << "mode=" << int(mode) << " geometry=" << geometry
                  << " group=" << group << " k=" << k << " analytic=" << actual
                  << " numeric=" << numeric << '\n';
        throw std::runtime_error("3DGUT finite difference mismatch");
      }
    }
  // A real parameter update should reduce the differentiable objective.
  const double before = objective();
  const auto original_means = means;
  bool decreased = false;
  for (float step = .02f; step > 1e-6f; step *= .25f) {
    for (int k = 0; k < 3 * N; ++k)
      means[k] = original_means[k] - step * analytic[0][k];
    input[0].put(means);
    const double after = objective();
    if (!std::isfinite(before) || !std::isfinite(after))
      throw std::runtime_error("Nonfinite GUT objective");
    // Broad opaque splats have a locally flat position objective in float.
    if (after < before || thin || occluded) {
      decreased = true;
      break;
    }
  }
  if (!decreased)
    throw std::runtime_error("3DGUT gradient step failed to decrease loss");
  const auto training_color = output[0].get(3 * P);
  s.record_backward_state = false;
  objective();
  const auto inference_color = output[0].get(3 * P);
  for (int k = 0; k < 3 * P; ++k)
    if (std::abs(training_color[k] - inference_color[k]) > 1e-6f)
      throw std::runtime_error("3DGUT inference differs from training forward");
  if (!geometry) {
    s.record_backward_state = true;
    // Keep nonempty draw lists but reject every candidate at the alpha floor.
    const float alpha_clip = cam.alpha_clip;
    cam.alpha_clip = 1e-4f;
    for (int k = 0; k < 5; ++k)
      check(cudaMemset(grads[k].p, 0, gs[k] * sizeof(float)));
    const auto empty = Rasterizer::forward(pools, g, cam, s, out);
    if (empty.instance_count == 0 || empty.snapshot_buckets != 0)
      throw std::runtime_error("GUT empty contribution compaction failed");
    zero_auxiliary();
    Rasterizer::backward(pools, g, cam, s, empty, fo, loss, grad);
    if (auxiliary)
      for (auto &buffer : auxiliary_grads)
        for (float value : buffer.get(N))
          if (value != 0.f)
            throw std::runtime_error("Empty GUT contributions produced ADC/densify output");
    for (int k = 0; k < 5; ++k)
      for (float value : grads[k].get(gs[k]))
        if (value != 0.f)
          throw std::runtime_error("GUT empty contributions produced gradients");
    cam.alpha_clip = alpha_clip;
    s.record_backward_state = false;
  }
  opacity.assign(N, 0.f);
  input[3].put(opacity);
  objective();
  for (float alpha : output[1].get(P))
    if (alpha != 0.f)
      throw std::runtime_error("3DGUT all-culled render retained stale pixels");
  std::cout << (sanitizer_smoke ? "3DGUT sanitizer smoke passed mode=" : thin ? "3DGUT thin-scale finiteness passed mode="
                     : "3DGUT finite differences passed mode=")
            << int(mode) << " geometry=" << geometry << " compact=" << compact
            << " SH=" << sh_colors << '\n';
}
int main(int argc, char** argv) {
  sanitizer_smoke = argc > 1 && std::string(argv[1]) == "--sanitizer-smoke";
  try {
    run(CameraMode::pinhole, false, false, false);
    run(CameraMode::pinhole, true, true, false);
    run(CameraMode::opencv_fisheye, true, true, true);
    run(CameraMode::equirectangular, false, true, false);
    run(CameraMode::orthographic, true, false, false);
    run(CameraMode::pinhole, true, true, true, true);
    run(CameraMode::pinhole, false, true, true, true);
    run(CameraMode::opencv_fisheye, false, true, true);
    run(CameraMode::orthographic, false, true, false);
    run(CameraMode::pinhole, false, true, false, false, true);
    run(CameraMode::pinhole, false, true, false, false, true, true);
    run(CameraMode::pinhole, false, true, false, false, false, false, true);
    run(CameraMode::pinhole, false, true, false, false, true, true, false, true);
    run(CameraMode::opencv_fisheye, false, true, false,
        false, false, false, false, false, true);
    run(CameraMode::opencv_fisheye, true, false, false,
        false, false, false, false, false, true);
    run(CameraMode::opencv_fisheye, false, true, false,
        false, false, false, false, false, 2);
    // Multi-phase 128 buckets and 32 buckets, rotated/translated cameras,
    // partial tiles, orthographic origins and nonconstant densification maps.
    for (auto mode : {CameraMode::pinhole, CameraMode::opencv_fisheye,
                      CameraMode::equirectangular, CameraMode::orthographic})
      run(mode, false, true, true, true, true, true, false, false, 0, true);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
