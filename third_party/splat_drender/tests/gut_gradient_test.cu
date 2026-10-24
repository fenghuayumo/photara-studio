#include "splat_drender/api.h"
#include <array>
#include <cmath>
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
void run(CameraMode mode, bool geometry, bool compact, bool sh_colors,
         bool transformed = false) {
  constexpr int N = 2, W = 16, H = 12, P = W * H;
  std::array<Buffer, 6> workspace;
  Buffer compact_pool;
  WorkspacePools pools;
  pools.gaussian = [&](auto n) { return workspace[0].resize(n); };
  pools.grad = [&](auto n) { return workspace[1].resize(n); };
  pools.instance = [&](auto n) { return workspace[2].resize(n); };
  pools.pixel = [&](auto n) { return workspace[3].resize(n); };
  pools.tile = [&](auto n) { return workspace[4].resize(n); };
  if (compact)
    pools.compact_instance = [&](auto n) { return compact_pool.resize(n); };
  std::vector<float> means = {-.12f, .08f, 2.f, .2f, -.1f, 2.6f};
  if (mode == CameraMode::equirectangular) {
    means[2] = -2.f;
    means[5] = -2.6f;
  }
  std::vector<float> scales = {.36f, .28f, .20f, .4f, .3f, .25f};
  std::vector<float> rotations = {.98f, .08f, -.12f, .09f, 1.f, 0.f, 0.f, 0.f};
  const float norm =
      std::sqrt(.98f * .98f + .08f * .08f + .12f * .12f + .09f * .09f);
  for (int k = 0; k < 4; ++k)
    rotations[k] /= norm;
  std::vector<float> opacity = {.88f, .65f},
                     colors = {.8f, .2f, .4f, .1f, .6f, .3f};
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
    view.put({c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0,
              -.15f*c-.1f*s, .05f, .15f*s-.1f*c, 1});
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
  cam.fx = 12;
  cam.fy = 12;
  cam.cx = 7.5f;
  cam.cy = 5.5f;
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
      // Keep the transformed-camera objective inside the smooth central
      // support, away from opacity-floor and median-selection boundaries.
      dl[k][j] = transformed && (x < 7 || x > 8 || y < 5 || y > 6)
                      ? 0.f : .01f * std::sin(float(j + 7 * k) * .37f);
    }
    lossbuf[k].put(dl[k]);
  }
  LossGradients loss;
  loss.color = lossbuf[0].floats();
  loss.alpha = lossbuf[1].floats();
  loss.median_depth = lossbuf[2].floats();
  loss.normal = lossbuf[3].floats();
  std::array<Buffer, 5> grads;
  const int gs[5] = {6, 6, 8, 2, static_cast<int>(colors.size())};
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
  auto fwd = Rasterizer::forward(pools, g, cam, s, out);
  if (fwd.instance_count == 0)
    throw std::runtime_error("3DGUT produced no draw list");
  // Independent double-precision CPU oracle, traversing every Gaussian
  // without projected bounds or tile lists. Also checks UT culling coverage.
  if (mode == CameraMode::pinhole && !sh_colors) {
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
  Rasterizer::backward(pools, g, cam, s, fwd, fo, loss, grad);
  check(cudaDeviceSynchronize());
  std::array<std::vector<float>, 5> analytic;
  for (int k = 0; k < 5; ++k)
    analytic[k] = grads[k].get(gs[k]);
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
  for (int group = 0; group < 5; ++group)
    for (int k = 0; k < gs[group]; ++k) {
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
    for (int k = 0; k < 6; ++k)
      means[k] = original_means[k] - step * analytic[0][k];
    input[0].put(means);
    if (objective() < before) {
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
  opacity = {0.f, 0.f};
  input[3].put(opacity);
  objective();
  for (float alpha : output[1].get(P))
    if (alpha != 0.f)
      throw std::runtime_error("3DGUT all-culled render retained stale pixels");
  std::cout << "3DGUT finite differences passed mode=" << int(mode)
            << " geometry=" << geometry << " compact=" << compact
            << " SH=" << sh_colors << '\n';
}
int main() {
  try {
    run(CameraMode::pinhole, false, false, false);
    run(CameraMode::pinhole, true, true, false);
    run(CameraMode::opencv_fisheye, true, true, true);
    run(CameraMode::equirectangular, false, true, false);
    run(CameraMode::orthographic, true, false, false);
    run(CameraMode::pinhole, true, true, true, true);
    return 0;
  } catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
