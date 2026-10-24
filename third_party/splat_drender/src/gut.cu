#include "device/gut.cuh"
#include "device/sh.cuh"
#include "device/tiles.cuh"
#include "gut.h"
#include "splat_drender/config.h"

namespace splat_drender::gut {
namespace {
__global__ void prepare(Gaussians g, CameraView cam, RenderSettings settings,
                        CameraIntrinsics K, int gx, int gy,
                        ws::GaussianState st, int *radii) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.count)
    return;
  st.tiles_touched[i] = st.visible_flag[i] = 0;
  st.radius[i] = radii[i] = 0;
  const float3 mean =
      make_float3(g.means[3 * i], g.means[3 * i + 1], g.means[3 * i + 2]);
  const auto center = project(mat::xform_point(mean, cam.world_to_camera), K,
                              cam.width, cam.height);
  if (!center.valid || !(g.opacities[i] >= cfg::kAlphaFloor))
    return;
  const auto R =
      mat::quat_rotation(reinterpret_cast<const float4 *>(g.rotations)[i]);
  float2 points[7];
  points[0] = make_float2(center.pixel_x, center.pixel_y);
  float2 mu = make_float2(.25f * points[0].x, .25f * points[0].y);
  bool boundary = false;
  // Symmetric seven-point UT: lambda=1, weights 1/4 and 1/8.
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 2; ++b) {
      const float spread =
          (b ? 2.f : -2.f) * settings.scale_modifier * g.scales[3 * i + a];
      const float3 sigma =
          make_float3(mean.x + spread * R.m[a][0], mean.y + spread * R.m[a][1],
                      mean.z + spread * R.m[a][2]);
      auto p = project(mat::xform_point(sigma, cam.world_to_camera), K,
                       cam.width, cam.height);
      boundary |= !p.valid;
      const int j = 1 + 2 * a + b;
      points[j] =
          make_float2(center.pixel_x + wrap_dx(p.pixel_x - center.pixel_x,
                                               cam.width, K.mode),
                      p.pixel_y);
      mu.x += .125f * points[j].x;
      mu.y += .125f * points[j].y;
    }
  float xx = 0, xy = 0, yy = 0;
  for (int j = 0; j < 7; ++j) {
    const float w = j ? .125f : .25f, dx = points[j].x - mu.x,
                dy = points[j].y - mu.y;
    xx += w * dx * dx;
    xy += w * dx * dy;
    yy += w * dy * dy;
  }
  // UT is a culling approximation only. Expand support for the nonlinear
  // tails; near projection boundaries use a full-image conservative bin.
  xx = 4.f * xx + 1.f;
  xy *= 4.f;
  yy = 4.f * yy + 1.f;
  if (boundary) {
    mu = make_float2(cam.width * .5f, cam.height * .5f);
    xx = float(cam.width) * cam.width;
    yy = float(cam.height) * cam.height;
    xy = 0;
  }
  const float det = xx * yy - xy * xy;
  if (!(det > 0.f) || !isfinite(det))
    return;
  const float4 co = make_float4(yy / det, -xy / det, xx / det, g.opacities[i]);
  const int wrap = is_equirect(K.mode) ? cam.width : 0;
  const unsigned touched =
      tiles::enumerate(mu, co, gx, gy, i, 0, nullptr, nullptr, wrap);
  if (!touched)
    return;
  st.mean2d[i] = mu;
  st.conic_opacity[i] = co;
  const auto t = mat::xform_point(mean, cam.world_to_camera);
  st.depth_key[i] = __float_as_uint(norm3df(t.x, t.y, t.z));
  st.tiles_touched[i] = touched;
  st.visible_flag[i] = 1;
  st.radius[i] = radii[i] = int(ceilf(3.f * sqrtf(fmaxf(xx, yy))));
  if (g.sh)
    st.rgb[i] = sh::evaluate(
        i, g.sh_degree, g.sh_bases, reinterpret_cast<const float3 *>(g.means),
        make_float3(cam.center[0], cam.center[1], cam.center[2]), g.sh,
        st.clamped);
  atomicAdd(st.n_visible, 1u);
  atomicAdd(st.n_instances, touched);
}

template <bool BACKWARD>
__global__ void
blend(Gaussians g, CameraView cam, RenderSettings s, CameraIntrinsics K,
      ws::GaussianState st, ws::TileState tiles, const unsigned *ids,
      ws::PixelState ps, RenderOutputs out, ForwardOutputsView fo,
      LossGradients loss, ws::GradState grad, ModelGradients model_grad) {
  const int x = blockIdx.x * cfg::kTileWidth + threadIdx.x;
  const int y = blockIdx.y * cfg::kTileHeight + threadIdx.y;
  if (x >= cam.width || y >= cam.height)
    return;
  const int pix = y * cam.width + x, pixels = cam.width * cam.height;
  const uint2 range = tiles.range[blockIdx.y * gridDim.x + blockIdx.x];
  float3 origin, dir;
  float ray_z;
  ray(cam, K, x, y, origin, dir, ray_z);
  float T = 1.f, color[3] = {}, normal[3] = {}, depth = 0.f;
  unsigned last = 0, median = ~0u;
  // Backward replays front-to-back, subtracting accepted contributions from
  // the cached final color/normal. This avoids unstable reverse divisions.
  float remaining[3] = {}, remaining_n[3] = {};
  if constexpr (BACKWARD)
    for (int c = 0; c < 3; ++c) {
      remaining[c] = ps.total_color[c * pixels + pix] +
                     (1.f - fo.alpha[pix]) * s.background[c];
      remaining_n[c] =
          s.need_depth ? fo.normal[c * pixels + pix] * fo.alpha[pix] : 0.f;
    }
  const float alpha_total = BACKWARD ? fo.alpha[pix] : 0.f;
  for (unsigned j = range.x; j < range.y; ++j) {
    if constexpr (BACKWARD) {
      if (j - range.x >= ps.n_contrib[pix])
        break;
    }
    const unsigned i = ids[j];
    float p[10];
    parameters(g, i, p);
    auto r =
        response(p, origin, dir, ray_z, cam.world_to_camera, s.scale_modifier);
    const float raw = g.opacities[i] * r.weight;
    const float alpha = fminf(cfg::kAlphaClip, raw);
    if (!(alpha >= cfg::kAlphaFloor))
      continue;
    const float next = T * (1.f - alpha);
    if (next < cfg::kTransmittanceFloor)
      break;
    const float weight = alpha * T;
    const float3 rgb =
        g.colors ? reinterpret_cast<const float3 *>(g.colors)[i] : st.rgb[i];
    const float colors[3] = {rgb.x, rgb.y, rgb.z};
    const bool is_median = T >= .5f && next < .5f;
    if constexpr (BACKWARD) {
      float da = loss.alpha[pix] * (1.f - alpha_total) / (1.f - alpha);
      float dn[3] = {};
      for (int c = 0; c < 3; ++c) {
        remaining[c] -= weight * colors[c];
        da += loss.color[c * pixels + pix] *
              (T * colors[c] - remaining[c] / (1.f - alpha));
        atomicAdd(reinterpret_cast<float *>(grad.d_color) + 3 * i + c,
                  weight * loss.color[c * pixels + pix]);
        if (s.need_depth && alpha_total > 1e-8f) {
          remaining_n[c] -= weight * r.normal[c];
          const float d = loss.normal[c * pixels + pix] / alpha_total;
          da += d * (T * r.normal[c] - remaining_n[c] / (1.f - alpha) -
                     fo.normal[c * pixels + pix] * (1.f - alpha_total) /
                         (1.f - alpha));
          dn[c] = weight * d;
        }
      }
      Dual dp[10];
      for (int k = 0; k < 10; ++k)
        dp[k] = Dual::variable(p[k], k);
      const auto dr = response(dp, origin, dir, ray_z, cam.world_to_camera,
                               s.scale_modifier);
      const float dw = raw < cfg::kAlphaClip ? da * g.opacities[i] : 0.f;
      float derivatives[10];
      for (int k = 0; k < 10; ++k) {
        float d = dw * dr.weight.d[k];
        if (s.need_depth) {
          if (is_median)
            d += loss.median_depth[pix] * dr.depth.d[k];
          for (int c = 0; c < 3; ++c)
            d += dn[c] * dr.normal[c].d[k];
        }
        derivatives[k] = d;
      }
      for (int c = 0; c < 3; ++c) {
        atomicAdd(reinterpret_cast<float *>(grad.d_mean2d) + 3 * i + c,
                  derivatives[c]);
        atomicAdd(reinterpret_cast<float *>(grad.d_conic) + 4 * i + c,
                  derivatives[3 + c]);
      }
      for (int c = 0; c < 4; ++c)
        atomicAdd(reinterpret_cast<float *>(grad.d_ray_plane) + 4 * i + c,
                  derivatives[6 + c]);
      if (raw < cfg::kAlphaClip)
        atomicAdd(&grad.d_conic[i].w, da * r.weight);
      if (model_grad.refine_weight) {
        const auto proj =
            project(mat::xform_point(make_float3(p[0], p[1], p[2]),
                                     cam.world_to_camera),
                    K, cam.width, cam.height);
        const float3 gm =
            mat::transpose(
                mat::world_rotation_transposed(cam.world_to_camera)) *
            make_float3(derivatives[0], derivatives[1], derivatives[2]);
        // Convert the world mean derivative to a screen displacement
        // proxy used only by the adaptive density controller.
        const float dx = gm.x / (fabsf(proj.J.du[0]) + 1e-6f),
                    dy = gm.y / (fabsf(proj.J.dv[1]) + 1e-6f);
        atomicAdd(model_grad.refine_weight + i,
                  hypotf(dx * cam.width, dy * cam.height) /
                      fmaxf(alpha_total, 1e-6f));
      }
      if (loss.densify_map && model_grad.densify_weight &&
          model_grad.densify_weight_den) {
        atomicAdd(model_grad.densify_weight + i,
                  weight * loss.densify_map[pix]);
        atomicAdd(model_grad.densify_weight_den + i, weight);
      }
    } else {
      for (int c = 0; c < 3; ++c) {
        color[c] += weight * colors[c];
        normal[c] += weight * r.normal[c];
      }
      if (is_median) {
        depth = r.depth;
        median = i;
      }
      atomicExch(reinterpret_cast<unsigned *>(out.visibility + i),
                 __float_as_uint(1.f));
    }
    T = next;
    last = j - range.x + 1;
  }
  if constexpr (!BACKWARD) {
    out.alpha[pix] = 1.f - T;
    for (int c = 0; c < 3; ++c) {
      out.color[c * pixels + pix] = color[c] + T * s.background[c];
      if (ps.total_color)
        ps.total_color[c * pixels + pix] = color[c];
      if (s.need_depth)
        out.normal[c * pixels + pix] = normal[c] / fmaxf(1.f - T, 1e-8f);
    }
    if (s.need_depth)
      out.median_depth[pix] = median == ~0u ? 0.f : depth;
    if (ps.n_contrib)
      ps.n_contrib[pix] = last;
  }
}
} // namespace
void preprocess(const Gaussians &g, const CameraView &c,
                const RenderSettings &s, CameraIntrinsics K, int gx, int gy,
                ws::GaussianState st, int *radii) {
  prepare<<<(g.count + 255) / 256, 256>>>(g, c, s, K, gx, gy, st, radii);
}
void forward(const Gaussians &g, const CameraView &c, const RenderSettings &s,
             CameraIntrinsics K, ws::GaussianState st, ws::TileState ts,
             const unsigned *ids, ws::PixelState ps, const RenderOutputs &out) {
  blend<false><<<dim3((c.width + cfg::kTileWidth - 1) / cfg::kTileWidth,
                      (c.height + cfg::kTileHeight - 1) / cfg::kTileHeight),
                 dim3(cfg::kTileWidth, cfg::kTileHeight)>>>(
      g, c, s, K, st, ts, ids, ps, out, {}, {}, {}, {});
}
void backward(const Gaussians &g, const CameraView &c, const RenderSettings &s,
              CameraIntrinsics K, ws::GaussianState st, ws::TileState ts,
              const unsigned *ids, ws::PixelState ps,
              const ForwardOutputsView &fo, const LossGradients &loss,
              ws::GradState gs, const ModelGradients &mg) {
  blend<true><<<dim3((c.width + cfg::kTileWidth - 1) / cfg::kTileWidth,
                     (c.height + cfg::kTileHeight - 1) / cfg::kTileHeight),
                dim3(cfg::kTileWidth, cfg::kTileHeight)>>>(
      g, c, s, K, st, ts, ids, ps, {}, fo, loss, gs, mg);
}
} // namespace splat_drender::gut
