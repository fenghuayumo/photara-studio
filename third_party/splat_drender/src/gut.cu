#include "device/gut.cuh"
#include "device/sh.cuh"
#include "device/tiles.cuh"
#include "gut.h"
#include "splat_drender/config.h"
#include <cstdlib>

namespace splat_drender::gut {
namespace {
static_assert(cfg::kTileWidth == 16 && cfg::kTileHeight == 16,
              "3DGUT bucket traversal requires 16x16 tiles");
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
  const float3 origin = make_float3(
      cam.center[0] - mean.x, cam.center[1] - mean.y, cam.center[2] - mean.z);
  st.gut_refine[i] = make_float2(1.f / (fabsf(center.J.du[0]) + 1e-6f),
                                 1.f / (fabsf(center.J.dv[1]) + 1e-6f));
  for (int a = 0; a < 3; ++a) {
    const float inv = 1.f / (settings.scale_modifier * g.scales[3 * i + a]);
    const float3 b =
        make_float3(R.m[a][0] * inv, R.m[a][1] * inv, R.m[a][2] * inv);
    st.gut_rows[3 * i + a] = make_float4(b.x, b.y, b.z, mat::dot3(b, origin));
  }
  const unsigned mask = __activemask();
#if __CUDA_ARCH__ >= 800
  const unsigned sum = __reduce_add_sync(mask, touched);
  if (int(threadIdx.x & 31) == __ffs(mask) - 1) {
    atomicAdd(st.n_visible, unsigned(__popc(mask)));
    atomicAdd(st.n_instances, sum);
  }
#else
  atomicAdd(st.n_visible, 1u);
  atomicAdd(st.n_instances, touched);
#endif
}

template <bool BACKWARD>
__global__ void reference_blend(Gaussians g, CameraView cam, RenderSettings s,
                                CameraIntrinsics K, ws::GaussianState st,
                                ws::TileState tiles, const unsigned *ids,
                                ws::PixelState ps, RenderOutputs out,
                                ForwardOutputsView fo, LossGradients loss,
                                ws::GradState grad, ModelGradients model_grad) {
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
__global__ void blend_rgb(Gaussians g, CameraView cam, RenderSettings s,
                          CameraIntrinsics K, ws::GaussianState st,
                          ws::TileState tiles, const unsigned *ids,
                          ws::PixelState ps, RenderOutputs out) {
  constexpr int batch_size = cfg::kTileWidth * cfg::kTileHeight;
  __shared__ float4 rows0[batch_size], rows1[batch_size], rows2[batch_size];
  __shared__ float4 colors[batch_size];
  __shared__ unsigned gaussian_ids[batch_size];
  const int rank = threadIdx.y * cfg::kTileWidth + threadIdx.x;
  const int x = blockIdx.x * cfg::kTileWidth + threadIdx.x;
  const int y = blockIdx.y * cfg::kTileHeight + threadIdx.y;
  const bool inside = x < cam.width && y < cam.height;
  const int pix = y * cam.width + x, pixels = cam.width * cam.height;
  const uint2 range = tiles.range[blockIdx.y * gridDim.x + blockIdx.x];
  const unsigned tile = blockIdx.y * gridDim.x + blockIdx.x;
  const unsigned bucket_base =
      ps.snap_ct && tile > 0 ? tiles.bucket_offset[tile - 1] : 0;
  {
    if (ps.snap_ct) {
      for (unsigned b = rank; b < (range.y - range.x + 31) / 32;
           b += batch_size)
        tiles.bucket_tile[bucket_base + b] = tile;
    }
  }
  float3 origin, dir;
  float ray_z;
  ray(cam, K, x, y, origin, dir, ray_z);
  const float3 offset =
      make_float3(origin.x - cam.center[0], origin.y - cam.center[1],
                  origin.z - cam.center[2]);
  float T = 1.f;
  float3 color = make_float3(0, 0, 0);
  unsigned last = 0;
  bool done = !inside;
  for (unsigned base = range.x; base < range.y; base += batch_size) {
    if (__syncthreads_count(!done) == 0)
      break;
    const unsigned batch = min(unsigned(batch_size), range.y - base);
    if (unsigned(rank) < batch) {
      const unsigned i = ids[base + rank];
      gaussian_ids[rank] = i;
      rows0[rank] = st.gut_rows[3 * i];
      rows1[rank] = st.gut_rows[3 * i + 1];
      rows2[rank] = st.gut_rows[3 * i + 2];
      const float3 c =
          g.colors ? reinterpret_cast<const float3 *>(g.colors)[i] : st.rgb[i];
      colors[rank] = make_float4(c.x, c.y, c.z, g.opacities[i]);
    }
    __syncthreads();
    for (unsigned j = 0; j < batch && !done; ++j) {
      {
        if (ps.snap_ct && (j & 31) == 0) {
          const std::size_t slot =
              (std::size_t(bucket_base + (base - range.x + j) / 32) << 8) +
              rank;
          ps.snap_ct[slot] = make_float4(color.x, color.y, color.z, T);
        }
      }
      const float4 a = rows0[j], b = rows1[j], c = rows2[j];
      const float3 o =
          make_float3(a.w + a.x * offset.x + a.y * offset.y + a.z * offset.z,
                      b.w + b.x * offset.x + b.y * offset.y + b.z * offset.z,
                      c.w + c.x * offset.x + c.y * offset.y + c.z * offset.z);
      const float3 d = make_float3(a.x * dir.x + a.y * dir.y + a.z * dir.z,
                                   b.x * dir.x + b.y * dir.y + b.z * dir.z,
                                   c.x * dir.x + c.y * dir.y + c.z * dir.z);
      const float dd = mat::dot3(d, d);
      const float peak = -mat::dot3(o, d) / dd;
      if (!(peak > 0.f) || !(dd > 0.f))
        continue;
      const float3 q =
          make_float3(o.x + peak * d.x, o.y + peak * d.y, o.z + peak * d.z);
      const float response = expf(-.5f * mat::dot3(q, q));
      const float4 rgb = colors[j];
      const float raw = rgb.w * response, alpha = fminf(cfg::kAlphaClip, raw);
      if (!(alpha >= cfg::kAlphaFloor))
        continue;
      const float next = T * (1.f - alpha);
      if (next < cfg::kTransmittanceFloor) {
        done = true;
        break;
      }
      const float weight = T * alpha;
      const unsigned i = gaussian_ids[j];
      {
        color.x += weight * rgb.x;
        color.y += weight * rgb.y;
        color.z += weight * rgb.z;
        const unsigned active = __activemask();
        if ((rank & 31) == __ffs(active) - 1)
          atomicExch(reinterpret_cast<unsigned *>(out.visibility + i),
                     __float_as_uint(1.f));
      }
      T = next;
      last = base + j - range.x + 1;
    }
    __syncthreads();
  }
  {
    if (ps.snap_ct) {
      __shared__ unsigned tile_max[8];
      unsigned maximum = last;
      for (int offset = 16; offset > 0; offset >>= 1)
        maximum = max(maximum, __shfl_down_sync(0xffffffffu, maximum, offset));
      if ((rank & 31) == 0)
        tile_max[rank >> 5] = maximum;
      __syncthreads();
      if (rank == 0) {
        unsigned maximum = 0;
        for (int w = 0; w < 8; ++w)
          maximum = max(maximum, tile_max[w]);
        tiles.max_contributor[tile] = maximum;
      }
    }
    if (inside) {
      if (ps.gut_rays)
        ps.gut_rays[pix] = make_float4(dir.x, dir.y, dir.z, 0.f);
      out.alpha[pix] = 1.f - T;
      out.color[pix] = color.x + T * s.background[0];
      out.color[pixels + pix] = color.y + T * s.background[1];
      out.color[2 * pixels + pix] = color.z + T * s.background[2];
      if (ps.total_color) {
        ps.total_color[pix] = color.x;
        ps.total_color[pixels + pix] = color.y;
        ps.total_color[2 * pixels + pix] = color.z;
      }
      if (ps.n_contrib)
        ps.n_contrib[pix] = last;
    }
  }
}

// One lane owns one Gaussian. Pixel state flows through its 32-Gaussian
// bucket, so gradients stay in registers and commit once per tile instance.
// This replaces atomics per accepted pixel with atomics per Gaussian/tile.
__global__ void backward_buckets(Gaussians g, CameraView cam, RenderSettings s,
                                 CameraIntrinsics K, ws::GaussianState st,
                                 ws::TileState ts, const unsigned *ids,
                                 ws::PixelState ps, ForwardOutputsView fo,
                                 LossGradients loss, ws::GradState grad,
                                 ModelGradients mg, int buckets) {
  const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
  const int bucket = blockIdx.x * 8 + warp;
  if (bucket >= buckets)
    return;
  const unsigned tile = ts.bucket_tile[bucket];
  const uint2 range = ts.range[tile];
  const unsigned first = tile == 0 ? 0 : ts.bucket_offset[tile - 1];
  const unsigned position = (bucket - first) * 32 + lane;
  if ((bucket - first) * 32 >= ts.max_contributor[tile])
    return;
  const bool valid = position < range.y - range.x;
  const unsigned id = valid ? ids[range.x + position] : 0;
  float4 a{}, b{}, c{}, rgb{};
  float2 rf{};
  if (valid) {
    a = st.gut_rows[3 * id];
    b = st.gut_rows[3 * id + 1];
    c = st.gut_rows[3 * id + 2];
    const float3 color =
        g.colors ? reinterpret_cast<const float3 *>(g.colors)[id] : st.rgb[id];
    rgb = make_float4(color.x, color.y, color.z, g.opacities[id]);
    rf = st.gut_refine[id];
  }
  const int grid_x = (cam.width + 15) / 16;
  const int x0 = (tile % grid_x) * 16, y0 = (tile / grid_x) * 16;
  const int pixels = cam.width * cam.height;
  __shared__ float4 state[8][32], constants[8][32], directions[8][32],
      offsets[8][32];
  __shared__ unsigned limits[8][32];
  float T = 0, rx = 0, ry = 0, rz = 0, gx = 0, gy = 0, gz = 0, dfinal = 0,
        final_T = 0;
  float dx = 0, dy = 0, dz = 0, ox = 0, oy = 0, oz = 0, error = 0;
  unsigned limit = 0;
  float acc_color[3] = {}, acc_rows[9] = {}, acc_o[3] = {};
  float acc_opacity = 0, acc_refine = 0, acc_error = 0, acc_den = 0;
  for (int step = 0; step < 256 + 31; ++step) {
    if ((step & 31) == 0 && step < 256) {
      // All lanes must finish consuming the previous stage before any lane
      // overwrites it; independent thread scheduling can run ahead otherwise.
      __syncwarp();
      const int local = step + lane, x = x0 + (local & 15),
                y = y0 + (local >> 4);
      const bool pin = x < cam.width && y < cam.height;
      const int pix = y * cam.width + x;
      const unsigned last = pin ? ps.n_contrib[pix] : 0;
      float4 initial = make_float4(0, 0, 0, 0),
             constant = make_float4(0, 0, 0, 0);
      float4 direction = make_float4(0, 0, 0, 1),
             offset = make_float4(0, 0, 0, 0);
      // Inactive snapshots are deliberately uninitialized; never load
      // them unless this pixel reached at least one lane of the bucket.
      if (pin && last > (bucket - first) * 32) {
        const float4 snap = ps.snap_ct[(std::size_t(bucket) << 8) + local];
        initial = make_float4(
            ps.total_color[pix] - snap.x, ps.total_color[pixels + pix] - snap.y,
            ps.total_color[2 * pixels + pix] - snap.z, snap.w);
        constant = make_float4(loss.color[pix], loss.color[pixels + pix],
                               loss.color[2 * pixels + pix], 0);
        constant.w = s.background[0] * constant.x +
                     s.background[1] * constant.y +
                     s.background[2] * constant.z - loss.alpha[pix];
        direction = ps.gut_rays[pix];
        direction.w = 1.f - fo.alpha[pix];
        if (is_orthographic(K.mode)) {
          const float3 off = mat::xform_dir_transpose(
              make_float3((x - K.cx) / K.fx, (y - K.cy) / K.fy, 0),
              cam.world_to_camera);
          offset = make_float4(off.x, off.y, off.z, 0);
        }
        if (loss.densify_map)
          offset.w = loss.densify_map[pix];
      }
      state[warp][lane] = initial;
      constants[warp][lane] = constant;
      directions[warp][lane] = direction;
      offsets[warp][lane] = offset;
      limits[warp][lane] = last;
      __syncwarp();
    }
    if (step > 0) {
      T = __shfl_up_sync(0xffffffffu, T, 1);
      rx = __shfl_up_sync(0xffffffffu, rx, 1);
      ry = __shfl_up_sync(0xffffffffu, ry, 1);
      rz = __shfl_up_sync(0xffffffffu, rz, 1);
      gx = __shfl_up_sync(0xffffffffu, gx, 1);
      gy = __shfl_up_sync(0xffffffffu, gy, 1);
      gz = __shfl_up_sync(0xffffffffu, gz, 1);
      dfinal = __shfl_up_sync(0xffffffffu, dfinal, 1);
      final_T = __shfl_up_sync(0xffffffffu, final_T, 1);
      dx = __shfl_up_sync(0xffffffffu, dx, 1);
      dy = __shfl_up_sync(0xffffffffu, dy, 1);
      dz = __shfl_up_sync(0xffffffffu, dz, 1);
      ox = __shfl_up_sync(0xffffffffu, ox, 1);
      oy = __shfl_up_sync(0xffffffffu, oy, 1);
      oz = __shfl_up_sync(0xffffffffu, oz, 1);
      limit = __shfl_up_sync(0xffffffffu, limit, 1);
      error = __shfl_up_sync(0xffffffffu, error, 1);
    }
    const int pixel = step - lane;
    if (lane == 0 && pixel >= 0 && pixel < 256) {
      const float4 v = state[warp][pixel & 31],
                   co = constants[warp][pixel & 31];
      const float4 d = directions[warp][pixel & 31],
                   o = offsets[warp][pixel & 31];
      T = v.w;
      rx = v.x;
      ry = v.y;
      rz = v.z;
      gx = co.x;
      gy = co.y;
      gz = co.z;
      dfinal = co.w;
      dx = d.x;
      dy = d.y;
      dz = d.z;
      final_T = d.w;
      ox = o.x;
      oy = o.y;
      oz = o.z;
      error = o.w;
      limit = limits[warp][pixel & 31];
    }
    if (!valid || pixel < 0 || pixel >= 256 || position >= limit)
      continue;
    const float3 o = make_float3(a.w + a.x * ox + a.y * oy + a.z * oz,
                                 b.w + b.x * ox + b.y * oy + b.z * oz,
                                 c.w + c.x * ox + c.y * oy + c.z * oz);
    const float3 d = make_float3(a.x * dx + a.y * dy + a.z * dz,
                                 b.x * dx + b.y * dy + b.z * dz,
                                 c.x * dx + c.y * dy + c.z * dz);
    const float dd = mat::dot3(d, d), peak = -mat::dot3(o, d) / dd;
    if (!(peak > 0.f) || !(dd > 0.f))
      continue;
    const float3 q =
        make_float3(o.x + peak * d.x, o.y + peak * d.y, o.z + peak * d.z);
    const float response = expf(-.5f * mat::dot3(q, q));
    const float raw = rgb.w * response, alpha = fminf(cfg::kAlphaClip, raw);
    if (!(alpha >= cfg::kAlphaFloor))
      continue;
    const float weight = T * alpha;
    rx -= weight * rgb.x;
    ry -= weight * rgb.y;
    rz -= weight * rgb.z;
    const float da =
        T * (gx * rgb.x + gy * rgb.y + gz * rgb.z) -
        (gx * rx + gy * ry + gz * rz + final_T * dfinal) / (1.f - alpha);
    const float factor = raw < cfg::kAlphaClip ? -da * raw : 0.f;
    // Retain the floating-point orthogonality residual. Dropping this
    // term is exact algebraically, but destabilizes very thin Gaussians.
    const float h = factor * mat::dot3(q, d) / dd;
    const float3 go = make_float3(
        factor * q.x - h * d.x, factor * q.y - h * d.y, factor * q.z - h * d.z);
    const float3 v =
        make_float3(ox + peak * dx, oy + peak * dy, oz + peak * dz);
    acc_color[0] += weight * gx;
    acc_color[1] += weight * gy;
    acc_color[2] += weight * gz;
    acc_o[0] += go.x;
    acc_o[1] += go.y;
    acc_o[2] += go.z;
    acc_rows[0] += go.x * v.x - h * q.x * dx;
    acc_rows[1] += go.x * v.y - h * q.x * dy;
    acc_rows[2] += go.x * v.z - h * q.x * dz;
    acc_rows[3] += go.y * v.x - h * q.y * dx;
    acc_rows[4] += go.y * v.y - h * q.y * dy;
    acc_rows[5] += go.y * v.z - h * q.y * dz;
    acc_rows[6] += go.z * v.x - h * q.z * dx;
    acc_rows[7] += go.z * v.y - h * q.z * dy;
    acc_rows[8] += go.z * v.z - h * q.z * dz;
    if (raw < cfg::kAlphaClip)
      acc_opacity += da * response;
    if (mg.refine_weight) {
      const float3 dm = make_float3(-a.x * go.x - b.x * go.y - c.x * go.z,
                                    -a.y * go.x - b.y * go.y - c.y * go.z,
                                    -a.z * go.x - b.z * go.y - c.z * go.z);
      const float3 gm =
          mat::transpose(mat::world_rotation_transposed(cam.world_to_camera)) *
          dm;
      acc_refine += hypotf(gm.x * rf.x * cam.width, gm.y * rf.y * cam.height) /
                    fmaxf(1.f - final_T, 1e-6f);
    }
    if (loss.densify_map) {
      acc_error += weight * error;
      acc_den += weight;
    }
    T *= 1.f - alpha;
  }
  if (valid) {
    for (int k = 0; k < 3; ++k) {
      atomicAdd(reinterpret_cast<float *>(grad.d_color) + 3 * id + k,
                acc_color[k]);
      atomicAdd(reinterpret_cast<float *>(grad.d_conic) + 4 * id + k,
                acc_rows[k]);
      atomicAdd(reinterpret_cast<float *>(grad.d_ray_plane) + 4 * id + k,
                acc_rows[3 + k]);
      atomicAdd(reinterpret_cast<float *>(grad.d_normal) + 3 * id + k,
                acc_rows[6 + k]);
      atomicAdd(reinterpret_cast<float *>(grad.d_mean2d) + 3 * id + k,
                acc_o[k]);
    }
    atomicAdd(&grad.d_conic[id].w, acc_opacity);
    if (mg.refine_weight)
      atomicAdd(mg.refine_weight + id, acc_refine);
    if (mg.densify_weight && mg.densify_weight_den) {
      atomicAdd(mg.densify_weight + id, acc_error);
      atomicAdd(mg.densify_weight_den + id, acc_den);
    }
  }
}

// Convert accumulated transform/origin gradients to activated geometry once
// per Gaussian, before the existing SH derivative and fused Adam finalizer.
__global__ void transform_backward(Gaussians g, CameraView cam,
                                   RenderSettings s, ws::GaussianState st,
                                   ws::GradState grad) {
  const int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i >= g.count || st.radius[i] == 0)
    return;
  const float3 go = grad.d_mean2d[i];
  const float4 b0 = st.gut_rows[3 * i], b1 = st.gut_rows[3 * i + 1],
               b2 = st.gut_rows[3 * i + 2];
  const float3 u = make_float3(cam.center[0] - g.means[3 * i],
                               cam.center[1] - g.means[3 * i + 1],
                               cam.center[2] - g.means[3 * i + 2]);
  float D[3][3] = {
      {grad.d_conic[i].x + go.x * u.x, grad.d_conic[i].y + go.x * u.y,
       grad.d_conic[i].z + go.x * u.z},
      {grad.d_ray_plane[i].x + go.y * u.x, grad.d_ray_plane[i].y + go.y * u.y,
       grad.d_ray_plane[i].z + go.y * u.z},
      {grad.d_normal[i].x + go.z * u.x, grad.d_normal[i].y + go.z * u.y,
       grad.d_normal[i].z + go.z * u.z}};
  const float3 ds = make_float3(
      -(D[0][0] * b0.x + D[0][1] * b0.y + D[0][2] * b0.z) / g.scales[3 * i],
      -(D[1][0] * b1.x + D[1][1] * b1.y + D[1][2] * b1.z) / g.scales[3 * i + 1],
      -(D[2][0] * b2.x + D[2][1] * b2.y + D[2][2] * b2.z) /
          g.scales[3 * i + 2]);
  for (int a = 0; a < 3; ++a)
    for (int b = 0; b < 3; ++b)
      D[a][b] /= g.scales[3 * i + a] * s.scale_modifier;
  const float4 q = reinterpret_cast<const float4 *>(g.rotations)[i];
  const float w = q.x, x = q.y, y = q.z, z = q.w;
  const float4 dq =
      make_float4(2.f * (z * (D[0][1] - D[1][0]) + y * (D[2][0] - D[0][2]) +
                         x * (D[1][2] - D[2][1])),
                  2.f * (y * (D[0][1] + D[1][0]) + z * (D[0][2] + D[2][0]) +
                         w * (D[1][2] - D[2][1])) -
                      4.f * x * (D[1][1] + D[2][2]),
                  2.f * (x * (D[0][1] + D[1][0]) + w * (D[2][0] - D[0][2]) +
                         z * (D[1][2] + D[2][1])) -
                      4.f * y * (D[0][0] + D[2][2]),
                  2.f * (w * (D[0][1] - D[1][0]) + x * (D[0][2] + D[2][0]) +
                         y * (D[1][2] + D[2][1])) -
                      4.f * z * (D[0][0] + D[1][1]));
  grad.d_mean2d[i] = make_float3(-b0.x * go.x - b1.x * go.y - b2.x * go.z,
                                 -b0.y * go.x - b1.y * go.y - b2.y * go.z,
                                 -b0.z * go.x - b1.z * go.y - b2.z * go.z);
  const float opacity = grad.d_conic[i].w;
  grad.d_conic[i] = make_float4(ds.x, ds.y, ds.z, opacity);
  grad.d_ray_plane[i] = dq;
}

bool reference_mode() {
  const char *value = std::getenv("PHOTARA_GUT_REFERENCE");
  return value && value[0] == '1';
}
} // namespace
bool use_rgb_snapshots(const RenderSettings &s) {
  return s.record_backward_state && !s.need_depth && !reference_mode();
}
void preprocess(const Gaussians &g, const CameraView &c,
                const RenderSettings &s, CameraIntrinsics K, int gx, int gy,
                ws::GaussianState st, int *radii) {
  prepare<<<(g.count + 255) / 256, 256>>>(g, c, s, K, gx, gy, st, radii);
}
void forward(const Gaussians &g, const CameraView &c, const RenderSettings &s,
             CameraIntrinsics K, ws::GaussianState st, ws::TileState ts,
             const unsigned *ids, ws::PixelState ps, const RenderOutputs &out) {
  if (!s.need_depth && !reference_mode()) {
    blend_rgb<<<dim3((c.width + cfg::kTileWidth - 1) / cfg::kTileWidth,
                     (c.height + cfg::kTileHeight - 1) / cfg::kTileHeight),
                dim3(cfg::kTileWidth, cfg::kTileHeight)>>>(g, c, s, K, st, ts,
                                                           ids, ps, out);
    return;
  }
  reference_blend<false>
      <<<dim3((c.width + cfg::kTileWidth - 1) / cfg::kTileWidth,
              (c.height + cfg::kTileHeight - 1) / cfg::kTileHeight),
         dim3(cfg::kTileWidth, cfg::kTileHeight)>>>(g, c, s, K, st, ts, ids, ps,
                                                    out, {}, {}, {}, {});
}
void backward(const Gaussians &g, const CameraView &c, const RenderSettings &s,
              CameraIntrinsics K, ws::GaussianState st, ws::TileState ts,
              const unsigned *ids, ws::PixelState ps,
              const ForwardOutputsView &fo, const LossGradients &loss,
              ws::GradState gs, const ModelGradients &mg, int buckets) {
  if (!s.need_depth && !reference_mode()) {
    backward_buckets<<<(buckets + 7) / 8, 256>>>(g, c, s, K, st, ts, ids, ps,
                                                 fo, loss, gs, mg, buckets);
    transform_backward<<<(g.count + 255) / 256, 256>>>(g, c, s, st, gs);
    return;
  }
  reference_blend<true>
      <<<dim3((c.width + cfg::kTileWidth - 1) / cfg::kTileWidth,
              (c.height + cfg::kTileHeight - 1) / cfg::kTileHeight),
         dim3(cfg::kTileWidth, cfg::kTileHeight)>>>(g, c, s, K, st, ts, ids, ps,
                                                    {}, fo, loss, gs, mg);
}
} // namespace splat_drender::gut
