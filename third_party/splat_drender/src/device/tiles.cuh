// Gaussian-to-tile enumeration.
//
// Uses the opacity-aware SnugBox/AccuTile bound: the iso-contour where
// alpha falls to 1/255 defines the exact tile footprint of a Gaussian,
// so tiles outside the truncated ellipse cannot affect any pixel.
#pragma once

#include "matrix.cuh"
#include "splat_drender/config.h"

namespace splat_drender::tiles {

// Walks the AccuTile footprint. When `emit` is null only the touched-tile
// count is produced; otherwise (tile_id, gaussian_id) pairs are written to
// `tile_out` / `value_out` starting at `offset`. `wrap_width` > 0 adds the
// seam-crossing coverage for equirectangular images. Each Gaussian/tile pair
// must occur once, even when its footprint covers more than one panorama.
SD_D2 inline unsigned enumerate(float2 mean, float4 conic_opacity, int grid_x, int grid_y,
                                unsigned gaussian_id, unsigned offset, unsigned* tile_out,
                                unsigned* value_out, int wrap_width,
                                unsigned long long* packed_out = nullptr, unsigned depth = 0,
                                const ushort4* clip = nullptr) {
    // GUT's full-sphere fallback has no finite longitude ellipse. Emit each
    // tile once: wrapping a huge ellipse can duplicate its Gaussian and
    // composite opacity repeatedly, and a low-opacity ellipse need not cover
    // the whole image even when its covariance was expanded to screen size.
    if (wrap_width > 0 && conic_opacity.x == -1.f && conic_opacity.y == 0.f &&
        conic_opacity.z == -1.f && conic_opacity.w >= 1.f / 255.f) {
        const unsigned count = unsigned(grid_x) * unsigned(grid_y);
        if (tile_out || packed_out)
            for (unsigned tile = 0; tile < count; ++tile) {
                if (packed_out)
                    packed_out[offset + tile] =
                        (static_cast<unsigned long long>(tile) << 32) | depth;
                else
                    tile_out[offset + tile] = tile;
                value_out[offset + tile] = gaussian_id;
            }
        return count;
    }
    const float a = conic_opacity.x, b = conic_opacity.y, c = conic_opacity.z;
    const float det = a * c - b * b;
    if (!(a > 0.f && c > 0.f) || !(conic_opacity.w >= 1.f / 255.f) || !(det > 0.f))
        return 0;
    const float threshold = 2.f * logf(conic_opacity.w * 255.f);
    if (!(threshold >= 0.f))
        return 0;

    unsigned count = 0;
    if (wrap_width > 0) {
        // Slice by tile row, where a convex ellipse has one horizontal
        // interval. Wrap that interval in pixel coordinates before converting
        // it to tiles, then union its at most two tile ranges. Walking shifted
        // ellipses separately duplicates candidates when ranges overlap (also
        // for sub-tile seams and widths not divisible by the tile size).
        const float w = float(wrap_width);
        const float ey = sqrtf(threshold * a / det);
        const float ex = sqrtf(threshold * c / det);
        const float arg_min = mean.y + b * ex / c;
        const float arg_max = mean.y - b * ex / c;
        const int y_min = max(0, min(grid_y, int(floorf((mean.y - ey) / cfg::kTileHeight))));
        const int y_max = max(0, min(grid_y, int(floorf((mean.y + ey) / cfg::kTileHeight)) + 1));
        auto emit_range = [&](int begin, int end, int ty) {
            for (int tx = begin; tx < end; ++tx) {
                const unsigned tile = unsigned(ty) * unsigned(grid_x) + unsigned(tx);
                if (tile_out || packed_out) {
                    if (packed_out)
                        packed_out[offset + count] =
                            (static_cast<unsigned long long>(tile) << 32) | depth;
                    else
                        tile_out[offset + count] = tile;
                    value_out[offset + count] = gaussian_id;
                }
                ++count;
            }
        };
        for (int ty = y_min; ty < y_max; ++ty) {
            const float s0 = fmaxf(mean.y - ey, float(ty * cfg::kTileHeight));
            const float s1 = fminf(mean.y + ey, float((ty + 1) * cfg::kTileHeight));
            const float d0 = s0 - mean.y, d1 = s1 - mean.y;
            const float r0 = sqrtf(fmaxf(0.f, a * threshold - det * d0 * d0));
            const float r1 = sqrtf(fmaxf(0.f, a * threshold - det * d1 * d1));
            const float left = arg_min >= s0 && arg_min <= s1 ? mean.x - ex
                                                              : fminf(mean.x + (-b * d0 - r0) / a,
                                                                      mean.x + (-b * d1 - r1) / a);
            const float right = arg_max >= s0 && arg_max <= s1 ? mean.x + ex
                                                               : fmaxf(mean.x + (-b * d0 + r0) / a,
                                                                       mean.x + (-b * d1 + r1) / a);
            const float span = fmaxf(0.f, right - left);
            if (span >= w) {
                emit_range(0, grid_x, ty);
                continue;
            }
            float lo = left - w * floorf(left / w);
            if (lo >= w)
                lo = 0.f;
            if (lo < 0.f)
                lo += w;
            const float hi = lo + span;
            const int begin = max(0, min(grid_x, int(floorf(lo / cfg::kTileWidth))));
            if (hi < w) {
                emit_range(begin, max(0, min(grid_x, int(floorf(hi / cfg::kTileWidth)) + 1)), ty);
            } else {
                const int end = max(0, min(grid_x, int(floorf((hi - w) / cfg::kTileWidth)) + 1));
                if (end >= begin)
                    emit_range(0, grid_x, ty);
                else {
                    emit_range(0, end, ty);
                    emit_range(begin, grid_x, ty);
                }
            }
        }
        return count;
    }
    auto walk = [&](float2 center) {
        // Iterate along the shorter screen axis so the per-slice interval
        // on the long axis is exact and tight.
        const float ex = sqrtf(threshold * c / det);
        const float ey = sqrtf(threshold * a / det);
        const bool iterate_x = (2.f * ex) / cfg::kTileWidth <= (2.f * ey) / cfg::kTileHeight;
        const float cu = iterate_x ? center.x : center.y;
        const float cv = iterate_x ? center.y : center.x;
        const float au = iterate_x ? a : c;
        const float av = iterate_x ? c : a;
        const float bu = iterate_x ? b : b;
        const int gu = iterate_x ? grid_x : grid_y;
        const int gv = iterate_x ? grid_y : grid_x;
        const float blk_u = iterate_x ? float(cfg::kTileWidth) : float(cfg::kTileHeight);
        const float blk_v = iterate_x ? float(cfg::kTileHeight) : float(cfg::kTileWidth);

        const float eu = sqrtf(threshold * av / det);
        const float ev = sqrtf(threshold * au / det);
        const int clip_u_min = clip ? (iterate_x ? clip->x : clip->z) : 0;
        const int clip_u_max = clip ? (iterate_x ? clip->y : clip->w) : gu;
        const int clip_v_min = clip ? (iterate_x ? clip->z : clip->x) : 0;
        const int clip_v_max = clip ? (iterate_x ? clip->w : clip->y) : gv;
        const int u_min = max(clip_u_min, min(gu, int(floorf((cu - eu) / blk_u))));
        const int u_max = min(clip_u_max, min(gu, int(floorf((cu + eu) / blk_u)) + 1));
        if (u_min >= u_max)
            return;

        // u coordinates where the v extrema of the ellipse occur.
        const float arg_v_min = cu + bu * ev / au;
        const float arg_v_max = cu - bu * ev / au;

        for (int tu = u_min; tu < u_max; ++tu) {
            const float s0 = fmaxf(cu - eu, tu * blk_u);
            const float s1 = fminf(cu + eu, (tu + 1) * blk_u);
            const float d0 = s0 - cu, d1 = s1 - cu;
            const float r0 = sqrtf(fmaxf(0.f, av * threshold - det * d0 * d0));
            const float r1 = sqrtf(fmaxf(0.f, av * threshold - det * d1 * d1));
            const float lo0 = cv + (-bu * d0 - r0) / av;
            const float lo1 = cv + (-bu * d1 - r1) / av;
            const float hi0 = cv + (-bu * d0 + r0) / av;
            const float hi1 = cv + (-bu * d1 + r1) / av;
            const float sv_min = (arg_v_min >= s0 && arg_v_min <= s1) ? cv - ev : fminf(lo0, lo1);
            const float sv_max = (arg_v_max >= s0 && arg_v_max <= s1) ? cv + ev : fmaxf(hi0, hi1);
            const int v_min = max(clip_v_min, min(gv, int(floorf(sv_min / blk_v))));
            const int v_max = min(clip_v_max, min(gv, int(floorf(sv_max / blk_v)) + 1));
            for (int tv = v_min; tv < v_max; ++tv) {
                if (tile_out || packed_out) {
                    const int tx = iterate_x ? tu : tv;
                    const int ty = iterate_x ? tv : tu;
                    const unsigned tile = unsigned(ty) * unsigned(grid_x) + unsigned(tx);
                    if (packed_out)
                        packed_out[offset + count] =
                            (static_cast<unsigned long long>(tile) << 32) | depth;
                    else
                        tile_out[offset + count] = tile;
                    value_out[offset + count] = gaussian_id;
                }
                ++count;
            }
        }
    };

    walk(mean);
    return count;
}

} // namespace splat_drender::tiles
