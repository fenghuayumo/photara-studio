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
// seam-crossing duplicates for equirectangular images.
SD_D2 inline unsigned enumerate(
    float2 mean, float4 conic_opacity, int grid_x, int grid_y,
    unsigned gaussian_id, unsigned offset, unsigned* tile_out,
    unsigned* value_out, int wrap_width) {
    const float a = conic_opacity.x, b = conic_opacity.y, c = conic_opacity.z;
    const float det = a * c - b * b;
    if (!(a > 0.f && c > 0.f) || !(conic_opacity.w >= 1.f / 255.f) ||
        !(det > 0.f))
        return 0;
    const float threshold = 2.f * logf(conic_opacity.w * 255.f);
    if (!(threshold >= 0.f)) return 0;

    unsigned count = 0;
    auto walk = [&](float2 center) {
        // Iterate along the shorter screen axis so the per-slice interval
        // on the long axis is exact and tight.
        const float ex = sqrtf(threshold * c / det);
        const float ey = sqrtf(threshold * a / det);
        const bool iterate_x = (2.f * ex) / cfg::kTileWidth <=
                               (2.f * ey) / cfg::kTileHeight;
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
        const int u_min = max(0, min(gu, int(floorf((cu - eu) / blk_u))));
        const int u_max = max(0, min(gu, int(floorf((cu + eu) / blk_u)) + 1));
        if (u_min >= u_max) return;

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
            const float sv_min =
                (arg_v_min >= s0 && arg_v_min <= s1) ? cv - ev : fminf(lo0, lo1);
            const float sv_max =
                (arg_v_max >= s0 && arg_v_max <= s1) ? cv + ev : fmaxf(hi0, hi1);
            const int v_min = max(0, min(gv, int(floorf(sv_min / blk_v))));
            const int v_max = max(0, min(gv, int(floorf(sv_max / blk_v)) + 1));
            for (int tv = v_min; tv < v_max; ++tv) {
                if (tile_out) {
                    const int tx = iterate_x ? tu : tv;
                    const int ty = iterate_x ? tv : tu;
                    tile_out[offset + count] =
                        unsigned(ty) * unsigned(grid_x) + unsigned(tx);
                    value_out[offset + count] = gaussian_id;
                }
                ++count;
            }
        }
    };

    walk(mean);
    if (wrap_width > 0) {
        const float radius = 3.f * sqrtf(fmaxf(c / det, 0.f));
        const float w = float(wrap_width);
        if (mean.x - radius < 0.f)
            walk(make_float2(mean.x + w, mean.y));
        if (mean.x + radius > w)
            walk(make_float2(mean.x - w, mean.y));
    }
    return count;
}

}  // namespace splat_drender::tiles
