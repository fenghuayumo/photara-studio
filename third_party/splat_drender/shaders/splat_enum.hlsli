// Opacity-aware tile walk. emit_mode 0 counts, 1 writes tile keys, 2 writes packed keys.
#ifndef PHOTARA_SPLAT_ENUM_HLSLI
#define PHOTARA_SPLAT_ENUM_HLSLI

uint walk_tiles(float2 center, int grid_x, int grid_y, float a, float b, float c, float det, float threshold,
                uint gaussian_id, uint offset, uint depth, uint emit_mode) {
    float ex = sqrt(threshold * c / det);
    float ey = sqrt(threshold * a / det);
    bool iterate_x = (2.0f * ex) / kTileSize <= (2.0f * ey) / kTileSize;
    float cu = iterate_x ? center.x : center.y;
    float cv = iterate_x ? center.y : center.x;
    float au = iterate_x ? a : c;
    float av = iterate_x ? c : a;
    float bu = b;
    int gu = iterate_x ? grid_x : grid_y;
    int gv = iterate_x ? grid_y : grid_x;
    float blk_u = kTileSize;
    float blk_v = kTileSize;
    float eu = sqrt(threshold * av / det);
    float ev = sqrt(threshold * au / det);
    int u_min = max(0, min(gu, int(floor((cu - eu) / blk_u))));
    int u_max = max(0, min(gu, int(floor((cu + eu) / blk_u)) + 1));
    if (u_min >= u_max) return 0;

    float arg_v_min = cu + bu * ev / au;
    float arg_v_max = cu - bu * ev / au;
    uint count = 0;
    [loop] for (int tu = u_min; tu < u_max; ++tu) {
        float s0 = max(cu - eu, float(tu) * blk_u);
        float s1 = min(cu + eu, float(tu + 1) * blk_u);
        float d0 = s0 - cu;
        float d1 = s1 - cu;
        float r0 = sqrt(max(0.0f, av * threshold - det * d0 * d0));
        float r1 = sqrt(max(0.0f, av * threshold - det * d1 * d1));
        float lo0 = cv + (-bu * d0 - r0) / av;
        float lo1 = cv + (-bu * d1 - r1) / av;
        float hi0 = cv + (-bu * d0 + r0) / av;
        float hi1 = cv + (-bu * d1 + r1) / av;
        float sv_min = (arg_v_min >= s0 && arg_v_min <= s1) ? cv - ev : min(lo0, lo1);
        float sv_max = (arg_v_max >= s0 && arg_v_max <= s1) ? cv + ev : max(hi0, hi1);
        int v_min = max(0, min(gv, int(floor(sv_min / blk_v))));
        int v_max = max(0, min(gv, int(floor(sv_max / blk_v)) + 1));
        [loop] for (int tv = v_min; tv < v_max; ++tv) {
            if (emit_mode != 0) {
                int tx = iterate_x ? tu : tv;
                int ty = iterate_x ? tv : tu;
                uint tile = uint(ty) * uint(grid_x) + uint(tx);
                uint slot = offset + count;
                if (emit_mode == 2) {
                    key_lo[slot] = depth;
                    key_hi[slot] = tile;
                } else {
                    key_lo[slot] = tile;
                }
                values[slot] = gaussian_id;
            }
            ++count;
        }
    }
    return count;
}

uint enumerate_tiles(float2 mean, float4 conic, int grid_x, int grid_y, uint gaussian_id, uint offset,
                     int wrap_width, uint depth, uint emit_mode) {
    float a = conic.x;
    float b = conic.y;
    float c = conic.z;
    float det = a * c - b * b;
    if (!(a > 0.0f && c > 0.0f) || !(conic.w >= kAlphaFloor) || !(det > 0.0f)) return 0;
    float threshold = 2.0f * log(conic.w * 255.0f);
    if (!(threshold >= 0.0f)) return 0;
    uint total = walk_tiles(mean, grid_x, grid_y, a, b, c, det, threshold, gaussian_id, offset, depth, emit_mode);
    if (wrap_width > 0) {
        float radius = 3.0f * sqrt(max(c / det, 0.0f));
        float w = float(wrap_width);
        if (mean.x - radius < 0.0f) {
            total += walk_tiles(float2(mean.x + w, mean.y), grid_x, grid_y, a, b, c, det, threshold,
                                gaussian_id, offset + total, depth, emit_mode);
        }
        if (mean.x + radius > w) {
            total += walk_tiles(float2(mean.x - w, mean.y), grid_x, grid_y, a, b, c, det, threshold,
                                gaussian_id, offset + total, depth, emit_mode);
        }
    }
    return total;
}

#endif
