// Opacity-aware tile walk. emit_mode 0 counts, 1 writes tile keys, 2 writes packed keys.
#ifndef PHOTARA_SPLAT_ENUM_HLSLI
#define PHOTARA_SPLAT_ENUM_HLSLI

// Per-invocation GUT context. Both count and emit initialize it from the
// identical cached rows. GS and unsupported camera/condition cases leave it off.
static bool gut_support_enabled = false;
static float3 gut_c0, gut_c1, gut_c2, gut_origin;
static float gut_sign, gut_support2;
static float4 gut_intrinsics;
static uint2 gut_extent;
static int4 gut_tile_bounds;

bool gut_outside(float3 n) {
    precise float3 local = gut_sign*float3(dot(gut_c0,n),dot(gut_c1,n),dot(gut_c2,n));
    precise float center = -dot(local,gut_origin);
    center += 1e-3f*dot(abs(local),abs(gut_origin));
    return center < 0.0f && center*center > gut_support2*dot(local,local)*1.0001f;
}
float3 gut_axis_plane(uint axis, float pixel, bool lower) {
    float slope = (pixel-gut_intrinsics[2u+axis])/gut_intrinsics[axis];
    return axis == 0u ? (lower ? float3(1,0,-slope) : float3(-1,0,slope))
                      : (lower ? float3(0,1,-slope) : float3(0,-1,slope));
}
// Clip the screen interval with support halfspaces, even when the ellipsoid
// crosses z=0. The negative support set is convex in the screen coordinate;
// bisect only when an endpoint is outside and the opposite one is inside.
int2 gut_axis_bounds(uint axis, uint extent) {
    float lo = 0.0f, hi = float(extent-1u), left = lo, right = hi;
    if (gut_outside(gut_axis_plane(axis,lo,true)) ||
        gut_outside(gut_axis_plane(axis,hi,false))) return int2(1,0);
    if (gut_outside(gut_axis_plane(axis,lo,false))) {
        [loop] for (uint k=0u;k<12u;++k) {
            float mid = (lo+hi)*0.5f;
            if (gut_outside(gut_axis_plane(axis,mid,false))) lo=mid; else hi=mid;
        }
        left = max(0.0f,lo-1.0f);
    }
    lo=0.0f; hi=float(extent-1u);
    if (gut_outside(gut_axis_plane(axis,hi,true))) {
        [loop] for (uint k=0u;k<12u;++k) {
            float mid=(lo+hi)*0.5f;
            if (gut_outside(gut_axis_plane(axis,mid,true))) hi=mid; else lo=mid;
        }
        right=min(float(extent-1u),hi+1.0f);
    }
    return int2(int(floor(left/16.0f)),int(floor(right/16.0f))+1);
}
void gut_init_support(float4 row0,float4 row1,float4 row2,float opacity,
                      float support2,float4 intrinsics,uint2 extent,bool enabled) {
    gut_support_enabled = false;
    gut_c0=0.0f;gut_c1=0.0f;gut_c2=0.0f;gut_origin=0.0f;
    gut_support2=0.0f;gut_sign=0.0f;gut_tile_bounds=0;
    float3 lengths=float3(dot(row0.xyz,row0.xyz),dot(row1.xyz,row1.xyz),dot(row2.xyz,row2.xyz));
    if (!enabled || any(extent>1048560u) || !(min(intrinsics.x,intrinsics.y)>0.0f) ||
        !(max(lengths.x,max(lengths.y,lengths.z)) < 1e8f*min(lengths.x,min(lengths.y,lengths.z)))) return;
    gut_c0=cross(row1.xyz,row2.xyz); gut_c1=cross(row2.xyz,row0.xyz); gut_c2=cross(row0.xyz,row1.xyz);
    float det=dot(row0.xyz,gut_c0);
    if (!finite1(det) || abs(det)<1e-20f) return;
    gut_sign=det>=0.0f ? 1.0f : -1.0f;
    gut_origin=float3(row0.w,row1.w,row2.w);
    gut_support2=max(0.0f,support2)+1e-3f;
    gut_intrinsics=intrinsics; gut_extent=extent;
    gut_support_enabled=true;
    int2 x=gut_axis_bounds(0u,extent.x),y=gut_axis_bounds(1u,extent.y);
    gut_tile_bounds=int4(x.x,y.x,x.y,y.y);
}
bool gut_accept_tile(int tx,int ty) {
    if (!gut_support_enabled) return true;
    if (tx<gut_tile_bounds.x || ty<gut_tile_bounds.y || tx>=gut_tile_bounds.z || ty>=gut_tile_bounds.w) return false;
    float2 lo=float2(tx,ty)*16.0f, hi=min(lo+15.0f,float2(gut_extent)-1.0f);
    return !gut_outside(gut_axis_plane(0u,lo.x,true)) && !gut_outside(gut_axis_plane(0u,hi.x,false)) &&
           !gut_outside(gut_axis_plane(1u,lo.y,true)) && !gut_outside(gut_axis_plane(1u,hi.y,false));
}

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
    if (gut_support_enabled) {
        u_min=max(u_min,iterate_x ? gut_tile_bounds.x : gut_tile_bounds.y);
        u_max=min(u_max,iterate_x ? gut_tile_bounds.z : gut_tile_bounds.w);
    }
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
        if (gut_support_enabled) {
            v_min=max(v_min,iterate_x ? gut_tile_bounds.y : gut_tile_bounds.x);
            v_max=min(v_max,iterate_x ? gut_tile_bounds.w : gut_tile_bounds.z);
        }
        [loop] for (int tv = v_min; tv < v_max; ++tv) {
            int tx = iterate_x ? tu : tv;
            int ty = iterate_x ? tv : tu;
            if (!gut_accept_tile(tx,ty)) continue;
            if (emit_mode != 0) {
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
