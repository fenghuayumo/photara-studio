#pragma once

#include "matrix.cuh"

namespace splat_drender::footprint {

// Capped ellipse semi-major axis: sqrt(8 * lambda_max(C)). The screen
// kernel is included in C; opacity compensation still uses the uncapped C.
struct Cap { float q, lambda, da, db, dc; };

SD_D2 inline Cap policy(float a, float b, float c, int width, int height) {
    const float half = .5f * (a - c);
    const float radius = hypotf(half, b);
    const float lambda = .5f * a + .5f * c + radius;
    const float limit = fminf(1024.f, float(min(width, height)));
    const float variance = .125f * limit * limit;
    if (!(lambda > variance) || !isfinite(lambda)) return {1.f, lambda, 0.f, 0.f, 0.f};
    // At an isotropic covariance use the symmetric eigenvalue subgradient.
    const float h = radius > 0.f ? half / radius : 0.f;
    return {variance / lambda, lambda, .5f + .5f*h,
            radius > 0.f ? b / radius : 0.f, .5f - .5f*h};
}

SD_D2 inline void apply(float& a, float& b, float& c, int width, int height) {
    const auto cap = policy(a,b,c,width,height);
    if (cap.q < 1.f) { a *= cap.q; b *= cap.q; c *= cap.q; }
}

// Gradient of inverse(q*C) followed by the complete q(C) VJP. dconic.y
// stores half of the off-diagonal derivative, as in the raster backward.
SD_D2 inline float3 conic_backward(float a, float b, float c, Cap cap, float4 dconic) {
    const float x = a*cap.q, y = b*cap.q, z = c*cap.q;
    const float det = x*z-y*y;
    const float u = z/det, v = -y/det, w = x/det;
    const float ga = -(u*u*dconic.x + 2.f*u*v*dconic.y + v*v*dconic.z);
    const float gb = -2.f*(u*v*dconic.x + (u*w+v*v)*dconic.y + v*w*dconic.z);
    const float gc = -(v*v*dconic.x + 2.f*v*w*dconic.y + w*w*dconic.z);
    const float pull = (ga*a + gb*b + gc*c) / cap.lambda;
    return make_float3(cap.q*(ga-pull*cap.da), cap.q*(gb-pull*cap.db),
                       cap.q*(gc-pull*cap.dc));
}

} // namespace splat_drender::footprint
