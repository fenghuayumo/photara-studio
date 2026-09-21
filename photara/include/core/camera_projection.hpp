#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define AETHER_CAMERA_HD __host__ __device__
#else
#define AETHER_CAMERA_HD
#endif

namespace photara {
// automatic is a frontend request only; scene cameras always have a concrete model.
enum class CameraModel : std::uint32_t {
    pinhole = 0,
    opencv_fisheye = 1,
    automatic = 2,
    equirectangular = 3
};

AETHER_CAMERA_HD [[nodiscard]] constexpr bool uses_native_splat_projection(
    const CameraModel model) noexcept {
    return model == CameraModel::opencv_fisheye ||
           model == CameraModel::equirectangular;
}

// Models whose pixels live on the full sphere: their projection has no
// cheirality (a point behind the camera is observable) and pixel distances are
// only meaningful in the angular domain (they wrap at the azimuth seam and
// compress toward the poles). Every consumer that compares a reprojection with
// an observation must use the angular/tangent-plane helpers below instead of a
// raw pixel difference.
AETHER_CAMERA_HD [[nodiscard]] constexpr bool uses_bearing_projection(
    const CameraModel model) noexcept {
    return model == CameraModel::equirectangular;
}

inline constexpr double k_pi = 3.14159265358979323846;

// Angular tightening applied when a *pixel* budget is converted into an angular
// tolerance for an equirectangular camera. A sphere chart covers 2 pi radians
// across the same number of pixels a rectilinear lens spends on its (much
// narrower) field of view, so one chart pixel is far coarser in angle: with the
// default focal ratio f = 1.2 * width, a pinhole pixel spans 1 / (1.2 * width)
// radians while a panorama pixel spans 2 pi / width, a ratio of 2 pi * 1.2
// (~7.5). Applying the full ratio would demand sub-pixel precision from every
// individual pair, which starves the two-view fits and the model comparison
// (the pipeline's final BA does reach that precision, a single short-baseline
// pair cannot). A third of the ratio keeps most of the intended precision at a
// fraction of the cost; see docs/SFM_EQUIRECTANGULAR.md for the measurements.
// It affects threshold conversions only, never a reported pixel error.
inline constexpr double k_equirect_threshold_scale = 1.0 / 3.0;

// Four coefficient slots: Brown k1,k2,p1,p2 or fisheye k1,k2,k3,k4.
// Coordinates and derivatives are on the positive-depth normalized plane.
struct CameraProjection {
    double x, y, xx, xy, yx, yy;
    double dx[4], dy[4];
};
AETHER_CAMERA_HD inline CameraProjection project_camera_plane(
    CameraModel model, double x, double y,
    double k1, double k2, double p1, double p2) {
    CameraProjection out{};
    const double r2 = x*x + y*y;
    if (model == CameraModel::opencv_fisheye) {
        const double r = ::sqrt(r2);
        const double theta = ::atan(r);
        const double t2 = theta*theta;
        const double poly = 1.0 + t2*(k1 + t2*(k2 + t2*(p1 + t2*p2)));
        const double s = r > 1e-8 ? theta*poly/r : 1.0;
        const double derivative = 1.0 + t2*(3*k1 + t2*(5*k2 + t2*(7*p1 + t2*9*p2)));
        const double slope = r > 1e-8 ? (derivative/(1+r2)-s)/r2 : 2*(k1-1.0/3.0);
        out.x = x*s; out.y = y*s;
        out.xx = s + x*x*slope; out.xy = x*y*slope;
        out.yx = out.xy; out.yy = s + y*y*slope;
        double power = t2;
        for (int i=0; i<4; ++i) {
            const double factor = r > 1e-8 ? theta*power/r : 0.0;
            out.dx[i] = x*factor; out.dy[i] = y*factor;
            power *= t2;
        }
    } else {
        const double radial = 1 + k1*r2 + k2*r2*r2;
        const double slope = 2*(k1 + 2*k2*r2);
        out.x = x*radial + 2*p1*x*y + p2*(r2+2*x*x);
        out.y = y*radial + p1*(r2+2*y*y) + 2*p2*x*y;
        out.xx = radial+x*x*slope+2*p1*y+6*p2*x;
        out.xy = x*y*slope+2*p1*x+2*p2*y;
        out.yx = out.xy;
        out.yy = radial+y*y*slope+6*p1*y+2*p2*x;
        out.dx[0]=x*r2; out.dx[1]=x*r2*r2;
        out.dx[2]=2*x*y; out.dx[3]=r2+2*x*x;
        out.dy[0]=y*r2; out.dy[1]=y*r2*r2;
        out.dy[2]=r2+2*y*y; out.dy[3]=2*x*y;
    }
    return out;
}

struct CameraRay {
    double x, y, z;
    bool valid;
};

struct CameraPixel {
    double u, v;
    bool valid;
};

// OpenCV fisheye: theta_d = theta * (1 + k1 theta^2 + ... + k4 theta^8).
AETHER_CAMERA_HD inline CameraPixel project_fisheye_camera(
    double X, double Y, double Z,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double k3, double k4) {
    CameraPixel out{};
    if (!(Z > 1e-8)) return out;
    const double radius = ::hypot(X, Y);
    const double theta = ::atan2(radius, Z);
    const double t2 = theta * theta;
    const double poly = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)));
    const double theta_d = theta * poly;
    if (theta_d < 0.0) return out;
    if (radius < 1e-12) {
        out.u = cx;
        out.v = cy;
        out.valid = true;
        return out;
    }
    const double scale = theta_d / radius;
    out.u = fx * scale * X + cx;
    out.v = fy * scale * Y + cy;
    out.valid = true;
    return out;
}

AETHER_CAMERA_HD inline CameraPixel project_equirectangular_camera(
    double X, double Y, double Z, int width, int height) {
    CameraPixel out{};
    const double length = ::sqrt(X * X + Y * Y + Z * Z);
    if (!(length > 1e-12)) return out;
    const double azimuth = ::atan2(X, Z);
    const double elevation = ::asin(
        ::fmax(-1.0, ::fmin(1.0, Y / length)));
    out.u = (azimuth / (2.0 * k_pi) + 0.5) * static_cast<double>(width);
    out.v = (elevation / k_pi + 0.5) * static_cast<double>(height);
    out.valid = true;
    return out;
}

AETHER_CAMERA_HD inline CameraRay unproject_fisheye_camera(
    double u, double v,
    double fx, double fy, double cx, double cy,
    double k1, double k2, double k3, double k4) {
    CameraRay out{};
    const double xn = (u - cx) / fx;
    const double yn = (v - cy) / fy;
    const double radius = ::hypot(xn, yn);
    if (radius < 1e-12) {
        out.x = 0.0;
        out.y = 0.0;
        out.z = 1.0;
        out.valid = true;
        return out;
    }
    auto distorted = [&](double theta) {
        const double t2 = theta * theta;
        return theta * (1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))));
    };
    const double hi = 1.5707963267948966 - 1e-8;
    if (radius >= distorted(hi)) return out;
    double theta = radius < hi ? radius : 0.5 * hi;
    for (int i = 0; i < 50; ++i) {
        const double t2 = theta * theta;
        const double poly = 1.0 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)));
        const double derivative =
            1.0 + t2 * (3.0 * k1 + t2 * (5.0 * k2 + t2 * (7.0 * k3 + t2 * 9.0 * k4)));
        const double error = theta * poly - radius;
        if (::fabs(error) < 1e-12) break;
        theta -= error / ::fmax(derivative, 1e-12);
        theta = ::fmax(0.0, ::fmin(hi, theta));
    }
    const double scale = ::sin(theta) / radius;
    out.x = scale * xn;
    out.y = scale * yn;
    out.z = ::cos(theta);
    out.valid = true;
    return out;
}

AETHER_CAMERA_HD inline CameraRay unproject_equirectangular_camera(
    double u, double v, int width, int height) {
    CameraRay out{};
    if (width <= 0 || height <= 0) return out;
    const double azimuth =
        2.0 * k_pi * (u / static_cast<double>(width) - 0.5);
    const double elevation =
        k_pi * (v / static_cast<double>(height) - 0.5);
    const double cos_el = ::cos(elevation);
    out.x = cos_el * ::sin(azimuth);
    out.y = ::sin(elevation);
    out.z = cos_el * ::cos(azimuth);
    out.valid = true;
    return out;
}

// Equirectangular images are compared in the tangent plane at the observed
// bearing instead of in pixels: the pixel difference wraps by a full width at
// the azimuth seam and its metric meaning shrinks as cos(elevation) toward the
// poles, so neither the residual nor the Jacobian is usable there. The tangent
// basis below is the analytic d(bearing)/d(azimuth) and d(bearing)/d(elevation)
// at the observation, which makes the two residuals well defined everywhere on
// the sphere.
struct EquirectTangentBasis {
    double t1x, t1y, t1z;  // d(bearing)/d(azimuth), unit
    double t2x, t2y, t2z;  // d(bearing)/d(elevation), unit
    double scale_x;        // pixels per radian along t1
    double scale_y;        // pixels per radian along t2
    bool valid;
};

// Pixel scale of an equirectangular image: a full turn covers `width` pixels
// (and a half turn `height`), which is exactly what fx = width / (2 pi) and
// fy = height / pi encode in the camera intrinsics.
AETHER_CAMERA_HD inline double equirect_pixel_scale_x(const int width) {
    return static_cast<double>(width) / (2.0 * k_pi);
}

AETHER_CAMERA_HD inline double equirect_pixel_scale_y(const int height) {
    return static_cast<double>(height) / k_pi;
}

// Tangent basis of the equirectangular chart at an observed pixel. The chart
// angles are recovered from the intrinsics themselves (longitude = (u - cx)/fx,
// elevation = (v - cy)/fy), which is what makes the BA able to evaluate the
// residual without carrying the image size: for an equirectangular camera
// fx = width / (2 pi) and fy = height / pi by construction.
AETHER_CAMERA_HD inline EquirectTangentBasis equirect_tangent_basis(
    double u, double v,
    double fx, double fy, double cx, double cy) {
    EquirectTangentBasis out{};
    out.valid = false;
    if (!(fx > 1e-12) || !(fy > 1e-12)) return out;
    const double azimuth = (u - cx) / fx;
    const double elevation = (v - cy) / fy;
    const double sin_lon = ::sin(azimuth), cos_lon = ::cos(azimuth);
    const double sin_lat = ::sin(elevation), cos_lat = ::cos(elevation);
    out.t1x = cos_lon;
    out.t1y = 0.0;
    out.t1z = -sin_lon;
    out.t2x = -sin_lat * sin_lon;
    out.t2y = cos_lat;
    out.t2z = -sin_lat * cos_lon;
    out.scale_x = fx;
    out.scale_y = fy;
    out.valid = true;
    return out;
}

// Tangent-plane reprojection residual and its 2x3 Jacobian with respect to the
// camera-space point. Residuals are in pixels (the tangent basis is pre-scaled)
// so Huber thresholds and reprojection limits keep their usual units.
// Singular only exactly at the poles, where the chart itself is degenerate for
// the residual direction along t1; the basis still returns a usable value.
struct EquirectLocalReprojection {
    double residual_x, residual_y;
    double j00, j01, j02;  // d residual_x / d p_cam
    double j10, j11, j12;  // d residual_y / d p_cam
    bool valid;
};

AETHER_CAMERA_HD inline EquirectLocalReprojection equirect_local_reprojection(
    double px, double py, double pz,
    const EquirectTangentBasis& basis) {
    EquirectLocalReprojection out{};
    out.residual_x = 0.0;
    out.residual_y = 0.0;
    out.j00 = out.j01 = out.j02 = 0.0;
    out.j10 = out.j11 = out.j12 = 0.0;
    out.valid = false;
    if (!basis.valid) return out;
    const double length = ::sqrt(px * px + py * py + pz * pz);
    if (!(length > 1e-12) || !::isfinite(length)) return out;
    const double inv_length = 1.0 / length;
    const double bx = px * inv_length;
    const double by = py * inv_length;
    const double bz = pz * inv_length;

    // (I - b b^T) / |p|: derivative of the predicted unit bearing.
    const double d00 = (1.0 - bx * bx) * inv_length;
    const double d01 = -bx * by * inv_length;
    const double d02 = -bx * bz * inv_length;
    const double d10 = -by * bx * inv_length;
    const double d11 = (1.0 - by * by) * inv_length;
    const double d12 = -by * bz * inv_length;
    const double d20 = -bz * bx * inv_length;
    const double d21 = -bz * by * inv_length;
    const double d22 = (1.0 - bz * bz) * inv_length;

    // Residual: the predicted bearing projected onto the tangent basis of the
    // observation, scaled to pixels. Zero exactly when the rays coincide.
    out.residual_x =
        basis.scale_x * (bx * basis.t1x + by * basis.t1y + bz * basis.t1z);
    out.residual_y =
        basis.scale_y * (bx * basis.t2x + by * basis.t2y + bz * basis.t2z);
    out.j00 = basis.scale_x * (basis.t1x * d00 + basis.t1y * d10 + basis.t1z * d20);
    out.j01 = basis.scale_x * (basis.t1x * d01 + basis.t1y * d11 + basis.t1z * d21);
    out.j02 = basis.scale_x * (basis.t1x * d02 + basis.t1y * d12 + basis.t1z * d22);
    out.j10 = basis.scale_y * (basis.t2x * d00 + basis.t2y * d10 + basis.t2z * d20);
    out.j11 = basis.scale_y * (basis.t2x * d01 + basis.t2y * d11 + basis.t2z * d21);
    out.j12 = basis.scale_y * (basis.t2x * d02 + basis.t2y * d12 + basis.t2z * d22);
    out.valid = ::isfinite(out.residual_x) && ::isfinite(out.residual_y) &&
                ::isfinite(out.j00) && ::isfinite(out.j01) && ::isfinite(out.j02) &&
                ::isfinite(out.j10) && ::isfinite(out.j11) && ::isfinite(out.j12);
    return out;
}

// Angular distance (radians) between two unit bearings.
AETHER_CAMERA_HD inline double bearing_angle(
    const double ax, const double ay, const double az,
    const double bx, const double by, const double bz) {
    const double dot = ax * bx + ay * by + az * bz;
    return ::acos(::fmax(-1.0, ::fmin(1.0, dot)));
}
} // namespace photara
#undef AETHER_CAMERA_HD
