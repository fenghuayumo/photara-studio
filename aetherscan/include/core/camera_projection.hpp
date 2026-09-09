#pragma once
#include <cmath>
#include <cstdint>

#if defined(__CUDACC__)
#define AETHER_CAMERA_HD __host__ __device__
#else
#define AETHER_CAMERA_HD
#endif

namespace aetherscan {
// automatic is a frontend request only; scene cameras always have a concrete model.
enum class CameraModel : std::uint32_t {
    pinhole = 0,
    opencv_fisheye = 1,
    automatic = 2,
    equirectangular = 3
};

[[nodiscard]] constexpr bool uses_native_splat_projection(
    const CameraModel model) noexcept {
    return model == CameraModel::opencv_fisheye ||
           model == CameraModel::equirectangular;
}

inline constexpr double k_pi = 3.14159265358979323846;

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
} // namespace aetherscan
#undef AETHER_CAMERA_HD
