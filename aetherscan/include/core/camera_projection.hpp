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
enum class CameraModel : std::uint32_t { pinhole = 0, opencv_fisheye = 1, automatic = 2 };

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
} // namespace aetherscan
#undef AETHER_CAMERA_HD
