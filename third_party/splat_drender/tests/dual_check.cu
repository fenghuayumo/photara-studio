// Host-side sanity check: dual-number derivatives of the native fisheye
// geometry vs central finite differences.
#include "device/fisheye.cuh"

#include <cstdio>

static inline float rnorm3df(float a, float b, float c) {
    return 1.f / sqrtf(a * a + b * b + c * c);
}

using namespace splat_drender;

int main() {
    CameraIntrinsics K;
    K.fx = 29.f; K.fy = 31.f; K.cx = 39.2f; K.cy = 31.1f;
    K.mode = CameraMode::opencv_fisheye;
    K.k1 = 0.06f; K.k2 = -0.012f; K.k3 = 0.003f; K.k4 = -0.0004f;

    const float angle = 0.23f, c = cosf(angle), s = sinf(angle);
    float view[16] = {c, 0, -s, 0, 0, 1, 0, 0, s, 0, c, 0, 0, 0, 0, 1};
    mat::Mat3 W = mat::world_rotation_transposed(view);

    float t[3] = {0.9f, 0.32f, 1.3f};
    // Sigma = R^T S^2 R with the test scales/rotation.
    float q[4] = {0.94f, 0.13f, -0.18f, 0.21f};
    mat::Mat3 R = mat::quat_rotation(make_float4(q[0], q[1], q[2], q[3]));
    const float sc[3] = {0.16f, 0.10f, 0.22f};
    mat::Mat3 S2 = mat::Mat3::diag(sc[0] * sc[0], sc[1] * sc[1], sc[2] * sc[2]);
    mat::Mat3 Sig = mat::transpose(R) * S2 * R;
    float sig[6] = {Sig.m[0][0], Sig.m[0][1], Sig.m[0][2],
                    Sig.m[1][1], Sig.m[1][2], Sig.m[2][2]};

    // Dual evaluation: d conic / d Sigma.
    fisheye::Dual td[3] = {fisheye::Dual::variable(t[0], 0),
                           fisheye::Dual::variable(t[1], 1),
                           fisheye::Dual::variable(t[2], 2)};
    fisheye::Dual sd[6];
    for (int i = 0; i < 6; ++i) sd[i] = fisheye::Dual::variable(sig[i], 3 + i);
    const auto dual = fisheye::evaluate(td, sd, W, K, 0.f);

    // Chain: d conic / d scale_y = sum_i d conic / d Sigma_i * dSigma_i/ds_y.
    // dSigma/ds_y = 2 s_y R^T e_yy R  (entries use row 1 of R).
    auto sigma_grad_sy = [&](int entry) {
        const float k = 2.f * sc[1];
        static const int row = 1;
        const int i = (entry == 0 || entry == 1 || entry == 2) ? 0
                     : (entry == 3 || entry == 4) ? 1 : 2;
        const int j = (entry == 0) ? 0
                     : (entry == 1 || entry == 3) ? 1
                     : (entry == 2 || entry == 4) ? 2 : 2;
        // entry index maps: 0:(0,0) 1:(0,1) 2:(0,2) 3:(1,1) 4:(1,2) 5:(2,2)
        const int ii = entry == 0 ? 0 : entry == 1 ? 0 : entry == 2 ? 0
                                                          : entry == 3 ? 1
                                                          : entry == 4 ? 1 : 2;
        const int jj = entry == 0 ? 0 : entry == 1 ? 1 : entry == 2 ? 2
                                                          : entry == 3 ? 1
                                                          : entry == 4 ? 2 : 2;
        (void)i; (void)j;
        return k * R.m[row][ii] * R.m[row][jj];
    };

    const float w[3] = {-1.32303e-05f, 2.f * 0.000495287f, -0.0185415f};
    float dual_total = 0.f;
    for (int e = 0; e < 6; ++e) {
        float g = w[0] * dual.conic[0].d[3 + e] + w[1] * dual.conic[1].d[3 + e] +
                  w[2] * dual.conic[2].d[3 + e];
        dual_total += g * sigma_grad_sy(e);
    }

    // Central differences of the float evaluation.
    auto eval_conic_loss = [&](float sy) {
        mat::Mat3 S2p = mat::Mat3::diag(sc[0] * sc[0], sy * sy, sc[2] * sc[2]);
        mat::Mat3 Sigp = mat::transpose(R) * S2p * R;
        float sp[6] = {Sigp.m[0][0], Sigp.m[0][1], Sigp.m[0][2],
                       Sigp.m[1][1], Sigp.m[1][2], Sigp.m[2][2]};
        const auto out = fisheye::evaluate(t, sp, W, K, 0.f);
        return w[0] * out.conic[0] + w[1] * out.conic[1] + w[2] * out.conic[2];
    };
    const float h = 1e-4f;
    const float fd = (eval_conic_loss(sc[1] + h) - eval_conic_loss(sc[1] - h)) / (2 * h);

    printf("dual dL/ds_y = %.6f   fd = %.6f   ratio = %.4f\n", dual_total, fd,
           fd / dual_total);
    printf("conic = (%.6f, %.6f, %.6f) coef=%.6f plane=(%.6f,%.6f,%.6f,%.6f)\n",
           fisheye::scalar(dual.conic[0]), fisheye::scalar(dual.conic[1]),
           fisheye::scalar(dual.conic[2]), fisheye::scalar(dual.coefficient),
           fisheye::scalar(dual.plane[0]), fisheye::scalar(dual.plane[1]),
           fisheye::scalar(dual.plane[2]), fisheye::scalar(dual.plane[3]));
    return 0;
}
