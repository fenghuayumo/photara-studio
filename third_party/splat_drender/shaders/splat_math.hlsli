// Shared EWA math for the Vulkan splat forward. Formulas follow
// splat_drender so the blended image matches the CUDA backend.
#ifndef PHOTARA_SPLAT_MATH_HLSLI
#define PHOTARA_SPLAT_MATH_HLSLI

static const uint kModePinhole = 0;
static const uint kModeFisheye = 1;
static const uint kModeEquirect = 3;
static const uint kModeOrtho = 4;
static const float kAlphaClip = 0.99f;
static const float kAlphaFloor = 1.0f / 255.0f;
static const float kTransmittanceFloor = 1.0e-4f;
static const float kDepthSeedWindow = 0.4f;
static const float kDepthMinTransmittance = 0.45f;
static const int kDepthSplit = 8;
static const int kDepthRefinements = 5;
static const float kTileSize = 16.0f;
static const float kPi = 3.14159265358979323846f;

struct PushConstants {
    uint u0;
    uint u1;
    uint u2;
    uint u3;
    uint u4;
    uint u5;
    uint u6;
    uint u7;
    uint u8;
    uint u9;
    uint u10;
    uint u11;
    uint u12;
    uint u13;
    uint u14;
    uint u15;
    uint u16;
    uint u17;
    uint u18;
    uint u19;
};

[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

struct Mat3 {
    float m[3][3];
};

float mat_at(Mat3 a, int col, int row) {
    return a.m[row][col];
}

Mat3 mat_zero() {
    Mat3 o;
    [unroll] for (int i = 0; i < 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j) o.m[i][j] = 0.0f;
    return o;
}

Mat3 mat_diag(float x, float y, float z) {
    Mat3 o = mat_zero();
    o.m[0][0] = x;
    o.m[1][1] = y;
    o.m[2][2] = z;
    return o;
}

Mat3 mat_mul(Mat3 a, Mat3 b) {
    Mat3 o = mat_zero();
    [unroll] for (int i = 0; i < 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j)
            o.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j];
    return o;
}

float3 mat_mul_vec(Mat3 a, float3 v) {
    return float3(
        a.m[0][0] * v.x + a.m[0][1] * v.y + a.m[0][2] * v.z,
        a.m[1][0] * v.x + a.m[1][1] * v.y + a.m[1][2] * v.z,
        a.m[2][0] * v.x + a.m[2][1] * v.y + a.m[2][2] * v.z);
}

Mat3 mat_transpose(Mat3 a) {
    Mat3 o = mat_zero();
    [unroll] for (int i = 0; i < 3; ++i)
        [unroll] for (int j = 0; j < 3; ++j) o.m[i][j] = a.m[j][i];
    return o;
}

Mat3 mat_outer(float3 u, float3 v) {
    Mat3 o = mat_zero();
    o.m[0][0] = u.x * v.x;
    o.m[0][1] = u.x * v.y;
    o.m[0][2] = u.x * v.z;
    o.m[1][0] = u.y * v.x;
    o.m[1][1] = u.y * v.y;
    o.m[1][2] = u.y * v.z;
    o.m[2][0] = u.z * v.x;
    o.m[2][1] = u.z * v.y;
    o.m[2][2] = u.z * v.z;
    return o;
}

float3 xform_point(float3 p, float v0, float v1, float v2, float v4, float v5, float v6,
                   float v8, float v9, float v10, float v12, float v13, float v14) {
    return float3(
        v0 * p.x + v4 * p.y + v8 * p.z + v12,
        v1 * p.x + v5 * p.y + v9 * p.z + v13,
        v2 * p.x + v6 * p.y + v10 * p.z + v14);
}

Mat3 world_rotation_transposed(float v0, float v1, float v2, float v4, float v5, float v6,
                               float v8, float v9, float v10) {
    Mat3 w = mat_zero();
    w.m[0][0] = v0;
    w.m[0][1] = v1;
    w.m[0][2] = v2;
    w.m[1][0] = v4;
    w.m[1][1] = v5;
    w.m[1][2] = v6;
    w.m[2][0] = v8;
    w.m[2][1] = v9;
    w.m[2][2] = v10;
    return w;
}

Mat3 quat_rotation(float4 q) {
    float r = q.x;
    float x = q.y;
    float y = q.z;
    float z = q.w;
    Mat3 o = mat_zero();
    o.m[0][0] = 1.0f - 2.0f * (y * y + z * z);
    o.m[0][1] = 2.0f * (x * y + r * z);
    o.m[0][2] = 2.0f * (x * z - r * y);
    o.m[1][0] = 2.0f * (x * y - r * z);
    o.m[1][1] = 1.0f - 2.0f * (x * x + z * z);
    o.m[1][2] = 2.0f * (y * z + r * x);
    o.m[2][0] = 2.0f * (x * z + r * y);
    o.m[2][1] = 2.0f * (y * z - r * x);
    o.m[2][2] = 1.0f - 2.0f * (x * x + y * y);
    return o;
}

bool finite1(float v) {
    return !isnan(v) && !isinf(v);
}

bool finite3(float3 v) {
    return finite1(v.x) && finite1(v.y) && finite1(v.z);
}

bool camera_visible(float3 t, uint mode) {
    if (mode == kModeEquirect) {
        return finite3(t) && sqrt(dot(t, t)) > 1.0e-6f;
    }
    return t.z > (mode == kModeFisheye ? 1.0e-6f : 0.2f);
}

struct Projection {
    float pixel_x;
    float pixel_y;
    float du0, du1, du2;
    float dv0, dv1, dv2;
    bool valid;
};

Projection project_ortho(float3 t, float fx, float fy, float cx, float cy) {
    Projection p;
    p.pixel_x = 0.0f;
    p.pixel_y = 0.0f;
    p.du0 = 0.0f;
    p.du1 = 0.0f;
    p.du2 = 0.0f;
    p.dv0 = 0.0f;
    p.dv1 = 0.0f;
    p.dv2 = 0.0f;
    p.valid = false;
    if (!(t.z > 0.2f)) return p;
    p.pixel_x = t.x * fx + cx;
    p.pixel_y = t.y * fy + cy;
    p.du0 = fx;
    p.dv1 = fy;
    p.valid = finite1(p.pixel_x) && finite1(p.pixel_y);
    return p;
}

Projection project_pinhole(float3 t, float fx, float fy, float cx, float cy) {
    Projection p;
    p.pixel_x = 0.0f;
    p.pixel_y = 0.0f;
    p.du0 = 0.0f;
    p.du1 = 0.0f;
    p.du2 = 0.0f;
    p.dv0 = 0.0f;
    p.dv1 = 0.0f;
    p.dv2 = 0.0f;
    p.valid = false;
    if (!(t.z > 0.2f)) return p;
    float iz = 1.0f / t.z;
    p.pixel_x = t.x * iz * fx + cx;
    p.pixel_y = t.y * iz * fy + cy;
    p.du0 = fx * iz;
    p.du1 = 0.0f;
    p.du2 = -(fx * t.x) * iz * iz;
    p.dv0 = 0.0f;
    p.dv1 = fy * iz;
    p.dv2 = -(fy * t.y) * iz * iz;
    p.valid = true;
    return p;
}

Projection project_fisheye(float3 t, float fx, float fy, float cx, float cy,
                           float k1, float k2, float k3, float k4) {
    Projection p;
    p.pixel_x = 0.0f;
    p.pixel_y = 0.0f;
    p.du0 = 0.0f;
    p.du1 = 0.0f;
    p.du2 = 0.0f;
    p.dv0 = 0.0f;
    p.dv1 = 0.0f;
    p.dv2 = 0.0f;
    p.valid = false;
    if (!(t.z > 1.0e-6f)) return p;
    float r = sqrt(t.x * t.x + t.y * t.y);
    float theta = atan2(r, t.z);
    if (!(theta < 1.57079632679f)) return p;
    float th2 = theta * theta;
    float poly = 1.0f + th2 * (k1 + th2 * (k2 + th2 * (k3 + th2 * k4)));
    float theta_d = theta * poly;
    if (!(theta_d >= 0.0f)) return p;
    float dtheta_d = 1.0f + th2 * (3.0f * k1 + th2 * (5.0f * k2 + th2 * (7.0f * k3 + th2 * 9.0f * k4)));
    if (!(dtheta_d > 1.0e-6f)) return p;

    float r2 = t.x * t.x + t.y * t.y;
    float l2 = r2 + t.z * t.z;
    float du_dx, du_dy, du_dz, dv_dx, dv_dy, dv_dz;
    if (r2 < 1.0e-6f * t.z * t.z) {
        float iz = 1.0f / t.z;
        float q = r2 * iz * iz;
        float a = k1 - 1.0f / 3.0f;
        float b = k2 - k1 + 1.0f / 5.0f;
        float s = iz * (1.0f + a * q + b * q * q);
        float ds = 2.0f * iz * iz * iz * (a + 2.0f * b * q);
        float dz = -iz * iz * (1.0f + 3.0f * a * q + 5.0f * b * q * q);
        p.pixel_x = cx + fx * t.x * s;
        p.pixel_y = cy + fy * t.y * s;
        du_dx = fx * (s + t.x * t.x * ds);
        du_dy = fx * t.x * t.y * ds;
        du_dz = fx * t.x * dz;
        dv_dx = fy * t.x * t.y * ds;
        dv_dy = fy * (s + t.y * t.y * ds);
        dv_dz = fy * t.y * dz;
    } else {
        float rho = theta_d / r;
        p.pixel_x = fx * rho * t.x + cx;
        p.pixel_y = fy * rho * t.y + cy;
        float dth_dx = t.z * t.x / (l2 * r);
        float dth_dy = t.z * t.y / (l2 * r);
        float dth_dz = -r / l2;
        float dtd_dx = dtheta_d * dth_dx;
        float dtd_dy = dtheta_d * dth_dy;
        float dtd_dz = dtheta_d * dth_dz;
        float ir3 = 1.0f / (r * r * r);
        float drho_dx = dtd_dx / r - theta_d * t.x * ir3;
        float drho_dy = dtd_dy / r - theta_d * t.y * ir3;
        float drho_dz = dtd_dz / r;
        du_dx = fx * (rho + t.x * drho_dx);
        du_dy = fx * t.x * drho_dy;
        du_dz = fx * t.x * drho_dz;
        dv_dx = fy * t.y * drho_dx;
        dv_dy = fy * (rho + t.y * drho_dy);
        dv_dz = fy * t.y * drho_dz;
    }
    p.du0 = du_dx;
    p.du1 = du_dy;
    p.du2 = du_dz;
    p.dv0 = dv_dx;
    p.dv1 = dv_dy;
    p.dv2 = dv_dz;
    p.valid = !isnan(p.pixel_x) && !isnan(p.pixel_y) && !isinf(p.pixel_x) && !isinf(p.pixel_y);
    return p;
}

Projection project_equirect(float3 t, int width, int height) {
    Projection p;
    p.pixel_x = 0.0f;
    p.pixel_y = 0.0f;
    p.du0 = 0.0f;
    p.du1 = 0.0f;
    p.du2 = 0.0f;
    p.dv0 = 0.0f;
    p.dv1 = 0.0f;
    p.dv2 = 0.0f;
    p.valid = false;
    float l = sqrt(dot(t, t));
    if (!(l > 1.0e-6f) || width <= 0 || height <= 0) return p;
    float horiz = sqrt(t.x * t.x + t.z * t.z);
    if (horiz < 1.0e-5f) return p;
    float az = atan2(t.x, t.z);
    float el = atan2(t.y, horiz);
    p.pixel_x = (az / (2.0f * kPi) + 0.5f) * float(width);
    p.pixel_y = (el / kPi + 0.5f) * float(height);
    float l2 = l * l;
    float h2 = horiz * horiz;
    float daz_dx = t.z / h2;
    float daz_dz = -t.x / h2;
    float del_dx = -t.y * t.x / (horiz * l2);
    float del_dy = horiz / l2;
    float del_dz = -t.y * t.z / (horiz * l2);
    float su = float(width) / (2.0f * kPi);
    float sv = float(height) / kPi;
    p.du0 = su * daz_dx;
    p.du1 = 0.0f;
    p.du2 = su * daz_dz;
    p.dv0 = sv * del_dx;
    p.dv1 = sv * del_dy;
    p.dv2 = sv * del_dz;
    p.valid = !isnan(p.pixel_x) && !isnan(p.pixel_y) && !isinf(p.pixel_x) && !isinf(p.pixel_y);
    return p;
}

Projection project_camera(float3 t, uint mode, int width, int height, float fx, float fy, float cx, float cy,
                          float k1, float k2, float k3, float k4) {
    if (mode == kModeFisheye) return project_fisheye(t, fx, fy, cx, cy, k1, k2, k3, k4);
    if (mode == kModeEquirect) return project_equirect(t, width, height);
    return project_pinhole(t, fx, fy, cx, cy);
}

float3 pixel_unit_ray(float px, float py, uint mode, float fx, float fy, float cx, float cy,
                      float k1, float k2, float k3, float k4) {
    float x = (px - cx) / fx;
    float y = (py - cy) / fy;
    if (mode == kModeEquirect) {
        float cos_elevation = cos(y);
        return float3(cos_elevation * sin(x), sin(y), cos_elevation * cos(x));
    }
    if (mode != kModeFisheye) {
        float inv = rsqrt(x * x + y * y + 1.0f);
        return float3(x * inv, y * inv, inv);
    }
    float r = sqrt(x * x + y * y);
    float lo = 0.0f;
    float hi = 1.57079632679f;
    [loop] for (int i = 0; i < 28; ++i) {
        float mid = 0.5f * (lo + hi);
        float q = mid * mid;
        float td = mid * (1.0f + q * (k1 + q * (k2 + q * (k3 + q * k4))));
        if (td < r) lo = mid;
        else hi = mid;
    }
    float theta = 0.5f * (lo + hi);
    float scale = r > 1.0e-8f ? sin(theta) / r : 1.0f;
    return float3(x * scale, y * scale, max(0.0f, cos(theta)));
}

float pixel_ray_z(float px, float py, uint mode, float fx, float fy, float cx, float cy,
                  float k1, float k2, float k3, float k4) {
    if (mode == kModeEquirect) return 1.0f;
    return pixel_unit_ray(px, py, mode, fx, fy, cx, cy, k1, k2, k3, k4).z;
}

float wrap_dx(float dx, int wrap_width, uint mode) {
    if (mode != kModeEquirect || wrap_width <= 0) return dx;
    float w = float(wrap_width);
    return dx - w * floor((dx + 0.5f * w) / w);
}

float gaussian_power(float4 conic, float dx, float dy) {
    precise float xx = conic.x * dx;
    precise float yy = (conic.z * dy) * dy;
    precise float xy = (conic.y * dx) * dy;
    precise float fused = mad(dx, xx, yy);
    precise float half_term = -0.5f * fused;
    return half_term - xy;
}

void sym_eig3(inout Mat3 a, out float eig0, out float eig1, out float eig2, out Mat3 vecs) {
    vecs = mat_diag(1.0f, 1.0f, 1.0f);
    [loop] for (int sweep = 0; sweep < 16; ++sweep) {
        float off = abs(a.m[0][1]) + abs(a.m[0][2]) + abs(a.m[1][2]);
        if (off < 1.0e-12f) break;
        [unroll] for (int p = 0; p < 2; ++p) {
            [unroll] for (int q = p + 1; q < 3; ++q) {
                if (abs(a.m[p][q]) < 1.0e-15f) continue;
                float theta = 0.5f * (a.m[q][q] - a.m[p][p]) / a.m[p][q];
                float t = (theta >= 0.0f ? 1.0f : -1.0f) / (abs(theta) + sqrt(theta * theta + 1.0f));
                float c = 1.0f / sqrt(t * t + 1.0f);
                float s = t * c;
                [unroll] for (int k = 0; k < 3; ++k) {
                    float akp = a.m[k][p];
                    float akq = a.m[k][q];
                    a.m[k][p] = c * akp - s * akq;
                    a.m[k][q] = s * akp + c * akq;
                }
                [unroll] for (int k2 = 0; k2 < 3; ++k2) {
                    float apk = a.m[p][k2];
                    float aqk = a.m[q][k2];
                    a.m[p][k2] = c * apk - s * aqk;
                    a.m[q][k2] = s * apk + c * aqk;
                }
                [unroll] for (int k3 = 0; k3 < 3; ++k3) {
                    float vkp = vecs.m[k3][p];
                    float vkq = vecs.m[k3][q];
                    vecs.m[k3][p] = c * vkp - s * vkq;
                    vecs.m[k3][q] = s * vkp + c * vkq;
                }
            }
        }
    }
    eig0 = a.m[0][0];
    eig1 = a.m[1][1];
    eig2 = a.m[2][2];
}

int min_index3(float a, float b, float c) {
    int i = 0;
    float v = a;
    if (b < v) { i = 1; v = b; }
    if (c < v) i = 2;
    return i;
}

struct FishResult {
    float cov0, cov1, cov2;
    float coef;
    float4 plane;
    float3 normal;
    bool ok;
};

FishResult fisheye_evaluate(float3 t, float c0, float c1, float c2, float c3, float c4, float c5,
                            Mat3 W, float fx, float fy, float k1, float k2, float k3, float k4, float kernel) {
    FishResult built;
    built.cov0 = 0.0f;
    built.cov1 = 0.0f;
    built.cov2 = 0.0f;
    built.coef = 0.0f;
    built.plane = 0.0f;
    built.normal = 0.0f;
    built.ok = false;
    float x = t.x;
    float y = t.y;
    float z = t.z;
    float r2 = x * x + y * y;
    float l2 = r2 + z * z;
    float ray_len = sqrt(l2);
    float j00, j01, j02, j10, j11, j12;
    if (r2 < 1.0e-6f * z * z) {
        float iz = 1.0f / z;
        float q = r2 * iz * iz;
        float a = k1 - 1.0f / 3.0f;
        float b = k2 - k1 + 1.0f / 5.0f;
        float s = iz * (1.0f + a * q + b * q * q);
        float ds = 2.0f * iz * iz * iz * (a + 2.0f * b * q);
        float dz = -iz * iz * (1.0f + 3.0f * a * q + 5.0f * b * q * q);
        j00 = fx * (s + x * x * ds);
        j01 = fx * x * y * ds;
        j02 = fx * x * dz;
        j10 = fy * x * y * ds;
        j11 = fy * (s + y * y * ds);
        j12 = fy * y * dz;
    } else {
        float r = sqrt(r2);
        float theta = atan2(r, z);
        float q = theta * theta;
        float td = theta * (1.0f + q * (k1 + q * (k2 + q * (k3 + q * k4))));
        float dt = 1.0f + q * (3.0f * k1 + q * (5.0f * k2 + q * (7.0f * k3 + q * 9.0f * k4)));
        float s = td / r;
        float ds = (dt * z / l2 - s) / r2;
        j00 = fx * (s + x * x * ds);
        j01 = fx * x * y * ds;
        j02 = -fx * dt * x / l2;
        j10 = fy * x * y * ds;
        j11 = fy * (s + y * y * ds);
        j12 = -fy * dt * y / l2;
    }

    float C[3][3];
    [unroll] for (int i = 0; i < 3; ++i) {
        [unroll] for (int j = 0; j < 3; ++j) {
            C[i][j] = c0 * mat_at(W, i, 0) * mat_at(W, j, 0) +
                      c1 * (mat_at(W, i, 0) * mat_at(W, j, 1) + mat_at(W, i, 1) * mat_at(W, j, 0)) +
                      c2 * (mat_at(W, i, 0) * mat_at(W, j, 2) + mat_at(W, i, 2) * mat_at(W, j, 0)) +
                      c3 * mat_at(W, i, 1) * mat_at(W, j, 1) +
                      c4 * (mat_at(W, i, 1) * mat_at(W, j, 2) + mat_at(W, i, 2) * mat_at(W, j, 1)) +
                      c5 * mat_at(W, i, 2) * mat_at(W, j, 2);
        }
    }

    float a = 0.0f;
    float b = 0.0f;
    float c = 0.0f;
    [unroll] for (int ii = 0; ii < 3; ++ii) {
        [unroll] for (int jj = 0; jj < 3; ++jj) {
            float j0 = ii == 0 ? j00 : (ii == 1 ? j01 : j02);
            float j1 = ii == 0 ? j10 : (ii == 1 ? j11 : j12);
            float jt0 = jj == 0 ? j00 : (jj == 1 ? j01 : j02);
            float jt1 = jj == 0 ? j10 : (jj == 1 ? j11 : j12);
            a += j0 * C[ii][jj] * jt0;
            b += j0 * C[ii][jj] * jt1;
            c += j1 * C[ii][jj] * jt1;
        }
    }
    float det0 = max(a * c - b * b, 1.0e-6f);
    a += kernel;
    c += kernel;
    float det = a * c - b * b;
    built.cov0 = a;
    built.cov1 = b;
    built.cov2 = c;
    built.coef = sqrt(det0 / max(det, 1.0e-6f));

    float A[3][3];
    A[0][0] = C[1][1] * C[2][2] - C[1][2] * C[2][1];
    A[0][1] = C[0][2] * C[2][1] - C[0][1] * C[2][2];
    A[0][2] = C[0][1] * C[1][2] - C[0][2] * C[1][1];
    A[1][0] = A[0][1];
    A[1][1] = C[0][0] * C[2][2] - C[0][2] * C[2][0];
    A[1][2] = C[0][2] * C[1][0] - C[0][0] * C[1][2];
    A[2][0] = A[0][2];
    A[2][1] = A[1][2];
    A[2][2] = C[0][0] * C[1][1] - C[0][1] * C[1][0];
    float determinant = C[0][0] * A[0][0] + C[0][1] * A[0][1] + C[0][2] * A[0][2];
    float n0 = x / ray_len;
    float n1 = y / ray_len;
    float n2 = z / ray_len;
    float q0 = (A[0][0] * n0 + A[0][1] * n1 + A[0][2] * n2) / determinant;
    float q1 = (A[1][0] * n0 + A[1][1] * n1 + A[1][2] * n2) / determinant;
    float q2 = (A[2][0] * n0 + A[2][1] * n1 + A[2][2] * n2) / determinant;
    float vb = n0 * q0 + n1 * q1 + n2 * q2;
    float qlen = sqrt(q0 * q0 + q1 * q1 + q2 * q2);
    built.normal = float3(-q0 / qlen, -q1 / qlen, -q2 / qlen);

    float g00 = j00 * j00 + j01 * j01 + j02 * j02;
    float g01 = j00 * j10 + j01 * j11 + j02 * j12;
    float g11 = j10 * j10 + j11 * j11 + j12 * j12;
    float qj0 = q0 * j00 + q1 * j01 + q2 * j02;
    float qj1 = q0 * j10 + q1 * j11 + q2 * j12;
    float gd = g00 * g11 - g01 * g01;
    built.plane.x = (qj0 * g11 - qj1 * g01) / (vb * gd);
    built.plane.y = (qj1 * g00 - qj0 * g01) / (vb * gd);
    built.plane.z = ray_len;
    built.plane.w = sqrt(vb);
    built.ok = true;
    return built;
}

struct SplatGeom {
    float cov0, cov1, cov2;
    float coef;
    float4 ray_plane;
    float3 normal;
    bool ok;
};

SplatGeom project_splat(float3 mean, float v0, float v1, float v2, float v4, float v5, float v6,
                        float v8, float v9, float v10, float v12, float v13, float v14,
                        uint mode, int width, int height, float fx, float fy, float cx, float cy,
                        float k1, float k2, float k3, float k4, float kernel, float scale_modifier,
                        bool has_scales, float3 scale, float4 rotation, float c0, float c1, float c2,
                        float c3, float c4, float c5) {
    SplatGeom built;
    built.cov0 = 0.0f;
    built.cov1 = 0.0f;
    built.cov2 = 0.0f;
    built.coef = 1.0f;
    built.ray_plane = 0.0f;
    built.normal = 0.0f;
    built.ok = false;

    float3 t = xform_point(mean, v0, v1, v2, v4, v5, v6, v8, v9, v10, v12, v13, v14);
    float tc = sqrt(dot(t, t));
    float u = 0.0f;
    float v = 0.0f;
    Mat3 J = mat_zero();
    if (mode == kModeOrtho) {
        if (!(t.z > 0.2f)) return built;
        J.m[0][0] = fx;
        J.m[1][1] = fy;
    } else if (mode == kModePinhole) {
        float tan_fovx = float(width) / (2.0f * fx);
        float tan_fovy = float(height) / (2.0f * fy);
        float limx = 1.3f * tan_fovx;
        float limy = 1.3f * tan_fovy;
        u = t.x / t.z;
        v = t.y / t.z;
        t.x = min(limx, max(-limx, u)) * t.z;
        t.y = min(limy, max(-limy, v)) * t.z;
        u = t.x / t.z;
        v = t.y / t.z;
        J.m[0][0] = fx / t.z;
        J.m[1][1] = fy / t.z;
        J.m[2][0] = -(fx * t.x) / (t.z * t.z);
        J.m[2][1] = -(fy * t.y) / (t.z * t.z);
    } else {
        Projection proj = project_camera(t, mode, width, height, fx, fy, cx, cy, k1, k2, k3, k4);
        if (!proj.valid) return built;
        J.m[0][0] = proj.du0;
        J.m[0][1] = proj.dv0;
        J.m[1][0] = proj.du1;
        J.m[1][1] = proj.dv1;
        J.m[2][0] = proj.du2;
        J.m[2][1] = proj.dv2;
        u = t.x / max(t.z, 1.0e-6f);
        v = t.y / max(t.z, 1.0e-6f);
    }

    Mat3 W = world_rotation_transposed(v0, v1, v2, v4, v5, v6, v8, v9, v10);
    Mat3 T = mat_mul(W, J);
    Mat3 cov = mat_zero();
    Mat3 Vrk = mat_zero();
    Mat3 Vrk_inv = mat_zero();
    Mat3 cov_cam_inv = mat_zero();
    bool well = false;
    if (has_scales) {
        float3 sl = scale_modifier * scale;
        Mat3 S = mat_diag(sl.x, sl.y, sl.z);
        precise float invx = 1.0f / sl.x;
        precise float invy = 1.0f / sl.y;
        precise float invz = 1.0f / sl.z;
        Mat3 S_inv = mat_diag(invx, invy, invz);
        Mat3 R = quat_rotation(rotation);
        Mat3 SR = mat_mul(S, R);
        Mat3 M = mat_mul(SR, T);
        cov = mat_mul(mat_transpose(M), M);
        Vrk = mat_mul(mat_transpose(SR), SR);
        Mat3 M_inv = mat_mul(mat_mul(S_inv, R), W);
        cov_cam_inv = mat_mul(mat_transpose(M_inv), M_inv);
        Mat3 M_inv2 = mat_mul(S_inv, R);
        Vrk_inv = mat_mul(mat_transpose(M_inv2), M_inv2);
        well = true;
    } else {
        Vrk.m[0][0] = c0;
        Vrk.m[0][1] = c1;
        Vrk.m[0][2] = c2;
        Vrk.m[1][0] = c1;
        Vrk.m[1][1] = c3;
        Vrk.m[1][2] = c4;
        Vrk.m[2][0] = c2;
        Vrk.m[2][1] = c4;
        Vrk.m[2][2] = c5;
        cov = mat_mul(mat_transpose(T), mat_mul(mat_transpose(Vrk), T));
        Mat3 eig_in = Vrk;
        float e0, e1, e2;
        Mat3 vecs;
        sym_eig3(eig_in, e0, e1, e2, vecs);
        int mi = min_index3(e0, e1, e2);
        float emin = mi == 0 ? e0 : (mi == 1 ? e1 : e2);
        well = emin > 1.0e-8f;
        if (well) {
            Mat3 diag = mat_diag(1.0f / e0, 1.0f / e1, 1.0f / e2);
            Vrk_inv = mat_mul(mat_mul(vecs, diag), mat_transpose(vecs));
        } else {
            float3 evec = float3(mat_at(vecs, mi, 0), mat_at(vecs, mi, 1), mat_at(vecs, mi, 2));
            Vrk_inv = mat_outer(evec, evec);
        }
        cov_cam_inv = mat_mul(mat_mul(mat_transpose(W), Vrk_inv), W);
    }

    if (mode == kModeFisheye) {
        FishResult native = fisheye_evaluate(
            t, Vrk.m[0][0], Vrk.m[0][1], Vrk.m[0][2], Vrk.m[1][1], Vrk.m[1][2], Vrk.m[2][2],
            W, fx, fy, k1, k2, k3, k4, kernel);
        if (!native.ok) return built;
        if (!finite1(native.cov0) || !finite1(native.cov1) || !finite1(native.cov2)) return built;
        if (!finite3(native.normal)) return built;
        if (!finite1(native.plane.x) || !finite1(native.plane.y) || !finite1(native.plane.z) ||
            !finite1(native.plane.w) || !finite1(native.coef)) return built;
        if (!(native.cov0 * native.cov2 > native.cov1 * native.cov1)) return built;
        built.cov0 = native.cov0;
        built.cov1 = native.cov1;
        built.cov2 = native.cov2;
        built.coef = native.coef;
        built.ray_plane = native.plane;
        built.normal = native.normal;
        built.ok = true;
        return built;
    }

    built.cov0 = cov.m[0][0] + kernel;
    built.cov1 = cov.m[0][1];
    built.cov2 = cov.m[1][1] + kernel;
    float det0 = max(1.0e-6f, cov.m[0][0] * cov.m[1][1] - cov.m[0][1] * cov.m[0][1]);
    float det1 = max(1.0e-6f, built.cov0 * built.cov2 - built.cov1 * built.cov1);
    built.coef = sqrt(det0 / det1);

    float3 uvh = float3(u, v, 1.0f);
    float3 uvh_m = mat_mul_vec(cov_cam_inv, uvh);
    float u_sq = u * u;
    float v_sq = v * v;
    float l = sqrt(dot(t, t));
    Mat3 nJ_inv = mat_zero();
    nJ_inv.m[0][0] = v_sq + 1.0f;
    nJ_inv.m[0][1] = -u * v;
    nJ_inv.m[0][2] = -u;
    nJ_inv.m[1][0] = -u * v;
    nJ_inv.m[1][1] = u_sq + 1.0f;
    nJ_inv.m[1][2] = -v;
    float vb = dot(uvh_m, uvh);
    float ray_len2 = u_sq + v_sq + 1.0f;
    float factor_normal = l / ray_len2;
    float3 plane = mat_mul_vec(nJ_inv, uvh_m / vb);
    float rsigmat = well ? sqrt(vb / ray_len2) : 0.0f;
    built.ray_plane = float4(plane.x * factor_normal / fx, plane.y * factor_normal / fy, tc, rsigmat);
    float3 ray_normal = float3(-plane.x * factor_normal, -plane.y * factor_normal, -1.0f);
    Mat3 nJ = mat_zero();
    float iz = 1.0f / t.z;
    nJ.m[0][0] = iz;
    nJ.m[0][2] = t.x / l;
    nJ.m[1][1] = iz;
    nJ.m[1][2] = t.y / l;
    nJ.m[2][0] = -t.x * iz * iz;
    nJ.m[2][1] = -t.y * iz * iz;
    nJ.m[2][2] = t.z / l;
    float3 cam_normal = mat_mul_vec(nJ, ray_normal);
    float inv = rsqrt(dot(cam_normal, cam_normal));
    built.normal = cam_normal * inv;
    built.ok = well;
    return built;
}

float3 evaluate_sh(uint degree, float3 pos, float3 campos, StructuredBuffer<float> coeffs, uint coeff_offset,
                   out float clamped0, out float clamped1, out float clamped2) {
    float3 dir = pos - campos;
    float inv = rsqrt(dot(dir, dir));
    dir *= inv;
    float3 c0 = float3(coeffs[coeff_offset], coeffs[coeff_offset + 1], coeffs[coeff_offset + 2]);
    float3 result = float3(0.28209479177387814f * c0.x, 0.28209479177387814f * c0.y, 0.28209479177387814f * c0.z);
    float x = dir.x;
    float y = dir.y;
    float z = dir.z;
    if (degree > 0) {
        float3 c1 = float3(coeffs[coeff_offset + 3], coeffs[coeff_offset + 4], coeffs[coeff_offset + 5]);
        float3 c2 = float3(coeffs[coeff_offset + 6], coeffs[coeff_offset + 7], coeffs[coeff_offset + 8]);
        float3 c3 = float3(coeffs[coeff_offset + 9], coeffs[coeff_offset + 10], coeffs[coeff_offset + 11]);
        result += float3(-0.4886025119029199f * y * c1.x + 0.4886025119029199f * z * c2.x - 0.4886025119029199f * x * c3.x,
                         -0.4886025119029199f * y * c1.y + 0.4886025119029199f * z * c2.y - 0.4886025119029199f * x * c3.y,
                         -0.4886025119029199f * y * c1.z + 0.4886025119029199f * z * c2.z - 0.4886025119029199f * x * c3.z);
        if (degree > 1) {
            float xx = x * x;
            float yy = y * y;
            float zz = z * z;
            float xy = x * y;
            float yz = y * z;
            float xz = x * z;
            float q2[5] = {xy, yz, 2.0f * zz - xx - yy, xz, xx - yy};
            float kC2[5] = {1.0925484305920792f, -1.0925484305920792f, 0.31539156525252005f,
                            -1.0925484305920792f, 0.5462742152960396f};
            [unroll] for (int s = 0; s < 5; ++s) {
                uint base = coeff_offset + uint(4 + s) * 3;
                result += kC2[s] * q2[s] * float3(coeffs[base], coeffs[base + 1], coeffs[base + 2]);
            }
            if (degree > 2) {
                float q3[7] = {y * (3.0f * xx - yy), xy * z, y * (4.0f * zz - xx - yy),
                               z * (2.0f * zz - 3.0f * xx - 3.0f * yy), x * (4.0f * zz - xx - yy),
                               z * (xx - yy), x * (xx - 3.0f * yy)};
                float kC3[7] = {-0.5900435899266435f, 2.890611442640554f, -0.4570457994644658f,
                                0.3731763325901154f, -0.4570457994644658f, 1.445305721320277f,
                                -0.5900435899266435f};
                [unroll] for (int s3 = 0; s3 < 7; ++s3) {
                    uint base3 = coeff_offset + uint(9 + s3) * 3;
                    result += kC3[s3] * q3[s3] * float3(coeffs[base3], coeffs[base3 + 1], coeffs[base3 + 2]);
                }
            }
        }
    }
    result += 0.5f;
    clamped0 = result.x < 0.0f ? 1.0f : 0.0f;
    clamped1 = result.y < 0.0f ? 1.0f : 0.0f;
    clamped2 = result.z < 0.0f ? 1.0f : 0.0f;
    return max(result, 0.0f);
}

#endif
