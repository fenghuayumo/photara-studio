#include "aetherscan/geometry/pose.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <numeric>
#include <random>

namespace aetherscan::geometry {
namespace {

constexpr double kEps = 1e-12;

Mat3 skew(const Vec3& v) {
    Mat3 S;
    S << 0.0, -v.z(), v.y(), v.z(), 0.0, -v.x(), -v.y(), v.x(), 0.0;
    return S;
}

Mat3 enforce_rotation(const Mat3& M) {
    Eigen::JacobiSVD<Mat3> svd(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 R = svd.matrixU() * svd.matrixV().transpose();
    if (R.determinant() < 0.0) {
        Mat3 U = svd.matrixU();
        U.col(2) *= -1.0;
        R = U * svd.matrixV().transpose();
    }
    return R;
}

void normalize_points(
    const std::vector<Vec2>& pts,
    const std::vector<std::size_t>& sample,
    std::vector<Vec2>& out,
    Mat3& T) {
    Vec2 mean = Vec2::Zero();
    for (std::size_t i : sample) mean += pts[i];
    mean /= static_cast<double>(sample.size());

    double mean_dist = 0.0;
    out.resize(sample.size());
    for (std::size_t k = 0; k < sample.size(); ++k) {
        out[k] = pts[sample[k]] - mean;
        mean_dist += out[k].norm();
    }
    mean_dist = std::max(mean_dist / static_cast<double>(sample.size()), kEps);
    const double scale = std::sqrt(2.0) / mean_dist;
    for (Vec2& p : out) p *= scale;

    T.setIdentity();
    T(0, 0) = scale;
    T(1, 1) = scale;
    T(0, 2) = -scale * mean.x();
    T(1, 2) = -scale * mean.y();
}

std::size_t adaptive_iters(
    double confidence,
    std::size_t sample_size,
    std::size_t inlier_count,
    std::size_t total,
    std::size_t max_iters) {
    if (inlier_count == 0 || total == 0) return max_iters;
    const double w = static_cast<double>(inlier_count) / static_cast<double>(total);
    const double p_all_inliers = std::pow(w, static_cast<double>(sample_size));
    if (p_all_inliers >= 1.0 - kEps) return 1;
    if (p_all_inliers <= kEps) return max_iters;
    const double numer = std::log(std::max(1.0 - confidence, kEps));
    const double denom = std::log(1.0 - p_all_inliers);
    if (denom >= 0.0) return max_iters;
    const double n = numer / denom;
    if (!std::isfinite(n) || n >= static_cast<double>(max_iters)) return max_iters;
    return static_cast<std::size_t>(std::max(1.0, std::ceil(n)));
}

bool unique_sample(
    std::mt19937& rng,
    std::size_t n,
    std::size_t k,
    std::vector<std::size_t>& sample) {
    if (n < k) return false;
    sample.resize(k);
    std::uniform_int_distribution<std::size_t> dist(0, n - 1);
    for (std::size_t i = 0; i < k; ++i) {
        bool ok = false;
        for (int attempt = 0; attempt < 64 && !ok; ++attempt) {
            const std::size_t v = dist(rng);
            ok = true;
            for (std::size_t j = 0; j < i; ++j) {
                if (sample[j] == v) {
                    ok = false;
                    break;
                }
            }
            if (ok) sample[i] = v;
        }
        if (!ok) return false;
    }
    return true;
}

double sampson_error_F(const Mat3& F, const Vec2& x1, const Vec2& x2) {
    const Vec3 X1(x1.x(), x1.y(), 1.0);
    const Vec3 X2(x2.x(), x2.y(), 1.0);
    const Vec3 Fx1 = F * X1;
    const Vec3 Ftx2 = F.transpose() * X2;
    const double num = X2.dot(Fx1);
    const double den =
        Fx1.x() * Fx1.x() + Fx1.y() * Fx1.y() + Ftx2.x() * Ftx2.x() + Ftx2.y() * Ftx2.y();
    if (den < kEps) return std::abs(num);
    return std::abs(num) / std::sqrt(den);
}

double sampson_error_E(const Mat3& E, const Vec2& x1, const Vec2& x2) {
    return sampson_error_F(E, x1, x2);
}

Vec2 apply_H(const Mat3& H, const Vec2& p) {
    const Vec3 q = H * Vec3(p.x(), p.y(), 1.0);
    if (std::abs(q.z()) < kEps) return Vec2(1e6, 1e6);
    return Vec2(q.x() / q.z(), q.y() / q.z());
}

double homography_error(const Mat3& H, const Vec2& x1, const Vec2& x2) {
    const Vec2 hx = apply_H(H, x1);
    const Mat3 Hinv = H.inverse();
    const Vec2 hy = apply_H(Hinv, x2);
    return 0.5 * ((hx - x2).norm() + (hy - x1).norm());
}

bool triangulate_two(
    const Mat34& P1,
    const Mat34& P2,
    const Vec2& x1,
    const Vec2& x2,
    Vec3& X) {
    Eigen::Matrix4d A;
    A.row(0) = x1.x() * P1.row(2) - P1.row(0);
    A.row(1) = x1.y() * P1.row(2) - P1.row(1);
    A.row(2) = x2.x() * P2.row(2) - P2.row(0);
    A.row(3) = x2.y() * P2.row(2) - P2.row(1);
    Eigen::JacobiSVD<Eigen::Matrix4d> svd(A, Eigen::ComputeFullV);
    const Vec4 h = svd.matrixV().col(3);
    if (std::abs(h.w()) < kEps) return false;
    X = h.head<3>() / h.w();
    return X.allFinite();
}

std::size_t cheirality_count(
    const Mat3& R,
    const Vec3& t,
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::uint8_t>& mask,
    std::vector<std::uint8_t>* updated_mask) {
    Mat34 P1 = Mat34::Zero();
    P1.leftCols<3>() = Mat3::Identity();
    Mat34 P2;
    P2.leftCols<3>() = R;
    P2.col(3) = t;

    std::size_t count = 0;
    if (updated_mask) *updated_mask = mask;
    for (std::size_t i = 0; i < p1.size(); ++i) {
        if (!mask.empty() && i < mask.size() && mask[i] == 0) continue;
        Vec3 X;
        if (!triangulate_two(P1, P2, p1[i], p2[i], X)) {
            if (updated_mask && i < updated_mask->size()) (*updated_mask)[i] = 0;
            continue;
        }
        const Vec3 Xc2 = R * X + t;
        if (X.z() > kEps && Xc2.z() > kEps) {
            ++count;
        } else if (updated_mask && i < updated_mask->size()) {
            (*updated_mask)[i] = 0;
        }
    }
    return count;
}

Mat3 essential_from_sample(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    Eigen::Matrix<double, Eigen::Dynamic, 9> A(sample.size(), 9);
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const Vec2& a = p1[sample[i]];
        const Vec2& b = p2[sample[i]];
        A.row(static_cast<Eigen::Index>(i)) << a.x() * b.x(), a.y() * b.x(), b.x(),
            a.x() * b.y(), a.y() * b.y(), b.y(), a.x(), a.y(), 1.0;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> e = svd.matrixV().col(8);
    Mat3 E;
    E << e(0), e(1), e(2), e(3), e(4), e(5), e(6), e(7), e(8);

    Eigen::JacobiSVD<Mat3> esvd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = esvd.singularValues();
    const double scale = 0.5 * (s(0) + s(1));
    if (scale > kEps) {
        s(0) = scale;
        s(1) = scale;
    } else {
        s(0) = 1.0;
        s(1) = 1.0;
    }
    s(2) = 0.0;
    return esvd.matrixU() * s.asDiagonal() * esvd.matrixV().transpose();
}

Mat3 fundamental_from_sample(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    std::vector<Vec2> n1, n2;
    Mat3 T1, T2;
    normalize_points(p1, sample, n1, T1);
    normalize_points(p2, sample, n2, T2);

    Eigen::Matrix<double, Eigen::Dynamic, 9> A(sample.size(), 9);
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const Vec2& a = n1[i];
        const Vec2& b = n2[i];
        A.row(static_cast<Eigen::Index>(i)) << a.x() * b.x(), a.y() * b.x(), b.x(),
            a.x() * b.y(), a.y() * b.y(), b.y(), a.x(), a.y(), 1.0;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> f = svd.matrixV().col(8);
    Mat3 Fn;
    Fn << f(0), f(1), f(2), f(3), f(4), f(5), f(6), f(7), f(8);

    Eigen::JacobiSVD<Mat3> fsvd(Fn, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Vector3d s = fsvd.singularValues();
    s(2) = 0.0;
    Fn = fsvd.matrixU() * s.asDiagonal() * fsvd.matrixV().transpose();
    Mat3 F = T2.transpose() * Fn * T1;
    const double nrm = F.norm();
    if (nrm > kEps) F /= nrm;
    return F;
}

Mat3 homography_from_sample(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    std::vector<Vec2> n1, n2;
    Mat3 T1, T2;
    normalize_points(p1, sample, n1, T1);
    normalize_points(p2, sample, n2, T2);

    Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(sample.size()), 9);
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const double x = n1[i].x(), y = n1[i].y();
        const double u = n2[i].x(), v = n2[i].y();
        const Eigen::Index r = static_cast<Eigen::Index>(2 * i);
        A.row(r) << -x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u;
        A.row(r + 1) << 0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v;
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    Eigen::Matrix<double, 9, 1> h = svd.matrixV().col(8);
    Mat3 Hn;
    Hn << h(0), h(1), h(2), h(3), h(4), h(5), h(6), h(7), h(8);
    Mat3 H = T2.inverse() * Hn * T1;
    if (std::abs(H(2, 2)) > kEps) H /= H(2, 2);
    return H;
}

bool solve_pnp_dlt(
    const std::vector<Vec3>& obj,
    const std::vector<Vec2>& img,
    const std::vector<std::size_t>& sample,
    Mat3& R,
    Vec3& t) {
    if (sample.size() < 6) return false;
    Eigen::MatrixXd A(2 * static_cast<Eigen::Index>(sample.size()), 12);
    A.setZero();
    for (std::size_t i = 0; i < sample.size(); ++i) {
        const Vec3& X = obj[sample[i]];
        const Vec2& x = img[sample[i]];
        const Eigen::Index r = static_cast<Eigen::Index>(2 * i);
        A.row(r) << X.x(), X.y(), X.z(), 1.0, 0, 0, 0, 0, -x.x() * X.x(),
            -x.x() * X.y(), -x.x() * X.z(), -x.x();
        A.row(r + 1) << 0, 0, 0, 0, X.x(), X.y(), X.z(), 1.0, -x.y() * X.x(),
            -x.y() * X.y(), -x.y() * X.z(), -x.y();
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);

    auto decompose = [](const Mat34& projection, Mat3& rotation, Vec3& translation) -> bool {
        Mat3 M = projection.leftCols<3>();
        Eigen::JacobiSVD<Mat3> svd_m(M, Eigen::ComputeFullU | Eigen::ComputeFullV);
        rotation = svd_m.matrixU() * svd_m.matrixV().transpose();
        if (rotation.determinant() < 0.0) {
            Mat3 U = svd_m.matrixU();
            U.col(2) *= -1.0;
            rotation = U * svd_m.matrixV().transpose();
        }
        const double scale = (rotation.transpose() * M).trace() / 3.0;
        if (std::abs(scale) < kEps) return false;
        translation = projection.col(3) / scale;
        return rotation.allFinite() && translation.allFinite();
    };

    const double largest = svd.singularValues()(0);
    double best_error = std::numeric_limits<double>::infinity();
    bool found = false;
    Mat3 best_R = Mat3::Identity();
    Vec3 best_t = Vec3::Zero();

    // Near-planar scenes make the DLT nullspace multi-dimensional; try every
    // near-null right singular vector and keep the best calibrated pose.
    for (int column = 11; column >= 0; --column) {
        if (svd.singularValues()(column) > largest * 1e-4) break;
        Eigen::Matrix<double, 12, 1> p = svd.matrixV().col(column);
        Mat34 P;
        P << p(0), p(1), p(2), p(3), p(4), p(5), p(6), p(7), p(8), p(9), p(10), p(11);
        for (double sign : {1.0, -1.0}) {
            Mat3 candidate_R;
            Vec3 candidate_t;
            if (!decompose(sign * P, candidate_R, candidate_t)) continue;
            std::size_t positive = 0;
            double error = 0.0;
            for (std::size_t idx : sample) {
                const Vec3 Xc = candidate_R * obj[idx] + candidate_t;
                if (Xc.z() <= kEps) {
                    error = std::numeric_limits<double>::infinity();
                    break;
                }
                ++positive;
                const Vec2 proj(Xc.x() / Xc.z(), Xc.y() / Xc.z());
                error += (proj - img[idx]).squaredNorm();
            }
            if (!(error < best_error) || positive * 2 < sample.size()) continue;
            best_error = error;
            best_R = candidate_R;
            best_t = candidate_t;
            found = true;
        }
    }
    if (!found) return false;
    R = best_R;
    t = best_t;
    return true;
}

double pnp_reproj_error(
    const Mat3& R,
    const Vec3& t,
    const Vec3& X,
    const Vec2& x,
    double fx,
    double fy,
    double cx,
    double cy) {
    const Vec3 Xc = R * X + t;
    if (Xc.z() <= kEps) return 1e6;
    const Vec2 proj(fx * Xc.x() / Xc.z() + cx, fy * Xc.y() / Xc.z() + cy);
    return (proj - x).norm();
}

template <typename FitFn, typename ErrFn>
RansacResult ransac_mat3(
    std::size_t n,
    std::size_t sample_size,
    double threshold,
    double confidence,
    std::size_t max_iters,
    FitFn fit,
    ErrFn err) {
    RansacResult best;
    best.inliers.assign(n, 0);
    if (n < sample_size) return best;

    std::mt19937 rng(42);
    std::vector<std::size_t> sample;
    std::size_t iters = max_iters;
    for (std::size_t it = 0; it < iters; ++it) {
        if (!unique_sample(rng, n, sample_size, sample)) continue;
        Mat3 model;
        try {
            model = fit(sample);
        } catch (...) {
            continue;
        }
        if (!model.allFinite()) continue;

        std::vector<std::uint8_t> mask(n, 0);
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            if (err(model, i) <= threshold) {
                mask[i] = 1;
                ++count;
            }
        }
        if (count > best.inlier_count) {
            best.model = model;
            best.inliers = std::move(mask);
            best.inlier_count = count;
            best.valid = count >= sample_size;
            iters = std::min(iters, adaptive_iters(confidence, sample_size, count, n, max_iters));
        }
    }

    // Optional refit on all inliers when enough support.
    if (best.valid && best.inlier_count >= sample_size) {
        std::vector<std::size_t> inl;
        inl.reserve(best.inlier_count);
        for (std::size_t i = 0; i < n; ++i)
            if (best.inliers[i]) inl.push_back(i);
        if (inl.size() >= sample_size) {
            const Mat3 refined = fit(inl);
            if (refined.allFinite()) {
                best.model = refined;
                std::size_t count = 0;
                for (std::size_t i = 0; i < n; ++i) {
                    const bool ok = err(best.model, i) <= threshold;
                    best.inliers[i] = ok ? 1 : 0;
                    if (ok) ++count;
                }
                best.inlier_count = count;
                best.valid = count >= sample_size;
            }
        }
    }
    return best;
}

}  // namespace

Mat3 rodrigues(const Vec3& rvec) {
    const double theta = rvec.norm();
    if (theta < 1e-10) {
        return Mat3::Identity() + skew(rvec);
    }
    const Vec3 k = rvec / theta;
    const Mat3 K = skew(k);
    return Mat3::Identity() + std::sin(theta) * K + (1.0 - std::cos(theta)) * (K * K);
}

Vec3 inverse_rodrigues(const Mat3& R) {
    const double tr = (R.trace() - 1.0) * 0.5;
    const double c = std::clamp(tr, -1.0, 1.0);
    const double theta = std::acos(c);
    if (theta < 1e-10) {
        return 0.5 * Vec3(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
    }
    if (std::abs(theta - 3.14159265358979323846) < 1e-6) {
        // Near 180 deg: recover from diagonal of R.
        Vec3 axis;
        axis.x() = std::sqrt(std::max(0.0, (R(0, 0) + 1.0) * 0.5));
        axis.y() = std::sqrt(std::max(0.0, (R(1, 1) + 1.0) * 0.5));
        axis.z() = std::sqrt(std::max(0.0, (R(2, 2) + 1.0) * 0.5));
        if (R(0, 1) < 0.0) axis.y() = -axis.y();
        if (R(0, 2) < 0.0) axis.z() = -axis.z();
        if (axis.norm() < kEps) axis = Vec3(1.0, 0.0, 0.0);
        else axis.normalize();
        return axis * theta;
    }
    const double s = 1.0 / (2.0 * std::sin(theta));
    return theta * s * Vec3(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
}

Mat3 essential_from_normalized_8point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    return essential_from_sample(p1, p2, sample);
}

Mat3 fundamental_from_8point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    return fundamental_from_sample(p1, p2, sample);
}

Mat3 homography_from_4point(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    const std::vector<std::size_t>& sample) {
    return homography_from_sample(p1, p2, sample);
}

RansacResult estimate_essential_ransac(
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    double threshold_normalized,
    double confidence,
    std::size_t max_iters) {
    const std::size_t n = std::min(p1.size(), p2.size());
    return ransac_mat3(
        n, 8, threshold_normalized, confidence, max_iters,
        [&](const std::vector<std::size_t>& s) {
            return essential_from_sample(p1, p2, s);
        },
        [&](const Mat3& E, std::size_t i) { return sampson_error_E(E, p1[i], p2[i]); });
}

RansacResult estimate_fundamental_ransac(
    const std::vector<Vec2>& p1_pixels,
    const std::vector<Vec2>& p2_pixels,
    double threshold_pixels,
    double confidence,
    std::size_t max_iters) {
    const std::size_t n = std::min(p1_pixels.size(), p2_pixels.size());
    return ransac_mat3(
        n, 8, threshold_pixels, confidence, max_iters,
        [&](const std::vector<std::size_t>& s) {
            return fundamental_from_sample(p1_pixels, p2_pixels, s);
        },
        [&](const Mat3& F, std::size_t i) {
            return sampson_error_F(F, p1_pixels[i], p2_pixels[i]);
        });
}

RansacResult estimate_homography_ransac(
    const std::vector<Vec2>& p1_pixels,
    const std::vector<Vec2>& p2_pixels,
    double threshold_pixels,
    double confidence,
    std::size_t max_iters) {
    const std::size_t n = std::min(p1_pixels.size(), p2_pixels.size());
    return ransac_mat3(
        n, 4, threshold_pixels, confidence, max_iters,
        [&](const std::vector<std::size_t>& s) {
            return homography_from_sample(p1_pixels, p2_pixels, s);
        },
        [&](const Mat3& H, std::size_t i) {
            return homography_error(H, p1_pixels[i], p2_pixels[i]);
        });
}

RelativePose recover_pose(
    const Mat3& E,
    const std::vector<Vec2>& p1,
    const std::vector<Vec2>& p2,
    std::vector<std::uint8_t>& inlier_mask) {
    RelativePose best;
    Eigen::JacobiSVD<Mat3> svd(E, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Mat3 U = svd.matrixU();
    Mat3 Vt = svd.matrixV().transpose();
    if (U.determinant() < 0.0) U.col(2) *= -1.0;
    if (Vt.determinant() < 0.0) Vt.row(2) *= -1.0;

    Mat3 W;
    W << 0.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0;
    const Mat3 R1 = U * W * Vt;
    const Mat3 R2 = U * W.transpose() * Vt;
    Vec3 t = U.col(2);
    t.normalize();

    const Mat3 Rs[2] = {enforce_rotation(R1), enforce_rotation(R2)};
    const Vec3 ts[2] = {t, -t};

    std::vector<std::uint8_t> best_mask;
    for (const Mat3& R : Rs) {
        for (const Vec3& ti : ts) {
            std::vector<std::uint8_t> updated;
            const std::size_t c =
                cheirality_count(R, ti, p1, p2, inlier_mask, &updated);
            if (c > best.cheirality_count) {
                best.R = R;
                best.t = ti;
                best.cheirality_count = c;
                best.valid = c > 0;
                best_mask = std::move(updated);
            }
        }
    }
    if (best.valid && !best_mask.empty()) inlier_mask = std::move(best_mask);
    return best;
}

AbsolutePose estimate_pnp_ransac(
    const std::vector<Vec3>& object_points,
    const std::vector<Vec2>& image_points,
    double fx,
    double fy,
    double cx,
    double cy,
    double threshold_pixels,
    double confidence,
    std::size_t max_iters,
    bool use_pixel_and_K) {
    AbsolutePose best;
    const std::size_t n = std::min(object_points.size(), image_points.size());
    if (n < 6) return best;

    // DLT always on normalized rays; reprojection error in pixel space when K given.
    std::vector<Vec2> normalized(n);
    std::vector<Vec2> pixels = image_points;
    double err_fx = fx, err_fy = fy, err_cx = cx, err_cy = cy;
    if (use_pixel_and_K) {
        if (fx <= kEps || fy <= kEps) return best;
        for (std::size_t i = 0; i < n; ++i) {
            normalized[i] = Vec2(
                (image_points[i].x() - cx) / fx, (image_points[i].y() - cy) / fy);
        }
    } else {
        normalized = image_points;
        err_fx = 1.0;
        err_fy = 1.0;
        err_cx = 0.0;
        err_cy = 0.0;
        for (std::size_t i = 0; i < n; ++i) pixels[i] = normalized[i];
    }
    const double thr = threshold_pixels;

    std::mt19937 rng(7);
    std::vector<std::size_t> sample;
    std::vector<std::uint8_t> best_mask(n, 0);
    std::size_t best_count = 0;
    std::size_t iters = max_iters;

    // Prefer a full-set DLT when the correspondences are already clean (unit tests /
    // well-conditioned resection). Fall back to RANSAC otherwise.
    {
        sample.resize(n);
        std::iota(sample.begin(), sample.end(), 0);
        Mat3 R;
        Vec3 t;
        if (solve_pnp_dlt(object_points, normalized, sample, R, t)) {
            std::size_t count = 0;
            std::vector<std::uint8_t> mask(n, 0);
            for (std::size_t i = 0; i < n; ++i) {
                const double e = pnp_reproj_error(
                    R, t, object_points[i], pixels[i], err_fx, err_fy, err_cx, err_cy);
                if (e <= thr) {
                    mask[i] = 1;
                    ++count;
                }
            }
            if (count >= 6) {
                best.R = R;
                best.t = t;
                best.valid = true;
                best_mask = std::move(mask);
                best_count = count;
            }
        }
    }

    for (std::size_t it = 0; it < iters; ++it) {
        if (!unique_sample(rng, n, 6, sample)) continue;
        Mat3 R;
        Vec3 t;
        if (!solve_pnp_dlt(object_points, normalized, sample, R, t)) continue;

        std::vector<std::uint8_t> mask(n, 0);
        std::size_t count = 0;
        for (std::size_t i = 0; i < n; ++i) {
            const double e = pnp_reproj_error(
                R, t, object_points[i], pixels[i], err_fx, err_fy, err_cx, err_cy);
            if (e <= thr) {
                mask[i] = 1;
                ++count;
            }
        }
        if (count > best_count) {
            best.R = R;
            best.t = t;
            best.valid = count >= 6;
            best_mask = std::move(mask);
            best_count = count;
            iters = std::min(iters, adaptive_iters(confidence, 6, count, n, max_iters));
        }
    }

    if (best.valid && best_count >= 6) {
        std::vector<std::size_t> inl;
        for (std::size_t i = 0; i < n; ++i)
            if (best_mask[i]) inl.push_back(i);
        Mat3 R;
        Vec3 t;
        if (solve_pnp_dlt(object_points, normalized, inl, R, t)) {
            best.R = R;
            best.t = t;
        }
        best = refine_pnp_lm(
            best, object_points, pixels, err_fx, err_fy, err_cx, err_cy, &best_mask);
    }
    return best;
}

AbsolutePose refine_pnp_lm(
    const AbsolutePose& initial,
    const std::vector<Vec3>& object_points,
    const std::vector<Vec2>& image_points,
    double fx,
    double fy,
    double cx,
    double cy,
    const std::vector<std::uint8_t>* inlier_mask,
    std::size_t max_iters) {
    AbsolutePose pose = initial;
    if (!pose.valid) return pose;

    std::vector<std::size_t> idx;
    const std::size_t n = std::min(object_points.size(), image_points.size());
    idx.reserve(n);
    for (std::size_t i = 0; i < n; ++i) {
        if (inlier_mask && i < inlier_mask->size() && (*inlier_mask)[i] == 0) continue;
        idx.push_back(i);
    }
    if (idx.size() < 6) return pose;

    Mat3 R = pose.R;
    Vec3 t = pose.t;
    double lambda = 1e-3;

    auto cost_rt = [&](const Mat3& Rm, const Vec3& tm) {
        double c = 0.0;
        for (std::size_t i : idx) {
            const double e =
                pnp_reproj_error(Rm, tm, object_points[i], image_points[i], fx, fy, cx, cy);
            c += e * e;
        }
        return c;
    };

    double current = cost_rt(R, t);
    for (std::size_t iter = 0; iter < max_iters; ++iter) {
        Eigen::Matrix<double, 6, 6> JtJ = Eigen::Matrix<double, 6, 6>::Zero();
        Eigen::Matrix<double, 6, 1> Jte = Eigen::Matrix<double, 6, 1>::Zero();

        for (std::size_t i : idx) {
            const Vec3 X = object_points[i];
            const Vec2 x = image_points[i];
            const Vec3 Xc = R * X + t;
            if (Xc.z() <= kEps) continue;
            const double invz = 1.0 / Xc.z();
            const double invz2 = invz * invz;
            const Eigen::Matrix<double, 2, 1> resid(
                fx * Xc.x() * invz + cx - x.x(), fy * Xc.y() * invz + cy - x.y());

            Eigen::Matrix<double, 2, 3> Jcam;
            Jcam << fx * invz, 0.0, -fx * Xc.x() * invz2, 0.0, fy * invz,
                -fy * Xc.y() * invz2;

            // Left perturbation: Xc' ≈ Xc + [-skew(Xc)] omega + dt
            Eigen::Matrix<double, 2, 6> J;
            J.leftCols<3>() = Jcam * (-skew(Xc));
            J.rightCols<3>() = Jcam;

            JtJ.noalias() += J.transpose() * J;
            Jte.noalias() += J.transpose() * resid;
        }

        Eigen::Matrix<double, 6, 6> H = JtJ;
        H.diagonal() *= (1.0 + lambda);
        Eigen::LDLT<Eigen::Matrix<double, 6, 6>> ldlt(H);
        if (ldlt.info() != Eigen::Success) break;
        const Eigen::Matrix<double, 6, 1> dp = -ldlt.solve(Jte);
        if (!dp.allFinite()) break;

        const Mat3 R_trial = rodrigues(dp.head<3>()) * R;
        const Vec3 t_trial = t + dp.tail<3>();
        const double trial_cost = cost_rt(R_trial, t_trial);
        if (trial_cost < current) {
            R = R_trial;
            t = t_trial;
            current = trial_cost;
            lambda = std::max(1e-7, lambda * 0.1);
            if (dp.norm() < 1e-8) break;
        } else {
            lambda = std::min(1e7, lambda * 10.0);
        }
    }

    pose.R = enforce_rotation(R);
    pose.t = t;
    pose.valid = pose.R.allFinite() && pose.t.allFinite();
    return pose;
}

Vec2 project_point(
    const Mat3& R,
    const Vec3& t,
    const Vec3& X,
    double fx,
    double fy,
    double cx,
    double cy) {
    const Vec3 Xc = R * X + t;
    if (std::abs(Xc.z()) < kEps) return Vec2(1e6, 1e6);
    return Vec2(fx * Xc.x() / Xc.z() + cx, fy * Xc.y() / Xc.z() + cy);
}

bool triangulate_dlt(
    const std::vector<Mat34>& projections,
    const std::vector<Vec2>& observations,
    Vec3& X) {
    const std::size_t m = std::min(projections.size(), observations.size());
    if (m < 2) return false;

    Eigen::MatrixXd A(static_cast<Eigen::Index>(2 * m), 4);
    for (std::size_t i = 0; i < m; ++i) {
        const Mat34& P = projections[i];
        const Vec2& x = observations[i];
        const Eigen::Index r = static_cast<Eigen::Index>(2 * i);
        A.row(r) = x.x() * P.row(2) - P.row(0);
        A.row(r + 1) = x.y() * P.row(2) - P.row(1);
    }
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeFullV);
    const Vec4 h = svd.matrixV().col(3);
    if (std::abs(h.w()) < kEps) return false;
    X = h.head<3>() / h.w();
    return X.allFinite();
}

}  // namespace aetherscan::geometry
