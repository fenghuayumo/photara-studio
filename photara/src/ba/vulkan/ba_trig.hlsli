// Double sin, cos, and atan. HLSL's transcendentals convert to float. The
// polynomials are the fdlibm/Cephes expansions; a Python check against the
// platform libm stays within 1 ulp on the ranges bundle adjustment uses.

double floor_d(double x) {
    if (isnan_d(x) || !isfinite_d(x)) return x;
    uint lo;
    uint hi;
    asuint(x, lo, hi);
    int exponent = int((hi >> 20) & 0x7ffu) - 1023;
    if (exponent < 0) {
        if ((hi & 0x80000000u) == 0u) return 0.0;
        if ((hi & 0x7fffffffu) == 0u && lo == 0u) return x;
        return -1.0;
    }
    if (exponent >= 52) return x;
    int fractional = 52 - exponent;
    uint lo2 = lo;
    uint hi2 = hi;
    if (fractional >= 32) {
        lo2 = 0u;
        int high_fraction = fractional - 32;
        if (high_fraction > 0) hi2 = hi & ~((1u << high_fraction) - 1u);
    } else {
        lo2 = lo & ~((1u << fractional) - 1u);
    }
    double truncated = asdouble(lo2, hi2);
    if ((hi & 0x80000000u) != 0u && (lo2 != lo || hi2 != hi)) truncated -= 1.0;
    return truncated;
}

double round_d(double x) {
    if (isnan_d(x) || !isfinite_d(x)) return x;
    if (x >= 0.0) return floor_d(x + 0.5);
    return -floor_d((-x) + 0.5);
}

int double_to_int_positive(double value) {
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    int exponent = int((hi >> 20) & 0x7ffu) - 1023;
    if (exponent < 0) return 0;
    if (exponent >= 31) return 2147483647;
    uint top = (hi & 0xfffffu) | 0x100000u;
    int shift = 52 - exponent;
    if (shift >= 32) return int(top >> uint(shift - 32));
    return int((top << uint(32 - shift)) | (lo >> uint(shift)));
}

int double_to_int(double value) {
    if (value < 0.0) return -double_to_int_positive(-value);
    return double_to_int_positive(value);
}

double sin_poly(double r) {
    double z = r * r;
    return r + r * z * (-1.66666666666666324348e-01 + z * (
        8.33333333332248946124e-03 + z * (-1.98412698298579493134e-04 + z * (
            2.75573137070700676789e-06 + z * (-2.50507602534068634195e-08 +
                z * 1.58969099521155010221e-10)))));
}

double cos_poly(double r) {
    double z = r * r;
    return 1.0 + z * (-0.5 + z * (4.16666666666666019037e-02 + z * (
        -1.38888888888741095749e-03 + z * (2.48015872894767294178e-05 + z * (
            -2.75573143513906633035e-07 + z * (2.08757232129817482790e-09 +
                z * -1.13596475577881948265e-11))))));
}

void reduce_half_pi(double x, out double sine, out double cosine, out int quadrant) {
    double n = round_d(x * 0.63661977236758134308);
    double r = (x - n * 1.57079632679489655800e+00) - n * 6.12323399573676603587e-17;
    quadrant = double_to_int(n) % 4;
    if (quadrant < 0) quadrant += 4;
    sine = sin_poly(r);
    cosine = cos_poly(r);
}

double sin_d(double x) {
    if (isnan_d(x)) return x;
    if (!isfinite_d(x) || abs_d(x) > 1.0e8) return asdouble(0u, 0x7ff80000u);
    double sine;
    double cosine;
    int quadrant;
    reduce_half_pi(x, sine, cosine, quadrant);
    if (quadrant == 0) return sine;
    if (quadrant == 1) return cosine;
    if (quadrant == 2) return -sine;
    return -cosine;
}

double cos_d(double x) {
    if (isnan_d(x)) return x;
    if (!isfinite_d(x) || abs_d(x) > 1.0e8) return asdouble(0u, 0x7ff80000u);
    double sine;
    double cosine;
    int quadrant;
    reduce_half_pi(x, sine, cosine, quadrant);
    if (quadrant == 0) return cosine;
    if (quadrant == 1) return -sine;
    if (quadrant == 2) return -cosine;
    return sine;
}

double atan_d(double x) {
    if (isnan_d(x)) return x;
    uint lo;
    uint hi;
    asuint(x, lo, hi);
    uint ix = hi & 0x7fffffffu;
    bool negative = (hi & 0x80000000u) != 0u;
    if (ix >= 0x44100000u) {
        double limit = 1.57079632679489655800e+00 + 6.12323399573676603587e-17;
        return negative ? -limit : limit;
    }
    int id = -1;
    if (ix >= 0x3fdc0000u) {
        x = abs_d(x);
        if (ix < 0x3ff30000u) {
            if (ix < 0x3fe60000u) {
                id = 0;
                x = (2.0 * x - 1.0) / (2.0 + x);
            } else {
                id = 1;
                x = (x - 1.0) / (x + 1.0);
            }
        } else if (ix < 0x40038000u) {
            id = 2;
            x = (x - 1.5) / (1.0 + 1.5 * x);
        } else {
            id = 3;
            x = -1.0 / x;
        }
    }
    double z = x * x;
    double w = z * z;
    double s1 = z * (3.33333333333329318027e-01 + w * (
        1.42857142725034663711e-01 + w * (9.09088713343650656196e-02 + w * (
            6.66107313738753120669e-02 + w * (4.97687799461593236017e-02 +
                w * 1.62858201153657823623e-02)))));
    double s2 = w * (-1.99999999998764832476e-01 + w * (
        -1.11111104054623557880e-01 + w * (-7.69187620504482999495e-02 + w * (
            -5.83357013379057348645e-02 + w * -3.65315727442169155270e-02))));
    if (id < 0) return x - x * (s1 + s2);
    double hi_table[4] = {
        4.63647609000806093515e-01, 7.85398163397448278999e-01,
        9.82793723247329054082e-01, 1.57079632679489655800e+00};
    double lo_table[4] = {
        2.26987774529616870924e-17, 3.06161699786838301793e-17,
        1.39033110312309984516e-17, 6.12323399573676603587e-17};
    z = hi_table[id] - ((x * (s1 + s2) - lo_table[id]) - x);
    return negative ? -z : z;
}
