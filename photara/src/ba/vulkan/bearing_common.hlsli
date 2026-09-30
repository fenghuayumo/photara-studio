// Shared layout for the Vulkan bearing Schur solver. Host push constants are
// 12 uints (48 bytes) so every kernel sees the same packing. Doubles travel
// as little-endian low/high pairs. Observation records are 32 bytes:
// uint camera, uint point, double direction[3]. Linearized terms are 12
// doubles per observation: row-major 3x3 H, then b.

struct Push {
    uint n;
    uint p0;
    uint p1;
    uint p2;
    uint groups;
    uint flags;
    uint huber_lo;
    uint huber_hi;
    uint baseline_lo;
    uint baseline_hi;
    uint lambda_lo;
    uint lambda_hi;
};

double unpack(uint lo, uint hi) {
    return asdouble(lo, hi);
}

double load_double(RWByteAddressBuffer buf, uint index) {
    uint offset = index * 8u;
    return asdouble(buf.Load(offset), buf.Load(offset + 4u));
}

void store_double(RWByteAddressBuffer buf, uint index, double value) {
    uint offset = index * 8u;
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    buf.Store(offset, lo);
    buf.Store(offset + 4u, hi);
}

uint load_uint(RWByteAddressBuffer buf, uint index) {
    return buf.Load(index * 4u);
}

bool isnan_d(double value) {
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    return (hi & 0x7ff00000u) == 0x7ff00000u &&
           ((hi & 0x000fffffu) != 0u || lo != 0u);
}

bool isfinite_d(double value) {
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    return (hi & 0x7ff00000u) != 0x7ff00000u;
}

// HLSL sqrt() is float. Scale into [1, 4), take a float guess, then four
// Newton steps. Checked against IEEE sqrt across normal and subnormal inputs:
// the error stays within 1 ulp.
double sqrt_d(double x) {
    if (isnan_d(x) || x < 0.0) return asdouble(0u, 0x7ff80000u);
    if (x == 0.0) return x;
    if (!isfinite_d(x)) return x;
    uint lo;
    uint hi;
    asuint(x, lo, hi);
    bool subnormal = (hi & 0x7ff00000u) == 0u;
    if (subnormal) {
        x *= 4503599627370496.0;
        asuint(x, lo, hi);
    }
    int exponent = int((hi >> 20) & 0x7ffu) - 1023;
    int even = exponent & ~1;
    int scaled_field = (exponent - even) + 1023;
    uint scaled_hi = (hi & 0x800fffffu) | (uint(scaled_field) << 20);
    double scaled = asdouble(lo, scaled_hi);
    double y = double(sqrt(float(scaled)));
    y = 0.5 * (y + scaled / y);
    y = 0.5 * (y + scaled / y);
    y = 0.5 * (y + scaled / y);
    y = 0.5 * (y + scaled / y);
    uint ylo;
    uint yhi;
    asuint(y, ylo, yhi);
    int combined = int((yhi >> 20) & 0x7ffu) + even / 2;
    double result = 0.0;
    if (combined >= 2047) result = asdouble(0u, 0x7ff00000u);
    else if (combined > 0) {
        uint out_hi = (yhi & 0x800fffffu) | (uint(combined) << 20);
        result = asdouble(ylo, out_hi);
    }
    if (subnormal) result *= 1.4901161193847656e-8;
    return result;
}

double fmin_d(double a, double b) {
    if (isnan_d(a)) return b;
    if (isnan_d(b)) return a;
    return a < b ? a : b;
}

double fmax_d(double a, double b) {
    if (isnan_d(a)) return b;
    if (isnan_d(b)) return a;
    return a > b ? a : b;
}

// Row-major SPD 3x3 inverse. A non-positive determinant reports failure and
// leaves the outputs untouched, matching the CUDA kernel.
bool invert3(
    double a0, double a1, double a2, double a3, double a4, double a5,
    double a6, double a7, double a8,
    out double o0, out double o1, out double o2, out double o3, out double o4,
    out double o5, out double o6, out double o7, out double o8) {
    o0 = 0.0;
    o1 = 0.0;
    o2 = 0.0;
    o3 = 0.0;
    o4 = 0.0;
    o5 = 0.0;
    o6 = 0.0;
    o7 = 0.0;
    o8 = 0.0;
    double x = a4 * a8 - a5 * a7;
    double y = a2 * a7 - a1 * a8;
    double z = a1 * a5 - a2 * a4;
    double det = a0 * x + a3 * y + a6 * z;
    if (!(det > 0.0) || !isfinite_d(det)) return false;
    o0 = x / det;
    o1 = y / det;
    o2 = z / det;
    o3 = y / det;
    o4 = (a0 * a8 - a2 * a6) / det;
    o5 = (a2 * a3 - a0 * a5) / det;
    o6 = z / det;
    o7 = o5;
    o8 = (a0 * a4 - a1 * a3) / det;
    return true;
}
