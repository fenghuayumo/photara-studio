// Shared layout for the Vulkan joint bundle adjustment. The host push block
// is 32 uints (128 bytes). Doubles travel as little-endian low/high pairs.
// Pose records are 7 doubles (56 bytes). Point records are 3 doubles (24).
// Intrinsics are 8 doubles followed by a uint model and 4 bytes of padding
// (72). A linearized observation is 37 doubles (residual, 2x6 pose, 2x3
// point, 2x8 intrinsics, robust weight) then a uint valid flag at byte 296
// (304 bytes, matching the host struct).

struct Push {
    uint n;
    uint p0;
    uint p1;
    uint p2;
    uint groups;
    uint flags;
    uint huber_lo;
    uint huber_hi;
    uint depth_lo;
    uint depth_hi;
    uint damp_lo;
    uint damp_hi;
    uint prior_lo;
    uint prior_hi;
    uint min_f_lo;
    uint min_f_hi;
    uint max_f_lo;
    uint max_f_hi;
    uint obs_lo;
    uint obs_hi;
    uint dof;
    uint camera_values;
    uint pad0;
    uint pad1;
    uint pad2;
    uint pad3;
    uint pad4;
    uint pad5;
    uint pad6;
    uint pad7;
    uint pad8;
    uint pad9;
};

static const uint kFixPose = 1u;
static const uint kFixPoint = 2u;
static const uint kOptRot = 4u;
static const uint kOptTrans = 8u;
static const uint kOptPoints = 16u;
static const uint kOptFocal = 32u;
static const uint kOptAspect = 64u;
static const uint kOptPrincipal = 128u;
static const uint kOptDistortion = 256u;
static const uint kPoseBytes = 56u;
static const uint kPointBytes = 24u;
static const uint kIntrinsicBytes = 72u;
static const uint kLinearBytes = 304u;
static const uint kIntrinsicParams = 8u;

double unpack(uint lo, uint hi) {
    return asdouble(lo, hi);
}

double load_at(RWByteAddressBuffer buf, uint byte_offset) {
    return asdouble(buf.Load(byte_offset), buf.Load(byte_offset + 4u));
}

void store_at(RWByteAddressBuffer buf, uint byte_offset, double value) {
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    buf.Store(byte_offset, lo);
    buf.Store(byte_offset + 4u, hi);
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

double abs_d(double value) {
    uint lo;
    uint hi;
    asuint(value, lo, hi);
    return asdouble(lo, hi & 0x7fffffffu);
}

// HLSL sqrt() is float. Scale into [1, 4), take a float guess, then four
// Newton steps. The error stays within 1 ulp of IEEE sqrt.
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

double load_lin(RWByteAddressBuffer buf, uint observation, uint element) {
    return load_at(buf, observation * kLinearBytes + element * 8u);
}

void store_lin(RWByteAddressBuffer buf, uint observation, uint element, double value) {
    store_at(buf, observation * kLinearBytes + element * 8u, value);
}

bool lin_valid(RWByteAddressBuffer buf, uint observation) {
    return (buf.Load(observation * kLinearBytes + 296u) & 0xffu) != 0u;
}

void store_lin_valid(RWByteAddressBuffer buf, uint observation, bool valid) {
    buf.Store(observation * kLinearBytes + 296u, valid ? 1u : 0u);
}
