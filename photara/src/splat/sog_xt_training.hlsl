// Frozen SOG-XT codec with STE quantization. Gather reductions avoid float
// atomics, which are optional on Vulkan and unavailable on many Metal devices.
[[vk::binding(0)]] StructuredBuffer<float> src;
[[vk::binding(1)]] RWStructuredBuffer<float> dst;
[[vk::binding(2)]] StructuredBuffer<float> low;
[[vk::binding(3)]] StructuredBuffer<float> high;
[[vk::binding(4)]] RWByteAddressBuffer aux;
[[vk::binding(5)]] StructuredBuffer<int> counts;
[[vk::binding(6)]] StructuredBuffer<int> offsets;
[[vk::binding(7)]] StructuredBuffer<int> indices;
struct Push { uint mode, n, width, stride, first, dims, total; float tv, qtv; };
[[vk::push_constant]] Push pc;
float encode(float v, uint d) { return d < 3 ? sign(v) * log(1 + abs(v)) : v; }
float quantize(float v, float lo, float hi, uint levels) {
    if (hi <= lo) return lo;
    return lo + (hi - lo) * floor(saturate((v - lo) / (hi - lo)) * (levels - 1) + 0.5) / (levels - 1);
}
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_GroupID, uint3 tid : SV_GroupThreadID) {
    uint at = (gid.y * 65535 + gid.x) * 256 + tid.x;
    if (at >= pc.total) return;
    if (pc.mode == 0 || pc.mode == 3 || pc.mode == 4) {
        uint i = at / pc.width, c = at % pc.width, d = pc.first + c;
        uint pos = i * pc.stride + c;
        float value = src[pos], v = encode(value, d), lo = low[d], hi = high[d];
        if (pc.mode == 0) {
            float q = quantize(v, lo, hi, d < 3 ? 65536 : d >= 6 && d < 10 ? 100 : 256);
            dst[pos] = d < 3 ? sign(q) * (exp(abs(q)) - 1) : q;
        } else if (pc.mode == 3) {
            if (v < lo || v > hi || hi <= lo) dst[pos] = 0;
            else if (d < 3) dst[pos] *= (1 + abs(asfloat(aux.Load(pos * 4)))) / (1 + abs(value));
        } else if (hi > lo) {
            float g = 0;
            // Outgoing edges, followed by incoming edges of the fixed image layout.
            for (uint axis = 0; axis < 2; ++axis) {
                int j = asint(aux.Load((i * 2 + axis) * 4));
                if (j >= 0) g += sign(v - encode(src[j * pc.stride + c], d));
            }
            for (int k = offsets[i]; k < offsets[i + 1]; ++k)
                g += sign(v - encode(src[indices[k] * pc.stride + c], d));
            float weight = (d >= 6 && d < 10 ? pc.qtv : pc.tv) / (float(pc.n) * 28 * (hi - lo));
            if (d < 3) weight /= 1 + abs(value);
            dst[pos] += g * weight;
        }
    } else if (pc.mode == 1 || pc.mode == 5) {
        uint center = at / pc.dims, d = at % pc.dims;
        float sum = 0;
        for (int k = offsets[center]; k < offsets[center + 1]; ++k)
            sum += src[indices[k] * pc.stride + 3 + d];
        float v = counts[center] > 0 ? sum / counts[center] : 0;
        if (pc.mode == 1) {
            aux.Store(at * 4, asuint(v));
            dst[at] = quantize(v, low[d], high[d], 100);
        } else {
            float raw = asfloat(aux.Load(at * 4));
            dst[at] = raw >= low[d] && raw <= high[d] && high[d] > low[d] ? v : 0;
        }
    } else if (pc.mode == 2) {
        uint i = at / pc.dims, d = at % pc.dims;
        uint label = aux.Load(i * 4);
        dst[i * pc.stride + 3 + d] = src[label * pc.dims + d];
    }
}
