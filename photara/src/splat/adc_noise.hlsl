RWStructuredBuffer<float> means : register(u0);
StructuredBuffer<float> opacity_logits : register(t1);
StructuredBuffer<float> visibility : register(t2);
struct Push { uint size, seed; float standard_deviation, maximum_noise; uint o0, o1, o2; };
[[vk::push_constant]] Push pc;

// Preserve TinyTensor's existing PCG/Box-Muller sequence for each flattened
// [N,3] element. Fusion does not replace the Vulkan noise distribution.
uint pcg_hash(uint value) {
    uint state = value * 747796405u + 2891336453u;
    uint word = ((state >> ((state >> 28u) + 4u)) ^ state) * 277803737u;
    return (word >> 22u) ^ word;
}
float rand01(uint index) {
    return float(pcg_hash(pc.seed ^ pcg_hash(index + 1u)) >> 8u) / 16777216.0f;
}
float normal_sample(uint index) {
    float u1 = max(rand01(index * 2u), 1e-7f);
    float u2 = rand01(index * 2u + 1u);
    return sqrt(-2.0f * log(u1)) * cos(6.28318530718f * u2);
}
[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint i = (group.y * 65535u + group.x) * 256u + lane;
    if (i >= pc.size || !(visibility[pc.o2+i] > 0.0f)) return;
    float opacity = 1.0f / (1.0f + exp(-opacity_logits[pc.o1+i]));
    float weight = pow(1.0f - opacity, 150.0f) * pc.standard_deviation;
    if (weight == 0.0f || pc.maximum_noise == 0.0f) return;
    [unroll] for (uint axis = 0; axis < 3; ++axis) {
        uint index = 3u * i + axis;
        means[pc.o0+index] += clamp(normal_sample(index) * weight,
            -pc.maximum_noise, pc.maximum_noise);
    }
}
