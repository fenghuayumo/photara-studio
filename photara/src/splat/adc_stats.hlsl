StructuredBuffer<float> refine_weight : register(t0);
StructuredBuffer<float> visibility : register(t1);
StructuredBuffer<int> radii : register(t2);
StructuredBuffer<float> geometry : register(t3);
StructuredBuffer<float> image_error : register(t4);
StructuredBuffer<float> normalized_size : register(t5);
RWStructuredBuffer<float> gradient : register(u6);
RWStructuredBuffer<float> count : register(u7);
RWStructuredBuffer<float> max_screen : register(u8);
RWStructuredBuffer<float> priority : register(u9);
RWStructuredBuffer<float> max_geometry : register(u10);
RWStructuredBuffer<float> accumulated_error : register(u11);

struct Push {
    uint size, flags; float inverse_resolution, power, screen_threshold;
    uint o0, o1, o2, o3, o4, o5, o6, o7, o8, o9, o10, o11;
};
[[vk::push_constant]] Push pc;

[numthreads(256, 1, 1)]
void main(uint3 group : SV_GroupID, uint lane : SV_GroupIndex) {
    uint i = (group.y * 65535u + group.x) * 256u + lane;
    if (i >= pc.size || radii[pc.o2+i] <= 0 ||
        ((pc.flags & 2u) && !(visibility[pc.o1+i] > 0.0f))) return;
    float value = refine_weight[pc.o0+i];
    float raw = isfinite(value) ? max(value, 0.0f) : 0.0f;
    float weight = pc.power == 1.0f ? raw : pow(raw, pc.power);
    gradient[pc.o6+i] = (pc.flags & 1u) ? max(gradient[pc.o6+i], weight) : gradient[pc.o6+i] + weight;
    count[pc.o7+i] += 1.0f;
    float screen = (pc.flags & 16u) ? normalized_size[pc.o5+i] : float(radii[pc.o2+i]) * pc.inverse_resolution;
    max_screen[pc.o8+i] = max(max_screen[pc.o8+i], screen);
    priority[pc.o9+i] += pc.screen_threshold > 0.0f
        ? log2(max(screen / pc.screen_threshold, 1.0f)) : weight * (1.0f + screen);
    if ((pc.flags & 4u) && isfinite(geometry[pc.o3+i]))
        max_geometry[pc.o10+i] = max(max_geometry[pc.o10+i], geometry[pc.o3+i]);
    if ((pc.flags & 8u) && isfinite(image_error[pc.o4+i]))
        accumulated_error[pc.o11+i] += max(image_error[pc.o4+i], 0.0f);
}
