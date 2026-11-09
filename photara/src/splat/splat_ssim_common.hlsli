#include "splat_math.hlsli"
StructuredBuffer<float> prediction : register(t0);
StructuredBuffer<float> target : register(t1);
StructuredBuffer<float> mask_image : register(t2);
RWStructuredBuffer<float> derivative0 : register(u3);
RWStructuredBuffer<float> derivative1 : register(u4);
RWStructuredBuffer<float> derivative2 : register(u5);
RWStructuredBuffer<float> output : register(u6);

static const float gaussian[11] = {
    0.001028380123898387f, 0.0075987582094967365f, 0.036000773310661316f, 0.10936068743467331f,
    0.21300552785396576f,  0.26601171493530273f,   0.21300552785396576f,  0.10936068743467331f,
    0.036000773310661316f, 0.0075987582094967365f, 0.001028380123898387f};
float push_float(uint value) {
    return asfloat(value);
}

float prediction_at(int c, int x, int y, uint w, uint h, bool masked) {
    if (x < 0 || y < 0 || x >= (int)w || y >= (int)h)
        return 0.0f;
    uint p = (uint)y * w + (uint)x;
    float v = prediction[(uint)c * w * h + p];
    return masked ? v * mask_image[p] : v;
}
float target_at(int c, int x, int y, uint w, uint h, bool masked) {
    if (x < 0 || y < 0 || x >= (int)w || y >= (int)h)
        return 0.0f;
    uint p = (uint)y * w + (uint)x;
    float v = target[(uint)c * w * h + p];
    return masked ? v * mask_image[p] : v;
}
