// Stretch planar float RGB [3,H,W] into a packed RGBA8 buffer the size of the
// editor's shared preview image. The editor samples that image full-frame, so
// letterboxing here would show up as empty bars.
struct PreviewPush {
    uint src_w;
    uint src_h;
    uint dst_w;
    uint dst_h;
};

[[vk::push_constant]] ConstantBuffer<PreviewPush> pc;

[[vk::binding(0, 0)]] StructuredBuffer<float> image;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> rgba;

[numthreads(16, 16, 1)]
void main(uint3 id : SV_DispatchThreadID) {
    if (id.x >= pc.dst_w || id.y >= pc.dst_h) return;
    if (pc.src_w == 0 || pc.src_h == 0) return;
    uint sx = min(
        (uint)((float(id.x) + 0.5f) * float(pc.src_w) / float(pc.dst_w)),
        pc.src_w - 1);
    uint sy = min(
        (uint)((float(id.y) + 0.5f) * float(pc.src_h) / float(pc.dst_h)),
        pc.src_h - 1);
    uint pixels = pc.src_w * pc.src_h;
    uint index = sy * pc.src_w + sx;
    uint r = (uint)(saturate(image[index]) * 255.0f + 0.5f);
    uint g = (uint)(saturate(image[pixels + index]) * 255.0f + 0.5f);
    uint b = (uint)(saturate(image[pixels * 2 + index]) * 255.0f + 0.5f);
    rgba[id.y * pc.dst_w + id.x] = r | (g << 8) | (b << 16) | (255u << 24);
}
