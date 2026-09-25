#include "splat_math.hlsli"

// 8-bit RGBA of the color channels the blender just wrote. The editor preview
// wants bytes in its own image, not planar floats on the host, so the
// quantization happens here and the result never leaves the device.
[[vk::binding(0, 0)]] StructuredBuffer<float> image;
[[vk::binding(1, 0)]] RWStructuredBuffer<uint> rgba;

[numthreads(256, 1, 1)]
void main(uint3 dispatch_id : SV_DispatchThreadID) {
    uint pixels = pc.u0;
    uint index = dispatch_id.x;
    if (index >= pixels) return;
    uint r = uint(saturate(image[index]) * 255.0f + 0.5f);
    uint g = uint(saturate(image[pixels + index]) * 255.0f + 0.5f);
    uint b = uint(saturate(image[2u * pixels + index]) * 255.0f + 0.5f);
    // R8G8B8A8 in memory order, stored as one little-endian word.
    rgba[index] = r | (g << 8) | (b << 16) | (255u << 24);
}
