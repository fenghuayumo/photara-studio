#include "sift_common.hlsli"

// FilterImage: separable Gaussian convolution with clamped borders. The
// kernel itself is generated on the host exactly like SiftGPU's
// CreateFilterKernel and uploaded as `kernel` (radius + 1 floats). Weights are
// accumulated in ascending order like the CUDA filter loops.
struct PushConstants
{
    uint width;
    uint height;
    uint radius;
    uint direction; // 0 = horizontal, 1 = vertical
};

[[vk::binding(0, 0)]] RWByteAddressBuffer source;
[[vk::binding(1, 0)]] RWByteAddressBuffer destination;
[[vk::binding(2, 0)]] RWByteAddressBuffer kernel;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(128, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint x = dtid.x;
    const uint y = dtid.y;
    if (x >= pc.width) return;

    float value = 0.0f;
    for (int k = -int(pc.radius); k <= int(pc.radius); ++k)
    {
        uint index;
        if (pc.direction == 0u)
        {
            const int sx = clampi(int(x) + k, 0, int(pc.width) - 1);
            index = y * pc.width + uint(sx);
        }
        else
        {
            const int sy = clampi(int(y) + k, 0, int(pc.height) - 1);
            index = uint(sy) * pc.width + x;
        }
        value += load_f32(source, index * 4u) *
                 load_f32(kernel, uint(k + int(pc.radius)) * 4u);
    }
    store_f32(destination, (y * pc.width + x) * 4u, value);
}
