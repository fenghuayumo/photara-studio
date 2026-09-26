#include "common.hlsli"

// Training images reach the loader packed as one uint per pixel (r | g << 8 |
// b << 16 | a << 24). Unpacking them into planar float RGB on the host costs a
// scalar conversion over every pixel and forces a 3x larger staging copy
// (float planes instead of packed words). This kernel does the same conversion
// on the device, so the upload stays at the packed size and the planar tensors
// are written by the GPU:
//
//   rgb[p * pixels + i] = channel(p) / 255   for p in {0, 1, 2}
//   gray[i] = 0.299 * red + 0.587 * green + 0.114 * blue
//   mask[i] = alpha / 255
//
// gray/mask are optional; their bindings stay bound to the runtime's dummy
// buffer when the caller does not need them, and the flag bit keeps the kernel
// from writing there.
struct PushConstants
{
    uint pixels;
    uint flags;             // bit0: write gray, bit1: write mask
    uint packed_offset;
    uint rgb_offset;
    uint gray_offset;
    uint mask_offset;
};

static const uint kFlagGray = 1u;
static const uint kFlagMask = 2u;

[[vk::binding(0, 0)]] RWByteAddressBuffer packed;
[[vk::binding(1, 0)]] RWByteAddressBuffer rgb;
[[vk::binding(2, 0)]] RWByteAddressBuffer gray;
[[vk::binding(3, 0)]] RWByteAddressBuffer mask;
[[vk::push_constant]] ConstantBuffer<PushConstants> pc;

[numthreads(TT_GROUP_SIZE, 1, 1)]
void main(uint3 dtid : SV_DispatchThreadID)
{
    const uint index = dtid.x;
    if (index >= pc.pixels)
    {
        return;
    }

    const uint value = packed.Load(pc.packed_offset + index * 4u);
    const float inverse_255 = 1.0 / 255.0;
    const float red = float(value & 0xFFu) * inverse_255;
    const float green = float((value >> 8u) & 0xFFu) * inverse_255;
    const float blue = float((value >> 16u) & 0xFFu) * inverse_255;

    rgb.Store(pc.rgb_offset + index * 4u, asuint(red));
    rgb.Store(pc.rgb_offset + (pc.pixels + index) * 4u, asuint(green));
    rgb.Store(pc.rgb_offset + (2u * pc.pixels + index) * 4u, asuint(blue));

    if ((pc.flags & kFlagGray) != 0u)
    {
        gray.Store(pc.gray_offset + index * 4u,
                   asuint(0.299 * red + 0.587 * green + 0.114 * blue));
    }
    if ((pc.flags & kFlagMask) != 0u)
    {
        mask.Store(pc.mask_offset + index * 4u,
                   asuint(float((value >> 24u) & 0xFFu) * inverse_255));
    }
}
