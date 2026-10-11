// Dedicated forward-only variant: ordinary training shaders keep their
// Gaussian alpha and backward-state behavior unchanged.
#define SPLAT_BLEND_NO_GEOMETRY 1
#define SPLAT_BLEND_GAUSS_SLOTS 5u
#define SPLAT_BLEND_RINGS 1
#include "splat_blend.hlsl"
