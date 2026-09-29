// Generated variant of the descriptor-tile matcher; see sift_match_tiles.hlsl.
//
// Uses the SM 6.4 integer dot product (SPV_KHR_integer_dot_product), so it
// compiles to SPIR-V 1.5 and is only selected on devices that report
// shaderIntegerDotProduct and Vulkan 1.2 or newer.
#define PHOTARA_MATCH_DP4A 1

#define TILE_ROWS 128
#include "sift_match_tiles.hlsl"
