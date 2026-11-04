#ifndef SPLAT_DRENDER_GUT_RAY_HLSLI
#define SPLAT_DRENDER_GUT_RAY_HLSLI
#include "splat_gut_numerics.hlsli"
// GUT uses the same world-space ray/row convention as CUDA gut::ray.
// The including shader binds the packed camera as StructuredBuffer<float>.
float3 gut_world_direction(float3 d) {
    return float3(gut_dot(float3(camera[0], camera[1], camera[2]), d),
                  gut_dot(float3(camera[4], camera[5], camera[6]), d),
                  gut_dot(float3(camera[8], camera[9], camera[10]), d));
}
float3 gut_camera_point(float3 p) {
    return float3(gut_affine(p, camera[0], camera[4], camera[8], camera[12]),
                  gut_affine(p, camera[1], camera[5], camera[9], camera[13]),
                  gut_affine(p, camera[2], camera[6], camera[10], camera[14]));
}

float3 gut_camera_direction(float3 d) {
    return float3(camera[0] * d.x + camera[4] * d.y + camera[8] * d.z,
                  camera[1] * d.x + camera[5] * d.y + camera[9] * d.z,
                  camera[2] * d.x + camera[6] * d.y + camera[10] * d.z);
}
#endif
