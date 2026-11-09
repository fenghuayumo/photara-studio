// Shared by standalone shape gradients and the rasterizer's fused Adam.
// Keep the clamp and tie rules identical to the CUDA shape regularizer.
#ifndef PHOTARA_SHAPE_REGULARIZERS_MATH
#define PHOTARA_SHAPE_REGULARIZERS_MATH
void shape_regularizer_gradients(
    float3 raw_scales, float4 raw_quat,
    inout float3 scale_gradient, inout float4 quaternion_gradient,
    float erank_weight, float erank_s3_weight, float quat_weight,
    float axis_ratio_weight, float max_axis_ratio) {
    // Quaternion norm pull: |q| - 1 - log|q| has its minimum on the unit
    // sphere and never saturates.
    if (quat_weight > 0.F) {
        const float norm_squared =
            raw_quat[0] * raw_quat[0] +
            raw_quat[1] * raw_quat[1] +
            raw_quat[2] * raw_quat[2] +
            raw_quat[3] * raw_quat[3];
        if (norm_squared > 1e-20F) {
            const float norm = sqrt(norm_squared);
            const float factor = quat_weight * (norm - 1.F) / norm_squared;
            for (int component = 0; component < 4; ++component)
                quaternion_gradient[component] +=
                    factor * raw_quat[component];
        }
    }

    // Soft hinge in log-ratio space; no parameter projection after Adam.
    if (axis_ratio_weight > 0.F) {
        int longest = 0, shortest = 0;
        for (int axis = 1; axis < 3; ++axis) {
            if (raw_scales[axis] > raw_scales[longest]) longest = axis;
            if (raw_scales[axis] < raw_scales[shortest]) shortest = axis;
        }
        const float excess = raw_scales[longest] -
            raw_scales[shortest] - log(max_axis_ratio);
        if (excess > 0.F && isfinite(excess)) {
            scale_gradient[longest] += axis_ratio_weight * excess;
            scale_gradient[shortest] -= axis_ratio_weight * excess;
        }
    }

    if (erank_weight <= 0.F && erank_s3_weight <= 0.F) return;

    // Effective-rank penalty on the relative squared scale shares
    // x_i = exp(2 (ls_i - ls_max)); r = exp(entropy(p)), penalized below
    // rank two. Sub-gradients keep the max/min axis selection fixed.
    float log_scale[3] = {0, 0, 0};
    int largest = 0;
    for (int axis = 0; axis < 3; ++axis) {
        log_scale[axis] = raw_scales[axis];
        if (!(log_scale[axis] > -1e4F)) log_scale[axis] = -1e4F;
        if (axis > 0 && log_scale[axis] > log_scale[largest])
            largest = axis;
    }
    float x[3] = {0, 0, 0};
    for (int axis = 0; axis < 3; ++axis)
        x[axis] = exp(2.F * (log_scale[axis] - log_scale[largest]));
    const float sum = x[0] + x[1] + x[2];
    if (!(sum > 0.F)) return;
    int smallest = 0;
    for (int axis = 1; axis < 3; ++axis)
        if (x[axis] < x[smallest]) smallest = axis;
    const float share_largest = x[largest] / sum;
    const float raw_share_smallest = x[smallest] / sum;
    const bool smallest_clamped = !(raw_share_smallest > 1e-30F);
    const float share_smallest =
        smallest_clamped ? 1e-30F : raw_share_smallest;
    const float middle_complement = 1.F - share_largest - share_smallest;
    const bool middle_clamped = !(middle_complement > 1e-30F);
    const float share_middle = middle_clamped ? 1e-30F : middle_complement;
    const float entropy = -(
        share_largest * log(share_largest) +
        share_middle * log(share_middle) +
        share_smallest * log(share_smallest));
    const float rank = exp(entropy);
    // reg = max(-log(r - 0.99999), 0): only splats flatter than rank two
    // carry gradient.
    float dreg_dshare[3] = {0, 0, 0};  // largest, middle, smallest order.
    if (erank_weight > 0.F && rank - 0.99999F < 1.F) {
        const float shares[3] = {share_largest, share_middle, share_smallest};
        for (int k = 0; k < 3; ++k)
            dreg_dshare[k] =
                erank_weight * rank * (log(shares[k]) + 1.F) / (rank - 0.99999F);
    }
    const float dloss_dshare_smallest = dreg_dshare[2] + erank_s3_weight;
    float dloss_dx[3] = {0, 0, 0};
    for (int axis = 0; axis < 3; ++axis) {
        float d_largest = 0.F;
        float d_smallest = 0.F;
        if (axis == largest) d_largest = (1.F - share_largest) / sum;
        else d_largest = -share_largest / sum;
        if (axis == smallest) {
            if (!smallest_clamped)
                d_smallest = (1.F - share_smallest) / sum;
        } else {
            d_smallest = -share_smallest / sum;
        }
        const float d_middle = middle_clamped
            ? 0.F
            : -(d_largest + d_smallest);
        const float axis_dloss_dx =
            dreg_dshare[0] * d_largest +
            dreg_dshare[1] * d_middle +
            dloss_dshare_smallest * d_smallest;
        dloss_dx[axis] = axis_dloss_dx;
    }
    // x_largest is identically one, but every other x_j carries
    // dx_j/dls_largest = -2 x_j through the shared max subtraction.
    for (int axis = 0; axis < 3; ++axis) {
        if (axis == largest) {
            float pull = 0.F;
            for (int other = 0; other < 3; ++other)
                if (other != largest) pull += dloss_dx[other] * x[other];
            scale_gradient[axis] +=
                (-2.F * pull);
        } else {
            scale_gradient[axis] +=
                dloss_dx[axis] * 2.F * x[axis];
        }
    }
}
#endif
