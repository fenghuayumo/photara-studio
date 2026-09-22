#include "sam3.h"
#include "test_utils.h"

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

static std::string format_shape(const std::vector<int> & shape) {
    std::string out = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) {
            out += ", ";
        }
        out += std::to_string(shape[i]);
    }
    out += "]";
    return out;
}

static std::vector<int> trim_shape(const int64_t ne[4]) {
    std::vector<int> shape = {
        (int) ne[0],
        (int) ne[1],
        (int) ne[2],
        (int) ne[3],
    };
    while (!shape.empty() && shape.back() == 1) {
        shape.pop_back();
    }
    if (shape.empty()) {
        shape.push_back(1);
    }
    return shape;
}

static ref_tensor_f32 run_prefix_stage_or_die(const sam3_model          & model,
                                              sam3_vit_prefix_stage       stage,
                                              const ref_tensor_f32      & input,
                                              int                         n_threads) {
    int64_t in_ne[4] = {1, 1, 1, 1};
    for (size_t i = 0; i < input.shape.size() && i < 4; ++i) {
        in_ne[i] = input.shape[i];
    }

    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_prefix_stage(model, stage, input.data.data(), in_ne, out.data, out_ne, n_threads)) {
        fprintf(stderr, "prefix stage %d failed\n", (int) stage);
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static ref_tensor_f32 make_image_input(const ref_tensor_f32 & preprocessed, int img_size) {
    ref_tensor_f32 image = preprocessed;
    image.shape = {img_size, img_size, 3, 1};
    return image;
}

static ref_tensor_f32 run_block0_input_or_die(const sam3_model   & model,
                                              const ref_tensor_f32 & image,
                                              int                   img_size,
                                              int                   n_threads) {
    int64_t out_ne[4] = {0, 0, 0, 0};
    ref_tensor_f32 out;
    if (!sam3_test_run_vit_block0_input(model, image.data.data(), img_size, out.data, out_ne, n_threads)) {
        fprintf(stderr, "block0 input prefix failed\n");
        std::exit(1);
    }
    out.shape = trim_shape(out_ne);
    return out;
}

static void print_diff_row(const char * label,
                           const ref_tensor_f32 & a,
                           const ref_tensor_f32 & b,
                           float atol = 1e-4f) {
    const compare_result r = compare_tensors(a.data.data(), b.data.data(), a.numel(), atol);
    fprintf(stderr, "%-18s %-18s %12.6f %12.6f %10d\n",
            label,
            format_shape(a.shape).c_str(),
            r.max_diff,
            r.mean_diff,
            r.n_bad);
}

int main(int argc, char ** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <model.ggml> [ref_dir]\n", argv[0]);
        fprintf(stderr, "Default ref_dir: tests/ref_phase3\n");
        return 1;
    }

    const std::string model_path = argv[1];
    const std::string ref_dir = argc >= 3 ? argv[2] : "tests/ref_phase3";

    const ref_tensor_f32 preprocessed = load_ref_f32(ref_dir + "/preprocessed");
    if (preprocessed.data.empty() || preprocessed.shape.size() != 4) {
        fprintf(stderr, "failed to load %s/preprocessed\n", ref_dir.c_str());
        return 1;
    }

    const int img_size = preprocessed.shape[2];
    const int n_threads = 8;

    sam3_params cpu_params;
    cpu_params.model_path = model_path;
    cpu_params.use_gpu = false;
    cpu_params.n_threads = n_threads;

    sam3_params metal_params = cpu_params;
    metal_params.use_gpu = true;

    auto cpu_model = sam3_load_model(cpu_params);
    auto metal_model = sam3_load_model(metal_params);
    if (!cpu_model || !metal_model) {
        fprintf(stderr, "failed to load CPU or Metal model\n");
        return 1;
    }

    fprintf(stderr, "\n=== Exact ViT prefix compare ===\n");
    fprintf(stderr, "%-18s %-18s %12s %12s %10s\n",
            "stage", "shape", "max_abs_diff", "mean_abs", "n_bad");

    const ref_tensor_f32 image_input = make_image_input(preprocessed, img_size);

    const ref_tensor_f32 cpu_im2col = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL, image_input, n_threads);
    const ref_tensor_f32 metal_im2col = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_PATCH_IM2COL, image_input, n_threads);
    print_diff_row("patch_im2col", cpu_im2col, metal_im2col);

    const ref_tensor_f32 cpu_patch_raw_shared = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW, cpu_im2col, n_threads);
    const ref_tensor_f32 metal_patch_raw_shared = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT_RAW, cpu_im2col, n_threads);
    print_diff_row("patch_mulmat_raw", cpu_patch_raw_shared, metal_patch_raw_shared);

    {
        ref_tensor_f32 host_patch_raw_f32;
        ref_tensor_f32 host_patch_raw_f64;
        int64_t out_ne_f32[4] = {0, 0, 0, 0};
        int64_t out_ne_f64[4] = {0, 0, 0, 0};
        int64_t in_ne[4] = {1, 1, 1, 1};
        for (size_t i = 0; i < cpu_im2col.shape.size() && i < 4; ++i) {
            in_ne[i] = cpu_im2col.shape[i];
        }

        if (!sam3_test_run_patch_mulmat_host_ref(*cpu_model, cpu_im2col.data.data(), in_ne, false, host_patch_raw_f32.data, out_ne_f32) ||
            !sam3_test_run_patch_mulmat_host_ref(*cpu_model, cpu_im2col.data.data(), in_ne, true,  host_patch_raw_f64.data, out_ne_f64)) {
            fprintf(stderr, "host patch_mulmat reference failed\n");
            return 1;
        }

        host_patch_raw_f32.shape = trim_shape(out_ne_f32);
        host_patch_raw_f64.shape = trim_shape(out_ne_f64);

        print_diff_row("patch_raw_cpu_f32", cpu_patch_raw_shared, host_patch_raw_f32);
        print_diff_row("patch_raw_cpu_f64", cpu_patch_raw_shared, host_patch_raw_f64);
        print_diff_row("patch_raw_mtl_f32", metal_patch_raw_shared, host_patch_raw_f32);
        print_diff_row("patch_raw_mtl_f64", metal_patch_raw_shared, host_patch_raw_f64);
    }

    const ref_tensor_f32 cpu_patch_shared = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT, cpu_im2col, n_threads);
    const ref_tensor_f32 metal_patch_shared = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_PATCH_MULMAT, cpu_im2col, n_threads);
    print_diff_row("patch_mulmat", cpu_patch_shared, metal_patch_shared);

    const ref_tensor_f32 cpu_patch = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_PATCH_EMBED, image_input, n_threads);
    const ref_tensor_f32 metal_patch = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_PATCH_EMBED, image_input, n_threads);
    print_diff_row("patch_embed", cpu_patch, metal_patch);

    const ref_tensor_f32 cpu_pos_shared = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_POS_ADD, cpu_patch, n_threads);
    const ref_tensor_f32 metal_pos_shared = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_POS_ADD, cpu_patch, n_threads);
    print_diff_row("after_pos_embed", cpu_pos_shared, metal_pos_shared);

    const ref_tensor_f32 cpu_ln_norm_shared = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM, cpu_pos_shared, n_threads);
    const ref_tensor_f32 metal_ln_norm_shared = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_LN_PRE_NORM, cpu_pos_shared, n_threads);
    print_diff_row("ln_pre_norm", cpu_ln_norm_shared, metal_ln_norm_shared);

    const ref_tensor_f32 cpu_ln_pre_shared = run_prefix_stage_or_die(*cpu_model, SAM3_VIT_PREFIX_STAGE_LN_PRE, cpu_pos_shared, n_threads);
    const ref_tensor_f32 metal_ln_pre_shared = run_prefix_stage_or_die(*metal_model, SAM3_VIT_PREFIX_STAGE_LN_PRE, cpu_pos_shared, n_threads);
    print_diff_row("after_ln_pre", cpu_ln_pre_shared, metal_ln_pre_shared);

    const ref_tensor_f32 cpu_block0_input = run_block0_input_or_die(*cpu_model, preprocessed, img_size, n_threads);
    const ref_tensor_f32 metal_block0_input = run_block0_input_or_die(*metal_model, preprocessed, img_size, n_threads);
    print_diff_row("block0_input", cpu_block0_input, metal_block0_input);

    return 0;
}
