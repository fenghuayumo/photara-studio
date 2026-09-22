#import "ggml-metal-device.h"

#import "ggml-impl.h"

#include <Foundation/Foundation.h>

#include <Metal/Metal.h>

#include <stdatomic.h>
#include <inttypes.h>
#include <pthread.h>
#include <string.h>

#ifndef TARGET_OS_VISION
#define TARGET_OS_VISION 0
#endif

// create residency sets only on macOS >= 15.0
#if !TARGET_CPU_X86_64 && TARGET_OS_OSX && __MAC_OS_X_VERSION_MAX_ALLOWED >= 150000 || \
    TARGET_OS_IOS && __IPHONE_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_TV && __TV_OS_VERSION_MAX_ALLOWED >= 180000 || \
    TARGET_OS_VISION && __VISION_OS_VERSION_MAX_ALLOWED >= 200000
#define GGML_METAL_HAS_RESIDENCY_SETS 1
#endif

// overload of MTLGPUFamilyMetalX (not available in some environments)
static const NSInteger MTLGPUFamilyMetal3_GGML = 5001;
static const NSInteger MTLGPUFamilyMetal4_GGML = 5002;

#define GGML_METAL_PROFILE_MAX_CBS         9
#define GGML_METAL_PROFILE_MAX_KERNELS   512
#define GGML_METAL_PROFILE_MAX_NAME       96

struct ggml_metal_profile_dispatch {
    enum ggml_op op;
    char pipeline[GGML_METAL_PROFILE_MAX_NAME];
    uint32_t tg0;
    uint32_t tg1;
    uint32_t tg2;
    uint32_t tptg0;
    uint32_t tptg1;
    uint32_t tptg2;
    uint32_t sample0;
    uint32_t sample1;
};

struct ggml_metal_profile_cb {
    int cb_idx;
    id<MTLCommandBuffer> cmd_buf;
    id<MTLCounterSampleBuffer> sample_buf;
    uint32_t sample_next;
    uint32_t dispatch_count;
    uint32_t dispatch_cap;
    struct ggml_metal_profile_dispatch * dispatches;
};

struct ggml_metal_profile_op_stats {
    uint64_t encode_count;
    uint64_t host_encode_us;
    uint64_t concurrency_us;
    uint64_t set_pipeline_us;
    uint64_t set_bytes_us;
    uint64_t set_buffer_us;
    uint64_t dispatch_us;
    uint64_t dispatch_count;
    uint64_t gpu_ns;
};

struct ggml_metal_profile_kernel_stats {
    char name[GGML_METAL_PROFILE_MAX_NAME];
    uint64_t dispatch_count;
    uint64_t gpu_ns;
};

struct ggml_metal_profile_state {
    bool active;
    bool sample_dispatch;
    bool printed;

    size_t dispatch_cap;

    id<MTLCounterSet> timestamp_counter_set;

    uint64_t graph_compute_us;

    uint64_t cmd_buf_create_count;
    uint64_t cmd_buf_create_us;
    uint64_t cmd_buf_commit_count;
    uint64_t cmd_buf_commit_us;
    uint64_t cmd_buf_wait_count;
    uint64_t cmd_buf_wait_us;

    uint64_t encoder_create_count;
    uint64_t encoder_create_us;
    uint64_t encoder_end_count;
    uint64_t encoder_end_us;
    uint64_t memory_barrier_count;
    uint64_t memory_barrier_us;

    uint64_t pipeline_lookup_count;
    uint64_t pipeline_lookup_hit_count;
    uint64_t pipeline_lookup_us;
    uint64_t pipeline_compile_count;
    uint64_t pipeline_compile_us;

    uint64_t set_pipeline_count;
    uint64_t set_pipeline_us;
    uint64_t set_bytes_count;
    uint64_t set_bytes_us;
    uint64_t set_buffer_count;
    uint64_t set_buffer_us;
    uint64_t dispatch_count;
    uint64_t dispatch_us;

    uint64_t gpu_kernel_ns;
    uint64_t gpu_dispatch_profiled;

    struct ggml_metal_profile_op_stats ops[GGML_OP_COUNT];

    struct ggml_metal_profile_kernel_stats kernels[GGML_METAL_PROFILE_MAX_KERNELS];
    int n_kernels;

    struct ggml_metal_profile_cb cbs[GGML_METAL_PROFILE_MAX_CBS];
};

static pthread_mutex_t g_ggml_metal_profile_lock = PTHREAD_MUTEX_INITIALIZER;
static struct ggml_metal_profile_state g_ggml_metal_profile = {};

static bool ggml_metal_profile_env_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char * val = getenv("GGML_METAL_PROFILE");
        cached = val && atoi(val) != 0;
    }
    return cached != 0;
}

static bool ggml_metal_log_pipelines_enabled(void) {
    static int cached = -1;
    if (cached == -1) {
        const char * val = getenv("GGML_METAL_LOG_PIPELINES");
        cached = val && atoi(val) != 0;
    }
    return cached != 0;
}

static void ggml_metal_profile_reset_locked(void) {
    for (int i = 0; i < GGML_METAL_PROFILE_MAX_CBS; ++i) {
        if (g_ggml_metal_profile.cbs[i].sample_buf) {
            [g_ggml_metal_profile.cbs[i].sample_buf release];
        }
        free(g_ggml_metal_profile.cbs[i].dispatches);
    }

    memset(&g_ggml_metal_profile, 0, sizeof(g_ggml_metal_profile));
}

static id<MTLCounterSet> ggml_metal_profile_get_timestamp_counter_set(id<MTLDevice> device) {
    if (![device supportsCounterSampling:MTLCounterSamplingPointAtDispatchBoundary]) {
        return nil;
    }

    for (id<MTLCounterSet> counter_set in device.counterSets) {
        if ([counter_set.name isEqualToString:MTLCommonCounterSetTimestamp]) {
            return counter_set;
        }
    }

    return nil;
}

static void ggml_metal_profile_add_kernel_gpu_locked(const char * name, uint64_t gpu_ns) {
    if (!name || name[0] == '\0') {
        name = "(unknown)";
    }

    for (int i = 0; i < g_ggml_metal_profile.n_kernels; ++i) {
        if (strncmp(g_ggml_metal_profile.kernels[i].name, name, GGML_METAL_PROFILE_MAX_NAME) == 0) {
            g_ggml_metal_profile.kernels[i].dispatch_count++;
            g_ggml_metal_profile.kernels[i].gpu_ns += gpu_ns;
            return;
        }
    }

    if (g_ggml_metal_profile.n_kernels >= GGML_METAL_PROFILE_MAX_KERNELS) {
        return;
    }

    struct ggml_metal_profile_kernel_stats * kernel = &g_ggml_metal_profile.kernels[g_ggml_metal_profile.n_kernels++];
    snprintf(kernel->name, sizeof(kernel->name), "%s", name);
    kernel->dispatch_count = 1;
    kernel->gpu_ns = gpu_ns;
}

static int ggml_metal_profile_cmp_kernel_count_desc(const void * a, const void * b) {
    const struct ggml_metal_profile_kernel_stats * ka = (const struct ggml_metal_profile_kernel_stats *) a;
    const struct ggml_metal_profile_kernel_stats * kb = (const struct ggml_metal_profile_kernel_stats *) b;
    if (ka->dispatch_count < kb->dispatch_count) return 1;
    if (ka->dispatch_count > kb->dispatch_count) return -1;
    return strcmp(ka->name, kb->name);
}

static int ggml_metal_profile_cmp_kernel_gpu_desc(const void * a, const void * b) {
    const struct ggml_metal_profile_kernel_stats * ka = (const struct ggml_metal_profile_kernel_stats *) a;
    const struct ggml_metal_profile_kernel_stats * kb = (const struct ggml_metal_profile_kernel_stats *) b;
    if (ka->gpu_ns < kb->gpu_ns) return 1;
    if (ka->gpu_ns > kb->gpu_ns) return -1;
    return strcmp(ka->name, kb->name);
}

bool ggml_metal_profile_enabled(void) {
    return ggml_metal_profile_env_enabled();
}

void ggml_metal_profile_begin_graph(ggml_metal_device_t dev, int n_nodes) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    ggml_metal_profile_reset_locked();
    g_ggml_metal_profile.active = true;
    g_ggml_metal_profile.dispatch_cap = MAX((size_t) 4096, (size_t) n_nodes * 8);
    g_ggml_metal_profile.timestamp_counter_set = ggml_metal_profile_get_timestamp_counter_set(ggml_metal_device_get_obj(dev));
    g_ggml_metal_profile.sample_dispatch = g_ggml_metal_profile.timestamp_counter_set != nil;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void * ggml_metal_profile_get_cb(int cb_idx) {
    if (!ggml_metal_profile_enabled()) {
        return NULL;
    }

    if (cb_idx < 0 || cb_idx >= GGML_METAL_PROFILE_MAX_CBS) {
        return NULL;
    }

    return &g_ggml_metal_profile.cbs[cb_idx];
}

void ggml_metal_profile_note_graph_compute_us(uint64_t us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.graph_compute_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_command_buffer_create(int cb_idx, ggml_metal_cmd_buf_t cmd_buf_raw, uint64_t us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);

    g_ggml_metal_profile.cmd_buf_create_count++;
    g_ggml_metal_profile.cmd_buf_create_us += us;

    if (cb_idx >= 0 && cb_idx < GGML_METAL_PROFILE_MAX_CBS) {
        struct ggml_metal_profile_cb * cb = &g_ggml_metal_profile.cbs[cb_idx];
        cb->cb_idx = cb_idx;
        cb->cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
        if (cb->dispatch_cap == 0) {
            cb->dispatch_cap = (uint32_t) g_ggml_metal_profile.dispatch_cap;
            cb->dispatches = calloc(cb->dispatch_cap, sizeof(struct ggml_metal_profile_dispatch));
        }

        if (g_ggml_metal_profile.sample_dispatch && cb->sample_buf == nil) {
            id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;
            id<MTLDevice> device = cmd_buf.commandQueue.device;

            MTLCounterSampleBufferDescriptor * desc = [MTLCounterSampleBufferDescriptor new];
            desc.counterSet = g_ggml_metal_profile.timestamp_counter_set;
            desc.storageMode = MTLStorageModeShared;
            desc.sampleCount = g_ggml_metal_profile.dispatch_cap * 2;
            desc.label = [NSString stringWithFormat:@"ggml-metal-profile-cb-%d", cb_idx];

            NSError * error = nil;
            cb->sample_buf = [device newCounterSampleBufferWithDescriptor:desc error:&error];
            [desc release];

            if (cb->sample_buf) {
            } else {
                g_ggml_metal_profile.sample_dispatch = false;
                if (error) {
                    GGML_LOG_WARN("%s: unable to create counter sample buffer: %s\n", __func__, [[error localizedDescription] UTF8String]);
                }
            }
        }
    }

    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_command_buffer_commit(int cb_idx, uint64_t us) {
    GGML_UNUSED(cb_idx);

    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.cmd_buf_commit_count++;
    g_ggml_metal_profile.cmd_buf_commit_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_command_buffer_wait(int cb_idx, uint64_t us) {
    GGML_UNUSED(cb_idx);

    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.cmd_buf_wait_count++;
    g_ggml_metal_profile.cmd_buf_wait_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_encoder_create(uint64_t us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.encoder_create_count++;
    g_ggml_metal_profile.encoder_create_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_encoder_end(uint64_t us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.encoder_end_count++;
    g_ggml_metal_profile.encoder_end_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_memory_barrier(uint64_t us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.memory_barrier_count++;
    g_ggml_metal_profile.memory_barrier_us += us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_note_op_encode(enum ggml_op op, uint64_t total_us, uint64_t concurrency_us) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);
    g_ggml_metal_profile.ops[op].encode_count++;
    g_ggml_metal_profile.ops[op].host_encode_us += total_us;
    g_ggml_metal_profile.ops[op].concurrency_us += concurrency_us;
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

void ggml_metal_profile_finalize_and_log(void) {
    if (!ggml_metal_profile_enabled()) {
        return;
    }

    pthread_mutex_lock(&g_ggml_metal_profile_lock);

    if (!g_ggml_metal_profile.active || g_ggml_metal_profile.printed) {
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
        return;
    }

    g_ggml_metal_profile.printed = true;

    uint64_t gpu_span_ns = 0;
    double gpu_first_s = 0.0;
    double gpu_last_s = 0.0;

    uint64_t bucket_lt10 = 0;
    uint64_t bucket_10_50 = 0;
    uint64_t bucket_50_100 = 0;
    uint64_t bucket_100_500 = 0;
    uint64_t bucket_gt500 = 0;
    uint64_t work_lt1k = 0;
    uint64_t work_1k_16k = 0;
    uint64_t work_16k_256k = 0;
    uint64_t work_gt256k = 0;

    for (int i = 0; i < GGML_METAL_PROFILE_MAX_CBS; ++i) {
        struct ggml_metal_profile_cb * cb = &g_ggml_metal_profile.cbs[i];
        if (!cb->cmd_buf) {
            continue;
        }

        const NSTimeInterval kernel_start = cb->cmd_buf.kernelStartTime;
        const NSTimeInterval kernel_end = cb->cmd_buf.kernelEndTime;
        const uint64_t cb_gpu_ns = kernel_end > kernel_start ? (uint64_t) ((kernel_end - kernel_start) * 1e9) : 0;
        g_ggml_metal_profile.gpu_kernel_ns += cb_gpu_ns;

        if (kernel_end > kernel_start) {
            if (gpu_first_s == 0.0 || kernel_start < gpu_first_s) {
                gpu_first_s = kernel_start;
            }
            if (kernel_end > gpu_last_s) {
                gpu_last_s = kernel_end;
            }
        }

        if (cb->dispatch_count == 0) {
            continue;
        }

        NSData * resolved = nil;
        const MTLCounterResultTimestamp * samples = NULL;
        if (cb->sample_buf != nil && cb->sample_next > 0) {
            resolved = [cb->sample_buf resolveCounterRange:NSMakeRange(0, cb->sample_next)];
            if (resolved && resolved.length >= cb->sample_next * sizeof(MTLCounterResultTimestamp)) {
                samples = (const MTLCounterResultTimestamp *) resolved.bytes;
            }
        }

        uint64_t tick_first = UINT64_MAX;
        uint64_t tick_last = 0;
        for (uint32_t j = 0; j < cb->dispatch_count; ++j) {
            const struct ggml_metal_profile_dispatch * rec = &cb->dispatches[j];
            const uint64_t threads_total =
                    (uint64_t) rec->tg0 * (uint64_t) rec->tg1 * (uint64_t) rec->tg2 *
                    (uint64_t) rec->tptg0 * (uint64_t) rec->tptg1 * (uint64_t) rec->tptg2;

            if (threads_total < 1024) {
                work_lt1k++;
            } else if (threads_total < 16384) {
                work_1k_16k++;
            } else if (threads_total < 262144) {
                work_16k_256k++;
            } else {
                work_gt256k++;
            }

            ggml_metal_profile_add_kernel_gpu_locked(rec->pipeline, 0);

            if (samples == NULL || rec->sample0 == UINT32_MAX || rec->sample1 == UINT32_MAX) {
                continue;
            }

            const uint64_t t0 = samples[rec->sample0].timestamp;
            const uint64_t t1 = samples[rec->sample1].timestamp;
            if (t0 == MTLCounterErrorValue || t1 == MTLCounterErrorValue || t1 <= t0) {
                continue;
            }
            tick_first = MIN(tick_first, t0);
            tick_last = MAX(tick_last, t1);
        }

        const bool have_dispatch_timestamps = tick_last > tick_first && cb_gpu_ns > 0;
        const double ns_per_tick = have_dispatch_timestamps ? (double) cb_gpu_ns / (double) (tick_last - tick_first) : 0.0;

        for (uint32_t j = 0; j < cb->dispatch_count; ++j) {
            const struct ggml_metal_profile_dispatch * rec = &cb->dispatches[j];
            if (!have_dispatch_timestamps || rec->sample0 == UINT32_MAX || rec->sample1 == UINT32_MAX) {
                continue;
            }

            const uint64_t t0 = samples[rec->sample0].timestamp;
            const uint64_t t1 = samples[rec->sample1].timestamp;
            if (t0 == MTLCounterErrorValue || t1 == MTLCounterErrorValue || t1 <= t0) {
                continue;
            }

            const uint64_t gpu_ns = (uint64_t) ((double) (t1 - t0) * ns_per_tick);
            g_ggml_metal_profile.gpu_dispatch_profiled++;

            if (rec->op >= 0 && rec->op < GGML_OP_COUNT) {
                g_ggml_metal_profile.ops[rec->op].gpu_ns += gpu_ns;
            }
            ggml_metal_profile_add_kernel_gpu_locked(rec->pipeline, gpu_ns);

            const double gpu_us = gpu_ns / 1000.0;
            if (gpu_us < 10.0) {
                bucket_lt10++;
            } else if (gpu_us < 50.0) {
                bucket_10_50++;
            } else if (gpu_us < 100.0) {
                bucket_50_100++;
            } else if (gpu_us < 500.0) {
                bucket_100_500++;
            } else {
                bucket_gt500++;
            }
        }
    }

    if (gpu_last_s > gpu_first_s) {
        gpu_span_ns = (uint64_t) ((gpu_last_s - gpu_first_s) * 1e9);
    }

    fprintf(stderr, "\nGGML_METAL_PROFILE SUMMARY\n");
    fprintf(stderr, "  graph_compute_submit_ms: %.3f\n", g_ggml_metal_profile.graph_compute_us / 1000.0);
    fprintf(stderr, "  gpu_kernel_sum_ms:       %.3f\n", g_ggml_metal_profile.gpu_kernel_ns / 1e6);
    fprintf(stderr, "  gpu_kernel_span_ms:      %.3f\n", gpu_span_ns / 1e6);
    fprintf(stderr, "  command_buffers:         created=%" PRIu64 " committed=%" PRIu64 " waited=%" PRIu64 "\n",
            g_ggml_metal_profile.cmd_buf_create_count,
            g_ggml_metal_profile.cmd_buf_commit_count,
            g_ggml_metal_profile.cmd_buf_wait_count);
    fprintf(stderr, "  compute_encoders:        created=%" PRIu64 " ended=%" PRIu64 "\n",
            g_ggml_metal_profile.encoder_create_count,
            g_ggml_metal_profile.encoder_end_count);
    fprintf(stderr, "  dispatches:              total=%" PRIu64 " profiled=%" PRIu64 " avg_gpu_us=%.3f\n",
            g_ggml_metal_profile.dispatch_count,
            g_ggml_metal_profile.gpu_dispatch_profiled,
            g_ggml_metal_profile.dispatch_count ? (double) g_ggml_metal_profile.gpu_kernel_ns / 1000.0 / (double) g_ggml_metal_profile.dispatch_count : 0.0);
    fprintf(stderr, "  pipeline_lookup:         count=%" PRIu64 " hits=%" PRIu64 " time_ms=%.3f\n",
            g_ggml_metal_profile.pipeline_lookup_count,
            g_ggml_metal_profile.pipeline_lookup_hit_count,
            g_ggml_metal_profile.pipeline_lookup_us / 1000.0);
    fprintf(stderr, "  pipeline_compile:        count=%" PRIu64 " time_ms=%.3f\n",
            g_ggml_metal_profile.pipeline_compile_count,
            g_ggml_metal_profile.pipeline_compile_us / 1000.0);
    fprintf(stderr, "  host_components_ms:\n");
    fprintf(stderr, "    cmd_buffer_create: %.3f\n", g_ggml_metal_profile.cmd_buf_create_us / 1000.0);
    fprintf(stderr, "    cmd_buffer_commit: %.3f\n", g_ggml_metal_profile.cmd_buf_commit_us / 1000.0);
    fprintf(stderr, "    cmd_buffer_wait:   %.3f\n", g_ggml_metal_profile.cmd_buf_wait_us / 1000.0);
    fprintf(stderr, "    encoder_create:    %.3f\n", g_ggml_metal_profile.encoder_create_us / 1000.0);
    fprintf(stderr, "    encoder_end:       %.3f\n", g_ggml_metal_profile.encoder_end_us / 1000.0);
    fprintf(stderr, "    memory_barrier:    %.3f\n", g_ggml_metal_profile.memory_barrier_us / 1000.0);
    fprintf(stderr, "    set_pipeline:      %.3f\n", g_ggml_metal_profile.set_pipeline_us / 1000.0);
    fprintf(stderr, "    set_bytes:         %.3f\n", g_ggml_metal_profile.set_bytes_us / 1000.0);
    fprintf(stderr, "    set_buffer:        %.3f\n", g_ggml_metal_profile.set_buffer_us / 1000.0);
    fprintf(stderr, "    dispatch_submit:   %.3f\n", g_ggml_metal_profile.dispatch_us / 1000.0);
    fprintf(stderr, "  dispatch_gpu_histogram:\n");
    fprintf(stderr, "    <10us:    %" PRIu64 "\n", bucket_lt10);
    fprintf(stderr, "    10-50us:  %" PRIu64 "\n", bucket_10_50);
    fprintf(stderr, "    50-100us: %" PRIu64 "\n", bucket_50_100);
    fprintf(stderr, "    100-500us:%" PRIu64 "\n", bucket_100_500);
    fprintf(stderr, "    >500us:   %" PRIu64 "\n", bucket_gt500);
    fprintf(stderr, "  dispatch_work_histogram:\n");
    fprintf(stderr, "    <1k threads:      %" PRIu64 "\n", work_lt1k);
    fprintf(stderr, "    1k-16k threads:   %" PRIu64 "\n", work_1k_16k);
    fprintf(stderr, "    16k-256k threads: %" PRIu64 "\n", work_16k_256k);
    fprintf(stderr, "    >256k threads:    %" PRIu64 "\n", work_gt256k);
    fprintf(stderr, "  op_breakdown:\n");
    for (int op = 0; op < GGML_OP_COUNT; ++op) {
        const struct ggml_metal_profile_op_stats * st = &g_ggml_metal_profile.ops[op];
        if (st->encode_count == 0 && st->dispatch_count == 0) {
            continue;
        }
        fprintf(stderr, "    %-18s count=%" PRIu64 " host_encode_ms=%.3f concurrency_ms=%.3f host_dispatch_ms=%.3f gpu_ms=%.3f dispatches=%" PRIu64 "\n",
                ggml_op_name((enum ggml_op) op),
                st->encode_count,
                st->host_encode_us / 1000.0,
                st->concurrency_us / 1000.0,
                st->dispatch_us / 1000.0,
                st->gpu_ns / 1e6,
                st->dispatch_count);
    }

    if (g_ggml_metal_profile.n_kernels > 0) {
        struct ggml_metal_profile_kernel_stats by_count[GGML_METAL_PROFILE_MAX_KERNELS];
        struct ggml_metal_profile_kernel_stats by_gpu[GGML_METAL_PROFILE_MAX_KERNELS];
        memcpy(by_count, g_ggml_metal_profile.kernels, sizeof(by_count));
        memcpy(by_gpu,   g_ggml_metal_profile.kernels, sizeof(by_gpu));
        qsort(by_count, g_ggml_metal_profile.n_kernels, sizeof(by_count[0]), ggml_metal_profile_cmp_kernel_count_desc);
        qsort(by_gpu,   g_ggml_metal_profile.n_kernels, sizeof(by_gpu[0]),   ggml_metal_profile_cmp_kernel_gpu_desc);

        fprintf(stderr, "  top_kernels_by_count:\n");
        for (int i = 0; i < MIN(20, g_ggml_metal_profile.n_kernels); ++i) {
            fprintf(stderr, "    %-40s dispatches=%" PRIu64 " gpu_ms=%.3f\n",
                    by_count[i].name, by_count[i].dispatch_count, by_count[i].gpu_ns / 1e6);
        }

        fprintf(stderr, "  top_kernels_by_gpu:\n");
        if (g_ggml_metal_profile.gpu_dispatch_profiled == 0) {
            fprintf(stderr, "    (per-dispatch GPU timestamps unavailable on this run)\n");
        } else {
            for (int i = 0; i < MIN(20, g_ggml_metal_profile.n_kernels); ++i) {
                fprintf(stderr, "    %-40s dispatches=%" PRIu64 " gpu_ms=%.3f\n",
                        by_gpu[i].name, by_gpu[i].dispatch_count, by_gpu[i].gpu_ns / 1e6);
            }
        }
    }

    fprintf(stderr, "END_GGML_METAL_PROFILE\n\n");

    ggml_metal_profile_reset_locked();
    pthread_mutex_unlock(&g_ggml_metal_profile_lock);
}

#if !GGML_METAL_EMBED_LIBRARY
// Here to assist with NSBundle Path Hack
@interface GGMLMetalClass : NSObject
@end
@implementation GGMLMetalClass
@end
#endif

//
// MTLFunctionConstantValues wrapper
//

struct ggml_metal_cv {
    MTLFunctionConstantValues * obj;
};

ggml_metal_cv_t ggml_metal_cv_init(void) {
    ggml_metal_cv_t res = calloc(1, sizeof(struct ggml_metal_cv));

    res->obj = [[MTLFunctionConstantValues alloc] init];

    return res;
}

void ggml_metal_cv_free(ggml_metal_cv_t cv) {
    [cv->obj release];
    free(cv);
}

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeShort atIndex:idx];
}

void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeInt atIndex:idx];
}

void ggml_metal_cv_set_bool(ggml_metal_cv_t cv, bool value, int32_t idx) {
    [cv->obj setConstantValue:&value type:MTLDataTypeBool atIndex:idx];
}

//
// MTLComputePipelineState wrapper
//

struct ggml_metal_pipeline {
    id<MTLComputePipelineState> obj;
    char name[GGML_METAL_PROFILE_MAX_NAME];
};

ggml_metal_pipeline_t ggml_metal_pipeline_init(void) {
    ggml_metal_pipeline_t res = calloc(1, sizeof(struct ggml_metal_pipeline));

    *res = (struct ggml_metal_pipeline) {
        /*.obj  =*/ nil,
        /*.name =*/ { 0 },
    };

    return res;
}

void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline) {
    [pipeline->obj release];

    free(pipeline);
}

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline) {
    return pipeline.pipeline->obj.maxTotalThreadsPerThreadgroup;
}

struct ggml_metal_library {
    id<MTLLibrary> obj;
    id<MTLDevice> device;

    ggml_metal_pipelines_t pipelines; // cache of compiled pipelines

    NSLock * lock;
};

ggml_metal_library_t ggml_metal_library_init(ggml_metal_device_t dev) {
    id<MTLLibrary> library = nil;
    id<MTLDevice> device = ggml_metal_device_get_obj(dev);

    // load library
    //
    // - first check if the library is embedded
    // - then check if the library is in the bundle
    // - if not found, load the source and compile it
    // - if that fails, return NULL
    //
    // TODO: move to a function
    {
        const int64_t t_start = ggml_time_us();

        NSError * error = nil;
        NSString * src = nil;

#if GGML_METAL_EMBED_LIBRARY
        GGML_LOG_INFO("%s: using embedded metal library\n", __func__);

        extern const char ggml_metallib_start[];
        extern const char ggml_metallib_end[];

        src = [[NSString alloc] initWithBytes:ggml_metallib_start length:(ggml_metallib_end-ggml_metallib_start) encoding:NSUTF8StringEncoding];
#else

#ifdef SWIFT_PACKAGE
        NSBundle * bundle = SWIFTPM_MODULE_BUNDLE;
#else
        NSBundle * bundle = [NSBundle bundleForClass:[GGMLMetalClass class]];
#endif

        NSString * path_lib = [bundle pathForResource:@"default" ofType:@"metallib"];
        if (path_lib == nil) {
            // Try to find the resource in the directory where the current binary located.
            NSString * bin_cur = [[NSProcessInfo processInfo] arguments][0];
            NSString * bin_dir = [bin_cur stringByDeletingLastPathComponent];

            NSString * path_lib_default = [NSString pathWithComponents:@[bin_dir, @"default.metallib"]];
            if ([[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                GGML_LOG_INFO("%s: found '%s'\n", __func__, [path_lib_default UTF8String]);

                NSDictionary * atts = [[NSFileManager defaultManager] attributesOfItemAtPath:path_lib_default error:&error];
                if (atts && atts[NSFileType] == NSFileTypeSymbolicLink) {
                    // Optionally, if this is a symlink, try to resolve it.
                    path_lib_default = [[NSFileManager defaultManager] destinationOfSymbolicLinkAtPath:path_lib_default error:&error];
                    if (path_lib_default && [path_lib_default length] > 0 && ![[path_lib_default substringToIndex:1] isEqualToString:@"/"]) {
                        // It is a relative path, adding the binary directory as directory prefix.
                        path_lib_default = [NSString pathWithComponents:@[bin_dir, path_lib_default]];
                    }
                    if (!path_lib_default || ![[NSFileManager defaultManager] isReadableFileAtPath:path_lib_default]) {
                        // Link to the resource could not be resolved.
                        path_lib_default = nil;
                    } else {
                        GGML_LOG_INFO("%s: symlink resolved '%s'\n", __func__, [path_lib_default UTF8String]);
                    }
                }
            } else {
                // The resource couldn't be found in the binary's directory.
                path_lib_default = nil;
            }

            path_lib = path_lib_default;
        }

        if (path_lib != nil) {
            // pre-compiled library found
            NSURL * libURL = [NSURL fileURLWithPath:path_lib];
            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_lib UTF8String]);

            library = [device newLibraryWithURL:libURL error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        } else {
            GGML_LOG_INFO("%s: default.metallib not found, loading from source\n", __func__);

            NSString * path_source;
            NSString * path_resource = [[NSProcessInfo processInfo].environment objectForKey:@"GGML_METAL_PATH_RESOURCES"];

            GGML_LOG_INFO("%s: GGML_METAL_PATH_RESOURCES = %s\n", __func__, path_resource ? [path_resource UTF8String] : "nil");

            if (path_resource) {
                path_source = [path_resource stringByAppendingPathComponent:@"ggml-metal.metal"];
            } else {
                path_source = [bundle pathForResource:@"ggml-metal" ofType:@"metal"];
            }

            if (path_source == nil) {
                GGML_LOG_WARN("%s: error: could not use bundle path to find ggml-metal.metal, falling back to trying cwd\n", __func__);
                path_source = @"ggml-metal.metal";
            }

            GGML_LOG_INFO("%s: loading '%s'\n", __func__, [path_source UTF8String]);

            src = [NSString stringWithContentsOfFile:path_source encoding:NSUTF8StringEncoding error:&error];
            if (error) {
                GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                return nil;
            }
        }
#endif

        if (!library) {
            @autoreleasepool {
                // dictionary of preprocessor macros
                NSMutableDictionary * prep = [NSMutableDictionary dictionary];

                if (ggml_metal_device_get_props(dev)->has_bfloat) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_BF16"];
                }

                if (ggml_metal_device_get_props(dev)->has_tensor) {
                    [prep setObject:@"1" forKey:@"GGML_METAL_HAS_TENSOR"];
                }

#if GGML_METAL_EMBED_LIBRARY
                [prep setObject:@"1" forKey:@"GGML_METAL_EMBED_LIBRARY"];
#endif

                MTLCompileOptions * options = [MTLCompileOptions new];
                options.preprocessorMacros = prep;
                if (@available(macOS 15.0, iOS 18.0, *)) {
                    options.mathMode = MTLMathModeSafe;
                    options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
                } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
                    options.fastMathEnabled = NO;
#pragma clang diagnostic pop
                }

                library = [device newLibraryWithSource:src options:options error:&error];
                if (error) {
                    GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
                    return nil;
                }

#if !__has_feature(objc_arc)
                [options release];
#endif
            }
        }

#if GGML_METAL_EMBED_LIBRARY
        [src release];
#endif // GGML_METAL_EMBED_LIBRARY

        GGML_LOG_INFO("%s: loaded in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));

    res->obj       = library;
    res->device    = device;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose) {
    if (source == NULL) {
        GGML_LOG_ERROR("%s: source is NULL\n", __func__);
        return NULL;
    }

    id<MTLDevice> device = ggml_metal_device_get_obj(dev);
    id<MTLLibrary> library = nil;
    NSError * error = nil;

    const int64_t t_start = ggml_time_us();

    NSString * src = [[NSString alloc] initWithBytes:source
                                              length:strlen(source)
                                            encoding:NSUTF8StringEncoding];
    if (!src) {
        GGML_LOG_ERROR("%s: failed to create NSString from source\n", __func__);
        return NULL;
    }

    @autoreleasepool {
        NSMutableDictionary * prep = [NSMutableDictionary dictionary];

        MTLCompileOptions * options = [MTLCompileOptions new];
        options.preprocessorMacros = prep;
        if (@available(macOS 15.0, iOS 18.0, *)) {
            options.mathMode = MTLMathModeSafe;
            options.mathFloatingPointFunctions = MTLMathFloatingPointFunctionsPrecise;
        } else {
#pragma clang diagnostic push
#pragma clang diagnostic ignored "-Wdeprecated-declarations"
            options.fastMathEnabled = NO;
#pragma clang diagnostic pop
        }

        library = [device newLibraryWithSource:src options:options error:&error];
        if (error) {
            if (verbose) {
                GGML_LOG_ERROR("%s: error compiling source: %s\n", __func__, [[error description] UTF8String]);
            } else {
                GGML_LOG_ERROR("%s: error compiling source\n", __func__);
            }
            library = nil;
        }

        [options release];
    }

    [src release];

    if (!library) {
        if (verbose) {
            GGML_LOG_ERROR("%s: failed to create Metal library from source\n", __func__);
        }

        return NULL;
    }

    if (verbose) {
        GGML_LOG_INFO("%s: compiled in %.3f sec\n", __func__, (ggml_time_us() - t_start) / 1e6);
    }

    ggml_metal_library_t res = calloc(1, sizeof(struct ggml_metal_library));
    if (!res) {
        GGML_LOG_ERROR("%s: calloc failed\n", __func__);
        return NULL;
    }

    res->obj       = library;
    res->device    = device;
    res->pipelines = ggml_metal_pipelines_init();
    res->lock      = [NSLock new];

    return res;
}

void ggml_metal_library_free(ggml_metal_library_t lib) {
    if (!lib) {
        return;
    }

    if (lib->obj) {
        [lib->obj release];
    }

    ggml_metal_pipelines_free(lib->pipelines);

    [lib->lock release];

    free(lib);
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline(ggml_metal_library_t lib, const char * name) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;

    [lib->lock lock];

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);

    [lib->lock unlock];

    if (ggml_metal_profile_enabled()) {
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.pipeline_lookup_count++;
        g_ggml_metal_profile.pipeline_lookup_us += ggml_time_us() - t_start;
        if (res.pipeline) {
            g_ggml_metal_profile.pipeline_lookup_hit_count++;
        }
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }

    return res;
}

struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;

    struct ggml_metal_pipeline_with_params res = {
        /*.pipeline =*/ nil,
        /*.nsg      =*/ 0,
        /*.nr0      =*/ 0,
        /*.nr1      =*/ 0,
        /*.smem     =*/ 0,
        /*.c4       =*/ false,
        /*.cnt      =*/ false,
    };

    [lib->lock lock];

    res.pipeline = ggml_metal_pipelines_get(lib->pipelines, name);
    if (res.pipeline) {
        [lib->lock unlock];

        return res;
    }

    @autoreleasepool {
        NSError * error = nil;

        NSString * base_func = [NSString stringWithUTF8String:base];

        if (ggml_metal_log_pipelines_enabled()) {
            GGML_LOG_DEBUG("%s: compiling pipeline: base = '%s', name = '%s'\n", __func__, base, name);
        }

        id<MTLFunction> mtl_function;
        if (!cv) {
            mtl_function = [lib->obj newFunctionWithName:base_func];
        } else {
            mtl_function = [lib->obj newFunctionWithName:base_func constantValues:cv->obj error:&error];
        }
        if (!mtl_function) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to compile pipeline: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        id<MTLComputePipelineState> obj = [lib->device newComputePipelineStateWithFunction:mtl_function error:&error];

        [mtl_function release];

        if (!obj) {
            [lib->lock unlock];

            GGML_LOG_ERROR("%s: failed to create pipeline state: base = '%s', name = '%s'\n", __func__, base, name);
            if (error) {
                GGML_LOG_ERROR("%s: %s\n", __func__, [[error description] UTF8String]);
            }

            return res;
        }

        if (ggml_metal_log_pipelines_enabled()) {
            GGML_LOG_DEBUG("%s: loaded %-40s %16p | th_max = %4d | th_width = %4d\n", __func__, name,
                    (void *) obj,
                    (int)    obj.maxTotalThreadsPerThreadgroup,
                    (int)    obj.threadExecutionWidth);
        }

        if (obj.maxTotalThreadsPerThreadgroup == 0 || obj.threadExecutionWidth == 0) {
            [obj release];

            [lib->lock unlock];

            GGML_LOG_ERROR("%s: incompatible pipeline %s\n", __func__, name);

            return res;
        }

        res.pipeline = ggml_metal_pipeline_init();
        res.pipeline->obj = obj;
        snprintf(res.pipeline->name, sizeof(res.pipeline->name), "%s", name);

        ggml_metal_pipelines_add(lib->pipelines, name, res.pipeline);
    }

    [lib->lock unlock];

    if (ggml_metal_profile_enabled()) {
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.pipeline_compile_count++;
        g_ggml_metal_profile.pipeline_compile_us += ggml_time_us() - t_start;
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }

    return res;
}

//
// MTLComputeCommandEncoder wrapper
//

struct ggml_metal_encoder {
    id<MTLComputeCommandEncoder> obj;
    void * profile_cb;
    enum ggml_op current_op;
    int current_node_idx;
    char current_pipeline[GGML_METAL_PROFILE_MAX_NAME];
};

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent, void * profile_cb) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;

    ggml_metal_encoder_t res = calloc(1, sizeof(struct ggml_metal_encoder));

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    if (concurrent) {
        res->obj = [cmd_buf computeCommandEncoderWithDispatchType: MTLDispatchTypeConcurrent];
    } else {
        res->obj = [cmd_buf computeCommandEncoder];
    }

    [res->obj retain];
    res->profile_cb = profile_cb;
    res->current_op = GGML_OP_NONE;
    res->current_node_idx = -1;
    res->current_pipeline[0] = '\0';

    if (ggml_metal_profile_enabled()) {
        ggml_metal_profile_note_encoder_create(ggml_time_us() - t_start);
    }

    return res;
}

void ggml_metal_encoder_free(ggml_metal_encoder_t encoder) {
    [encoder->obj release];
    free(encoder);
}

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name) {
    [encoder->obj pushDebugGroup:[NSString stringWithCString:name encoding:NSUTF8StringEncoding]];
}

void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder) {
    [encoder->obj popDebugGroup];
}

void ggml_metal_encoder_set_current_op(ggml_metal_encoder_t encoder, enum ggml_op op, int node_idx) {
    encoder->current_op = op;
    encoder->current_node_idx = node_idx;
    encoder->current_pipeline[0] = '\0';
}

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;
    [encoder->obj setComputePipelineState:pipeline.pipeline->obj];
    snprintf(encoder->current_pipeline, sizeof(encoder->current_pipeline), "%s", pipeline.pipeline ? pipeline.pipeline->name : "(unknown)");

    if (ggml_metal_profile_enabled()) {
        const uint64_t us = ggml_time_us() - t_start;
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.set_pipeline_count++;
        g_ggml_metal_profile.set_pipeline_us += us;
        if (encoder->current_op >= 0 && encoder->current_op < GGML_OP_COUNT) {
            g_ggml_metal_profile.ops[encoder->current_op].set_pipeline_us += us;
        }
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }
}

void ggml_metal_encoder_set_bytes(ggml_metal_encoder_t encoder, void * data, size_t size, int idx) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;
    [encoder->obj setBytes:data length:size atIndex:idx];
    GGML_UNUSED(size);
    GGML_UNUSED(idx);

    if (ggml_metal_profile_enabled()) {
        const uint64_t us = ggml_time_us() - t_start;
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.set_bytes_count++;
        g_ggml_metal_profile.set_bytes_us += us;
        if (encoder->current_op >= 0 && encoder->current_op < GGML_OP_COUNT) {
            g_ggml_metal_profile.ops[encoder->current_op].set_bytes_us += us;
        }
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }
}

void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;
    [encoder->obj setBuffer:buffer.metal offset:buffer.offs atIndex:idx];
    GGML_UNUSED(idx);

    if (ggml_metal_profile_enabled()) {
        const uint64_t us = ggml_time_us() - t_start;
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.set_buffer_count++;
        g_ggml_metal_profile.set_buffer_us += us;
        if (encoder->current_op >= 0 && encoder->current_op < GGML_OP_COUNT) {
            g_ggml_metal_profile.ops[encoder->current_op].set_buffer_us += us;
        }
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }
}

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx) {
    [encoder->obj setThreadgroupMemoryLength:size atIndex:idx];
    GGML_UNUSED(encoder);
    GGML_UNUSED(size);
    GGML_UNUSED(idx);
}

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2) {
    const bool do_profile = ggml_metal_profile_enabled();
    const uint64_t t_start = do_profile ? ggml_time_us() : 0;

    struct ggml_metal_profile_cb * cb = (struct ggml_metal_profile_cb *) encoder->profile_cb;
    struct ggml_metal_profile_dispatch * rec = NULL;

    if (do_profile && cb && cb->dispatches && cb->dispatch_count < cb->dispatch_cap) {
        rec = &cb->dispatches[cb->dispatch_count];
        rec->op = encoder->current_op;
        snprintf(rec->pipeline, sizeof(rec->pipeline), "%s", encoder->current_pipeline[0] ? encoder->current_pipeline : "(unknown)");
        rec->tg0 = tg0;
        rec->tg1 = tg1;
        rec->tg2 = tg2;
        rec->tptg0 = tptg0;
        rec->tptg1 = tptg1;
        rec->tptg2 = tptg2;
        rec->sample0 = UINT32_MAX;
        rec->sample1 = UINT32_MAX;

        if (cb->sample_buf != nil && cb->sample_next + 2 <= cb->sample_buf.sampleCount) {
            rec->sample0 = cb->sample_next++;
            [encoder->obj sampleCountersInBuffer:cb->sample_buf atSampleIndex:rec->sample0 withBarrier:NO];
        }
    }

    [encoder->obj dispatchThreadgroups:MTLSizeMake(tg0, tg1, tg2) threadsPerThreadgroup:MTLSizeMake(tptg0, tptg1, tptg2)];

    if (do_profile) {
        if (cb && rec != NULL && cb->sample_buf != nil && rec->sample0 != UINT32_MAX && cb->sample_next + 1 <= cb->sample_buf.sampleCount) {
            rec->sample1 = cb->sample_next++;
            [encoder->obj sampleCountersInBuffer:cb->sample_buf atSampleIndex:rec->sample1 withBarrier:NO];
        }

        if (cb && cb->dispatch_count < cb->dispatch_cap) {
            cb->dispatch_count++;
        }

        const uint64_t us = ggml_time_us() - t_start;
        pthread_mutex_lock(&g_ggml_metal_profile_lock);
        g_ggml_metal_profile.dispatch_count++;
        g_ggml_metal_profile.dispatch_us += us;
        if (encoder->current_op >= 0 && encoder->current_op < GGML_OP_COUNT) {
            g_ggml_metal_profile.ops[encoder->current_op].dispatch_count++;
            g_ggml_metal_profile.ops[encoder->current_op].dispatch_us += us;
        }
        pthread_mutex_unlock(&g_ggml_metal_profile_lock);
    }
}

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;
    [encoder->obj memoryBarrierWithScope:MTLBarrierScopeBuffers];
    if (ggml_metal_profile_enabled()) {
        ggml_metal_profile_note_memory_barrier(ggml_time_us() - t_start);
    }
}

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder) {
    const uint64_t t_start = ggml_metal_profile_enabled() ? ggml_time_us() : 0;
    [encoder->obj endEncoding];
    if (ggml_metal_profile_enabled()) {
        ggml_metal_profile_note_encoder_end(ggml_time_us() - t_start);
    }
}

struct ggml_metal_device {
    id<MTLDevice> mtl_device;

    // a single global queue shared by all Metal backends
    // technically not needed for devices with unified memory, but enables discrete GPUs support
    // ref: https://github.com/ggml-org/llama.cpp/pull/15906
    id<MTLCommandQueue> mtl_queue;

    ggml_metal_rsets_t rsets;

    ggml_metal_library_t library;

    struct ggml_metal_device_props props;

    // virtual address for GPU memory allocations
    atomic_uintptr_t addr_virt;
};

//
// MTLResidenceSet wrapper
//

struct ggml_metal_rsets {
    NSLock * lock;

    NSMutableArray * data;

    // number of seconds since the last graph computation
    // keep the residency sets wired for that amount of time to avoid being collected by the OS
    int keep_alive_s;

    // background heartbeat thread to keep the residency sets alive
    atomic_bool d_stop;
    atomic_int  d_loop;

    dispatch_group_t d_group;
};

ggml_metal_rsets_t ggml_metal_rsets_init(void) {
    ggml_metal_rsets_t res = calloc(1, sizeof(struct ggml_metal_rsets));

    res->lock = [[NSLock alloc] init];
    res->data = [[NSMutableArray alloc] init];

    // by default keep the memory wired for 3 minutes
    res->keep_alive_s = 3*60;

    const char * GGML_METAL_RESIDENCY_KEEP_ALIVE_S = getenv("GGML_METAL_RESIDENCY_KEEP_ALIVE_S");
    if (GGML_METAL_RESIDENCY_KEEP_ALIVE_S) {
        res->keep_alive_s = atoi(GGML_METAL_RESIDENCY_KEEP_ALIVE_S);
    }

    if (res->keep_alive_s <= 0) {
        res->keep_alive_s = 3*60;
    }

    GGML_LOG_INFO("%s: creating a residency set collection (keep_alive = %d s)\n", __func__, res->keep_alive_s);

    atomic_store_explicit(&res->d_stop, false, memory_order_relaxed);
    atomic_store_explicit(&res->d_loop, 2*res->keep_alive_s, memory_order_relaxed);

    res->d_group = dispatch_group_create();

    // start a background thread that periodically requests residency for all the currently active sets in the collection
    // the requests stop after a certain amount of time (keep_alive_s) of inactivity
    dispatch_queue_t d_queue = dispatch_get_global_queue(QOS_CLASS_DEFAULT, 0);
    dispatch_group_async(res->d_group, d_queue, ^{
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
              while (!atomic_load_explicit(&res->d_stop, memory_order_relaxed)) {
                  if (atomic_load_explicit(&res->d_loop, memory_order_relaxed) > 0) {
                      [res->lock lock];

                      for (int i = 0; i < (int) res->data.count; ++i) {
                          [res->data[i] requestResidency];
                      }

                      atomic_fetch_sub_explicit(&res->d_loop, 1, memory_order_relaxed);

                      [res->lock unlock];
                  }

                  // half a second
                  usleep(500 * 1000);
              }
        }
#endif
    });

    return res;
}

void ggml_metal_rsets_free(ggml_metal_rsets_t rsets) {
    if (rsets == NULL) {
        return;
    }

    if ([rsets->data count] != 0) {
        GGML_LOG_WARN("%s: freeing %d residency sets that were still registered at shutdown\n",
                __func__, (int) [rsets->data count]);
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
        if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
            for (int i = 0; i < (int) [rsets->data count]; ++i) {
                id<MTLResidencySet> rset = rsets->data[i];
                [rset endResidency];
                [rset removeAllAllocations];
                [rset release];
            }
        }
#endif
        [rsets->data removeAllObjects];
    }

    atomic_store_explicit(&rsets->d_stop, true, memory_order_relaxed);

    dispatch_group_wait(rsets->d_group, DISPATCH_TIME_FOREVER);
    dispatch_release(rsets->d_group);

    [rsets->data release];
    [rsets->lock release];

    free(rsets);
}

ggml_metal_device_t ggml_metal_device_init(int device) {
    ggml_metal_device_t dev = calloc(1, sizeof(struct ggml_metal_device));

    assert(dev != NULL);

    if (dev->mtl_device == nil) {
        dev->mtl_device = MTLCreateSystemDefaultDevice();

        if (dev->mtl_device) {
            dev->mtl_queue = [dev->mtl_device newCommandQueue];
            if (dev->mtl_queue == nil) {
                GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            }

            dev->addr_virt = 0x000000400ULL;

            dev->props.device = device;
            dev->props.has_simdgroup_reduction  = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_simdgroup_reduction |= [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];

            dev->props.has_simdgroup_mm = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];
            dev->props.has_unified_memory = dev->mtl_device.hasUnifiedMemory;

            dev->props.has_bfloat  = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal3_GGML];
            dev->props.has_bfloat |= [dev->mtl_device supportsFamily:MTLGPUFamilyApple6];
            if (getenv("GGML_METAL_BF16_DISABLE") != NULL) {
                dev->props.has_bfloat = false;
            }

            dev->props.has_tensor = [dev->mtl_device supportsFamily:MTLGPUFamilyMetal4_GGML];
            if (getenv("GGML_METAL_TENSOR_DISABLE") != NULL) {
                dev->props.has_tensor = false;
            }

            // note: disable the tensor API by default for old chips because with the current implementation it is not useful
            // - M2 Ultra:   ~5% slower
            // - M4, M4 Max: no significant difference
            //
            // TODO: try to update the tensor API kernels to at least match the simdgroup performance
            if (getenv("GGML_METAL_TENSOR_ENABLE") == NULL &&
                ![[dev->mtl_device name] containsString:@"M5"] &&
                ![[dev->mtl_device name] containsString:@"M6"] &&
                ![[dev->mtl_device name] containsString:@"A19"] &&
                ![[dev->mtl_device name] containsString:@"A20"]) {
                GGML_LOG_WARN("%s: tensor API disabled for pre-M5 and pre-A19 devices\n", __func__);
                dev->props.has_tensor = false;
            }

            // double-check that the tensor API compiles
            if (dev->props.has_tensor) {
                const char * src_tensor_f16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device  half, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device  half, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for f16 support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_f16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                    dev->props.has_tensor = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API is not supported in this environment - disabling\n", __func__);
                        dev->props.has_tensor = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            // try to compile a dummy kernel to determine if the tensor API is supported for bfloat
            if (dev->props.has_tensor && dev->props.has_bfloat) {
                const char * src_tensor_bf16 = "\n"
                    "#include <metal_stdlib> \n"
                    "#include <metal_tensor> \n"
                    "#include <MetalPerformancePrimitives/MetalPerformancePrimitives.h> \n"
                    " \n"
                    "using namespace metal; \n"
                    "using namespace mpp::tensor_ops; \n"
                    " \n"
                    "kernel void dummy_kernel( \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> A [[buffer(0)]], \n"
                    "    tensor<device bfloat, dextents<int32_t, 2>> B [[buffer(1)]], \n"
                    "    device float * C [[buffer(2)]], \n"
                    "    uint2 tgid [[threadgroup_position_in_grid]]) \n"
                    "{ \n"
                    "    auto tA = A.slice(0, (int)tgid.y); \n"
                    "    auto tB = B.slice((int)tgid.x, 0); \n"
                    " \n"
                    "    matmul2d< \n"
                    "        matmul2d_descriptor(16, 16, dynamic_extent), \n"
                    "        execution_simdgroups<4>> mm; \n"
                    " \n"
                    "    auto cT = mm.get_destination_cooperative_tensor<decltype(tA), decltype(tB), float>(); \n"
                    " \n"
                    "    auto sA = tA.slice(0, 0); \n"
                    "    auto sB = tB.slice(0, 0); \n"
                    "    mm.run(sB, sA, cT); \n"
                    " \n"
                    "    auto tC = tensor<device float, dextents<int32_t, 2>, tensor_inline>(C, dextents<int32_t, 2>(4, 4)); \n"
                    " \n"
                    "    cT.store(tC); \n"
                    "}";

                GGML_LOG_INFO("%s: testing tensor API for bfloat support\n", __func__);
                ggml_metal_library_t lib = ggml_metal_library_init_from_source(dev, src_tensor_bf16, false);
                if (lib == NULL) {
                    GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                    dev->props.has_bfloat = false;
                } else {
                    struct ggml_metal_pipeline_with_params ppl = ggml_metal_library_compile_pipeline(lib, "dummy_kernel", "dummy_kernel", nil);
                    if (!ppl.pipeline) {
                        GGML_LOG_WARN("%s: - the tensor API does not support bfloat - disabling bfloat support\n", __func__);
                        dev->props.has_bfloat = false;
                    }

                    ggml_metal_library_free(lib);
                }
            }

            dev->props.use_residency_sets = true;
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
            dev->props.use_residency_sets = getenv("GGML_METAL_NO_RESIDENCY") == nil;
#endif

            dev->props.use_shared_buffers = dev->props.has_unified_memory;
#if TARGET_OS_OSX
            // In case of eGPU, shared memory may be preferable.
            dev->props.use_shared_buffers |= [dev->mtl_device location] == MTLDeviceLocationExternal;
#endif
            if (getenv("GGML_METAL_SHARED_BUFFERS_DISABLE") != NULL) {
                dev->props.use_shared_buffers = false;
            }
            if (getenv("GGML_METAL_SHARED_BUFFERS_ENABLE") != NULL) {
                dev->props.use_shared_buffers = true;
            }

            dev->props.supports_gpu_family_apple7 = [dev->mtl_device supportsFamily:MTLGPUFamilyApple7];

            dev->props.op_offload_min_batch_size  = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            dev->props.max_buffer_size            = dev->mtl_device.maxBufferLength;
            dev->props.max_theadgroup_memory_size = dev->mtl_device.maxThreadgroupMemoryLength;
            if (@available(macOS 10.12, iOS 16.0, *)) {
                dev->props.max_working_set_size   = dev->mtl_device.recommendedMaxWorkingSetSize;
            } else {
                dev->props.max_working_set_size   = dev->mtl_device.maxBufferLength;
            }

            snprintf(dev->props.name, sizeof(dev->props.name), "%s%d", "MTL", device);
            snprintf(dev->props.desc, sizeof(dev->props.desc), "%s", [[dev->mtl_device name] UTF8String]);

            dev->library = ggml_metal_library_init(dev);
            if (!dev->library) {
                GGML_LOG_ERROR("%s: error: failed to create library\n", __func__);
            }

            if (dev->props.use_residency_sets) {
                dev->rsets = ggml_metal_rsets_init();
            } else {
                dev->rsets = nil;
            }

            // print MTL GPU family:
            GGML_LOG_INFO("%s: GPU name:   %s\n", __func__, dev->props.name);

            // determine max supported GPU family
            // https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
            // https://developer.apple.com/metal/Metal-Feature-Set-Tables.pdf
            {
                for (int i = MTLGPUFamilyApple1 + 20; i >= MTLGPUFamilyApple1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyApple%d  (%d)\n", __func__, i - (int) MTLGPUFamilyApple1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyCommon1 + 5; i >= MTLGPUFamilyCommon1; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyCommon%d (%d)\n", __func__, i - (int) MTLGPUFamilyCommon1 + 1, i);
                        break;
                    }
                }

                for (int i = MTLGPUFamilyMetal3_GGML + 5; i >= MTLGPUFamilyMetal3_GGML; --i) {
                    if ([dev->mtl_device supportsFamily:i]) {
                        GGML_LOG_INFO("%s: GPU family: MTLGPUFamilyMetal%d  (%d)\n", __func__, i - (int) MTLGPUFamilyMetal3_GGML + 3, i);
                        break;
                    }
                }
            }

            GGML_LOG_INFO("%s: simdgroup reduction   = %s\n", __func__, dev->props.has_simdgroup_reduction ? "true" : "false");
            GGML_LOG_INFO("%s: simdgroup matrix mul. = %s\n", __func__, dev->props.has_simdgroup_mm        ? "true" : "false");
            GGML_LOG_INFO("%s: has unified memory    = %s\n", __func__, dev->props.has_unified_memory      ? "true" : "false");
            GGML_LOG_INFO("%s: has bfloat            = %s\n", __func__, dev->props.has_bfloat              ? "true" : "false");
            GGML_LOG_INFO("%s: has tensor            = %s\n", __func__, dev->props.has_tensor              ? "true" : "false");
            GGML_LOG_INFO("%s: use residency sets    = %s\n", __func__, dev->props.use_residency_sets      ? "true" : "false");
            GGML_LOG_INFO("%s: use shared buffers    = %s\n", __func__, dev->props.use_shared_buffers      ? "true" : "false");

#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
            if (@available(macOS 10.12, iOS 16.0, *)) {
                GGML_LOG_INFO("%s: recommendedMaxWorkingSetSize  = %8.2f MB\n", __func__, dev->props.max_working_set_size / 1e6);
            }
#endif
        }
    }

    return dev;
}

void ggml_metal_device_free(ggml_metal_device_t dev) {
    assert(dev != NULL);

    ggml_metal_rsets_free(dev->rsets);

    ggml_metal_library_free(dev->library);
    dev->library = NULL;

    if (dev->mtl_queue) {
        [dev->mtl_queue release];
        dev->mtl_queue = nil;
    }

    if (dev->mtl_device) {
        [dev->mtl_device release];
        dev->mtl_device = nil;
    }

    free(dev);
}

void * ggml_metal_device_get_obj(ggml_metal_device_t dev) {
    return dev->mtl_device;
}

void * ggml_metal_device_get_queue(ggml_metal_device_t dev) {
    return dev->mtl_queue;
}

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev) {
    return dev->library;
}

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data addObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_rm(ggml_metal_device_t dev, ggml_metal_rset_t rset) {
    if (rset == nil) {
        return;
    }

    GGML_ASSERT(dev->rsets);

    [dev->rsets->lock lock];

    [dev->rsets->data removeObject:rset];

    [dev->rsets->lock unlock];
}

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev) {
    if (dev->rsets == NULL) {
        return;
    }

    atomic_store_explicit(&dev->rsets->d_loop, 2*dev->rsets->keep_alive_s, memory_order_relaxed);
}

struct ggml_metal_event {
    void * obj; // id<MTLEvent>

    atomic_int value;
};

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeSignalEvent:event value:atomic_fetch_add_explicit(&ev->value, 1, memory_order_relaxed) + 1];
}

void ggml_metal_event_encode_wait(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf_raw) {
    id<MTLEvent> event = (id<MTLEvent>)ev->obj;

    id<MTLCommandBuffer> cmd_buf = (id<MTLCommandBuffer>) cmd_buf_raw;

    [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
}

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev) {
    id<MTLEvent> event = [dev->mtl_device newEvent];

    ggml_metal_event_t ev = calloc(1, sizeof(struct ggml_metal_event));

    ev->obj = (__bridge void *)event;
    ev->value = 0;

    return ev;
}

void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    id<MTLEvent> event = ev->obj;
    [event release];

    free(ev);

    GGML_UNUSED(dev);
}

void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLEvent> event = ev->obj;

        id<MTLCommandBuffer> cmd_buf = [dev->mtl_queue commandBuffer];
        [cmd_buf encodeWaitForEvent:event value:atomic_load_explicit(&ev->value, memory_order_relaxed)];
        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total) {
    if (@available(macOS 10.12, iOS 16.0, *)) {
        *total = dev->mtl_device.recommendedMaxWorkingSetSize;
        *free  = *total - dev->mtl_device.currentAllocatedSize;
    } else {
        *free = 0;
        *total = 0;
    }
}

static bool ggml_metal_device_supports_flash_attn_ext_shape(const struct ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t dq = op->src[0]->ne[0];
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    if (dq != dk) {
        return false;
    }

    switch (dk) {
        case 16:  return dv == 16;
        case 32:  return dv == 32;
        case 40:  return dv == 40;
        case 48:  return dv == 48;
        case 56:  return dv == 56;
        case 64:  return dv == 64;
        case 72:  return dv == 72;
        case 80:  return dv == 80;
        case 96:  return dv == 96;
        case 112: return dv == 112;
        case 128: return dv == 128;
        case 192: return dv == 192 || dv == 128;
        case 256: return dv == 256;
        case 320: return dv == 256;
        case 576: return dv == 512;
        default:  return false;
    }
}

bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op) {
    const bool has_simdgroup_mm        = dev->props.has_simdgroup_mm;
    const bool has_simdgroup_reduction = dev->props.has_simdgroup_reduction;
    const bool has_bfloat              = dev->props.has_bfloat;

    if (!has_bfloat) {
        if (op->type == GGML_TYPE_BF16) {
            return false;
        }

        for (size_t i = 0, n = 3; i < n; ++i) {
            if (op->src[i] != NULL && op->src[i]->type == GGML_TYPE_BF16) {
                return false;
            }
        }
    }

    switch (op->op) {
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
            return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    return ggml_is_contiguous_rows(op->src[0]) && (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16);
                default:
                    return false;
            }
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                    return ggml_is_contiguous_1(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
               default:
                    return false;
            }
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
        case GGML_OP_CONCAT:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
        case GGML_OP_ADD_ID:
        case GGML_OP_ACC:
            return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_REPEAT:
        case GGML_OP_CONV_TRANSPOSE_1D:
            return true;
        case GGML_OP_CONV_TRANSPOSE_2D:
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) &&
                (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_CONV_3D:
            return ggml_is_contiguous(op->src[0]) &&
                   ggml_is_contiguous(op->src[1]) &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32) &&
                   op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_SUM:
            return has_simdgroup_reduction && ggml_is_contiguous(op->src[0]);
        case GGML_OP_TRI:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_SUM_ROWS:
        case GGML_OP_CUMSUM:
        case GGML_OP_MEAN:
        case GGML_OP_SOFT_MAX:
        case GGML_OP_GROUP_NORM:
        case GGML_OP_L2_NORM:
            return has_simdgroup_reduction && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_COUNT_EQUAL:
            return has_simdgroup_reduction &&
                op->src[0]->type == GGML_TYPE_I32 &&
                op->src[1]->type == GGML_TYPE_I32 &&
                op->type == GGML_TYPE_I64;
        case GGML_OP_ARGMAX:
            return has_simdgroup_reduction;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            return has_simdgroup_reduction && (ggml_is_contiguous_rows(op->src[0]));
        case GGML_OP_ROPE:
            return true;
        case GGML_OP_IM2COL:
            return ggml_is_contiguous(op->src[1]) && op->src[1]->type == GGML_TYPE_F32 && (op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
        case GGML_OP_CONV_2D:
            return ggml_is_contiguous(op->src[0]) &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32 &&
                   (op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
        case GGML_OP_UPSCALE:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_WIN_PART:
        case GGML_OP_WIN_UNPART:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32 &&
                   op->src[1]->type == GGML_TYPE_F32 &&
                   op->type == GGML_TYPE_F32;
        case GGML_OP_POOL_1D:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_POOL_2D:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_PAD:
            // TODO: add circular padding support for metal, see https://github.com/ggml-org/llama.cpp/pull/16985
            if (ggml_get_op_params_i32(op, 8) != 0) {
                return false;
            }

            return (ggml_get_op_params_i32(op, 0) == 0) && (ggml_get_op_params_i32(op, 2) == 0) &&
                   (ggml_get_op_params_i32(op, 4) == 0) && (ggml_get_op_params_i32(op, 6) == 0);
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_ARGSORT:
        case GGML_OP_TOP_K:
        case GGML_OP_ARANGE:
            return true;
        case GGML_OP_FLASH_ATTN_EXT:
            if (op->src[1]->type != op->src[2]->type) {
                return false;
            }
            // for new head sizes, add checks here
            if (op->src[0]->ne[0] != 16 &&
                op->src[0]->ne[0] != 32 &&
                op->src[0]->ne[0] != 40 &&
                op->src[0]->ne[0] != 48 &&
                op->src[0]->ne[0] != 56 &&
                op->src[0]->ne[0] != 64 &&
                op->src[0]->ne[0] != 72 &&
                op->src[0]->ne[0] != 80 &&
                op->src[0]->ne[0] != 96 &&
                op->src[0]->ne[0] != 112 &&
                op->src[0]->ne[0] != 128 &&
                op->src[0]->ne[0] != 192 &&
                op->src[0]->ne[0] != 256 &&
                op->src[0]->ne[0] != 320 &&
                op->src[0]->ne[0] != 512 &&
                op->src[0]->ne[0] != 576) {
                return false;
            }
            if (!ggml_metal_device_supports_flash_attn_ext_shape(op)) {
                return false;
            }
            return has_simdgroup_mm; // TODO: over-restricted for vec-kernels
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
            return has_simdgroup_reduction;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            return has_simdgroup_reduction && op->src[2]->ne[0] % 32 == 0;
        case GGML_OP_SOLVE_TRI:
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            return has_simdgroup_reduction && op->src[0]->type != GGML_TYPE_NVFP4;
        case GGML_OP_SET:
        case GGML_OP_CPY:
        case GGML_OP_DUP:
        case GGML_OP_CONT:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F32:
                        switch (op->type) {
                           case GGML_TYPE_F32:
                           case GGML_TYPE_F16:
                           case GGML_TYPE_BF16:
                           case GGML_TYPE_Q8_0:
                           case GGML_TYPE_Q4_0:
                           case GGML_TYPE_Q4_1:
                           case GGML_TYPE_Q5_0:
                           case GGML_TYPE_Q5_1:
                           case GGML_TYPE_IQ4_NL:
                           case GGML_TYPE_I32:
                                return true;
                           default:
                                return false;
                        }
                    case GGML_TYPE_F16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_BF16:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_BF16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                        switch (op->type) {
                            case GGML_TYPE_F32:
                            case GGML_TYPE_F16:
                                return true;
                            default:
                                return false;
                        }
                    case GGML_TYPE_I32:
                        return op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_I32;
                    default:
                        return false;
                };
            }
        case GGML_OP_GET_ROWS:
            return op->src[0]->type != GGML_TYPE_NVFP4;
        case GGML_OP_SET_ROWS:
            {
                if (op->src[0]->type != GGML_TYPE_F32) {
                    return false;
                }

                switch (op->type) {
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_IQ4_NL:
                        return true;
                    default:
                        return false;
                };
            }
        case GGML_OP_DIAG:
            return true;
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
            return has_simdgroup_reduction;
        default:
            return false;
    }
}

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev) {
    return &dev->props;
}

//
// device buffers
//

// max memory buffers that can be mapped to the device
#define GGML_METAL_MAX_BUFFERS 64

struct ggml_metal_buffer_wrapper {
    void   * data;
    size_t   size;

    id<MTLBuffer> metal;
};

struct ggml_metal_buffer {
    void * all_data;
    size_t all_size;

    // if false, the Metal buffer data is allocated in private GPU memory and is not shared with the host
    bool is_shared;
    bool owned;

    // multiple buffers are used only to avoid the maximum buffer size limitation when using mmap
    int n_buffers;
    struct ggml_metal_buffer_wrapper buffers[GGML_METAL_MAX_BUFFERS];

    bool use_residency_sets;

    // optional MTLResidencySet
    // note: cannot use explicitly "id<MTLResidencySet>" here because it is not available on certain OSes
    id rset;

    // pointers to global device
    ggml_metal_device_t dev;
};

static void ggml_metal_log_allocated_size(id<MTLDevice> device, size_t size_aligned) {
#ifndef GGML_METAL_NDEBUG
#if TARGET_OS_OSX || (TARGET_OS_IOS && __clang_major__ >= 15)
    if (@available(macOS 10.12, iOS 16.0, *)) {
        GGML_LOG_DEBUG("%s: allocated buffer, size = %8.2f MiB, (%8.2f / %8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0,
                device.recommendedMaxWorkingSetSize / 1024.0 / 1024.0);

        if (device.currentAllocatedSize > device.recommendedMaxWorkingSetSize) {
            GGML_LOG_WARN("%s: warning: current allocated size is greater than the recommended max working set size\n", __func__);
        }
    } else {
        GGML_LOG_INFO("%s: allocated buffer, size = %8.2f MiB, (%8.2f)\n",
                __func__,
                size_aligned / 1024.0 / 1024.0,
                device.currentAllocatedSize / 1024.0 / 1024.0);
    }
#endif
#endif
    GGML_UNUSED(device);
    GGML_UNUSED(size_aligned);
}

// rset init
static bool ggml_metal_buffer_rset_init(ggml_metal_buffer_t buf) {
    buf->rset = nil;

    if (!buf->use_residency_sets) {
        return true;
    }

#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        MTLResidencySetDescriptor * desc = [[MTLResidencySetDescriptor alloc] init];
        desc.label = @"ggml_metal";
        desc.initialCapacity = buf->n_buffers;

        NSError * error;
        buf->rset = [buf->dev->mtl_device newResidencySetWithDescriptor:desc error:&error];
        if (error) {
            GGML_LOG_ERROR("%s: error: %s\n", __func__, [[error description] UTF8String]);
            [desc release];
            return false;
        }

        [desc release];

        for (int i = 0; i < buf->n_buffers; i++) {
            [buf->rset addAllocation:buf->buffers[i].metal];
        }

        [buf->rset commit];
        [buf->rset requestResidency];

        return true;
    }
#endif

    return true;
}

// rset free
static void ggml_metal_buffer_rset_free(ggml_metal_buffer_t buf) {
#if defined(GGML_METAL_HAS_RESIDENCY_SETS)
    if (@available(macOS 15.0, iOS 18.0, tvOS 18.0, visionOS 2.0, *)) {
        if (buf->rset) {
            [buf->rset endResidency];
            [buf->rset removeAllAllocations];
            [buf->rset release];
        }
    }
#else
    GGML_UNUSED(buf);
#endif
}

static void * ggml_metal_host_malloc(size_t n) {
    void * data = NULL;

#if TARGET_OS_OSX
    kern_return_t err = vm_allocate((vm_map_t) mach_task_self(), (void *) &data, n, VM_FLAGS_ANYWHERE);
    if (err != KERN_SUCCESS) {
        GGML_LOG_ERROR("%s: error: vm_allocate failed\n", __func__);
        return NULL;
    }
#else
    const int result = posix_memalign((void **) &data, sysconf(_SC_PAGESIZE), n);
    if (result != 0) {
        GGML_LOG_ERROR("%s: error: posix_memalign failed\n", __func__);
        return NULL;
    }
#endif

    return data;
}

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    shared = shared && props_dev->use_shared_buffers;

    // allocate shared buffer if the device supports it and it is required by the buffer type
    if (shared) {
        res->all_data = ggml_metal_host_malloc(size_aligned);
        res->is_shared = true;
    } else {
        // use virtual address
        res->all_data = (void *) atomic_fetch_add_explicit(&dev->addr_virt, size_aligned, memory_order_relaxed);
        res->is_shared = false;
    }
    res->all_size = size_aligned;

    res->owned = true;

    res->n_buffers = 1;

    if (res->all_data != NULL) {
        res->buffers[0].size  = size;
        res->buffers[0].metal = nil;

        if (size_aligned > 0) {
            if (props_dev->use_shared_buffers && shared) {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:res->all_data
                                                                  length:size_aligned
                                                                 options:MTLResourceStorageModeShared
                                                             deallocator:nil];
            } else {
                res->buffers[0].metal = [res->dev->mtl_device newBufferWithLength:size_aligned options:MTLResourceStorageModePrivate];
            }
        }

        res->buffers[0].data = res->all_data;
    }

    if (size_aligned > 0 && (res->all_data == NULL || res->buffers[0].metal == nil)) {
        GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
        free(res);
        return NULL;
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    //ggml_metal_log_allocated_size(device, size_aligned);

    return res;
}

ggml_metal_buffer_t ggml_metal_buffer_map(ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_buffer_t res = calloc(1, sizeof(struct ggml_metal_buffer));

    res->dev = dev;

    res->all_data = ptr;
    res->all_size = size;

    res->is_shared = true;
    res->owned = false;

    res->n_buffers = 0;

    const size_t size_page = sysconf(_SC_PAGESIZE);

    // page-align the data ptr
    {
        const uintptr_t offs = (uintptr_t) ptr % size_page;
        ptr  = (void *) ((char *) ptr - offs);
        size += offs;
    }

    size_t size_aligned = size;
    if ((size_aligned % size_page) != 0) {
        size_aligned += (size_page - (size_aligned % size_page));
    }

    const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

    // the buffer fits into the max buffer size allowed by the device
    if (size_aligned <= props_dev->max_buffer_size) {
        res->buffers[res->n_buffers].data  = ptr;
        res->buffers[res->n_buffers].size  = size;
        res->buffers[res->n_buffers].metal = nil;

        if (size_aligned > 0) {
            res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:ptr length:size_aligned options:MTLResourceStorageModeShared deallocator:nil];

            if (res->buffers[res->n_buffers].metal == nil) {
                GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_aligned / 1024.0 / 1024.0);
                free(res);
                return NULL;
            }
        }

        ggml_metal_log_allocated_size(res->dev->mtl_device, size_aligned);

        ++res->n_buffers;
    } else {
        // this overlap between the views will guarantee that the tensor with the maximum size will fully fit into
        // one of the views
        const size_t size_ovlp = ((max_tensor_size + size_page - 1) / size_page + 1) * size_page; // round-up 2 pages just in case
        const size_t size_step = props_dev->max_buffer_size - size_ovlp;
        const size_t size_view = props_dev->max_buffer_size;

        for (size_t i = 0; i < size; i += size_step) {
            const size_t size_step_aligned = (i + size_view <= size) ? size_view : (size_aligned - i);

            res->buffers[res->n_buffers].data  = (void *) ((uint8_t *) ptr + i);
            res->buffers[res->n_buffers].size  = size_step_aligned;
            res->buffers[res->n_buffers].metal = nil;

            if (size_step_aligned > 0) {
                res->buffers[res->n_buffers].metal = [res->dev->mtl_device newBufferWithBytesNoCopy:(void *) ((uint8_t *) ptr + i) length:size_step_aligned options:MTLResourceStorageModeShared deallocator:nil];

                if (res->buffers[res->n_buffers].metal == nil) {
                    GGML_LOG_ERROR("%s: error: failed to allocate buffer, size = %8.2f MiB\n", __func__, size_step_aligned / 1024.0 / 1024.0);
                    free(res);
                    return NULL;
                }
            }

            ggml_metal_log_allocated_size(res->dev->mtl_device, size_step_aligned);

            if (i + size_step < size) {
                GGML_LOG_INFO("\n");
            }

            ++res->n_buffers;
        }
    }

    res->use_residency_sets = props_dev->use_residency_sets;

    if (!ggml_metal_buffer_rset_init(res)) {
        GGML_LOG_ERROR("%s: error: failed to initialize residency set\n", __func__);
        free(res);
        return NULL;
    }

    ggml_metal_device_rsets_add(dev, res->rset);

    return res;
}

void ggml_metal_buffer_free(ggml_metal_buffer_t buf) {
    ggml_metal_device_rsets_rm(buf->dev, buf->rset);

    for (int i = 0; i < buf->n_buffers; i++) {
        [buf->buffers[i].metal release];
    }

    ggml_metal_buffer_rset_free(buf);

    if (buf->is_shared && buf->owned) {
#if TARGET_OS_OSX
        vm_deallocate((vm_map_t)mach_task_self(), (vm_address_t)buf->all_data, buf->all_size);
#else
        free(buf->all_data);
#endif
    }

    free(buf);
}

void * ggml_metal_buffer_get_base(ggml_metal_buffer_t buf) {
    return buf->all_data;
}

bool ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf) {
    return buf->is_shared;
}

void ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    if (buf->is_shared) {
        memset((char *) tensor->data + offset, value, size);
        return;
    }

    @autoreleasepool {
        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:bid_dst.metal
                          range:NSMakeRange(bid_dst.offs, bid_dst.offs + size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_set_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy((char *) tensor->data + offset, data, size);
        return;
    }

    @autoreleasepool {
        // src
        void * data_ptr = (void *)(uintptr_t) data; // "const cast" the src data
        id<MTLBuffer> buf_src = [buf->dev->mtl_device newBufferWithBytesNoCopy:data_ptr
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_src);

        // dst
        struct ggml_metal_buffer_id bid_dst = ggml_metal_buffer_get_id(buf, tensor);
        bid_dst.offs += offset;

        // note: for experimentation purposes, here we use a semaphore to wait for the copy to complete
        //       this is alternative to waitUntilCompleted, which should be faster, but don't seem to make much difference
        dispatch_semaphore_t completion_semaphore = dispatch_semaphore_create(0);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:buf_src
                       sourceOffset:0
                           toBuffer:bid_dst.metal
                  destinationOffset:bid_dst.offs
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
                             // TODO: can check for errors here
            GGML_UNUSED(cb);

            dispatch_semaphore_signal(completion_semaphore);
        }];

        [cmd_buf commit];

        dispatch_semaphore_wait(completion_semaphore, DISPATCH_TIME_FOREVER);
        dispatch_release(completion_semaphore);

        //[cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_get_tensor(ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (buf->is_shared) {
        memcpy(data, (const char *) tensor->data + offset, size);
        return;
    }

    @autoreleasepool {
        // src
        struct ggml_metal_buffer_id bid_src = ggml_metal_buffer_get_id(buf, tensor);
        bid_src.offs += offset;

        // dst
        id<MTLBuffer> buf_dst = [buf->dev->mtl_device newBufferWithBytesNoCopy:data
                                                               length:size
                                                              options:MTLResourceStorageModeShared
                                                          deallocator:nil];

        GGML_ASSERT(buf_dst);

        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

void ggml_metal_buffer_clear(ggml_metal_buffer_t buf, uint8_t value) {
    if (buf->is_shared) {
        memset(buf->all_data, value, buf->all_size);
        return;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = [buf->dev->mtl_queue commandBufferWithUnretainedReferences];

        {
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder fillBuffer:buf->buffers[0].metal
                          range:NSMakeRange(0, buf->buffers[0].size)
                          value:value];

            [encoder endEncoding];
        }

        [cmd_buf commit];
        [cmd_buf waitUntilCompleted];
    }
}

struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t) {
    struct ggml_metal_buffer_id res = { nil, 0 };

    const int64_t tsize = ggml_nbytes(t);

    // find the view that contains the tensor fully
    for (int i = 0; i < buf->n_buffers; ++i) {
        const int64_t ioffs = (int64_t) t->data - (int64_t) buf->buffers[i].data;

        //GGML_LOG_INFO("ioffs = %10ld, tsize = %10ld, sum = %10ld, buf->buffers[%d].size = %10ld\n", ioffs, tsize, ioffs + tsize, i, buf->buffers[i].size);
        if (ioffs >= 0 && ioffs + tsize <= (int64_t) buf->buffers[i].size) {
            res.metal = buf->buffers[i].metal;
            res.offs  = (size_t) ioffs;

            //GGML_LOG_INFO("%s: tensor '%16s', offs = %8ld\n", __func__, t->name, *offs);

            return res;
        }
    }

    GGML_LOG_ERROR("%s: error: tensor '%s' buffer is nil\n", __func__, t->name);

    return res;
}
