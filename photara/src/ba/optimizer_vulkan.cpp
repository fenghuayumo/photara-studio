#include "ba/optimizer.hpp"

#include "ba/linearizer.hpp"
#include "core/logging.hpp"
#include "photara_vk/command.hpp"
#include "photara_vk/device.hpp"

#include "ba_alpha.hlsl.embedded.hpp"
#include "ba_assemble_cameras.hlsl.embedded.hpp"
#include "ba_assemble_points.hlsl.embedded.hpp"
#include "ba_beta.hlsl.embedded.hpp"
#include "ba_camera_gather.hlsl.embedded.hpp"
#include "ba_camera_reduce.hlsl.embedded.hpp"
#include "ba_copy_intrinsic_rhs.hlsl.embedded.hpp"
#include "ba_damp_cameras.hlsl.embedded.hpp"
#include "ba_direction.hlsl.embedded.hpp"
#include "ba_dot.hlsl.embedded.hpp"
#include "ba_factor.hlsl.embedded.hpp"
#include "ba_finalize_intrinsics.hlsl.embedded.hpp"
#include "ba_intrinsic_reduce.hlsl.embedded.hpp"
#include "ba_intrinsic_schur.hlsl.embedded.hpp"
#include "ba_invert_points.hlsl.embedded.hpp"
#include "ba_linearize.hlsl.embedded.hpp"
#include "ba_linearized_cost.hlsl.embedded.hpp"
#include "ba_mask_intrinsics.hlsl.embedded.hpp"
#include "ba_point_gather.hlsl.embedded.hpp"
#include "ba_point_reduce.hlsl.embedded.hpp"
#include "ba_precondition.hlsl.embedded.hpp"
#include "ba_precondition_intrinsics.hlsl.embedded.hpp"
#include "ba_prior_cost.hlsl.embedded.hpp"
#include "ba_recover_points.hlsl.embedded.hpp"
#include "ba_reduced_rhs.hlsl.embedded.hpp"
#include "ba_schur_rhs.hlsl.embedded.hpp"
#include "ba_update_intrinsics.hlsl.embedded.hpp"
#include "ba_update_pcg.hlsl.embedded.hpp"
#include "ba_update_points.hlsl.embedded.hpp"
#include "ba_update_poses.hlsl.embedded.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>
#include <vector>

namespace photara::ba {
namespace {

static_assert(sizeof(Pose) == 56, "Vulkan BA pose record must be 56 bytes");
static_assert(sizeof(Point3) == 24, "Vulkan BA point record must be 24 bytes");
static_assert(sizeof(PinholeIntrinsics) == 72, "Vulkan BA intrinsics record must be 72 bytes");
static_assert(offsetof(PinholeIntrinsics, model) == 64, "camera model must follow eight doubles");
static_assert(sizeof(LinearizedObservation) == 304, "linearized observation must be 304 bytes");
static_assert(offsetof(LinearizedObservation, robust_weight) == 288, "robust weight offset");
static_assert(offsetof(LinearizedObservation, valid) == 296, "valid flag offset");

// Same 128-byte push block every BA shader reads.
struct Push {
    std::uint32_t n = 0;
    std::uint32_t p0 = 0;
    std::uint32_t p1 = 0;
    std::uint32_t p2 = 0;
    std::uint32_t groups = 1;
    std::uint32_t flags = 0;
    std::uint32_t huber_lo = 0;
    std::uint32_t huber_hi = 0;
    std::uint32_t depth_lo = 0;
    std::uint32_t depth_hi = 0;
    std::uint32_t damp_lo = 0;
    std::uint32_t damp_hi = 0;
    std::uint32_t prior_lo = 0;
    std::uint32_t prior_hi = 0;
    std::uint32_t min_f_lo = 0;
    std::uint32_t min_f_hi = 0;
    std::uint32_t max_f_lo = 0;
    std::uint32_t max_f_hi = 0;
    std::uint32_t obs_lo = 0;
    std::uint32_t obs_hi = 0;
    std::uint32_t dof = 0;
    std::uint32_t camera_values = 0;
    std::uint32_t pad0 = 0;
    std::uint32_t pad1 = 0;
    std::uint32_t pad2 = 0;
    std::uint32_t pad3 = 0;
    std::uint32_t pad4 = 0;
    std::uint32_t pad5 = 0;
    std::uint32_t pad6 = 0;
    std::uint32_t pad7 = 0;
    std::uint32_t pad8 = 0;
    std::uint32_t pad9 = 0;
};
static_assert(sizeof(Push) == 128, "BA push constants must be 128 bytes");

constexpr std::uint32_t kFixPose = 1u;
constexpr std::uint32_t kFixPoint = 2u;
constexpr std::uint32_t kOptRot = 4u;
constexpr std::uint32_t kOptTrans = 8u;
constexpr std::uint32_t kOptPoints = 16u;
constexpr std::uint32_t kOptFocal = 32u;
constexpr std::uint32_t kOptAspect = 64u;
constexpr std::uint32_t kOptPrincipal = 128u;
constexpr std::uint32_t kOptDistortion = 256u;

void pack_double(const double value, std::uint32_t &lo, std::uint32_t &hi) {
    std::uint64_t bits = 0;
    std::memcpy(&bits, &value, sizeof(bits));
    lo = static_cast<std::uint32_t>(bits);
    hi = static_cast<std::uint32_t>(bits >> 32);
}

template <std::size_t N> std::span<const std::byte> spirv_bytes(const unsigned char (&words)[N]) {
    return std::as_bytes(std::span<const unsigned char>(words, N));
}

struct Session {
    vk::Device device;
    std::string name;
    std::uint32_t max_groups = 1;
    vk::ComputePipeline linearize;
    vk::ComputePipeline assemble_points;
    vk::ComputePipeline assemble_cameras;
    vk::ComputePipeline damp_cameras;
    vk::ComputePipeline finalize_intrinsics;
    vk::ComputePipeline invert_points;
    vk::ComputePipeline reduced_rhs;
    vk::ComputePipeline schur_rhs;
    vk::ComputePipeline copy_intrinsic_rhs;
    vk::ComputePipeline intrinsic_schur;
    vk::ComputePipeline point_gather;
    vk::ComputePipeline point_reduce;
    vk::ComputePipeline camera_gather;
    vk::ComputePipeline camera_reduce;
    vk::ComputePipeline factor;
    vk::ComputePipeline precondition;
    vk::ComputePipeline dot;
    vk::ComputePipeline intrinsic_reduce;
    vk::ComputePipeline precondition_intrinsics;
    vk::ComputePipeline update_pcg;
    vk::ComputePipeline direction;
    vk::ComputePipeline alpha;
    vk::ComputePipeline beta;
    vk::ComputePipeline recover_points;
    vk::ComputePipeline update_poses;
    vk::ComputePipeline update_points;
    vk::ComputePipeline mask_intrinsics;
    vk::ComputePipeline update_intrinsics;
    vk::ComputePipeline prior_cost;
    vk::ComputePipeline linearized_cost;
};

vk::ComputePipeline make_pipeline(const vk::Device &device, const std::span<const std::byte> spirv,
                                  const std::uint32_t bindings) {
    return device.create_compute(spirv, bindings, sizeof(Push));
}

Session *load_session() noexcept {
    try {
        auto created = std::make_unique<Session>();
        vk::DeviceRequest request;
        request.want.push_descriptors = true;
        request.want.shader_float64 = true;
        request.want.buffer_atomic_f64 = true;
        created->device = vk::Device::create(request);
        if (!created->device.valid())
            return nullptr;
        const vk::Capabilities &caps = created->device.caps();
        if (!caps.enabled.shader_float64 || !caps.enabled.buffer_atomic_f64)
            return nullptr;
        VkPhysicalDeviceProperties properties{};
        vkGetPhysicalDeviceProperties(created->device.physical(), &properties);
        const VkPhysicalDeviceLimits &limits = properties.limits;
        if (limits.maxComputeWorkGroupInvocations < 256)
            return nullptr;
        if (limits.maxComputeWorkGroupSize[0] < 256)
            return nullptr;
        if (limits.maxPerStageDescriptorStorageBuffers < 16)
            return nullptr;
        if (limits.maxPushConstantsSize < sizeof(Push))
            return nullptr;
        if (limits.maxComputeWorkGroupCount[0] < 1)
            return nullptr;
        created->name = caps.name;
        created->max_groups = limits.maxComputeWorkGroupCount[0];
        const vk::Device &device = created->device;
        created->linearize = make_pipeline(device, spirv_bytes(ba_linearize_hlsl_spv), 10);
        created->assemble_points =
            make_pipeline(device, spirv_bytes(ba_assemble_points_hlsl_spv), 7);
        created->assemble_cameras =
            make_pipeline(device, spirv_bytes(ba_assemble_cameras_hlsl_spv), 9);
        created->damp_cameras = make_pipeline(device, spirv_bytes(ba_damp_cameras_hlsl_spv), 1);
        created->finalize_intrinsics =
            make_pipeline(device, spirv_bytes(ba_finalize_intrinsics_hlsl_spv), 5);
        created->invert_points = make_pipeline(device, spirv_bytes(ba_invert_points_hlsl_spv), 2);
        created->reduced_rhs = make_pipeline(device, spirv_bytes(ba_reduced_rhs_hlsl_spv), 3);
        created->schur_rhs = make_pipeline(device, spirv_bytes(ba_schur_rhs_hlsl_spv), 7);
        created->copy_intrinsic_rhs =
            make_pipeline(device, spirv_bytes(ba_copy_intrinsic_rhs_hlsl_spv), 2);
        created->intrinsic_schur =
            make_pipeline(device, spirv_bytes(ba_intrinsic_schur_hlsl_spv), 7);
        created->point_gather = make_pipeline(device, spirv_bytes(ba_point_gather_hlsl_spv), 7);
        created->point_reduce = make_pipeline(device, spirv_bytes(ba_point_reduce_hlsl_spv), 3);
        created->camera_gather = make_pipeline(device, spirv_bytes(ba_camera_gather_hlsl_spv), 9);
        created->camera_reduce = make_pipeline(device, spirv_bytes(ba_camera_reduce_hlsl_spv), 6);
        created->factor = make_pipeline(device, spirv_bytes(ba_factor_hlsl_spv), 2);
        created->precondition = make_pipeline(device, spirv_bytes(ba_precondition_hlsl_spv), 4);
        created->dot = make_pipeline(device, spirv_bytes(ba_dot_hlsl_spv), 3);
        created->intrinsic_reduce =
            make_pipeline(device, spirv_bytes(ba_intrinsic_reduce_hlsl_spv), 6);
        created->precondition_intrinsics =
            make_pipeline(device, spirv_bytes(ba_precondition_intrinsics_hlsl_spv), 4);
        created->update_pcg = make_pipeline(device, spirv_bytes(ba_update_pcg_hlsl_spv), 5);
        created->direction = make_pipeline(device, spirv_bytes(ba_direction_hlsl_spv), 3);
        created->alpha = make_pipeline(device, spirv_bytes(ba_alpha_hlsl_spv), 1);
        created->beta = make_pipeline(device, spirv_bytes(ba_beta_hlsl_spv), 1);
        created->recover_points =
            make_pipeline(device, spirv_bytes(ba_recover_points_hlsl_spv), 10);
        created->update_poses = make_pipeline(device, spirv_bytes(ba_update_poses_hlsl_spv), 2);
        created->update_points = make_pipeline(device, spirv_bytes(ba_update_points_hlsl_spv), 2);
        created->mask_intrinsics =
            make_pipeline(device, spirv_bytes(ba_mask_intrinsics_hlsl_spv), 5);
        created->update_intrinsics =
            make_pipeline(device, spirv_bytes(ba_update_intrinsics_hlsl_spv), 4);
        created->prior_cost = make_pipeline(device, spirv_bytes(ba_prior_cost_hlsl_spv), 4);
        created->linearized_cost =
            make_pipeline(device, spirv_bytes(ba_linearized_cost_hlsl_spv), 3);
        return created.release();
    } catch (...) {
        return nullptr;
    }
}

Session *session() noexcept {
    static Session *instance = load_session();
    return instance;
}

Session &require_session() {
    Session *loaded = session();
    if (loaded == nullptr)
        throw std::runtime_error("Vulkan bundle device is unavailable");
    return *loaded;
}

vk::Buffer device_buffer(const vk::Device &device, const VkDeviceSize bytes) {
    return device.create_buffer(std::max<VkDeviceSize>(bytes, 4), vk::MemoryKind::device_local);
}

} // namespace

class VulkanOptimizer::Impl {
  public:
    explicit Impl(Session &loaded, OptimizerOptions value)
        : device_(loaded.device), session_(&loaded), options_(std::move(value)),
          max_groups_(loaded.max_groups), encoder_(device_.encoder()) {
        if (options_.fix_first_pose)
            flags_ |= kFixPose;
        if (options_.fix_first_point)
            flags_ |= kFixPoint;
        if (options_.optimize_rotations)
            flags_ |= kOptRot;
        if (options_.optimize_translations)
            flags_ |= kOptTrans;
        if (options_.optimize_points)
            flags_ |= kOptPoints;
        if (options_.optimize_focal)
            flags_ |= kOptFocal;
        if (options_.optimize_aspect_ratio)
            flags_ |= kOptAspect;
        if (options_.optimize_principal_point)
            flags_ |= kOptPrincipal;
        if (options_.optimize_distortion)
            flags_ |= kOptDistortion;
    }

    vk::Device device_;
    Session *session_ = nullptr;
    OptimizerOptions options_{};
    std::uint32_t max_groups_ = 1;
    std::uint32_t flags_ = 0;
    double damping_ = 1e-3;
    double observation_weight_ = 1.0;
    std::size_t camera_count_ = 0;
    std::size_t point_count_ = 0;
    std::size_t observation_count_ = 0;
    std::size_t group_count_ = 0;
    std::size_t intrinsic_dof_ = 0;
    std::size_t camera_values_ = 0;
    std::size_t intrinsic_values_ = 0;
    std::size_t total_values_ = 0;
    VkDeviceSize pose_bytes_ = 0;
    VkDeviceSize point_bytes_ = 0;
    VkDeviceSize intrinsic_bytes_ = 0;

    vk::Buffer poses_;
    vk::Buffer pose_backup_;
    vk::Buffer intrinsics_;
    vk::Buffer intrinsic_backup_;
    vk::Buffer initial_intrinsics_;
    vk::Buffer intrinsic_constant_;
    vk::Buffer points_;
    vk::Buffer point_backup_;
    vk::Buffer cameras_;
    vk::Buffer point_ids_;
    vk::Buffer pose_group_;
    vk::Buffer observed_x_;
    vk::Buffer observed_y_;
    vk::Buffer weights_;
    vk::Buffer point_offsets_;
    vk::Buffer point_observations_;
    vk::Buffer camera_offsets_;
    vk::Buffer camera_observations_;
    vk::Buffer linearized_;
    vk::Buffer candidate_linearized_;
    vk::Buffer camera_h_;
    vk::Buffer camera_b_;
    vk::Buffer point_inverse_;
    vk::Buffer point_b_;
    vk::Buffer cross_;
    vk::Buffer reduced_point_;
    vk::Buffer camera_factor_;
    vk::Buffer intrinsic_factor_;
    vk::Buffer intrinsic_h_;
    vk::Buffer intrinsic_b_;
    vk::Buffer pose_intr_;
    vk::Buffer point_intr_;
    vk::Buffer point_scratch_;
    vk::Buffer camera_scratch_;
    vk::Buffer intrinsic_scratch_;
    vk::Buffer rhs_;
    vk::Buffer solution_;
    vk::Buffer residual_;
    vk::Buffer z_;
    vk::Buffer direction_;
    vk::Buffer product_;
    vk::Buffer point_temporary_;
    vk::Buffer point_step_;
    vk::Buffer scalars_;
    vk::Buffer cost_scalar_;
    vk::Buffer staging_;
    vk::Buffer readback_;
    // Destroyed first, while the buffers and device it submits against still
    // exist.
    vk::CommandEncoder encoder_;

    [[nodiscard]] std::uint32_t groups_for(const std::size_t count,
                                           const std::uint32_t threads) const {
        if (count == 0)
            return 0;
        const std::size_t raw = (count + threads - 1) / threads;
        return static_cast<std::uint32_t>(
            std::min<std::size_t>(std::max<std::size_t>(raw, 1), max_groups_));
    }

    [[nodiscard]] Push push_for(const std::uint32_t n) const {
        Push push{};
        push.n = n;
        push.flags = flags_;
        pack_double(options_.huber_delta, push.huber_lo, push.huber_hi);
        pack_double(options_.minimum_depth, push.depth_lo, push.depth_hi);
        pack_double(damping_, push.damp_lo, push.damp_hi);
        pack_double(options_.focal_prior_weight, push.prior_lo, push.prior_hi);
        pack_double(options_.min_focal_ratio, push.min_f_lo, push.min_f_hi);
        pack_double(options_.max_focal_ratio, push.max_f_lo, push.max_f_hi);
        pack_double(observation_weight_, push.obs_lo, push.obs_hi);
        push.dof = static_cast<std::uint32_t>(intrinsic_dof_);
        push.camera_values = static_cast<std::uint32_t>(camera_values_);
        return push;
    }

    void dispatch(const vk::ComputePipeline &pipeline,
                  const std::initializer_list<vk::BufferBinding> bindings, Push push,
                  const std::uint32_t groups) {
        if (groups == 0)
            return;
        push.groups = groups;
        const std::vector<vk::BufferBinding> stored(bindings);
        encoder_.dispatch(pipeline, stored, &push, sizeof(push), groups);
    }

    void upload_bytes(vk::Buffer &destination, const void *source, const VkDeviceSize bytes) {
        if (bytes == 0)
            return;
        if (staging_.size() < bytes) {
            staging_ = device_.create_buffer(bytes, vk::MemoryKind::host_visible);
        }
        staging_.upload(source, static_cast<std::size_t>(bytes));
        encoder_.copy(destination, 0, staging_, 0, bytes);
        encoder_.submit_wait();
    }

    void zero(const vk::Buffer &buffer, const VkDeviceSize bytes) {
        if (bytes == 0)
            return;
        encoder_.fill_u32(buffer, 0, bytes, 0);
    }

    void copy_device(vk::Buffer &destination, const vk::Buffer &source, const VkDeviceSize bytes) {
        if (bytes == 0)
            return;
        encoder_.copy(destination, 0, source, 0, bytes);
    }

    double read_double(const vk::Buffer &source, const std::uint32_t index) {
        encoder_.copy(readback_, 0, source, static_cast<VkDeviceSize>(index) * 8, 8);
        encoder_.submit_wait();
        double value = 0.0;
        readback_.download(&value, sizeof(value));
        return value;
    }

    void dot_to(const vk::Buffer &left, const vk::Buffer &right, const std::size_t count,
                const std::uint32_t slot) {
        zero(scalars_, 0); // filled precisely below; keep the slot write explicit
        encoder_.fill_u32(scalars_, static_cast<VkDeviceSize>(slot) * 8, 8, 0);
        Push push = push_for(static_cast<std::uint32_t>(count));
        push.p0 = slot;
        dispatch(session_->dot, {vk::binding(left), vk::binding(right), vk::binding(scalars_)},
                 push, groups_for(count, 256));
    }

    double dot(const vk::Buffer &left, const vk::Buffer &right, const std::size_t count) {
        dot_to(left, right, count, 0);
        return read_double(scalars_, 0);
    }

    void linearize(const vk::Buffer &output) {
        dispatch(session_->linearize,
                 {vk::binding(poses_), vk::binding(intrinsics_), vk::binding(pose_group_),
                  vk::binding(points_), vk::binding(cameras_), vk::binding(point_ids_),
                  vk::binding(observed_x_), vk::binding(observed_y_), vk::binding(weights_),
                  vk::binding(output)},
                 push_for(static_cast<std::uint32_t>(observation_count_)),
                 groups_for(observation_count_, 256));
    }

    double cost(const vk::Buffer &values) {
        zero(cost_scalar_, 8);
        dispatch(session_->linearized_cost,
                 {vk::binding(values), vk::binding(weights_), vk::binding(cost_scalar_)},
                 push_for(static_cast<std::uint32_t>(observation_count_)),
                 groups_for(observation_count_, 256));
        if (intrinsic_dof_ > 0) {
            dispatch(session_->prior_cost,
                     {vk::binding(intrinsics_), vk::binding(initial_intrinsics_),
                      vk::binding(intrinsic_constant_), vk::binding(cost_scalar_)},
                     push_for(static_cast<std::uint32_t>(group_count_)), 1);
        }
        return read_double(cost_scalar_, 0);
    }

    void assemble() {
        dispatch(session_->assemble_points,
                 {vk::binding(linearized_), vk::binding(point_offsets_),
                  vk::binding(point_observations_), vk::binding(point_inverse_),
                  vk::binding(point_b_), vk::binding(cross_), vk::binding(point_intr_)},
                 push_for(static_cast<std::uint32_t>(point_count_)), groups_for(point_count_, 1));
        if (intrinsic_dof_ > 0) {
            zero(intrinsic_h_,
                 static_cast<VkDeviceSize>(group_count_ * intrinsic_dof_ * intrinsic_dof_) * 8);
            zero(intrinsic_b_, static_cast<VkDeviceSize>(intrinsic_values_) * 8);
        }
        dispatch(session_->assemble_cameras,
                 {vk::binding(linearized_), vk::binding(camera_offsets_),
                  vk::binding(camera_observations_), vk::binding(pose_group_),
                  vk::binding(camera_h_), vk::binding(camera_b_), vk::binding(pose_intr_),
                  vk::binding(intrinsic_h_), vk::binding(intrinsic_b_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)), groups_for(camera_count_, 1));
        dispatch(session_->damp_cameras, {vk::binding(camera_h_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)),
                 groups_for(camera_count_, 256));
        if (intrinsic_dof_ > 0) {
            Push mask = push_for(static_cast<std::uint32_t>(camera_count_));
            mask.p0 = static_cast<std::uint32_t>(observation_count_);
            dispatch(session_->mask_intrinsics,
                     {vk::binding(cameras_), vk::binding(pose_group_),
                      vk::binding(intrinsic_constant_), vk::binding(pose_intr_),
                      vk::binding(point_intr_)},
                     mask, groups_for(std::max(camera_count_, observation_count_), 256));
            dispatch(
                session_->finalize_intrinsics,
                {vk::binding(intrinsic_h_), vk::binding(intrinsic_b_), vk::binding(intrinsics_),
                 vk::binding(initial_intrinsics_), vk::binding(intrinsic_constant_)},
                push_for(static_cast<std::uint32_t>(group_count_)), groups_for(group_count_, 256));
        }
        dispatch(session_->invert_points, {vk::binding(point_inverse_), vk::binding(point_b_)},
                 push_for(static_cast<std::uint32_t>(point_count_)), groups_for(point_count_, 256));
        dispatch(session_->reduced_rhs,
                 {vk::binding(point_inverse_), vk::binding(point_b_), vk::binding(reduced_point_)},
                 push_for(static_cast<std::uint32_t>(point_count_)), groups_for(point_count_, 256));
        dispatch(session_->schur_rhs,
                 {vk::binding(camera_offsets_), vk::binding(camera_observations_),
                  vk::binding(point_ids_), vk::binding(camera_b_), vk::binding(reduced_point_),
                  vk::binding(cross_), vk::binding(rhs_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)),
                 groups_for(camera_count_, 256));
        if (intrinsic_dof_ > 0) {
            dispatch(session_->copy_intrinsic_rhs, {vk::binding(intrinsic_b_), vk::binding(rhs_)},
                     push_for(static_cast<std::uint32_t>(intrinsic_values_)),
                     groups_for(intrinsic_values_, 256));
            dispatch(session_->intrinsic_schur,
                     {vk::binding(point_offsets_), vk::binding(point_observations_),
                      vk::binding(cameras_), vk::binding(pose_group_), vk::binding(reduced_point_),
                      vk::binding(point_intr_), vk::binding(rhs_)},
                     push_for(static_cast<std::uint32_t>(point_count_)),
                     groups_for(point_count_, 256));
        }
        Push camera_factor = push_for(static_cast<std::uint32_t>(camera_count_));
        camera_factor.p0 = 6;
        camera_factor.p1 = 6;
        dispatch(session_->factor, {vk::binding(camera_h_), vk::binding(camera_factor_)},
                 camera_factor, groups_for(camera_count_, 256));
        if (intrinsic_dof_ > 0) {
            Push intrinsic_factor = push_for(static_cast<std::uint32_t>(group_count_));
            intrinsic_factor.p0 = static_cast<std::uint32_t>(intrinsic_dof_);
            intrinsic_factor.p1 = 8;
            dispatch(session_->factor, {vk::binding(intrinsic_h_), vk::binding(intrinsic_factor_)},
                     intrinsic_factor, groups_for(group_count_, 256));
        }
    }

    void precondition_all() {
        dispatch(session_->precondition,
                 {vk::binding(camera_h_), vk::binding(camera_factor_), vk::binding(residual_),
                  vk::binding(z_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)),
                 groups_for(camera_count_, 256));
        if (intrinsic_dof_ > 0) {
            dispatch(session_->precondition_intrinsics,
                     {vk::binding(intrinsic_h_), vk::binding(intrinsic_factor_),
                      vk::binding(residual_), vk::binding(z_)},
                     push_for(static_cast<std::uint32_t>(group_count_)),
                     groups_for(group_count_, 256));
        }
    }

    void multiply() {
        zero(point_scratch_, static_cast<VkDeviceSize>(point_count_) * 3 * 8);
        zero(camera_scratch_, static_cast<VkDeviceSize>(camera_count_) * 6 * 8);
        if (intrinsic_dof_ > 0) {
            zero(intrinsic_scratch_, static_cast<VkDeviceSize>(intrinsic_values_) * 8);
            encoder_.fill_u32(product_, static_cast<VkDeviceSize>(camera_values_) * 8,
                              static_cast<VkDeviceSize>(intrinsic_values_) * 8, 0);
        }
        dispatch(session_->point_gather,
                 {vk::binding(cameras_), vk::binding(point_ids_), vk::binding(pose_group_),
                  vk::binding(cross_), vk::binding(point_intr_), vk::binding(direction_),
                  vk::binding(point_scratch_)},
                 push_for(static_cast<std::uint32_t>(observation_count_)),
                 groups_for(observation_count_, 256));
        dispatch(session_->point_reduce,
                 {vk::binding(point_inverse_), vk::binding(point_scratch_),
                  vk::binding(point_temporary_)},
                 push_for(static_cast<std::uint32_t>(point_count_)), groups_for(point_count_, 256));
        dispatch(session_->camera_gather,
                 {vk::binding(camera_offsets_), vk::binding(camera_observations_),
                  vk::binding(point_ids_), vk::binding(pose_group_), vk::binding(cross_),
                  vk::binding(point_intr_), vk::binding(point_temporary_),
                  vk::binding(camera_scratch_), vk::binding(intrinsic_scratch_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)), groups_for(camera_count_, 1));
        dispatch(session_->camera_reduce,
                 {vk::binding(pose_group_), vk::binding(camera_h_), vk::binding(pose_intr_),
                  vk::binding(direction_), vk::binding(camera_scratch_), vk::binding(product_)},
                 push_for(static_cast<std::uint32_t>(camera_count_)),
                 groups_for(camera_count_, 256));
        if (intrinsic_dof_ > 0) {
            Push push = push_for(static_cast<std::uint32_t>(camera_count_));
            push.p0 = static_cast<std::uint32_t>(group_count_);
            dispatch(session_->intrinsic_reduce,
                     {vk::binding(pose_group_), vk::binding(intrinsic_h_), vk::binding(pose_intr_),
                      vk::binding(direction_), vk::binding(intrinsic_scratch_),
                      vk::binding(product_)},
                     push, groups_for(std::max(camera_count_, group_count_), 256));
        }
    }

    std::size_t solve_pcg() {
        const VkDeviceSize total_bytes = static_cast<VkDeviceSize>(total_values_) * 8;
        zero(solution_, total_bytes);
        copy_device(residual_, rhs_, total_bytes);
        precondition_all();
        copy_device(direction_, z_, total_bytes);
        dot_to(residual_, z_, total_values_, 0);
        dot_to(rhs_, rhs_, total_values_, 1);
        const double rhs_norm = read_double(scalars_, 1);
        const double target =
            options_.pcg_tolerance * options_.pcg_tolerance * std::max(rhs_norm, 1e-30);
        std::size_t pcg = 0;
        for (; pcg < options_.maximum_pcg_iterations; ++pcg) {
            multiply();
            dot_to(direction_, product_, total_values_, 1);
            dispatch(session_->alpha, {vk::binding(scalars_)}, push_for(1), 1);
            dispatch(session_->update_pcg,
                     {vk::binding(solution_), vk::binding(residual_), vk::binding(direction_),
                      vk::binding(product_), vk::binding(scalars_)},
                     push_for(static_cast<std::uint32_t>(total_values_)),
                     groups_for(total_values_, 256));
            if ((pcg + 1) % 10 == 0 || pcg + 1 == options_.maximum_pcg_iterations) {
                dot_to(residual_, residual_, total_values_, 2);
                const double residual_norm = read_double(scalars_, 2);
                if (!std::isfinite(residual_norm) || residual_norm <= target) {
                    ++pcg;
                    break;
                }
            }
            precondition_all();
            dot_to(residual_, z_, total_values_, 3);
            dispatch(session_->beta, {vk::binding(scalars_)}, push_for(1), 1);
            dispatch(session_->direction,
                     {vk::binding(direction_), vk::binding(z_), vk::binding(scalars_)},
                     push_for(static_cast<std::uint32_t>(total_values_)),
                     groups_for(total_values_, 256));
        }
        return pcg;
    }
};

VulkanOptimizer::VulkanOptimizer(OptimizerOptions options)
    : impl_(std::make_unique<Impl>(require_session(), std::move(options))) {}
VulkanOptimizer::~VulkanOptimizer() = default;
VulkanOptimizer::VulkanOptimizer(VulkanOptimizer &&) noexcept = default;
VulkanOptimizer &VulkanOptimizer::operator=(VulkanOptimizer &&) noexcept = default;

bool VulkanOptimizer::is_available() noexcept { return session() != nullptr; }

std::string VulkanOptimizer::device_name() {
    Session *loaded = session();
    if (loaded == nullptr)
        throw std::runtime_error("Vulkan bundle device is unavailable");
    return loaded->name;
}

void VulkanOptimizer::upload(const Problem &problem) {
    problem.validate();
    auto &solver = *impl_;
    const auto fits = [](const std::size_t count) {
        return count <= static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max());
    };
    if (!fits(problem.poses.size()) || !fits(problem.points.size()) ||
        !fits(problem.observations.size()) || !fits(problem.intrinsics.size()) ||
        !fits(problem.observations.size() + 1)) {
        throw std::invalid_argument("Unsupported Vulkan bundle problem");
    }
    solver.camera_count_ = problem.poses.size();
    solver.point_count_ = problem.points.size();
    solver.observation_count_ = problem.observations.size();
    solver.group_count_ = problem.intrinsics.size();
    const LinearizerOptions linear = {solver.options_.huber_delta,
                                      solver.options_.minimum_depth,
                                      solver.options_.optimize_focal,
                                      solver.options_.optimize_aspect_ratio,
                                      solver.options_.optimize_principal_point,
                                      solver.options_.optimize_distortion};
    solver.intrinsic_dof_ = intrinsic_dof(linear);
    solver.camera_values_ = solver.camera_count_ * 6;
    solver.intrinsic_values_ = solver.group_count_ * solver.intrinsic_dof_;
    solver.total_values_ = solver.camera_values_ + solver.intrinsic_values_;
    solver.observation_weight_ =
        static_cast<double>(std::max<std::size_t>(solver.observation_count_, 1));
    solver.pose_bytes_ = static_cast<VkDeviceSize>(solver.camera_count_) * sizeof(Pose);
    solver.point_bytes_ = static_cast<VkDeviceSize>(solver.point_count_) * sizeof(Point3);
    solver.intrinsic_bytes_ =
        static_cast<VkDeviceSize>(solver.group_count_) * sizeof(PinholeIntrinsics);

    const auto allocate = [&](const VkDeviceSize bytes) {
        return device_buffer(solver.device_, bytes);
    };
    solver.poses_ = allocate(solver.pose_bytes_);
    solver.pose_backup_ = allocate(solver.pose_bytes_);
    solver.intrinsics_ = allocate(solver.intrinsic_bytes_);
    solver.intrinsic_backup_ = allocate(solver.intrinsic_bytes_);
    solver.initial_intrinsics_ = allocate(solver.intrinsic_bytes_);
    solver.intrinsic_constant_ = allocate(static_cast<VkDeviceSize>(solver.group_count_) * 4);
    solver.points_ = allocate(solver.point_bytes_);
    solver.point_backup_ = allocate(solver.point_bytes_);
    solver.cameras_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.point_ids_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.pose_group_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_) * 4);
    solver.observed_x_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 8);
    solver.observed_y_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 8);
    solver.weights_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 8);
    solver.point_offsets_ = allocate(static_cast<VkDeviceSize>(solver.point_count_ + 1) * 4);
    solver.point_observations_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.camera_offsets_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_ + 1) * 4);
    solver.camera_observations_ =
        allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.linearized_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 304);
    solver.candidate_linearized_ =
        allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 304);
    solver.camera_h_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_) * 36 * 8);
    solver.camera_b_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_) * 6 * 8);
    solver.camera_factor_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_) * 36 * 8);
    solver.point_inverse_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 9 * 8);
    solver.point_b_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 3 * 8);
    solver.cross_ = allocate(static_cast<VkDeviceSize>(solver.observation_count_) * 18 * 8);
    const VkDeviceSize intrinsic_h_bytes =
        static_cast<VkDeviceSize>(std::max<std::size_t>(
            solver.group_count_ * solver.intrinsic_dof_ * solver.intrinsic_dof_, 1)) *
        8;
    const VkDeviceSize intrinsic_v_bytes =
        static_cast<VkDeviceSize>(std::max<std::size_t>(solver.intrinsic_values_, 1)) * 8;
    const VkDeviceSize pose_intr_bytes = static_cast<VkDeviceSize>(std::max<std::size_t>(
                                             solver.camera_count_ * 6 * solver.intrinsic_dof_, 1)) *
                                         8;
    const VkDeviceSize point_intr_bytes =
        static_cast<VkDeviceSize>(
            std::max<std::size_t>(solver.observation_count_ * 3 * solver.intrinsic_dof_, 1)) *
        8;
    solver.intrinsic_h_ = allocate(intrinsic_h_bytes);
    solver.intrinsic_b_ = allocate(intrinsic_v_bytes);
    solver.pose_intr_ = allocate(pose_intr_bytes);
    solver.point_intr_ = allocate(point_intr_bytes);
    solver.intrinsic_factor_ =
        allocate(static_cast<VkDeviceSize>(std::max<std::size_t>(solver.group_count_, 1)) * 64 * 8);
    solver.point_scratch_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 3 * 8);
    solver.camera_scratch_ = allocate(static_cast<VkDeviceSize>(solver.camera_count_) * 6 * 8);
    solver.intrinsic_scratch_ = allocate(intrinsic_v_bytes);
    const VkDeviceSize total_bytes =
        static_cast<VkDeviceSize>(std::max<std::size_t>(solver.total_values_, 1)) * 8;
    solver.reduced_point_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 3 * 8);
    solver.rhs_ = allocate(total_bytes);
    solver.solution_ = allocate(total_bytes);
    solver.residual_ = allocate(total_bytes);
    solver.z_ = allocate(total_bytes);
    solver.direction_ = allocate(total_bytes);
    solver.product_ = allocate(total_bytes);
    solver.point_temporary_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 3 * 8);
    solver.point_step_ = allocate(static_cast<VkDeviceSize>(solver.point_count_) * 3 * 8);
    solver.scalars_ = allocate(64);
    solver.cost_scalar_ = allocate(8);
    const VkDeviceSize readback_bytes = std::max(
        {solver.pose_bytes_, solver.point_bytes_, solver.intrinsic_bytes_, VkDeviceSize{64}});
    solver.readback_ = solver.device_.create_buffer(readback_bytes, vk::MemoryKind::host_cached);

    solver.upload_bytes(solver.poses_, problem.poses.data(), solver.pose_bytes_);
    std::vector<std::uint32_t> pose_group(solver.camera_count_);
    for (std::size_t pose = 0; pose < solver.camera_count_; ++pose)
        pose_group[pose] = problem.intrinsic_index(pose);
    solver.upload_bytes(solver.pose_group_, pose_group.data(),
                        static_cast<VkDeviceSize>(pose_group.size()) * 4);
    std::vector<PinholeIntrinsics> group_intrinsics = problem.intrinsics;
    if (solver.options_.optimize_focal && !solver.options_.optimize_aspect_ratio) {
        for (PinholeIntrinsics &intrinsics : group_intrinsics) {
            const double focal = 0.5 * (intrinsics.fx + intrinsics.fy);
            intrinsics.fx = focal;
            intrinsics.fy = focal;
        }
    }
    solver.upload_bytes(solver.intrinsics_, group_intrinsics.data(), solver.intrinsic_bytes_);
    const std::vector<PinholeIntrinsics> initial =
        problem.initial_intrinsics.empty() ? group_intrinsics : problem.initial_intrinsics;
    solver.upload_bytes(solver.initial_intrinsics_, initial.data(), solver.intrinsic_bytes_);
    std::vector<std::uint32_t> constant(solver.group_count_, 0);
    if (!problem.intrinsic_constant.empty()) {
        for (std::size_t group = 0; group < solver.group_count_; ++group)
            constant[group] = problem.intrinsic_constant[group] ? 1u : 0u;
    }
    solver.upload_bytes(solver.intrinsic_constant_, constant.data(),
                        static_cast<VkDeviceSize>(constant.size()) * 4);
    solver.upload_bytes(solver.points_, problem.points.data(), solver.point_bytes_);
    solver.upload_bytes(solver.cameras_, problem.observations.camera.data(),
                        static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.upload_bytes(solver.point_ids_, problem.observations.point.data(),
                        static_cast<VkDeviceSize>(solver.observation_count_) * 4);
    solver.upload_bytes(solver.observed_x_, problem.observations.x.data(),
                        static_cast<VkDeviceSize>(solver.observation_count_) * 8);
    solver.upload_bytes(solver.observed_y_, problem.observations.y.data(),
                        static_cast<VkDeviceSize>(solver.observation_count_) * 8);
    solver.upload_bytes(solver.weights_, problem.observations.weight.data(),
                        static_cast<VkDeviceSize>(solver.observation_count_) * 8);

    std::vector<std::uint32_t> point_offsets(solver.point_count_ + 1);
    std::vector<std::uint32_t> camera_offsets(solver.camera_count_ + 1);
    std::vector<std::uint32_t> point_indices(solver.observation_count_);
    std::vector<std::uint32_t> camera_indices(solver.observation_count_);
    for (std::size_t i = 0; i < solver.observation_count_; ++i) {
        ++point_offsets[problem.observations.point[i] + 1];
        ++camera_offsets[problem.observations.camera[i] + 1];
    }
    std::partial_sum(point_offsets.begin(), point_offsets.end(), point_offsets.begin());
    std::partial_sum(camera_offsets.begin(), camera_offsets.end(), camera_offsets.begin());
    auto point_cursor = point_offsets;
    auto camera_cursor = camera_offsets;
    for (std::size_t i = 0; i < solver.observation_count_; ++i) {
        point_indices[point_cursor[problem.observations.point[i]]++] =
            static_cast<std::uint32_t>(i);
        camera_indices[camera_cursor[problem.observations.camera[i]]++] =
            static_cast<std::uint32_t>(i);
    }
    solver.upload_bytes(solver.point_offsets_, point_offsets.data(),
                        static_cast<VkDeviceSize>(point_offsets.size()) * 4);
    solver.upload_bytes(solver.point_observations_, point_indices.data(),
                        static_cast<VkDeviceSize>(point_indices.size()) * 4);
    solver.upload_bytes(solver.camera_offsets_, camera_offsets.data(),
                        static_cast<VkDeviceSize>(camera_offsets.size()) * 4);
    solver.upload_bytes(solver.camera_observations_, camera_indices.data(),
                        static_cast<VkDeviceSize>(camera_indices.size()) * 4);
}

OptimizerSummary VulkanOptimizer::optimize() {
    core::StageScope stage("ba.vulkan");
    auto &solver = *impl_;
    if (solver.observation_count_ == 0) {
        throw std::logic_error("Upload a BA problem before optimization");
    }
    const auto started = std::chrono::steady_clock::now();
    OptimizerSummary summary;
    solver.linearize(solver.linearized_);
    summary.initial_cost = solver.cost(solver.linearized_);
    summary.final_cost = summary.initial_cost;
    solver.damping_ = solver.options_.initial_damping;
    for (std::size_t iteration = 0; iteration < solver.options_.maximum_iterations; ++iteration) {
        solver.assemble();
        const std::size_t pcg = solver.solve_pcg();
        solver.dispatch(solver.session_->recover_points,
                        {vk::binding(solver.point_offsets_),
                         vk::binding(solver.point_observations_), vk::binding(solver.cameras_),
                         vk::binding(solver.pose_group_), vk::binding(solver.cross_),
                         vk::binding(solver.point_intr_), vk::binding(solver.point_inverse_),
                         vk::binding(solver.point_b_), vk::binding(solver.solution_),
                         vk::binding(solver.point_step_)},
                        solver.push_for(static_cast<std::uint32_t>(solver.point_count_)),
                        solver.groups_for(solver.point_count_, 256));
        const double norm_sq =
            solver.dot(solver.solution_, solver.solution_, solver.total_values_) +
            solver.dot(solver.point_step_, solver.point_step_, solver.point_count_ * 3);
        const double norm = std::sqrt(norm_sq);
        solver.copy_device(solver.pose_backup_, solver.poses_, solver.pose_bytes_);
        solver.copy_device(solver.point_backup_, solver.points_, solver.point_bytes_);
        solver.copy_device(solver.intrinsic_backup_, solver.intrinsics_, solver.intrinsic_bytes_);
        solver.dispatch(solver.session_->update_poses,
                        {vk::binding(solver.poses_), vk::binding(solver.solution_)},
                        solver.push_for(static_cast<std::uint32_t>(solver.camera_count_)),
                        solver.groups_for(solver.camera_count_, 256));
        solver.dispatch(solver.session_->update_points,
                        {vk::binding(solver.points_), vk::binding(solver.point_step_)},
                        solver.push_for(static_cast<std::uint32_t>(solver.point_count_)),
                        solver.groups_for(solver.point_count_, 256));
        if (solver.intrinsic_dof_ > 0) {
            solver.dispatch(
                solver.session_->update_intrinsics,
                {vk::binding(solver.intrinsics_), vk::binding(solver.initial_intrinsics_),
                 vk::binding(solver.intrinsic_constant_), vk::binding(solver.solution_)},
                solver.push_for(static_cast<std::uint32_t>(solver.group_count_)),
                solver.groups_for(solver.group_count_, 256));
        }
        solver.linearize(solver.candidate_linearized_);
        const double candidate = solver.cost(solver.candidate_linearized_);
        const bool accepted = std::isfinite(candidate) && candidate < summary.final_cost;
        summary.iterations.push_back({iteration, accepted ? candidate : summary.final_cost,
                                      solver.damping_, norm, pcg, accepted});
        core::Logger::instance().debug("Vulkan BA iteration=", iteration,
                                       " cost=", summary.iterations.back().cost,
                                       " damping=", solver.damping_, " step_norm=", norm,
                                       " pcg_iterations=", pcg, " accepted=", accepted);
        if (accepted) {
            const double previous = summary.final_cost;
            summary.final_cost = candidate;
            std::swap(solver.linearized_, solver.candidate_linearized_);
            ++summary.successful_steps;
            solver.damping_ = std::max(solver.options_.minimum_damping, solver.damping_ / 3.0);
            if (norm <= solver.options_.step_tolerance ||
                previous - candidate <=
                    solver.options_.function_tolerance * std::max(1.0, previous)) {
                summary.termination = TerminationReason::converged;
                break;
            }
        } else {
            solver.copy_device(solver.poses_, solver.pose_backup_, solver.pose_bytes_);
            solver.copy_device(solver.points_, solver.point_backup_, solver.point_bytes_);
            solver.copy_device(solver.intrinsics_, solver.intrinsic_backup_,
                               solver.intrinsic_bytes_);
            ++summary.unsuccessful_steps;
            solver.damping_ = std::min(solver.options_.maximum_damping, solver.damping_ * 10.0);
            if (solver.damping_ >= solver.options_.maximum_damping) {
                summary.termination = summary.successful_steps > 0
                                          ? TerminationReason::maximum_iterations
                                          : TerminationReason::numerical_failure;
                break;
            }
        }
    }
    if (solver.encoder_.recording())
        solver.encoder_.submit_wait();
    summary.total_time_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();
    stage.finish(summary.brief_report());
    return summary;
}

void VulkanOptimizer::download(Problem &problem) const {
    auto &solver = *impl_;
    if (problem.poses.size() != solver.camera_count_ ||
        problem.points.size() != solver.point_count_) {
        throw std::invalid_argument("Download target topology differs from uploaded BA problem");
    }
    if (problem.intrinsics.size() != solver.group_count_) {
        throw std::invalid_argument("Download target intrinsics differ from uploaded BA problem");
    }
    auto &encoder = const_cast<vk::CommandEncoder &>(solver.encoder_);
    auto &readback = const_cast<vk::Buffer &>(solver.readback_);
    encoder.copy(readback, 0, solver.poses_, 0, solver.pose_bytes_);
    encoder.submit_wait();
    readback.download(problem.poses.data(), static_cast<std::size_t>(solver.pose_bytes_));
    encoder.copy(readback, 0, solver.points_, 0, solver.point_bytes_);
    encoder.submit_wait();
    readback.download(problem.points.data(), static_cast<std::size_t>(solver.point_bytes_));
    encoder.copy(readback, 0, solver.intrinsics_, 0, solver.intrinsic_bytes_);
    encoder.submit_wait();
    readback.download(problem.intrinsics.data(), static_cast<std::size_t>(solver.intrinsic_bytes_));
}

OptimizerSummary optimize_vulkan(Problem &problem, const OptimizerOptions &options) {
    VulkanOptimizer optimizer(options);
    optimizer.upload(problem);
    const OptimizerSummary summary = optimizer.optimize();
    optimizer.download(problem);
    return summary;
}

} // namespace photara::ba
