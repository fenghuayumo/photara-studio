#include "ba/bearing_vulkan.hpp"

#include "photara_vk/command.hpp"
#include "photara_vk/device.hpp"

#include "bearing_baseline_gauge.hlsl.embedded.hpp"
#include "bearing_camera_blocks.hlsl.embedded.hpp"
#include "bearing_chol_backward.hlsl.embedded.hpp"
#include "bearing_chol_forward.hlsl.embedded.hpp"
#include "bearing_chol_pivot.hlsl.embedded.hpp"
#include "bearing_chol_rank1.hlsl.embedded.hpp"
#include "bearing_diagonal_scale.hlsl.embedded.hpp"
#include "bearing_eliminate.hlsl.embedded.hpp"
#include "bearing_linearize.hlsl.embedded.hpp"
#include "bearing_model_decrease.hlsl.embedded.hpp"
#include "bearing_points_inverse.hlsl.embedded.hpp"
#include "bearing_schur.hlsl.embedded.hpp"
#include "bearing_sum_cost.hlsl.embedded.hpp"
#include "bearing_sum_model.hlsl.embedded.hpp"
#include "bearing_update_cameras.hlsl.embedded.hpp"
#include "bearing_update_points.hlsl.embedded.hpp"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <numeric>
#include <span>
#include <stdexcept>
#include <utility>

namespace photara::ba {
namespace {

// Same 48-byte push block every shader reads. Doubles are little-endian
// low/high pairs; the shaders unpack them with asdouble.
struct Push {
    std::uint32_t n = 0;
    std::uint32_t p0 = 0;
    std::uint32_t p1 = 0;
    std::uint32_t p2 = 0;
    std::uint32_t groups = 1;
    std::uint32_t flags = 0;
    std::uint32_t huber_lo = 0;
    std::uint32_t huber_hi = 0;
    std::uint32_t baseline_lo = 0;
    std::uint32_t baseline_hi = 0;
    std::uint32_t lambda_lo = 0;
    std::uint32_t lambda_hi = 0;
};
static_assert(sizeof(Push) == 48, "bearing push constants must be 48 bytes");

struct ObsGPU {
    std::uint32_t camera = 0;
    std::uint32_t point = 0;
    double direction[3]{};
};
static_assert(sizeof(ObsGPU) == 32, "bearing observation GPU record must be 32 bytes");
static_assert(offsetof(ObsGPU, direction) == 8, "bearing direction must follow two uints");

void pack_double(double value, std::uint32_t &lo, std::uint32_t &hi) {
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
    vk::ComputePipeline sum_cost;
    vk::ComputePipeline diagonal_scale;
    vk::ComputePipeline points_inverse;
    vk::ComputePipeline eliminate;
    vk::ComputePipeline camera_blocks;
    vk::ComputePipeline schur;
    vk::ComputePipeline baseline_gauge;
    vk::ComputePipeline chol_pivot;
    vk::ComputePipeline chol_rank1;
    vk::ComputePipeline chol_forward;
    vk::ComputePipeline chol_backward;
    vk::ComputePipeline update_points;
    vk::ComputePipeline update_cameras;
    vk::ComputePipeline model_decrease;
    vk::ComputePipeline sum_model;
};

vk::ComputePipeline make_pipeline(const vk::Device &device, std::span<const std::byte> spirv,
                                  std::uint32_t bindings) {
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
        created->linearize =
            make_pipeline(created->device, spirv_bytes(bearing_linearize_hlsl_spv), 6);
        created->sum_cost =
            make_pipeline(created->device, spirv_bytes(bearing_sum_cost_hlsl_spv), 3);
        created->diagonal_scale =
            make_pipeline(created->device, spirv_bytes(bearing_diagonal_scale_hlsl_spv), 5);
        created->points_inverse =
            make_pipeline(created->device, spirv_bytes(bearing_points_inverse_hlsl_spv), 7);
        created->eliminate =
            make_pipeline(created->device, spirv_bytes(bearing_eliminate_hlsl_spv), 4);
        created->camera_blocks =
            make_pipeline(created->device, spirv_bytes(bearing_camera_blocks_hlsl_spv), 10);
        created->schur = make_pipeline(created->device, spirv_bytes(bearing_schur_hlsl_spv), 6);
        created->baseline_gauge =
            make_pipeline(created->device, spirv_bytes(bearing_baseline_gauge_hlsl_spv), 3);
        created->chol_pivot =
            make_pipeline(created->device, spirv_bytes(bearing_chol_pivot_hlsl_spv), 2);
        created->chol_rank1 =
            make_pipeline(created->device, spirv_bytes(bearing_chol_rank1_hlsl_spv), 1);
        created->chol_forward =
            make_pipeline(created->device, spirv_bytes(bearing_chol_forward_hlsl_spv), 2);
        created->chol_backward =
            make_pipeline(created->device, spirv_bytes(bearing_chol_backward_hlsl_spv), 2);
        created->update_points =
            make_pipeline(created->device, spirv_bytes(bearing_update_points_hlsl_spv), 9);
        created->update_cameras =
            make_pipeline(created->device, spirv_bytes(bearing_update_cameras_hlsl_spv), 3);
        created->model_decrease =
            make_pipeline(created->device, spirv_bytes(bearing_model_decrease_hlsl_spv), 6);
        created->sum_model =
            make_pipeline(created->device, spirv_bytes(bearing_sum_model_hlsl_spv), 4);
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
    if (loaded == nullptr) {
        throw std::runtime_error("Vulkan bearing device is unavailable");
    }
    return *loaded;
}

vk::Buffer device_buffer(const vk::Device &device, VkDeviceSize bytes) {
    return device.create_buffer(std::max<VkDeviceSize>(bytes, 4), vk::MemoryKind::device_local);
}

} // namespace

class VulkanBearingOptimizer::Impl {
  public:
    explicit Impl(Session &loaded)
        : device_(loaded.device), session_(&loaded), encoder_(device_.encoder()),
          max_groups_(loaded.max_groups) {}

    vk::Device device_;
    Session *session_ = nullptr;
    vk::CommandEncoder encoder_;
    std::uint32_t max_groups_ = 1;
    int nc_ = 0;
    int np_ = 0;
    int no_ = 0;
    std::uint32_t anchor_ = 0;
    std::uint32_t first_ = 0;
    std::uint32_t second_ = 0;
    double baseline_ = 0;
    double huber_ = 0;
    double lambda_ = 1e-4;
    VkDeviceSize matrix_bytes_ = 0;
    VkDeviceSize camera_bytes_ = 0;
    VkDeviceSize point_bytes_ = 0;

    vk::Buffer cameras_;
    vk::Buffer points_;
    vk::Buffer candidate_cameras_;
    vk::Buffer candidate_points_;
    vk::Buffer weights_;
    vk::Buffer costs_;
    vk::Buffer total_;
    vk::Buffer inverse_;
    vk::Buffer point_rhs_;
    vk::Buffer eliminated_;
    vk::Buffer matrix_;
    vk::Buffer rhs_;
    vk::Buffer camera_scale_;
    vk::Buffer point_scale_;
    vk::Buffer linear_;
    vk::Buffer observations_;
    vk::Buffer point_offsets_;
    vk::Buffer point_indices_;
    vk::Buffer camera_offsets_;
    vk::Buffer camera_indices_;
    vk::Buffer info_;
    vk::Buffer staging_;
    vk::Buffer readback_;

    [[nodiscard]] std::uint32_t grid(int count) const {
        const auto needed = (static_cast<std::uint64_t>(std::max(count, 0)) + 255ull) / 256ull;
        const auto capped =
            std::min<std::uint64_t>(std::max<std::uint64_t>(needed, 1ull), max_groups_);
        return static_cast<std::uint32_t>(capped);
    }

    [[nodiscard]] Push constants(std::uint32_t n) const {
        Push push;
        push.n = n;
        push.p0 = first_;
        push.p1 = second_;
        pack_double(huber_, push.huber_lo, push.huber_hi);
        pack_double(baseline_, push.baseline_lo, push.baseline_hi);
        pack_double(lambda_, push.lambda_lo, push.lambda_hi);
        return push;
    }

    void launch(const vk::ComputePipeline &pipeline, const vk::BufferBinding *bindings,
                std::uint32_t binding_count, Push push, std::uint32_t groups) {
        push.groups = groups;
        encoder_.dispatch(pipeline, std::span<const vk::BufferBinding>(bindings, binding_count),
                          &push, sizeof(push), groups, 1, 1);
    }

    void copy_up(const void *source, std::size_t bytes, vk::Buffer &destination, bool wait) {
        if (bytes == 0)
            return;
        staging_.upload(source, bytes);
        encoder_.copy(destination, 0, staging_, 0, bytes);
        if (wait)
            encoder_.submit_wait();
    }

    void read(const vk::Buffer &source, void *destination, std::size_t bytes) {
        if (bytes == 0)
            return;
        encoder_.copy(readback_, 0, source, 0, bytes);
        encoder_.submit_wait();
        readback_.download(destination, bytes);
    }

    void zero(const vk::Buffer &buffer, VkDeviceSize bytes) {
        encoder_.fill_u32(buffer, 0, bytes, 0);
    }

    double cost(bool candidate, bool derivatives) {
        const vk::Buffer &cameras = candidate ? candidate_cameras_ : cameras_;
        const vk::Buffer &points = candidate ? candidate_points_ : points_;
        const vk::BufferBinding bindings[] = {
            vk::binding(cameras),  vk::binding(points),  vk::binding(observations_),
            vk::binding(weights_), vk::binding(linear_), vk::binding(costs_),
        };
        Push push = constants(static_cast<std::uint32_t>(no_));
        push.flags = derivatives ? 1u : 0u;
        launch(session_->linearize, bindings, 6, push, grid(no_));

        const vk::BufferBinding sum_bindings[] = {
            vk::binding(costs_),
            vk::binding(cameras),
            vk::binding(total_),
        };
        launch(session_->sum_cost, sum_bindings, 3, push, 1);
        double value = 0;
        read(total_, &value, sizeof(value));
        return value;
    }

    void diagonal_scales() {
        const vk::BufferBinding camera_bindings[] = {
            vk::binding(linear_),  vk::binding(camera_offsets_), vk::binding(camera_indices_),
            vk::binding(cameras_), vk::binding(camera_scale_),
        };
        Push camera_push = constants(static_cast<std::uint32_t>(nc_));
        camera_push.flags = 2u;
        launch(session_->diagonal_scale, camera_bindings, 5, camera_push, grid(nc_));

        const vk::BufferBinding point_bindings[] = {
            vk::binding(linear_),  vk::binding(point_offsets_), vk::binding(point_indices_),
            vk::binding(cameras_), vk::binding(point_scale_),
        };
        Push point_push = constants(static_cast<std::uint32_t>(np_));
        launch(session_->diagonal_scale, point_bindings, 5, point_push, grid(np_));
    }

    void factor(std::int32_t out_info[2]) {
        zero(info_, sizeof(std::uint32_t) * 2);
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_),      vk::binding(point_offsets_), vk::binding(point_indices_),
                vk::binding(point_scale_), vk::binding(inverse_),       vk::binding(point_rhs_),
                vk::binding(info_),
            };
            launch(session_->points_inverse, bindings, 7,
                   constants(static_cast<std::uint32_t>(np_)), grid(np_));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_),
                vk::binding(observations_),
                vk::binding(inverse_),
                vk::binding(eliminated_),
            };
            launch(session_->eliminate, bindings, 4, constants(static_cast<std::uint32_t>(no_)),
                   grid(no_));
        }
        zero(matrix_, matrix_bytes_);
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_),         vk::binding(observations_),
                vk::binding(eliminated_),     vk::binding(point_rhs_),
                vk::binding(camera_offsets_), vk::binding(camera_indices_),
                vk::binding(camera_scale_),   vk::binding(cameras_),
                vk::binding(matrix_),         vk::binding(rhs_),
            };
            launch(session_->camera_blocks, bindings, 10,
                   constants(static_cast<std::uint32_t>(nc_)), grid(nc_));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_),        vk::binding(observations_),  vk::binding(eliminated_),
                vk::binding(point_offsets_), vk::binding(point_indices_), vk::binding(matrix_),
            };
            Push push = constants(static_cast<std::uint32_t>(np_));
            push.p0 = static_cast<std::uint32_t>(nc_);
            const auto groups = std::min(static_cast<std::uint32_t>(np_), max_groups_);
            launch(session_->schur, bindings, 6, push, std::max(groups, 1u));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(cameras_),
                vk::binding(matrix_),
                vk::binding(rhs_),
            };
            Push push = constants(static_cast<std::uint32_t>(nc_));
            push.p0 = anchor_;
            push.p1 = first_;
            push.p2 = second_;
            launch(session_->baseline_gauge, bindings, 3, push, 1);
        }

        const int width = nc_ * 3;
        for (int column = 0; column < width; ++column) {
            const vk::BufferBinding pivot_bindings[] = {
                vk::binding(matrix_),
                vk::binding(info_),
            };
            Push pivot = constants(static_cast<std::uint32_t>(width));
            pivot.p0 = static_cast<std::uint32_t>(column);
            launch(session_->chol_pivot, pivot_bindings, 2, pivot, 1);
            if (column + 1 >= width)
                continue;
            const int trailing = width - column - 1;
            const std::int64_t cells = static_cast<std::int64_t>(trailing) * trailing;
            int rank_groups = static_cast<int>(std::min<std::int64_t>((cells + 255) / 256, 8192));
            if (static_cast<std::uint32_t>(rank_groups) > max_groups_) {
                rank_groups = static_cast<int>(max_groups_);
            }
            if (rank_groups < 1)
                rank_groups = 1;
            const vk::BufferBinding rank_bindings[] = {vk::binding(matrix_)};
            Push rank = constants(static_cast<std::uint32_t>(width));
            rank.p0 = static_cast<std::uint32_t>(column);
            launch(session_->chol_rank1, rank_bindings, 1, rank,
                   static_cast<std::uint32_t>(rank_groups));
        }
        read(info_, out_info, sizeof(std::int32_t) * 2);
    }

    double predicted_decrease() {
        const int width = nc_ * 3;
        const vk::BufferBinding solve_bindings[] = {
            vk::binding(matrix_),
            vk::binding(rhs_),
        };
        const Push solve = constants(static_cast<std::uint32_t>(width));
        launch(session_->chol_forward, solve_bindings, 2, solve, 1);
        launch(session_->chol_backward, solve_bindings, 2, solve, 1);
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(cameras_),
                vk::binding(rhs_),
                vk::binding(candidate_cameras_),
            };
            launch(session_->update_cameras, bindings, 3,
                   constants(static_cast<std::uint32_t>(width)), grid(width));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_),
                vk::binding(observations_),
                vk::binding(point_offsets_),
                vk::binding(point_indices_),
                vk::binding(inverse_),
                vk::binding(point_rhs_),
                vk::binding(rhs_),
                vk::binding(points_),
                vk::binding(candidate_points_),
            };
            launch(session_->update_points, bindings, 9, constants(static_cast<std::uint32_t>(np_)),
                   grid(np_));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(linear_), vk::binding(observations_),     vk::binding(rhs_),
                vk::binding(points_), vk::binding(candidate_points_), vk::binding(costs_),
            };
            launch(session_->model_decrease, bindings, 6,
                   constants(static_cast<std::uint32_t>(no_)), grid(no_));
        }
        {
            const vk::BufferBinding bindings[] = {
                vk::binding(costs_),
                vk::binding(cameras_),
                vk::binding(rhs_),
                vk::binding(total_),
            };
            launch(session_->sum_model, bindings, 4, constants(static_cast<std::uint32_t>(no_)), 1);
        }
        double value = 0;
        read(total_, &value, sizeof(value));
        return value;
    }
};

VulkanBearingOptimizer::VulkanBearingOptimizer(const BearingProblem &problem)
    : impl_(std::make_unique<Impl>(require_session())) {
    Impl &solver = *impl_;
    if (problem.cameras.empty() || problem.cameras.size() > 2000 || problem.points.empty() ||
        problem.observations.empty() ||
        problem.points.size() > static_cast<std::size_t>(std::numeric_limits<int>::max() / 9) ||
        problem.observations.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max() / 9) ||
        problem.anchor >= problem.cameras.size() ||
        problem.baseline_first >= problem.cameras.size() ||
        problem.baseline_second >= problem.cameras.size() ||
        problem.baseline_first == problem.baseline_second || !(problem.baseline > 1e-10) ||
        !std::isfinite(problem.baseline) || !(problem.huber > 0) || !std::isfinite(problem.huber)) {
        throw std::invalid_argument("Unsupported Vulkan bearing problem");
    }
    for (const auto &camera : problem.cameras) {
        for (double value : camera) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Nonfinite camera");
        }
    }
    for (const auto &point : problem.points) {
        for (double value : point) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Nonfinite point");
        }
    }

    solver.nc_ = static_cast<int>(problem.cameras.size());
    solver.np_ = static_cast<int>(problem.points.size());
    solver.no_ = static_cast<int>(problem.observations.size());
    solver.anchor_ = problem.anchor;
    solver.first_ = problem.baseline_first;
    solver.second_ = problem.baseline_second;
    solver.baseline_ = problem.baseline;
    solver.huber_ = problem.huber;
    solver.camera_bytes_ = static_cast<VkDeviceSize>(solver.nc_) * 3 * sizeof(double);
    solver.point_bytes_ = static_cast<VkDeviceSize>(solver.np_) * 3 * sizeof(double);
    const int width = solver.nc_ * 3;
    solver.matrix_bytes_ =
        static_cast<VkDeviceSize>(width) * static_cast<VkDeviceSize>(width) * sizeof(double);

    std::vector<ObsGPU> observations;
    std::vector<std::uint32_t> point_offsets(static_cast<std::size_t>(solver.np_) + 1);
    std::vector<std::uint32_t> camera_offsets(static_cast<std::size_t>(solver.nc_) + 1);
    std::vector<std::uint32_t> point_indices(static_cast<std::size_t>(solver.no_));
    std::vector<std::uint32_t> camera_indices(static_cast<std::size_t>(solver.no_));
    observations.reserve(problem.observations.size());
    for (const BearingObservation &observation : problem.observations) {
        if (observation.camera >= problem.cameras.size() ||
            observation.point >= problem.points.size()) {
            throw std::invalid_argument("Invalid bearing observation");
        }
        for (double value : observation.direction) {
            if (!std::isfinite(value))
                throw std::invalid_argument("Nonfinite bearing");
        }
        ObsGPU record;
        record.camera = observation.camera;
        record.point = observation.point;
        record.direction[0] = observation.direction[0];
        record.direction[1] = observation.direction[1];
        record.direction[2] = observation.direction[2];
        observations.push_back(record);
        ++point_offsets[observation.point + 1];
        ++camera_offsets[observation.camera + 1];
    }
    std::partial_sum(point_offsets.begin(), point_offsets.end(), point_offsets.begin());
    std::partial_sum(camera_offsets.begin(), camera_offsets.end(), camera_offsets.begin());
    std::vector<std::uint32_t> point_cursor = point_offsets;
    std::vector<std::uint32_t> camera_cursor = camera_offsets;
    for (std::uint32_t index = 0; index < observations.size(); ++index) {
        point_indices[point_cursor[observations[index].point]++] = index;
        camera_indices[camera_cursor[observations[index].camera]++] = index;
    }

    const VkDeviceSize observation_bytes = observations.size() * sizeof(ObsGPU);
    const VkDeviceSize weight_bytes = static_cast<VkDeviceSize>(solver.no_) * sizeof(double);
    const VkDeviceSize staging_bytes = std::max({
        observation_bytes,
        static_cast<VkDeviceSize>(point_offsets.size() * sizeof(std::uint32_t)),
        static_cast<VkDeviceSize>(point_indices.size() * sizeof(std::uint32_t)),
        static_cast<VkDeviceSize>(camera_offsets.size() * sizeof(std::uint32_t)),
        static_cast<VkDeviceSize>(camera_indices.size() * sizeof(std::uint32_t)),
        solver.camera_bytes_,
        solver.point_bytes_,
        weight_bytes,
    });
    const VkDeviceSize readback_bytes =
        std::max({solver.camera_bytes_, solver.point_bytes_, VkDeviceSize{16}});

    solver.cameras_ = device_buffer(solver.device_, solver.camera_bytes_);
    solver.points_ = device_buffer(solver.device_, solver.point_bytes_);
    solver.candidate_cameras_ = device_buffer(solver.device_, solver.camera_bytes_);
    solver.candidate_points_ = device_buffer(solver.device_, solver.point_bytes_);
    solver.weights_ = device_buffer(solver.device_, weight_bytes);
    solver.costs_ = device_buffer(solver.device_, weight_bytes);
    solver.total_ = device_buffer(solver.device_, sizeof(double));
    solver.inverse_ =
        device_buffer(solver.device_, static_cast<VkDeviceSize>(solver.np_) * 9 * sizeof(double));
    solver.point_rhs_ = device_buffer(solver.device_, solver.point_bytes_);
    solver.eliminated_ =
        device_buffer(solver.device_, static_cast<VkDeviceSize>(solver.no_) * 9 * sizeof(double));
    solver.matrix_ = device_buffer(solver.device_, solver.matrix_bytes_);
    solver.rhs_ = device_buffer(solver.device_, solver.camera_bytes_);
    solver.camera_scale_ = device_buffer(solver.device_, solver.camera_bytes_);
    solver.point_scale_ = device_buffer(solver.device_, solver.point_bytes_);
    solver.linear_ =
        device_buffer(solver.device_, static_cast<VkDeviceSize>(solver.no_) * 12 * sizeof(double));
    solver.observations_ = device_buffer(solver.device_, observation_bytes);
    solver.point_offsets_ =
        device_buffer(solver.device_, point_offsets.size() * sizeof(std::uint32_t));
    solver.point_indices_ =
        device_buffer(solver.device_, point_indices.size() * sizeof(std::uint32_t));
    solver.camera_offsets_ =
        device_buffer(solver.device_, camera_offsets.size() * sizeof(std::uint32_t));
    solver.camera_indices_ =
        device_buffer(solver.device_, camera_indices.size() * sizeof(std::uint32_t));
    solver.info_ = device_buffer(solver.device_, sizeof(std::uint32_t) * 2);
    solver.staging_ = solver.device_.create_buffer(staging_bytes, vk::MemoryKind::host_visible);
    solver.readback_ = solver.device_.create_buffer(readback_bytes, vk::MemoryKind::host_cached);

    solver.copy_up(observations.data(), observation_bytes, solver.observations_, true);
    solver.copy_up(point_offsets.data(), point_offsets.size() * sizeof(std::uint32_t),
                   solver.point_offsets_, true);
    solver.copy_up(point_indices.data(), point_indices.size() * sizeof(std::uint32_t),
                   solver.point_indices_, true);
    solver.copy_up(camera_offsets.data(), camera_offsets.size() * sizeof(std::uint32_t),
                   solver.camera_offsets_, true);
    solver.copy_up(camera_indices.data(), camera_indices.size() * sizeof(std::uint32_t),
                   solver.camera_indices_, true);
    solver.copy_up(problem.cameras.front().data(), solver.camera_bytes_, solver.cameras_, true);
    solver.copy_up(problem.points.front().data(), solver.point_bytes_, solver.points_, true);
}

VulkanBearingOptimizer::~VulkanBearingOptimizer() = default;

bool VulkanBearingOptimizer::is_available() noexcept { return session() != nullptr; }

std::string VulkanBearingOptimizer::device_name() {
    const Session *loaded = session();
    return loaded == nullptr ? std::string{} : loaded->name;
}

void VulkanBearingOptimizer::download(BearingProblem &problem) const {
    const Impl &solver = *impl_;
    if (problem.cameras.size() != static_cast<std::size_t>(solver.nc_) ||
        problem.points.size() != static_cast<std::size_t>(solver.np_)) {
        throw std::invalid_argument("Bearing download shape changed");
    }
    const_cast<Impl &>(solver).read(solver.cameras_, problem.cameras.front().data(),
                                    solver.camera_bytes_);
    const_cast<Impl &>(solver).read(solver.points_, problem.points.front().data(),
                                    solver.point_bytes_);
}

BearingSolveSummary VulkanBearingOptimizer::solve(const std::vector<double> &weights,
                                                  unsigned iterations, double tolerance,
                                                  double max_seconds) {
    Impl &solver = *impl_;
    if (weights.size() != static_cast<std::size_t>(solver.no_)) {
        throw std::invalid_argument("Bearing weight count mismatch");
    }
    if (!std::isfinite(tolerance) || tolerance < 0 || !std::isfinite(max_seconds) ||
        max_seconds < 0) {
        throw std::invalid_argument("Invalid bearing solve tolerance or time limit");
    }
    for (double weight : weights) {
        if (!std::isfinite(weight) || weight < 0)
            throw std::invalid_argument("Invalid bearing weight");
    }
    solver.copy_up(weights.data(), weights.size() * sizeof(double), solver.weights_, false);

    BearingSolveSummary result;
    const auto started = std::chrono::steady_clock::now();
    double current = solver.cost(false, true);
    solver.lambda_ = 1e-4;
    double rejection_factor = 2;
    result.initial_cost = current;
    if (!std::isfinite(current)) {
        result.seconds =
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
        return result;
    }
    result.usable = true;
    solver.diagonal_scales();
    for (unsigned iteration = 0; iteration < iterations; ++iteration) {
        if (max_seconds > 0 &&
            std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() >
                max_seconds) {
            break;
        }
        std::int32_t info[2] = {};
        solver.factor(info);
        ++result.iterations;
        if (info[0] < 0)
            throw std::runtime_error("Vulkan bearing factorization argument error");
        if (info[0] != 0 || info[1] != 0) {
            solver.lambda_ *= rejection_factor;
            rejection_factor *= 2;
            if (solver.lambda_ > 1e16) {
                result.usable = false;
                break;
            }
            continue;
        }
        const double predicted = solver.predicted_decrease();
        const double candidate = solver.cost(true, false);
        if (std::isfinite(candidate) && std::abs(current - candidate) <= tolerance * current)
            break;
        const double quality = predicted > 0 ? (current - candidate) / predicted : -1;
        if (std::isfinite(candidate) && quality > 1e-3) {
            current = candidate;
            std::swap(solver.cameras_, solver.candidate_cameras_);
            std::swap(solver.points_, solver.candidate_points_);
            solver.lambda_ = std::max(
                1e-16, solver.lambda_ * std::max(1.0 / 3.0, 1 - std::pow(2 * quality - 1, 3)));
            rejection_factor = 2;
            current = solver.cost(false, true);
        } else {
            solver.lambda_ *= rejection_factor;
            rejection_factor *= 2;
            if (solver.lambda_ > 1e16)
                break;
        }
    }
    result.final_cost = current;
    result.seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count();
    return result;
}

} // namespace photara::ba
