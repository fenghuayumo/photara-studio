#pragma once

#include "core/export.hpp"
#include <cstdint>

namespace tinytensor {

    struct LazyTelemetrySnapshot {
        uint64_t expr_nodes_created = 0;
        uint64_t materializations = 0;
        uint64_t kernel_launches = 0;
        uint64_t allocated_bytes = 0;
    };

    namespace internal {

        LFS_CORE_API void reset_lazy_telemetry();
        LFS_CORE_API LazyTelemetrySnapshot lazy_telemetry_snapshot();

        LFS_CORE_API void telemetry_record_expr_node(uint64_t count = 1);
        LFS_CORE_API void telemetry_record_materialization(uint64_t bytes);
        LFS_CORE_API void telemetry_record_kernel_launch(uint64_t count = 1);

    } // namespace internal

} // namespace tinytensor
