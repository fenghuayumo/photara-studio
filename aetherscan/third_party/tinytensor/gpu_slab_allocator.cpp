#include "internal/gpu_slab_allocator.hpp"

namespace tinytensor {

    GPUSlabAllocator& GPUSlabAllocator::instance() {
        static GPUSlabAllocator allocator;
        return allocator;
    }

} // namespace tinytensor
