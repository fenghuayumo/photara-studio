#pragma once

#include <memory>
#include <vector>
#include "data_types.hpp"

namespace tinytensor {

// Forward declarations
class Tensor;
class TensorIndexer;
class TensorError;
class MaskedTensorProxy;

using TensorPtr = std::shared_ptr<Tensor>;

} // namespace tinytensor
