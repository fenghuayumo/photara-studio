#pragma once

#include "splat/checkpoint.hpp"
#include "splat/options.hpp"
#include "mvs/types.hpp"
#include "internal/tensor_impl.hpp"

#include <cstring>
#include <string>
#include <stdexcept>
#include <type_traits>

namespace photara::splat::detail {

class CheckpointWriter {
public:
    std::vector<std::uint8_t> bytes;
    void raw(const void* data, std::size_t size) {
        if (!size) return;
        const auto* first = static_cast<const std::uint8_t*>(data);
        bytes.insert(bytes.end(), first, first + size);
    }
    template<class T> void pod(const T& value) {
        static_assert(std::is_trivially_copyable_v<T>);
        raw(&value, sizeof(value));
    }
    void string(const std::string& value) {
        pod(std::uint64_t(value.size())); raw(value.data(), value.size());
    }
};

class CheckpointReader {
public:
    explicit CheckpointReader(std::span<const std::uint8_t> bytes) : bytes_(bytes) {}
    std::span<const std::uint8_t> take(std::size_t size) {
        if (size > bytes_.size()) throw std::runtime_error("Truncated training checkpoint");
        auto result = bytes_.first(size); bytes_ = bytes_.subspan(size); return result;
    }
    template<class T> T pod() {
        static_assert(std::is_trivially_copyable_v<T>);
        T value; auto data = take(sizeof(T)); std::memcpy(&value, data.data(), sizeof(T)); return value;
    }
    std::string string() {
        const auto size = pod<std::uint64_t>();
        if (size > 1024 * 1024) throw std::runtime_error("Invalid checkpoint string");
        auto data = take(size); return {reinterpret_cast<const char*>(data.data()), data.size()};
    }
    void finish() const {
        if (!bytes_.empty()) throw std::runtime_error("Unexpected checkpoint data");
    }
private:
    std::span<const std::uint8_t> bytes_;
};

std::uint64_t training_fingerprint(const mvs::MvsScene& scene, const TrainingOptions& options);

} // namespace photara::splat::detail
