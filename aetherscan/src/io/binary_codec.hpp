#pragma once

#include <cstdint>
#include <cstring>
#include <limits>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <type_traits>
#include <vector>

namespace aetherscan::io {
namespace binary {

inline constexpr std::uint64_t k_fnv_offset = 14695981039346656037ULL;
inline constexpr std::uint64_t k_fnv_prime = 1099511628211ULL;

inline void hash_bytes(
    std::uint64_t& hash, const void* data, const std::size_t size) noexcept {
    const auto* bytes = static_cast<const unsigned char*>(data);
    for (std::size_t i = 0; i < size; ++i) {
        hash ^= bytes[i];
        hash *= k_fnv_prime;
    }
}

class BufferWriter {
public:
    void bytes(const void* data, const std::size_t size) {
        if (size == 0) return;
        const auto* begin = static_cast<const std::uint8_t*>(data);
        buffer_.insert(buffer_.end(), begin, begin + size);
        hash_bytes(checksum_, data, size);
    }

    template <class T>
    void value(const T& item) {
        static_assert(std::is_trivially_copyable_v<T>);
        bytes(&item, sizeof(item));
    }

    void value_size(const std::size_t count) {
        value(static_cast<std::uint64_t>(count));
    }

    void string(const std::string_view text) {
        value_size(text.size());
        bytes(text.data(), text.size());
    }

    // Length-prefixed inner payload. Readers parse known fields and ignore
    // trailing bytes inside the record, so later writers can append fields.
    template <class Fn>
    void record(Fn&& write_body) {
        BufferWriter inner;
        write_body(inner);
        value_size(inner.size());
        bytes(inner.buffer().data(), inner.buffer().size());
    }

    [[nodiscard]] std::uint64_t size() const noexcept { return buffer_.size(); }
    [[nodiscard]] std::uint64_t checksum() const noexcept { return checksum_; }
    [[nodiscard]] const std::vector<std::uint8_t>& buffer() const noexcept {
        return buffer_;
    }
    [[nodiscard]] std::vector<std::uint8_t> take() { return std::move(buffer_); }

private:
    std::vector<std::uint8_t> buffer_;
    std::uint64_t checksum_{k_fnv_offset};
};

class BufferReader {
public:
    explicit BufferReader(const std::span<const std::uint8_t> data)
        : data_(data) {}

    void bytes(void* destination, const std::size_t size) {
        if (size > remaining())
            throw std::runtime_error("Truncated binary payload");
        if (size > 0) {
            std::memcpy(destination, data_.data() + offset_, size);
            hash_bytes(checksum_, destination, size);
            offset_ += size;
        }
    }

    template <class T>
    T value() {
        static_assert(std::is_trivially_copyable_v<T>);
        T item{};
        bytes(&item, sizeof(item));
        return item;
    }

    [[nodiscard]] std::size_t size(
        const std::size_t minimum_record_size = 1,
        const std::uint64_t maximum_count = 1'000'000'000ULL) {
        const std::uint64_t count = value<std::uint64_t>();
        if (minimum_record_size == 0 || count > maximum_count ||
            count > remaining() / minimum_record_size)
            throw std::runtime_error("Binary container size is invalid");
        if (count > static_cast<std::uint64_t>(
                        (std::numeric_limits<std::size_t>::max)()))
            throw std::runtime_error("Binary container is too large");
        return static_cast<std::size_t>(count);
    }

    std::string string() {
        std::string text(size(1, 16'777'216), '\0');
        bytes(text.data(), text.size());
        return text;
    }

    template <class Fn>
    void record(Fn&& read_body) {
        const std::uint64_t count = value<std::uint64_t>();
        if (count > remaining())
            throw std::runtime_error("Truncated binary record");
        const auto span = data_.subspan(offset_, static_cast<std::size_t>(count));
        hash_bytes(checksum_, span.data(), span.size());
        offset_ += static_cast<std::size_t>(count);
        BufferReader inner(span);
        read_body(inner);
    }

    void skip_remaining() {
        if (remaining() == 0) return;
        hash_bytes(checksum_, data_.data() + offset_, remaining());
        offset_ = data_.size();
    }

    void finish(const std::uint64_t expected_checksum) {
        skip_remaining();
        if (checksum_ != expected_checksum)
            throw std::runtime_error("Binary checksum mismatch");
    }

    [[nodiscard]] std::size_t remaining() const noexcept {
        return data_.size() - offset_;
    }

private:
    std::span<const std::uint8_t> data_;
    std::size_t offset_{};
    std::uint64_t checksum_{k_fnv_offset};
};

}  // namespace binary
}  // namespace aetherscan::io
