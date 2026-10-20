#pragma once

#include <cstddef>
#include <cstdint>
#include <iterator>
#include <list>
#include <map>
#include <utility>

namespace photara::sfm::detail {

// Entries are immutable after insertion. Eviction affects work, never decisions.
template<class T> class RecoveryMemo {
    struct Entry { T value; std::size_t bytes; typename std::list<std::uint64_t>::iterator age; };
    std::map<std::uint64_t, Entry> entries_;
    std::list<std::uint64_t> ages_;
    std::size_t bytes_{};
    std::size_t limit_;
public:
    explicit RecoveryMemo(std::size_t limit) : limit_(limit) {}
    const T* find(std::uint64_t key) {
        const auto it = entries_.find(key);
        if (it == entries_.end()) return nullptr;
        ages_.splice(ages_.end(), ages_, it->second.age);
        return &it->second.value;
    }
    void insert(std::uint64_t key, T value, std::size_t bytes) {
        bytes += sizeof(Entry) + 64; // Account for map/list nodes as well as payload.
        if (bytes > limit_ || entries_.contains(key)) return;
        while (bytes_ + bytes > limit_) {
            const auto oldest = entries_.find(ages_.front());
            bytes_ -= oldest->second.bytes;
            entries_.erase(oldest);
            ages_.pop_front();
        }
        ages_.push_back(key);
        entries_.emplace(key, Entry{std::move(value), bytes, std::prev(ages_.end())});
        bytes_ += bytes;
    }
    std::size_t bytes() const { return bytes_; }
};

} // namespace photara::sfm::detail
