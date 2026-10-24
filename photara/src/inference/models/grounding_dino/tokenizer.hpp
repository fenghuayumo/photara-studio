#pragma once

#include <filesystem>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace photara::inference::detail {

class GroundingDinoTokenizer {
public:
    static constexpr std::int32_t k_cls = 101;
    static constexpr std::int32_t k_sep = 102;
    static constexpr std::int32_t k_period = 1012;
    static constexpr std::int32_t k_question = 1029;

    void load(const std::filesystem::path& path);
    [[nodiscard]] std::vector<std::int32_t> encode(
        const std::string& text) const;

private:
    [[nodiscard]] std::int32_t unknown() const noexcept;
    void wordpiece(const std::string& word,
        std::vector<std::int32_t>& output) const;

    std::unordered_map<std::string, std::int32_t> vocabulary_;
    std::vector<std::string> by_id_;
    std::int32_t unknown_ = 100;
};

}  // namespace photara::inference::detail
