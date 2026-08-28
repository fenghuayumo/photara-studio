#pragma once

#include "pipeline.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace editor {

enum class ConsoleFilter : int { all, info, warning, error };

struct ConsoleView {
    ConsoleFilter filter{ConsoleFilter::all};
    std::array<char, 128> search{};
    bool follow{true};
    bool jump_to_end{};
    bool request_search_focus{};
    bool at_bottom{true};
    int selected{-1};
    std::size_t seen_count{};

    std::vector<int> visible;
    std::uint64_t cached_generation{~0ull};
    ConsoleFilter cached_filter{ConsoleFilter::all};
    std::string cached_query;
};

void draw_console(
    bool& open, ConsoleView& view, LogStream& log, bool running, JobKind job,
    Stage stage);

}  // namespace editor
