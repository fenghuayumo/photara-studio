#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <format>
#include <iostream>
#include <string>
#include <string_view>
#include <stdexcept>
#include <utility>

// TinyTensor Logger - Minimal logging for the standalone library
//
// Default level is info: LOG_DEBUG / LOG_TRACE are silent unless enabled via
// TINYTENSOR_LOG_LEVEL (or PHOTARA_LOG_LEVEL) = debug|trace.
//
// Call sites use either a single streamable message or fmt/std::format style:
//   LOG_INFO("hello");
//   LOG_INFO("count={}", n);
namespace tinytensor {

enum class LogLevel : int {
    error = 0,
    warn = 1,
    info = 2,
    debug = 3,
    trace = 4,
    off = 5,
};

inline LogLevel parse_log_level(std::string value, const LogLevel fallback) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "error") return LogLevel::error;
    if (value == "warning" || value == "warn") return LogLevel::warn;
    if (value == "info") return LogLevel::info;
    if (value == "debug") return LogLevel::debug;
    if (value == "trace") return LogLevel::trace;
    if (value == "off") return LogLevel::off;
    return fallback;
}

inline LogLevel& current_log_level() {
    static LogLevel level = [] {
        if (const char* env = std::getenv("TINYTENSOR_LOG_LEVEL")) {
            return parse_log_level(env, LogLevel::info);
        }
        if (const char* env = std::getenv("PHOTARA_LOG_LEVEL")) {
            return parse_log_level(env, LogLevel::info);
        }
        return LogLevel::info;
    }();
    return level;
}

inline void set_log_level(const LogLevel level) {
    current_log_level() = level;
}

inline bool log_enabled(const LogLevel level) {
    return static_cast<int>(level) <= static_cast<int>(current_log_level());
}

// Single argument: stream as-is (supports pre-formatted std::string / literals).
template <typename T>
inline void log_write(std::ostream& os, T&& value) {
    os << std::forward<T>(value);
}

// Two or more arguments: treat the first as a std::format format string.
template <typename... Args>
requires (sizeof...(Args) >= 1)
#if defined(__CUDACC__) && defined(_MSC_VER)
// MSVC's format-argument aggregate initialization triggers an nvcc 12.8
// host-codegen ICE. CUDA logging call sites use plain {} placeholders; stream
// those values without instantiating the MSVC format machinery in a .cu file.
inline void log_write(std::ostream& os, std::string_view fmt, Args&&... args) {
    const auto append = [&](const auto& value) {
        const auto placeholder = fmt.find("{}");
        if (placeholder == std::string_view::npos)
            throw std::invalid_argument("CUDA log format requires {} placeholders");
        os << fmt.substr(0, placeholder) << value;
        fmt.remove_prefix(placeholder + 2);
    };
    (append(args), ...);
    if (fmt.find("{}") != std::string_view::npos)
        throw std::invalid_argument("CUDA log format has missing arguments");
    os << fmt;
}
#else
inline void log_write(std::ostream& os, std::format_string<Args...> fmt, Args&&... args) {
    os << std::format(fmt, std::forward<Args>(args)...);
}
#endif

} // namespace tinytensor

#define LOG_INFO(...) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::info)) { \
            std::cout << "[INFO] "; \
            ::tinytensor::log_write(std::cout, __VA_ARGS__); \
            std::cout << std::endl; \
        } \
    } while (0)

#define LOG_WARN(...) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::warn)) { \
            std::cout << "[WARN] "; \
            ::tinytensor::log_write(std::cout, __VA_ARGS__); \
            std::cout << std::endl; \
        } \
    } while (0)

#define LOG_ERROR(...) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::error)) { \
            std::cerr << "[ERROR] "; \
            ::tinytensor::log_write(std::cerr, __VA_ARGS__); \
            std::cerr << std::endl; \
        } \
    } while (0)

#define LOG_DEBUG(...) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::debug)) { \
            std::cout << "[DEBUG] "; \
            ::tinytensor::log_write(std::cout, __VA_ARGS__); \
            std::cout << std::endl; \
        } \
    } while (0)

#define LOG_TRACE(...) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::trace)) { \
            std::cout << "[TRACE] "; \
            ::tinytensor::log_write(std::cout, __VA_ARGS__); \
            std::cout << std::endl; \
        } \
    } while (0)
