#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#if defined(TINYTENSOR_USE_FMT_LOGGING)
#include <fmt/format.h>
#else
#include <format>
#endif
#include <iostream>
#include <string>
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
#if defined(TINYTENSOR_USE_FMT_LOGGING)
// Avoid CUDA 12.8's internal compiler error in MSVC std::format when
// --expt-relaxed-constexpr is enabled; retain the same formatting syntax.
inline void log_write(std::ostream& os, fmt::format_string<Args...> pattern, Args&&... args) {
    os << fmt::format(pattern, std::forward<Args>(args)...);
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
