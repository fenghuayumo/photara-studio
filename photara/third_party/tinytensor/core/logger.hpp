#pragma once

#include <algorithm>
#include <cctype>
#include <cstdlib>
#include <iostream>
#include <string>

// TinyTensor Logger - Minimal logging for the standalone library
//
// Default level is info: LOG_DEBUG / LOG_TRACE are silent unless enabled via
// TINYTENSOR_LOG_LEVEL (or PHOTARA_LOG_LEVEL) = debug|trace.
//
// Macros take a single streamable message (string / ostringstream / concat).
// Keep this shape for CUDA/MSVC host compilation compatibility.
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

} // namespace tinytensor

#define LOG_INFO(msg) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::info)) { \
            std::cout << "[INFO] " << (msg) << std::endl; \
        } \
    } while (0)

#define LOG_WARN(msg) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::warn)) { \
            std::cout << "[WARN] " << (msg) << std::endl; \
        } \
    } while (0)

#define LOG_ERROR(msg) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::error)) { \
            std::cerr << "[ERROR] " << (msg) << std::endl; \
        } \
    } while (0)

#define LOG_DEBUG(msg) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::debug)) { \
            std::cout << "[DEBUG] " << (msg) << std::endl; \
        } \
    } while (0)

#define LOG_TRACE(msg) \
    do { \
        if (::tinytensor::log_enabled(::tinytensor::LogLevel::trace)) { \
            std::cout << "[TRACE] " << (msg) << std::endl; \
        } \
    } while (0)
