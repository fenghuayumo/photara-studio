#pragma once

#include <iostream>
#include <format>

// TinyTensor Logger - Minimal logging for the standalone library
namespace tinytensor {

// Simple logging macros for compatibility
#define LOG_INFO(msg) std::cout << "[INFO] " << msg << std::endl
#define LOG_WARN(msg) std::cout << "[WARN] " << msg << std::endl
#define LOG_ERROR(msg) std::cerr << "[ERROR] " << msg << std::endl
#define LOG_DEBUG(msg) std::cout << "[DEBUG] " << msg << std::endl
#define LOG_TRACE(msg) std::cout << "[TRACE] " << msg << std::endl

// Template logging with format support
template<typename... Args>
void log_info(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << "[INFO] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

template<typename... Args>
void log_warn(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << "[WARN] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

template<typename... Args>
void log_error(std::format_string<Args...> fmt, Args&&... args) {
    std::cerr << "[ERROR] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

template<typename... Args>
void log_debug(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << "[DEBUG] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

template<typename... Args>
void log_trace(std::format_string<Args...> fmt, Args&&... args) {
    std::cout << "[TRACE] " << std::format(fmt, std::forward<Args>(args)...) << std::endl;
}

} // namespace tinytensor
