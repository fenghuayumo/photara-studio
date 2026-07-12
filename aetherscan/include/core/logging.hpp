#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cctype>
#include <cstdint>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <mutex>
#include <sstream>
#include <stop_token>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <utility>

namespace aetherscan::core {

enum class LogLevel : std::uint8_t { error, warning, info, debug, trace, off };

inline std::string_view log_level_name(const LogLevel level) noexcept {
    switch (level) {
        case LogLevel::error: return "error";
        case LogLevel::warning: return "warning";
        case LogLevel::info: return "info";
        case LogLevel::debug: return "debug";
        case LogLevel::trace: return "trace";
        case LogLevel::off: return "off";
    }
    return "unknown";
}

inline LogLevel parse_log_level(std::string value, const LogLevel fallback) {
    std::transform(
        value.begin(), value.end(), value.begin(),
        [](const unsigned char c) { return static_cast<char>(std::tolower(c)); });
    if (value == "error") return LogLevel::error;
    if (value == "warning" || value == "warn") return LogLevel::warning;
    if (value == "info") return LogLevel::info;
    if (value == "debug") return LogLevel::debug;
    if (value == "trace") return LogLevel::trace;
    if (value == "off") return LogLevel::off;
    return fallback;
}

inline std::tm local_time(const std::time_t value) {
    std::tm result{};
#if defined(_WIN32)
    localtime_s(&result, &value);
#else
    localtime_r(&value, &result);
#endif
    return result;
}

class Logger {
public:
    static Logger& instance() {
        static Logger logger;
        return logger;
    }

    std::filesystem::path configure(
        const std::filesystem::path& directory,
        const std::string_view prefix = "aetherscan",
        const LogLevel console_level = LogLevel::info,
        const LogLevel file_level = LogLevel::trace) {
        const auto now = std::chrono::system_clock::now();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        const std::tm tm = local_time(time);
        std::ostringstream name;
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        name << prefix << '-' << std::put_time(&tm, "%Y%m%d-%H%M%S") << '-'
             << std::setfill('0') << std::setw(3) << milliseconds.count() << ".log";
        std::lock_guard lock(mutex_);
        console_level_ = console_level;
        file_level_ = file_level;
        file_.close();
        path_.clear();
        if (!directory.empty()) {
            std::filesystem::create_directories(directory);
            path_ = directory / name.str();
            file_.open(path_, std::ios::out | std::ios::app);
            if (!file_) {
                path_.clear();
                throw std::runtime_error("Failed to create log file");
            }
        }
        return path_;
    }

    void set_levels(const LogLevel console_level, const LogLevel file_level) {
        std::lock_guard lock(mutex_);
        console_level_ = console_level;
        file_level_ = file_level;
    }

    [[nodiscard]] std::filesystem::path path() const {
        std::lock_guard lock(mutex_);
        return path_;
    }

    template <class... Values>
    void log(const LogLevel level, Values&&... values) {
        if (!enabled(level)) return;
        std::ostringstream message;
        (message << ... << std::forward<Values>(values));
        write(level, message.str());
    }

    template <class... Values> void error(Values&&... values) {
        log(LogLevel::error, std::forward<Values>(values)...);
    }
    template <class... Values> void warning(Values&&... values) {
        log(LogLevel::warning, std::forward<Values>(values)...);
    }
    template <class... Values> void info(Values&&... values) {
        log(LogLevel::info, std::forward<Values>(values)...);
    }
    template <class... Values> void debug(Values&&... values) {
        log(LogLevel::debug, std::forward<Values>(values)...);
    }
    template <class... Values> void trace(Values&&... values) {
        log(LogLevel::trace, std::forward<Values>(values)...);
    }

private:
    Logger() = default;

    [[nodiscard]] bool enabled(const LogLevel level) const noexcept {
        return level <= console_level_.load(std::memory_order_relaxed) ||
               level <= file_level_.load(std::memory_order_relaxed);
    }

    void write(const LogLevel level, const std::string& message) {
        const auto now = std::chrono::system_clock::now();
        const std::time_t time = std::chrono::system_clock::to_time_t(now);
        const std::tm tm = local_time(time);
        const auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(
            now.time_since_epoch()) % 1000;
        std::ostringstream line;
        line << std::put_time(&tm, "%Y-%m-%d %H:%M:%S") << '.'
             << std::setfill('0') << std::setw(3) << milliseconds.count()
             << " [" << log_level_name(level) << "] " << message << '\n';

        std::lock_guard lock(mutex_);
        if (level <= console_level_.load(std::memory_order_relaxed)) {
            std::ostream& output = level <= LogLevel::warning ? std::cerr : std::cout;
            output << line.str();
            output.flush();
        }
        if (file_ && level <= file_level_.load(std::memory_order_relaxed)) {
            file_ << line.str();
            file_.flush();
        }
    }

    mutable std::mutex mutex_;
    std::atomic<LogLevel> console_level_{LogLevel::warning};
    std::atomic<LogLevel> file_level_{LogLevel::off};
    std::ofstream file_;
    std::filesystem::path path_;
};

class StageScope {
public:
    explicit StageScope(std::string name, const LogLevel level = LogLevel::info)
        : name_(std::move(name)), level_(level), started_(Clock::now()) {
        Logger::instance().log(level_, "stage started: ", name_);
    }
    ~StageScope() { finish(); }
    StageScope(const StageScope&) = delete;
    StageScope& operator=(const StageScope&) = delete;

    void finish(const std::string_view summary = {}) {
        if (finished_) return;
        finished_ = true;
        const double seconds = std::chrono::duration<double>(Clock::now() - started_).count();
        Logger::instance().log(
            level_, "stage finished: ", name_, " elapsed_s=", std::fixed,
            std::setprecision(3), seconds,
            summary.empty() ? std::string{} : " " + std::string(summary));
    }

private:
    using Clock = std::chrono::steady_clock;
    std::string name_;
    LogLevel level_;
    Clock::time_point started_;
    bool finished_{false};
};

class ProgressReporter {
public:
    ProgressReporter(
        std::string label, const std::uint64_t total,
        const std::chrono::milliseconds interval = std::chrono::seconds(1),
        const LogLevel level = LogLevel::info)
        : label_(std::move(label)), total_(total), interval_(interval), level_(level),
          started_(Clock::now()), reporter_([this](const std::stop_token stop) {
              reporter_loop(stop);
          }) {
        Logger::instance().log(level_, "progress started: ", label_, " total=", total_);
    }

    ~ProgressReporter() { finish(); }
    ProgressReporter(const ProgressReporter&) = delete;
    ProgressReporter& operator=(const ProgressReporter&) = delete;

    void advance(const std::uint64_t count = 1) noexcept {
        completed_.fetch_add(count, std::memory_order_relaxed);
    }

    void finish() {
        bool expected = false;
        if (!finished_.compare_exchange_strong(expected, true)) return;
        condition_.notify_all();
        reporter_.request_stop();
        if (reporter_.joinable()) reporter_.join();
        report(true);
    }

    [[nodiscard]] std::uint64_t completed() const noexcept {
        return completed_.load(std::memory_order_relaxed);
    }

private:
    using Clock = std::chrono::steady_clock;

    void reporter_loop(const std::stop_token stop) {
        std::mutex wait_mutex;
        std::unique_lock lock(wait_mutex);
        while (!stop.stop_requested() && !finished_.load(std::memory_order_relaxed)) {
            condition_.wait_for(lock, stop, interval_, [this] {
                return finished_.load(std::memory_order_relaxed);
            });
            if (!stop.stop_requested() && !finished_.load(std::memory_order_relaxed))
                report(false);
        }
    }

    void report(const bool final) {
        const std::uint64_t raw = completed_.load(std::memory_order_relaxed);
        const std::uint64_t done = total_ == 0 ? raw : std::min(raw, total_);
        const double elapsed = std::max(
            std::chrono::duration<double>(Clock::now() - started_).count(), 1e-9);
        const double rate = static_cast<double>(done) / elapsed;
        const double percent = total_ == 0
            ? 0.0
            : 100.0 * static_cast<double>(done) / static_cast<double>(total_);
        std::ostringstream metrics;
        metrics << " completed=" << done << '/' << total_
                << " percent=" << std::fixed << std::setprecision(1) << percent
                << " items/s=" << std::setprecision(2) << rate
                << " elapsed_s=" << elapsed << " eta_s=";
        if (total_ == 0 || !(rate > 0.0))
            metrics << "n/a";
        else
            metrics << static_cast<double>(total_ - done) / rate;
        Logger::instance().log(
            level_, final ? "progress finished: " : "progress: ", label_,
            metrics.str());
    }

    std::string label_;
    std::uint64_t total_{};
    std::chrono::milliseconds interval_;
    LogLevel level_;
    std::atomic<std::uint64_t> completed_{0};
    std::atomic<bool> finished_{false};
    Clock::time_point started_;
    std::condition_variable_any condition_;
    std::jthread reporter_;
};

}  // namespace aetherscan::core
