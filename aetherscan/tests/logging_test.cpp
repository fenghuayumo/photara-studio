#include "core/logging.hpp"

#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

int main() {
    namespace fs = std::filesystem;
    using namespace aetherscan::core;
    const fs::path directory = fs::temp_directory_path() / "aetherscan-logging-test";
    std::error_code ignored;
    fs::remove_all(directory, ignored);
    const fs::path log_path = Logger::instance().configure(
        directory, "logging-test", LogLevel::off, LogLevel::trace);
    Logger::instance().error("error-message");
    Logger::instance().warning("warning-message");
    Logger::instance().info("info-message");
    Logger::instance().debug("debug-message");
    Logger::instance().trace("trace-message");
    {
        ProgressReporter progress("parallel-work", 400, std::chrono::milliseconds(5));
        std::vector<std::jthread> workers;
        for (int worker = 0; worker < 4; ++worker) {
            workers.emplace_back([&] {
                for (int item = 0; item < 100; ++item) progress.advance();
            });
        }
    }
    std::ifstream input(log_path);
    const std::string contents{
        std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>()};
    const bool valid = contents.find("[error] error-message") != std::string::npos &&
        contents.find("[warning] warning-message") != std::string::npos &&
        contents.find("[info] info-message") != std::string::npos &&
        contents.find("[debug] debug-message") != std::string::npos &&
        contents.find("[trace] trace-message") != std::string::npos &&
        contents.find("progress finished: parallel-work completed=400/400") !=
            std::string::npos;
    Logger::instance().configure({}, "logging-test", LogLevel::off, LogLevel::off);
    fs::remove_all(directory, ignored);
    if (!valid) {
        std::cerr << "logger/progress output validation failed\n";
        return 1;
    }
    return 0;
}
