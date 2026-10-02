#pragma once

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__APPLE__)
#include <mach-o/dyld.h>
#else
#include <unistd.h>
#endif

#ifndef PHOTARA_CLI_FILENAME
#if defined(_WIN32)
#define PHOTARA_CLI_FILENAME "photara.exe"
#else
#define PHOTARA_CLI_FILENAME "photara"
#endif
#endif

namespace editor {

// Resolve from the running Studio, never the build tree or working directory.
inline const std::string& cli_path() {
    static const std::string path = [] {
        std::filesystem::path executable;
#if defined(_WIN32)
        std::vector<wchar_t> buffer(256);
        for (;;) {
            const DWORD size = GetModuleFileNameW(
                nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
            if (size == 0)
                throw std::runtime_error("Cannot locate Studio executable");
            if (size < buffer.size()) {
                executable = std::wstring(buffer.data(), size);
                break;
            }
            buffer.resize(buffer.size() * 2);
        }
#elif defined(__APPLE__)
        std::uint32_t size = 0;
        _NSGetExecutablePath(nullptr, &size);
        std::vector<char> buffer(size);
        if (_NSGetExecutablePath(buffer.data(), &size) != 0)
            throw std::runtime_error("Cannot locate Studio executable");
        executable = std::filesystem::canonical(buffer.data());
#else
        executable = std::filesystem::read_symlink("/proc/self/exe");
#endif
        const auto cli = executable.parent_path() / PHOTARA_CLI_FILENAME;
        std::error_code error;
        if (!std::filesystem::is_regular_file(cli, error)) {
            const auto utf8 = cli.u8string();
            throw std::runtime_error(
                "Cannot find Photara CLI: " +
                std::string(utf8.begin(), utf8.end()));
        }
        const auto utf8 = cli.u8string();
        return std::string(utf8.begin(), utf8.end());
    }();
    return path;
}

} // namespace editor
