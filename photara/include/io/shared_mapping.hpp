#pragma once

#include <cstdint>
#include <filesystem>
#include <cmath>
#include <cstring>
#include <fstream>
#include <limits>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <unistd.h>
#endif

namespace photara::io::detail {

inline std::filesystem::path ack_path(std::filesystem::path path) {
    path += ".ack";
    return path;
}

struct Mapping {
    Mapping() = default;
    Mapping(const Mapping&) = delete;
    Mapping& operator=(const Mapping&) = delete;
    void* data{};
    std::size_t size{};
#ifdef _WIN32
    HANDLE handle{};
#else
    int handle{-1};
    std::string owned_name;
#endif
    ~Mapping() {
#ifdef _WIN32
        if (data) UnmapViewOfFile(data);
        if (handle) CloseHandle(handle);
#else
        if (data) munmap(data, size);
        if (handle >= 0) close(handle);
        if (!owned_name.empty()) shm_unlink(owned_name.c_str());
#endif
    }
    void open(const std::string& name, const std::size_t bytes, const bool create) {
        size = bytes;
#ifdef _WIN32
        handle = create
            ? CreateFileMappingA(INVALID_HANDLE_VALUE, nullptr, PAGE_READWRITE,
                static_cast<DWORD>(static_cast<std::uint64_t>(bytes) >> 32U),
                static_cast<DWORD>(bytes), name.c_str())
            : OpenFileMappingA(FILE_MAP_READ, FALSE, name.c_str());
        if (!handle) throw std::runtime_error("Cannot open handoff shared memory");
        data = MapViewOfFile(handle, create ? FILE_MAP_WRITE : FILE_MAP_READ, 0, 0, bytes);
#else
        handle = shm_open(name.c_str(), create ? O_CREAT | O_EXCL | O_RDWR : O_RDONLY, 0600);
        if (handle < 0) throw std::runtime_error("Cannot open handoff shared memory");
        if (create) {
            owned_name = name;
            struct statvfs available{};
            if (fstatvfs(handle, &available) != 0 ||
                bytes > static_cast<std::uint64_t>(available.f_bavail) * available.f_frsize)
                throw std::runtime_error("Insufficient shared-memory capacity for payload");
            if (ftruncate(handle, static_cast<off_t>(bytes)) != 0)
                throw std::runtime_error("Cannot size handoff shared memory");
        } else {
            struct stat info{};
            if (fstat(handle, &info) != 0 || info.st_size < 0 ||
                static_cast<std::uint64_t>(info.st_size) < bytes)
                throw std::runtime_error("Truncated handoff shared memory");
        }
        data = mmap(nullptr, bytes, create ? PROT_READ | PROT_WRITE : PROT_READ,
                    MAP_SHARED, handle, 0);
        if (data == MAP_FAILED) data = nullptr;
#endif
        if (!data) throw std::runtime_error("Cannot map handoff shared memory");
    }
};

inline void publish_text(const std::filesystem::path& path, const std::string& text) {
    auto temporary = path;
    temporary += ".tmp";
    {
        std::ofstream out(temporary, std::ios::binary | std::ios::trunc);
        out << text;
        out.close();
        if (!out) throw std::runtime_error("Cannot publish shared-payload handoff");
    }
    std::error_code ignored;
    std::filesystem::remove(path, ignored);
    std::filesystem::rename(temporary, path);
}

}  // namespace photara::io::detail
