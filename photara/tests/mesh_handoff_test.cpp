#include "mvs/mesh_handoff.hpp"
#include "mvs/export.hpp"
#include "io/shared_mapping.hpp"

#include <chrono>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#else
#include <sys/wait.h>
#include <unistd.h>
#endif

namespace {
void require(bool ok, const char* text) {
    if (!ok) throw std::runtime_error(text);
}

photara::mvs::Mesh make_mesh(unsigned flags) {
    photara::mvs::Mesh mesh;
    mesh.vertices = {{0.F, 0.F, 0.F}, {1.F, 0.F, 0.F}, {0.F, 1.F, 0.F}};
    mesh.faces = {{0, 1, 2}};
    if (flags & 1U) mesh.normals.assign(3, {0.F, 0.F, 1.F});
    if (flags & 2U) mesh.colors.assign(3, {0.25F, 0.5F, 1.F});
    return mesh;
}

void compare(const photara::mvs::Mesh& a, const photara::mvs::Mesh& b, unsigned) {
    require(photara::mvs::encode_mesh(a) == photara::mvs::encode_mesh(b),
            "Shared memory changed mesh attributes or face indices");
}

void expect_rejected(const std::filesystem::path& ready) {
    bool rejected = false;
    try { photara::mvs::receive_mesh_handoff(ready); } catch (const std::exception&) { rejected = true; }
    require(rejected, "Invalid mesh handoff was accepted");
}

}

int main(int argc, char** argv) {
    try {
        using namespace photara::mvs;
        if (argc == 4 && std::string(argv[1]) == "--publish") {
            const std::filesystem::path ready(argv[2]);
            const unsigned flags = std::stoul(argv[3]);
            MeshHandoff publisher(make_mesh(flags), ready);
            save_mesh_ply(make_mesh(flags), ready.parent_path() / "saved.ply");
            require(publisher.wait_for_reader(std::chrono::seconds(10)), "Reader did not attach");
            return 0;
        }
        const auto root = std::filesystem::temp_directory_path() /
            ("photara_mesh_handoff_test_" + std::to_string(
                std::chrono::steady_clock::now().time_since_epoch().count()));
        std::filesystem::create_directories(root);
        const auto exe = std::filesystem::absolute(argv[0]);
        for (unsigned flags = 0; flags < 4; ++flags) {
            const auto ready = root / ("mesh_" + std::to_string(flags) + ".ready");
#ifdef _WIN32
            std::wstring command = L"\"" + exe.wstring() + L"\" --publish \"" +
                ready.wstring() + L"\" " + std::to_wstring(flags);
            STARTUPINFOW startup{}; startup.cb = sizeof(startup);
            PROCESS_INFORMATION process{};
            require(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE,
                                  CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process),
                    "Cannot start publisher process");
            CloseHandle(process.hThread);
#else
            const auto child = fork();
            require(child >= 0, "Cannot fork publisher process");
            if (child == 0) {
                const auto flag = std::to_string(flags);
                execl(exe.c_str(), exe.c_str(), "--publish", ready.c_str(), flag.c_str(), nullptr);
                _exit(3);
            }
#endif
            const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
            while (!std::filesystem::exists(ready) && std::chrono::steady_clock::now() < deadline)
                std::this_thread::sleep_for(std::chrono::milliseconds(5));
            auto received = receive_mesh_handoff(ready);
#ifdef _WIN32
            require(WaitForSingleObject(process.hProcess, 15000) == WAIT_OBJECT_0,
                    "Publisher failed to exit after reader acknowledgement");
            DWORD code{};
            GetExitCodeProcess(process.hProcess, &code);
            CloseHandle(process.hProcess);
            require(code == 0, "Publisher failed");
#else
            int status{};
            require(waitpid(child, &status, 0) == child && WIFEXITED(status) && WEXITSTATUS(status) == 0,
                    "Publisher failed");
#endif
            // The publisher and its mapping are gone. The received mesh owns
            // its data; no disk mesh parser was needed to receive it.
            compare(make_mesh(flags), received, flags);
            bool expired = false;
            try { receive_mesh_handoff(ready); } catch (const std::exception&) { expired = true; }
            require(expired, "Publisher mapping was leaked");
            const auto saved = load_mesh_ply(root / "saved.ply");
            require(saved.vertices == received.vertices && saved.faces == received.faces &&
                saved.normals == received.normals, "Saved PLY fallback changed geometry");
        }
        const auto invalid = root / "invalid.ready";
        { std::ofstream out(invalid); out << "PHOTARA_MESH_HANDOFF 1\nmissing\n8\n"; }
        bool rejected = false;
        try { receive_mesh_handoff(invalid); } catch (const std::exception&) { rejected = true; }
        require(rejected, "Truncated manifest was accepted");
        // Exercise malformed mapped headers before any large allocations, and
        // invalid attributes/indices after decoding an otherwise valid layout.
        const auto malformed = root / "malformed.ready";
        for (unsigned variant = 0; variant < 7; ++variant) {
            auto bytes = encode_mesh(make_mesh(3));
            if (variant == 0) { const std::uint64_t n = 100'000'001; std::memcpy(bytes.data() + 4, &n, 8); }
            if (variant == 1) { const std::uint32_t f = 4; std::memcpy(bytes.data() + 20, &f, 4); }
            if (variant == 2) bytes.pop_back();
            if (variant == 3) { const float f = std::numeric_limits<float>::infinity(); std::memcpy(bytes.data() + 24, &f, 4); }
            if (variant == 4) { const std::int32_t i = 3; std::memcpy(bytes.data() + bytes.size() - 4, &i, 4); }
            if (variant == 5) { const std::uint32_t v = 99; std::memcpy(bytes.data(), &v, 4); }
            if (variant == 6) bytes.push_back(0);
            photara::io::detail::Mapping mapping;
#ifdef _WIN32
            const auto name = "Local\\Photara.mesh.test." + std::to_string(GetCurrentProcessId());
#else
            const auto name = "/photara_mesh_test_" + std::to_string(getpid());
#endif
            mapping.open(name, bytes.size(), true);
            std::memcpy(mapping.data, bytes.data(), bytes.size());
            photara::io::detail::publish_text(malformed,
                "PHOTARA_MESH_HANDOFF 1\n" + name + "\n" + std::to_string(bytes.size()) + "\n");
            expect_rejected(malformed);
        }
        for (unsigned variant = 0; variant < 4; ++variant) {
            auto mesh = make_mesh(3);
            if (variant == 0) mesh.normals.pop_back();
            if (variant == 1) mesh.vertices[0].x() = std::numeric_limits<float>::quiet_NaN();
            if (variant == 2) mesh.faces[0].x() = -1;
            if (variant == 3) mesh.faces.clear();
            bool invalid_source = false;
            try { MeshHandoff publisher(mesh, malformed); } catch (const std::exception&) { invalid_source = true; }
            require(invalid_source, "Invalid source mesh was published");
        }
        const auto timeout_ready = root / "timeout.ready";
        {
            MeshHandoff publisher(make_mesh(0), timeout_ready);
            require(!publisher.wait_for_reader(std::chrono::milliseconds(1)), "Missing reader was acknowledged");
            { std::ofstream out(photara::io::detail::ack_path(timeout_ready)); out << "stale-name\n"; }
            require(!publisher.wait_for_reader(std::chrono::milliseconds(1)), "Stale acknowledgment was accepted");
        }
        expect_rejected(timeout_ready);
        require(std::filesystem::weakly_canonical(root).parent_path() ==
                    std::filesystem::weakly_canonical(std::filesystem::temp_directory_path()),
                "Test cleanup escaped its temporary directory");
        std::filesystem::remove_all(root);
        std::cout << "Cross-process handoff, all optional attributes, publisher exit, PLY fallback and invalid manifests passed\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
