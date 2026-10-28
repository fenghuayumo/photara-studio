#include "../../raw_video_stream.hpp"
#include <chrono>
#include <fstream>
#include <iostream>
#include <iterator>
#include <string>

using Clock = std::chrono::steady_clock;
void require(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
std::wstring quote(const std::filesystem::path& path) { return L"\"" + path.wstring() + L"\""; }
class Child {
public:
    explicit Child(std::wstring command) {
        STARTUPINFOW startup{};
        startup.cb = sizeof(startup);
        PROCESS_INFORMATION info{};
        require(CreateProcessW(nullptr, command.data(), nullptr, nullptr, FALSE,
            CREATE_NO_WINDOW, nullptr, nullptr, &startup, &info), "Cannot start FFmpeg");
        process = info.hProcess;
        CloseHandle(info.hThread);
    }
    ~Child() {
        if (WaitForSingleObject(process, 0) == WAIT_TIMEOUT) {
            TerminateProcess(process, 2);
            WaitForSingleObject(process, INFINITE);
        }
        CloseHandle(process);
    }
    bool running() const { return WaitForSingleObject(process, 0) == WAIT_TIMEOUT; }
    void stop() { TerminateProcess(process, 2); }
    void finish() {
        require(WaitForSingleObject(process, 10000) == WAIT_OBJECT_0, "Encoder did not finish without pipe EOF");
        DWORD code = 1;
        GetExitCodeProcess(process, &code);
        require(code == 0, "FFmpeg failed");
    }
private:
    HANDLE process{};
};

void encode(const std::filesystem::path& ffmpeg, const std::filesystem::path& folder,
            int count, const std::wstring& codec, const std::wstring& suffix) {
    editor::RawVideoStream stream(folder / "frames.rgb");
    const auto output = folder / (L"roundtrip_" + std::to_wstring(count) + suffix);
    Child child(quote(ffmpeg) + L" -nostdin -hide_banner -loglevel error -y -probesize 32 -analyzeduration 0 -fpsprobesize 0"
        L" -f rawvideo -pixel_format rgb24 -video_size 320x240 -framerate 30 -i " + quote(stream.path()) +
        L" -frames:v " + std::to_wstring(count) + codec + L" " + quote(output));
    const auto deadline = Clock::now() + std::chrono::seconds(15);
    const auto check = [&] {
        require(Clock::now() < deadline, "Pipe operation timed out");
        require(child.running(), "Encoder exited early");
    };
    stream.connect(check);
    std::vector<std::uint8_t> frame(320 * 240 * 3), expected;
    for (int n = 0; n < count; ++n) {
        for (std::size_t i = 0; i < frame.size(); ++i)
            frame[i] = static_cast<std::uint8_t>((i * 31 + n * 17) % 256);
        if (suffix == L".rgb") expected.insert(expected.end(), frame.begin(), frame.end());
        stream.write(frame, check);
    }
    stream.finish([&] { require(Clock::now() < deadline, "Pipe drain timed out"); }, [&] { child.stop(); });
    child.finish();
    if (suffix == L".rgb") {
        std::ifstream input(output, std::ios::binary);
        std::vector<std::uint8_t> actual((std::istreambuf_iterator<char>(input)), {});
        require(actual == expected, "Frame order, colors or frame bytes changed");
    } else require(std::filesystem::file_size(output) > 0, "Encoded video missing");
}

template<class Operation> void expect_cancel(Operation operation) {
    const auto start = Clock::now();
    const auto cancel = [&] {
        if (Clock::now() - start > std::chrono::milliseconds(80)) throw std::runtime_error("Cancelled");
    };
    bool cancelled = false;
    try { operation(cancel); }
    catch (const std::runtime_error& e) { cancelled = std::string(e.what()) == "Cancelled"; }
    require(cancelled, "Expected cancellation");
    require(Clock::now() - start < std::chrono::seconds(2), "Cancellation blocked");
}

int wmain(int argc, wchar_t** argv) try {
    require(argc == 3, "Usage: video_stream_test ffmpeg output-folder");
    const std::filesystem::path ffmpeg(argv[1]), folder(argv[2]);
    std::filesystem::create_directories(folder);
    // One frame catches input probing/EOF deadlocks; 40 frames exceed pipe capacity.
    encode(ffmpeg, folder, 1, L" -c:v rawvideo -f rawvideo", L".rgb");
    encode(ffmpeg, folder, 40, L" -c:v rawvideo -f rawvideo", L".rgb");
    encode(ffmpeg, folder, 5, L" -c:v libx264 -pix_fmt yuv420p -movflags +faststart", L".mp4");
    encode(ffmpeg, folder, 5, L" -c:v libx265 -tag:v hvc1 -pix_fmt yuv420p -movflags +faststart", L"_hevc.mp4");
    {
        editor::RawVideoStream stream(folder / "waiting.rgb");
        expect_cancel([&](auto check) { stream.connect(check); });
    }
    {
        editor::RawVideoStream stream(folder / "blocked.rgb");
        HANDLE reader = CreateFileW(stream.path().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        require(reader != INVALID_HANDLE_VALUE, "Cannot open stalled reader");
        try {
            stream.connect([] {});
            expect_cancel([&](auto check) { stream.write(std::vector<std::uint8_t>(8 * 1024 * 1024), check); });
        } catch (...) { CloseHandle(reader); throw; }
        CloseHandle(reader);
    }
    {
        editor::RawVideoStream stream(folder / "draining.rgb");
        HANDLE reader = CreateFileW(stream.path().c_str(), GENERIC_READ, 0, nullptr, OPEN_EXISTING, 0, nullptr);
        require(reader != INVALID_HANDLE_VALUE, "Cannot open drain reader");
        try {
            stream.connect([] {});
            stream.write(std::vector<std::uint8_t>(32), [] {});
            expect_cancel([&](auto check) {
                stream.finish(check, [&] { CloseHandle(reader); reader = INVALID_HANDLE_VALUE; });
            });
        } catch (...) { if (reader != INVALID_HANDLE_VALUE) CloseHandle(reader); throw; }
        if (reader != INVALID_HANDLE_VALUE) CloseHandle(reader);
    }
    {
        editor::RawVideoStream stream(folder / "bad-encoder.rgb");
        Child child(quote(ffmpeg) + L" -photara-invalid-option");
        const auto deadline = Clock::now() + std::chrono::seconds(3);
        bool detected = false;
        try {
            stream.connect([&] {
                require(Clock::now() < deadline, "Encoder exit was not detected");
                if (!child.running()) throw std::runtime_error("Encoder exited");
            });
        } catch (const std::runtime_error& e) { detected = std::string(e.what()) == "Encoder exited"; }
        require(detected, "Encoder failure was not propagated");
    }
    // Independent export jobs must have independent pipe names.
    {
        editor::RawVideoStream a(folder / "a" / "frames.rgb"), b(folder / "b" / "frames.rgb");
        require(a.path() != b.path(), "Video pipe names collide");
    }
    std::cout << "Frame roundtrip, H.264/H.265, cancellation, backpressure, encoder failure and pipe isolation passed\n";
    return 0;
} catch (const std::exception& error) {
    std::cerr << error.what() << '\n';
    return 1;
}
