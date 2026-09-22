#include "sam_model.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "theme.hpp"

#include "sam/model_cache.hpp"

#include "imgui.h"

#include <cctype>
#include <filesystem>
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
#include <shellapi.h>
#endif

namespace editor {
namespace {

using i18n::tr;

#if defined(_WIN32)
std::wstring utf8_to_wide(const std::string& text) {
    if (text.empty()) return {};
    const int size = MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), nullptr, 0);
    std::wstring wide(static_cast<std::size_t>(size), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, text.data(), static_cast<int>(text.size()), wide.data(),
        size);
    return wide;
}

std::filesystem::path curl_executable() {
    wchar_t system[MAX_PATH]{};
    const UINT length = GetSystemDirectoryW(system, MAX_PATH);
    if (length > 0 && length < MAX_PATH) {
        const auto candidate =
            std::filesystem::path(system) / L"curl.exe";
        std::error_code error;
        if (std::filesystem::is_regular_file(candidate, error)) return candidate;
    }
    return L"curl.exe";
}

bool open_license_url() {
    const auto wide = utf8_to_wide(photara::sam::k_license_url);
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(
        nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (result > 32) return true;
    ImGui::SetClipboardText(photara::sam::k_license_url);
    return false;
}
#else
bool open_license_url() {
    ImGui::SetClipboardText(photara::sam::k_license_url);
    return false;
}
#endif

}  // namespace

SamModelDownload::~SamModelDownload() {
    cancel();
    if (worker_.joinable()) worker_.join();
}

void SamModelDownload::start() {
    if (state() == State::running) return;
    if (worker_.joinable()) worker_.join();
    cancel_ = false;
    progress_ = -1.F;
    {
        std::lock_guard lock(mutex_);
        status_ = "Starting download";
    }
    state_ = static_cast<int>(State::running);
    worker_ = std::thread([this] { run(); });
}

void SamModelDownload::cancel() { cancel_ = true; }

SamModelDownload::State SamModelDownload::state() const {
    return static_cast<State>(state_.load());
}

float SamModelDownload::progress() const { return progress_.load(); }

std::string SamModelDownload::status() const {
    std::lock_guard lock(mutex_);
    return status_;
}

void SamModelDownload::run() {
    auto fail = [&](const std::string& why, const State terminal) {
        std::lock_guard lock(mutex_);
        status_ = why;
        state_ = static_cast<int>(terminal);
    };

    const auto destination = photara::sam::user_model_path();
    std::error_code error;
    std::filesystem::create_directories(destination.parent_path(), error);
    if (error) {
        fail("Cannot create the model directory", State::failed);
        return;
    }
    auto partial = destination;
    partial += ".part";

#if !defined(_WIN32)
    fail("SAM model download is implemented for Windows curl", State::failed);
    return;
#else
    SECURITY_ATTRIBUTES security{};
    security.nLength = sizeof(security);
    security.bInheritHandle = TRUE;
    HANDLE read_pipe = nullptr;
    HANDLE write_pipe = nullptr;
    if (!CreatePipe(&read_pipe, &write_pipe, &security, 0)) {
        fail("Cannot capture curl output", State::failed);
        return;
    }
    SetHandleInformation(read_pipe, HANDLE_FLAG_INHERIT, 0);

    STARTUPINFOW startup{};
    startup.cb = sizeof(startup);
    startup.dwFlags = STARTF_USESTDHANDLES;
    startup.hStdInput = nullptr;
    startup.hStdOutput = write_pipe;
    startup.hStdError = write_pipe;
    PROCESS_INFORMATION process{};

    const std::wstring command =
        L"\"" + curl_executable().wstring() +
        L"\" -L -f --progress-bar -C - -o \"" + partial.wstring() + L"\" \"" +
        utf8_to_wide(photara::sam::k_model_url) + L"\"";
    std::vector<wchar_t> mutable_command(command.begin(), command.end());
    mutable_command.push_back(L'\0');
    const BOOL started = CreateProcessW(
        nullptr, mutable_command.data(), nullptr, nullptr, TRUE,
        CREATE_NO_WINDOW, nullptr, nullptr, &startup, &process);
    CloseHandle(write_pipe);
    if (!started) {
        CloseHandle(read_pipe);
        fail("curl was not found. Install curl or download the model by hand.",
             State::failed);
        return;
    }

    std::string pending;
    char buffer[256];
    while (!cancel_) {
        DWORD available = 0;
        if (!PeekNamedPipe(read_pipe, nullptr, 0, nullptr, &available, nullptr))
            break;
        if (available == 0) {
            if (WaitForSingleObject(process.hProcess, 50) == WAIT_OBJECT_0)
                break;
            continue;
        }
        DWORD read = 0;
        if (!ReadFile(
                read_pipe, buffer, sizeof(buffer), &read, nullptr) ||
            read == 0)
            break;
        pending.append(buffer, buffer + read);
        for (;;) {
            const auto mark = pending.find_first_of("\r\n");
            if (mark == std::string::npos) break;
            const std::string line = pending.substr(0, mark);
            pending.erase(0, mark + 1);
            const auto percent = line.find('%');
            if (percent == std::string::npos) continue;
            std::size_t start = percent;
            while (start > 0 &&
                   (std::isdigit(static_cast<unsigned char>(line[start - 1])) ||
                    line[start - 1] == '.'))
                --start;
            if (start >= percent) continue;
            const float value = std::strtof(
                line.substr(start, percent - start).c_str(), nullptr);
            progress_ = value / 100.F;
            std::lock_guard lock(mutex_);
            status_ = "Downloaded " + std::to_string(static_cast<int>(value)) + "%";
        }
    }

    if (cancel_) TerminateProcess(process.hProcess, 1);
    WaitForSingleObject(process.hProcess, 15000);
    DWORD code = 1;
    GetExitCodeProcess(process.hProcess, &code);
    CloseHandle(process.hThread);
    CloseHandle(process.hProcess);
    CloseHandle(read_pipe);

    if (cancel_) {
        fail("Download cancelled", State::cancelled);
        return;
    }
    if (code != 0) {
        std::filesystem::remove(partial, error);
        fail("Download failed (curl exit " + std::to_string(code) + ")",
             State::failed);
        return;
    }
    std::filesystem::remove(destination, error);
    error.clear();
    std::filesystem::rename(partial, destination, error);
    if (error || !photara::sam::model_file_ready(destination)) {
        fail("Downloaded file is incomplete", State::failed);
        return;
    }
    progress_ = 1.F;
    fail("SAM 3 model is ready", State::done);
#endif
}

bool ensure_sam_ready(App& app) {
    if (!app.settings.sam_masks) return true;
#if !defined(PHOTARA_HAS_SAM)
    set_message(
        app, "This build does not include SAM mask generation", theme::danger);
    return false;
#else
    if (app.settings.sam_text[0] == '\0') {
        set_message(
            app, "Enter a SAM3 prompt before running", theme::warning);
        return false;
    }
    const bool accepted = photara::sam::license_accepted();
    const bool ready = !photara::sam::locate_model().empty();
    if (accepted && ready) return true;
    app.sam_license_tick = accepted;
    app.show_sam_license = true;
    set_message(
        app,
        accepted ? "Download the SAM 3 model before running"
                 : "Accept the SAM 3 license and download the model before running",
        theme::warning);
    return false;
#endif
}

void draw_sam_license_modal(App& app) {
    constexpr const char* popup_id = "###Sam3License";
    if (app.show_sam_license) {
        ImGui::OpenPopup(popup_id);
        app.show_sam_license = false;
    }
    const ImGuiViewport* viewport = ImGui::GetMainViewport();
    ImGui::SetNextWindowPos(
        viewport->GetCenter(), ImGuiCond_Appearing, ImVec2(0.5F, 0.5F));
    ImGui::SetNextWindowSize({520.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(
            i18n::id("SAM 3 License (Meta)", popup_id), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize))
        return;

    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 490.F);
    ImGui::TextUnformatted(tr(
        "SAM 3 is Meta's model, not part of Photara, and it comes with its "
        "own licence. It is free to use, including commercially, but only on "
        "Meta's terms. Photara cannot ship the weights or accept the licence "
        "for you. Read the licence itself before continuing."));
    ImGui::PopTextWrapPos();
    ImGui::Spacing();
    if (ImGui::Button(tr("Read the license"), {160.F, 0.F})) {
        if (!open_license_url())
            set_message(
                app,
                std::string("Could not open a browser. The licence URL is on "
                            "the clipboard: ") +
                    photara::sam::k_license_url,
                theme::warning);
    }
    ImGui::SameLine();
    if (ImGui::Button(tr("Copy link"), {110.F, 0.F}))
        ImGui::SetClipboardText(photara::sam::k_license_url);
    ImGui::Spacing();
    ImGui::TextDisabled("%s", photara::sam::k_license_url);
    ImGui::TextDisabled("%s", tr("Download size about 707 MB"));
    ImGui::Spacing();
    ImGui::Checkbox(
        tr("I have read and accept the SAM 3 License"),
        &app.sam_license_tick);
    ImGui::Spacing();

    const auto phase = app.sam_download.state();
    const bool downloading = phase == SamModelDownload::State::running;
    ImGui::BeginDisabled(!app.sam_license_tick || downloading);
    if (ImGui::Button(tr("Download model"), {150.F, 0.F})) {
        try {
            photara::sam::accept_license();
            if (!photara::sam::locate_model().empty()) {
                set_message(app, "SAM 3 model is ready", theme::success);
                ImGui::CloseCurrentPopup();
            } else {
                app.sam_download.start();
            }
        } catch (const std::exception& failure) {
            set_message(app, failure.what(), theme::danger);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), {100.F, 0.F})) {
        if (downloading) app.sam_download.cancel();
        ImGui::CloseCurrentPopup();
    }
    if (downloading) {
        ImGui::Spacing();
        const float fraction = app.sam_download.progress();
        ImGui::ProgressBar(fraction < 0.F ? 0.F : fraction, {-1.F, 0.F});
        const std::string status = app.sam_download.status();
        ImGui::TextUnformatted(status.c_str());
    } else if (phase == SamModelDownload::State::done) {
        ImGui::Spacing();
        ImGui::TextColored(theme::success, "%s", tr("SAM 3 model is ready"));
    } else if (phase == SamModelDownload::State::failed) {
        ImGui::Spacing();
        const std::string status = app.sam_download.status();
        ImGui::TextColored(theme::danger, "%s", status.c_str());
    }
    ImGui::EndPopup();
}

}  // namespace editor
