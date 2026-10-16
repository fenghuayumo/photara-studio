#include "sam_model.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "theme.hpp"

#include "sam/model_cache.hpp"

#include "imgui.h"

#include <cctype>
#include <cstdlib>
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
#else
#include <cerrno>
#include <fcntl.h>
#include <signal.h>
#include <spawn.h>
#include <sys/wait.h>
#include <unistd.h>
extern char** environ;
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
    const char* argv[] = {
        "xdg-open", photara::sam::k_license_url, nullptr};
    pid_t pid = -1;
    const int spawned = ::posix_spawnp(
        &pid, "xdg-open", nullptr, nullptr, const_cast<char**>(argv), environ);
    int status = 1;
    if (spawned == 0 && pid > 0) {
        while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
        }
    }
    if (spawned == 0 && WIFEXITED(status) && WEXITSTATUS(status) == 0)
        return true;
    ImGui::SetClipboardText(photara::sam::k_license_url);
    return false;
}
#endif

}  // namespace

void SamModelDownload::start() {
    ModelDownloadSpec spec;
    spec.url=photara::sam::k_model_url;
    spec.destination=photara::sam::user_model_path();
    spec.minimum_bytes=photara::sam::k_model_bytes/2+1;
    spec.ready_message="SAM 3 model is ready";
    ModelDownload::start(std::move(spec));
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
