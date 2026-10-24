#include "sam_model.hpp"

#include "app.hpp"
#include "i18n.hpp"
#include "theme.hpp"

#include "sam/masks.hpp"
#include "sam/model_cache.hpp"

#include "imgui.h"

#include <algorithm>
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

bool open_license_url(const char* url) {
    const auto wide = utf8_to_wide(url);
    const auto result = reinterpret_cast<INT_PTR>(ShellExecuteW(
        nullptr, L"open", wide.c_str(), nullptr, nullptr, SW_SHOWNORMAL));
    if (result > 32) return true;
    ImGui::SetClipboardText(url);
    return false;
}
#else
bool open_license_url(const char* url) {
    const char* argv[] = {
        "xdg-open", url, nullptr};
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
    ImGui::SetClipboardText(url);
    return false;
}
#endif

}  // namespace

namespace {

std::filesystem::path selected_segmentation_path(const App& app) {
    std::filesystem::path path =
        path_from_utf8_field(app.settings.sam_model.data());
    if (!path.empty()) {
        if (const auto* entry = photara::sam::find_mask_model(path.string()))
            return photara::sam::user_model_path(entry->filename);
        return path;
    }
    const auto* entry = photara::sam::find_mask_model(
        app.settings.sam_engine == 1 ? photara::sam::k_default_sam2_model : "sam3-q4_0");
    return photara::sam::user_model_path(entry->filename);
}

const photara::sam::MaskModelEntry* selected_segmentation_entry(
    const App& app) {
    const auto path = selected_segmentation_path(app);
    const std::filesystem::path filename = path.filename();
    for (const auto& entry : photara::sam::mask_model_catalog()) {
        if (entry.kind == photara::sam::MaskModelKind::text_detector)
            continue;
        if (filename == std::filesystem::path(entry.filename)) return &entry;
    }
    return nullptr;
}

ModelDownloadSpec file_spec(
    const std::string& url,
    const std::filesystem::path& destination,
    const std::uint64_t bytes,
    const char* ready_message, const photara::sam::ModelLicense& license) {
    ModelDownloadSpec spec;
    spec.url = url;
    spec.destination = destination;
    spec.expected_bytes = bytes;
    spec.minimum_bytes = bytes;
    spec.ready_message = ready_message;
    spec.authorize = [id = std::string(license.id)] {
        photara::sam::require_model_license(*photara::sam::find_model_license(id));
    };
    return spec;
}

std::vector<ModelDownloadSpec> missing_selected_models(const App& app) {
    std::vector<ModelDownloadSpec> specs;

    const photara::sam::MaskModelEntry* entry =
        selected_segmentation_entry(app);
    if (entry != nullptr) {
        const auto path = selected_segmentation_path(app);
        if (!photara::sam::model_file_ready(path, entry->bytes))
            specs.push_back(file_spec(
                entry->url, path, entry->bytes, "Segmentation model is ready", photara::sam::model_license(entry->kind)));
    }

    if (app.settings.sam_engine != 1) return specs;
    const auto detector_file = selected_mask_detector(app);
    const photara::sam::MaskModelEntry* detector_entry = nullptr;
    for (const auto& model : photara::sam::mask_model_catalog())
        if (model.kind == photara::sam::MaskModelKind::text_detector &&
            detector_file.filename() == model.filename) detector_entry = &model;
    if (detector_entry && !photara::sam::model_file_ready(detector_file, detector_entry->bytes))
        specs.push_back(file_spec(detector_entry->url, detector_file,
            detector_entry->bytes, "Text detector is ready", photara::sam::model_license(detector_entry->kind)));
    if (photara::sam::locate_text_detector_vocab().empty())
        specs.push_back(file_spec(
            "https://huggingface.co/IDEA-Research/grounding-dino-tiny/resolve/main/vocab.txt",
            photara::sam::user_model_path(photara::sam::k_default_text_detector_vocab),
            231'508ULL, "Text detector vocabulary is ready",
            photara::sam::model_license(photara::sam::MaskModelKind::text_detector)));
    return specs;
}

std::vector<std::string> selected_agreements(const App& app) {
    auto model = selected_mask_model(app);
    if (model.empty()) model = selected_segmentation_path(app);
    std::vector<std::string> ids{photara::sam::segmentation_license(model).id};
    if (app.settings.sam_engine == 1)
        ids.push_back(photara::sam::model_license(photara::sam::MaskModelKind::text_detector).id);
    return ids;
}
bool request_agreements(App& app, bool download) {
    try {
        const auto ids = selected_agreements(app);
        bool missing = false;
        for (const auto& id : ids)
            missing |= !photara::sam::license_accepted(*photara::sam::find_model_license(id));
        if (!missing) return false;
        app.sam_license_families = ids;
        app.sam_license_checks.clear();
        for (const auto& id : ids)
            app.sam_license_checks.push_back(photara::sam::license_accepted(*photara::sam::find_model_license(id)));
        app.sam_license_download = download;
        app.show_sam_license = true;
        return true;
    } catch (const std::exception& error) {
        set_message(app, error.what(), theme::danger);
        return true;
    }
}

}  // namespace

void SamModelDownload::start() {
    ModelDownloadSpec spec;
    spec.url=photara::sam::k_model_url;
    spec.destination=photara::sam::user_model_path(photara::sam::k_model_file);
    spec.expected_bytes=photara::sam::k_model_bytes;
    spec.minimum_bytes=photara::sam::k_model_bytes;
    spec.ready_message="SAM 3 model is ready";
    spec.authorize = [] { photara::sam::require_model_license(
        photara::sam::model_license(photara::sam::MaskModelKind::sam3)); };
    ModelDownload::start(std::move(spec));
}

void start_sam_model_downloads(App& app) {
    if (app.sam_download.state() == SamModelDownload::State::running) return;
    if (request_agreements(app, true)) return;
    auto missing = missing_selected_models(app);
    if (missing.empty()) return;
    app.sam_download.pending = std::move(missing);
    auto first = std::move(app.sam_download.pending.front());
    app.sam_download.pending.erase(app.sam_download.pending.begin());
    app.sam_download.start(std::move(first));
}

void update_sam_model_downloads(App& app) {
    if (app.sam_download.state() != SamModelDownload::State::done ||
        app.sam_download.pending.empty()) return;
    auto next = std::move(app.sam_download.pending.front());
    app.sam_download.pending.erase(app.sam_download.pending.begin());
    app.sam_download.start(std::move(next));
}

std::filesystem::path selected_mask_model(const App& app) {
    const auto path = selected_segmentation_path(app);
    if (photara::sam::model_file_ready(path, 0)) return path;
    if (app.settings.sam_model[0] == '\0')
        return photara::sam::locate_model(app.settings.sam_engine == 1
            ? photara::sam::k_default_sam2_model : "sam3-q4_0");
    return photara::sam::resolve_model(path);
}

std::filesystem::path selected_mask_detector(const App& app) {
    const auto value = path_from_utf8_field(app.settings.sam_detector_model.data());
    if (!value.empty()) {
        if (const auto* entry = photara::sam::find_mask_model(value.string()))
            return photara::sam::user_model_path(entry->filename);
        return value;
    }
    const auto existing = photara::sam::locate_text_detector_model();
    return existing.empty() ? photara::sam::user_model_path(
        photara::sam::k_default_text_detector_model) : existing;
}

bool ensure_sam_ready(App& app) {
    if (!app.settings.sam_masks) return true;
    if (!photara::sam::built_with_sam()) {
        set_message(app, "This build does not include SAM mask generation", theme::danger);
        return false;
    }
    if (app.settings.sam_text[0] == '\0') {
        set_message(app, "Enter a mask prompt before running", theme::warning);
        return false;
    }
    if (app.settings.sam_engine == 1 && !photara::sam::language_masks_built()) {
        set_message(app, "Language masks require the GGML inference build", theme::danger);
        return false;
    }
    if (request_agreements(app, false)) return false;
    const auto* entry = selected_segmentation_entry(app);
    if (selected_mask_model(app).empty() ||
        (entry && !photara::sam::model_file_ready(selected_mask_model(app), entry->bytes))) {
        set_message(app, "Download the selected mask model before running", theme::warning);
        return false;
    }
    if (app.settings.sam_engine == 1 &&
        (!photara::sam::model_file_ready(selected_mask_detector(app), 0) ||
         photara::sam::locate_text_detector_vocab().empty())) {
        set_message(app, "Download the text detector and vocabulary before running", theme::warning);
        return false;
    }
    return true;
}

void draw_mask_model_controls(App& app) {
    auto& settings = app.settings;
    if (ImGui::Combo(tr("Mask engine"), &settings.sam_engine,
            "SAM 3\0Grounding DINO + SAM / EdgeTAM\0")) {
        settings.sam_model.fill('\0');
    }
    const auto selected = selected_segmentation_path(app);
    const auto* current = selected_segmentation_entry(app);
    ImGui::SetNextItemWidth(-1.F);
    if (ImGui::BeginCombo("##mask_model", current ? current->label : tr("Custom checkpoint"))) {
        for (const auto& entry : photara::sam::mask_model_catalog()) {
            if (entry.kind == photara::sam::MaskModelKind::text_detector) continue;
            if (settings.sam_engine == 0 && entry.kind != photara::sam::MaskModelKind::sam3) continue;
            if (ImGui::Selectable(entry.label, selected.filename() == entry.filename))
                store_path_field(settings.sam_model, photara::sam::user_model_path(entry.filename));
        }
        ImGui::EndCombo();
    }
    ImGui::InputText(tr("Checkpoint"), settings.sam_model.data(), settings.sam_model.size());
    if (settings.sam_engine == 1) {
        const auto detector = selected_mask_detector(app);
        ImGui::SetNextItemWidth(-1.F);
        if (ImGui::BeginCombo("##mask_detector", detector.filename().string().c_str())) {
            for (const auto& entry : photara::sam::mask_model_catalog()) {
                if (entry.kind != photara::sam::MaskModelKind::text_detector) continue;
                if (ImGui::Selectable(entry.label, detector.filename() == entry.filename))
                    store_path_field(settings.sam_detector_model, photara::sam::user_model_path(entry.filename));
            }
            ImGui::EndCombo();
        }
        ImGui::InputText(tr("Detector checkpoint"), settings.sam_detector_model.data(), settings.sam_detector_model.size());
        ImGui::SliderFloat(tr("Detector score"), &settings.sam_detector_threshold, 0.F, 1.F);
        ImGui::TextWrapped("%s", tr("Text detection runs again on each image; every box prompts one object."));
    }
    if (ImGui::Button(tr("Download missing models"), {-1.F, 0.F}))
        start_sam_model_downloads(app);
    if (ImGui::Button(tr("Model agreements"), {-1.F, 0.F})) {
        try {
            app.sam_license_families = selected_agreements(app);
            app.sam_license_checks.clear();
            for (const auto& id : app.sam_license_families)
                app.sam_license_checks.push_back(photara::sam::license_accepted(*photara::sam::find_model_license(id)));
            app.sam_license_download = false;
            app.show_sam_license = true;
        } catch (const std::exception& error) { set_message(app,error.what(),theme::danger); }
    }
    if (app.sam_download.state() == ModelDownload::State::running) {
        ImGui::ProgressBar(std::max(0.F, app.sam_download.progress()), {-1.F, 0.F});
        if (ImGui::Button(tr("Cancel download"))) app.sam_download.cancel();
    }
    if (app.sam_download.state() != ModelDownload::State::idle)
        ImGui::TextWrapped("%s", app.sam_download.status().c_str());
}

bool open_model_license_url(const char* url) { return open_license_url(url); }

void draw_sam_license_modal(App& app) {
    update_sam_model_downloads(app);
    constexpr const char* popup_id = "###ModelAgreements";
    if (app.show_sam_license) {
        ImGui::OpenPopup(popup_id);
        app.show_sam_license = false;
    }
    ImGui::SetNextWindowSize({610.F, 0.F}, ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(i18n::id("Model agreements", popup_id), nullptr,
            ImGuiWindowFlags_AlwaysAutoResize)) return;
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX() + 580.F);
    ImGui::TextUnformatted(tr("Model weights are not included with Photara. Review each provider's terms and explicitly accept them before downloading or using the selected models. Accepting Photara's application licence does not accept model terms."));
    ImGui::PopTextWrapPos();
    bool all_accepted = !app.sam_license_families.empty();
    for (std::size_t i = 0; i < app.sam_license_families.size(); ++i) {
        const auto* license = photara::sam::find_model_license(app.sam_license_families[i]);
        if (!license) { all_accepted = false; continue; }
        ImGui::PushID(static_cast<int>(i));
        ImGui::Separator();
        ImGui::TextUnformatted(license->label);
        if (ImGui::Button(tr("Read the license")) && !open_license_url(license->url))
            set_message(app, "Could not open a browser. The licence URL is on the clipboard.", theme::warning);
        ImGui::SameLine();
        if (ImGui::Button(tr("Copy link"))) ImGui::SetClipboardText(license->url);
        ImGui::TextWrapped("%s", license->url);
        bool checked = app.sam_license_checks[i];
        ImGui::Checkbox(tr("I have read and accept this model's terms"), &checked);
        app.sam_license_checks[i] = checked;
        all_accepted &= checked;
        ImGui::PopID();
    }
    ImGui::Spacing();
    ImGui::BeginDisabled(!all_accepted);
    if (ImGui::Button(tr("Accept and continue"), {180.F, 0.F})) {
        try {
            for (const auto& id : app.sam_license_families)
                photara::sam::accept_model_license(*photara::sam::find_model_license(id));
            if (app.sam_license_download) start_sam_model_downloads(app);
            else set_message(app, "Model agreements accepted. You can generate masks now.", theme::success);
            ImGui::CloseCurrentPopup();
        } catch (const std::exception& error) {
            set_message(app, error.what(), theme::danger);
        }
    }
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button(tr("Cancel"), {100.F, 0.F})) ImGui::CloseCurrentPopup();
    ImGui::EndPopup();
}

}  // namespace editor
