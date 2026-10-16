#include "moge_model.hpp"
#include "app.hpp"
#include "i18n.hpp"
#if defined(PHOTARA_HAS_INFERENCE)
#include "inference/models/moge.hpp"
#endif
#include <algorithm>

namespace editor {
ModelDownloadSpec moge_download_spec() {
    ModelDownloadSpec spec;
    spec.url=k_moge_model_url;
    spec.destination=moge_model_path();
    spec.expected_bytes=k_moge_model_bytes;
    spec.sha256=k_moge_model_sha256;
    spec.ready_message="MoGe model is ready";
#if defined(PHOTARA_HAS_INFERENCE)
    spec.validate=[](const auto& path) {(void)photara::inference::inspect_moge_model(path);};
#endif
    return spec;
}
bool moge_model_ready() {return verified_download_ready(moge_download_spec());}

void request_moge_download(App& app, bool train_after, bool resume_training) {
#if defined(PHOTARA_HAS_INFERENCE)
    app.moge_train_pending=train_after;
    app.moge_download_train_requested=train_after;
    app.moge_resume_pending=resume_training;
    app.show_moge_download=true;
    if (app.moge_download.state()!=ModelDownload::State::running)
        app.moge_download.start(moge_download_spec());
    set_message(app,"Downloading MoGe model...",theme::accent);
#else
    set_message(app,"MoGe inference is unavailable in this build. Choose an existing aligned cache.",theme::danger);
#endif
}
bool ensure_moge_ready(App& app, bool resume_training) {
    const auto& s=app.settings;
    if (!s.ai_priors || s.training_mode==0 || s.ai_prior_source!=0 ||
        !(s.ai_depth_weight>0.F || s.ai_normal_weight>0.F)) return true;
    if (moge_model_ready()) {
        store_utf8_path_field(app.settings.ai_prior_model,moge_model_path());
        return true;
    }
    request_moge_download(app,true,resume_training);
    return false;
}
void poll_moge_download(App& app) {
    if (!app.moge_train_pending) return;
    if (app.close_requested || app.show_close_confirmation || app.close_approved) {
        app.moge_train_pending=false;app.moge_download.cancel();return;
    }
    const auto state=app.moge_download.state();
    if (state==ModelDownload::State::failed || state==ModelDownload::State::cancelled) {
        app.moge_train_pending=false;
        set_message(app,app.moge_download.status(),state==ModelDownload::State::failed ? theme::danger : theme::text_muted);
        return;
    }
    if (state!=ModelDownload::State::done || app.job.running() || app.pending_train_prepare.valid()) return;
    app.moge_train_pending=false;
    if (!moge_model_ready()) {set_message(app,"MoGe model verification failed",theme::danger);return;}
    store_utf8_path_field(app.settings.ai_prior_model,moge_model_path());
    start_train(app,false,app.moge_resume_pending);
}
void draw_moge_download_modal(App& app) {
    constexpr auto popup="###MogeDownload";
    if (app.show_moge_download) {ImGui::OpenPopup(popup);app.show_moge_download=false;}
    ImGui::SetNextWindowSize({520.F,0.F},ImGuiCond_Appearing);
    if (!ImGui::BeginPopupModal(i18n::id("MoGe model download",popup),nullptr,ImGuiWindowFlags_AlwaysAutoResize)) return;
    ImGui::PushTextWrapPos(ImGui::GetCursorPosX()+480.F);
    ImGui::TextUnformatted(i18n::tr("MoGe-2 ViT-B Normal (MIT), 400 MiB. Downloaded once and reused across projects. Training continues automatically when the model is ready."));
    ImGui::PopTextWrapPos();
    const auto state=app.moge_download.state();
    const auto status=app.moge_download.status();
    ImGui::TextWrapped("%s",i18n::tr(status.c_str()));
    if (state==ModelDownload::State::running) {
        ImGui::ProgressBar(std::max(0.F,app.moge_download.progress()),{-1.F,0.F});
        if (ImGui::Button(i18n::tr("Cancel"))) {
            app.moge_train_pending=false;app.moge_resume_pending=false;app.moge_download.cancel();ImGui::CloseCurrentPopup();
        }
    } else {
        if (state==ModelDownload::State::done) ImGui::CloseCurrentPopup();
        else {
            if (ImGui::Button(i18n::tr("Retry download"))) {
                // A failed automatic training request can be retried without
                // requiring the user to press Train again.
                app.moge_train_pending=app.moge_download_train_requested;
                app.moge_download.start(moge_download_spec());
            }
            ImGui::SameLine();
            if (ImGui::Button(i18n::tr("Close"))) {app.moge_train_pending=false;ImGui::CloseCurrentPopup();}
        }
    }
    ImGui::EndPopup();
}
}
