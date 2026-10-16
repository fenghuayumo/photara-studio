#include "prior_preview.hpp"
#include "i18n.hpp"
#include "imgui.h"
#include <algorithm>
#include <chrono>
#include <cmath>

namespace editor {
namespace {
using i18n::tr;
std::string utf8(const std::filesystem::path& path) {
    const auto text=path.u8string(); return {text.begin(),text.end()};
}
}
void PriorPreview::reset_textures() {photo.reset();depth.reset();normal.reset();}
void ImageQaPriors::reset_textures() {
    depth.reset(); normal.reset(); file.clear(); stamp = {}; source_stamp = {};
}
void ImageQaPriors::update(const std::filesystem::path& folder, const std::filesystem::path& image) {
    const auto now = std::chrono::steady_clock::now();
    if (directory != folder || source != image) {
        directory = folder; source = image; next_check = {};
        reset_textures();
    }
    if (pending.valid() && pending.wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
        auto result = pending.get();
        if (result.directory == directory && result.source == source) {
            if (!result.unchanged) {
                reset_textures();
                if (result.data) {
                    depth.upload(result.data->depth); normal.upload(result.data->normal);
                    file = result.file; stamp = result.stamp; source_stamp = result.source_stamp;
                }
            }
            next_check = now + std::chrono::seconds(1);
        }
    }
    if (folder.empty() || image.empty() || pending.valid() || now < next_check) return;
    pending = std::async(std::launch::async, [folder, image, previous = file, previous_stamp = stamp,
                                           previous_source_stamp = source_stamp] {
        Result result; result.directory = folder; result.source = image;
        try {
            result.file = find_prior_preview(folder, image);
            if (!result.file.empty()) {
                result.stamp = std::filesystem::last_write_time(result.file);
                result.source_stamp = std::filesystem::last_write_time(image);
                result.unchanged = result.file == previous && result.stamp == previous_stamp &&
                    result.source_stamp == previous_source_stamp;
                if (!result.unchanged) result.data = load_prior_preview(result.file);
            }
        } catch (const std::exception&) { result.data.reset(); result.unchanged = false; }
        return result;
    });
}
bool PriorPreview::available(const std::filesystem::path& folder) {
    const auto now=std::chrono::steady_clock::now();
    if(folder!=availability_directory) {
        availability_directory=folder;cache_available=false;next_availability_check={};
    }
    if(pending_availability.valid() && pending_availability.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
        const bool found=pending_availability.get();
        if(checked_directory==folder) {
            cache_available=found;
            next_availability_check=now+std::chrono::seconds(1);
        }
    }
    if(folder.empty()) return false;
    if(!pending_availability.valid() && now>=next_availability_check) {
        checked_directory=folder;
        pending_availability=std::async(std::launch::async,[folder] {
            try {return !list_prior_previews(folder).empty();}
            catch(const std::exception&) {return false;}
        });
    }
    return cache_available;
}
void PriorPreview::show(const std::filesystem::path& folder) {
    open=true;
    requested_directory=folder;
    if(pending.valid() || pending_files.valid()) return;
    directory=folder; files.clear(); selected=-1; error.clear(); data={}; reset_textures();
    pending_files=std::async(std::launch::async,[folder]{return list_prior_previews(folder);});
}
void PriorPreview::draw() {
    if(!open) return;
    try {
        if(pending_files.valid() && pending_files.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            files=pending_files.get();
            if(!files.empty()) {
                selected=0;
                pending=std::async(std::launch::async,[file=files[0]]{return load_prior_preview(file);});
            }
        }
        if(pending.valid() && pending.wait_for(std::chrono::seconds(0))==std::future_status::ready) {
            data=pending.get();
            depth_range[0]=data.depth_near;depth_range[1]=data.depth_far;manual_range=false;
            displayed_range[0]=depth_range[0];displayed_range[1]=depth_range[1];
            photo.upload(data.photo); depth.upload(data.depth); normal.upload(data.normal);
        }
    } catch(const std::exception& failure) {error=failure.what();data={};reset_textures();}
    if(!pending.valid() && !pending_files.valid() && requested_directory!=directory)
        show(requested_directory);
    ImGui::SetNextWindowSize({1200,750},ImGuiCond_FirstUseEver);
    if(!ImGui::Begin(tr("AI prior preview"),&open)) {ImGui::End();return;}
    ImGui::TextWrapped("%s",utf8(directory).c_str());
    const bool loading=pending.valid() || pending_files.valid();
    ImGui::BeginDisabled(loading);
    if(ImGui::Button(tr("Refresh"))) show(directory);
    if(!files.empty()) {
        const auto choose=[&](int index) {
            selected=index;error.clear();data={};reset_textures();
            pending=std::async(std::launch::async,[file=files[index]]{return load_prior_preview(file);});
        };
        ImGui::SameLine();
        ImGui::BeginDisabled(selected<=0);
        if(ImGui::Button("<##prior_previous")) choose(selected-1);
        ImGui::EndDisabled(); ImGui::SameLine();
        ImGui::BeginDisabled(selected+1>=static_cast<int>(files.size()));
        if(ImGui::Button(">##prior_next")) choose(selected+1);
        ImGui::EndDisabled(); ImGui::SameLine();
        ImGui::SetNextItemWidth(240);
        if(ImGui::BeginCombo("##prior_view",selected>=0?utf8(files[selected].filename()).c_str():"")) {
            ImGuiListClipper clipper;clipper.Begin(static_cast<int>(files.size()));
            while(clipper.Step()) for(int i=clipper.DisplayStart;i<clipper.DisplayEnd;++i)
                if(ImGui::Selectable(utf8(files[i].filename()).c_str(),selected==i)) choose(i);
            ImGui::EndCombo();
        }
        ImGui::SameLine();ImGui::Text("%d / %d",selected+1,static_cast<int>(files.size()));
    }
    ImGui::EndDisabled();
    if(loading) ImGui::TextUnformatted(tr("Loading AI priors..."));
    else if(files.empty()) ImGui::TextWrapped("%s",tr("No prior maps yet. Start training to generate them, then refresh. Existing aligned caches can also be previewed."));
    if(!error.empty()) {
        ImGui::TextWrapped("%s",tr("Unable to preview this cache. It may be incomplete or its source image may have changed. Refresh after preparation finishes."));
        ImGui::TextWrapped("%s",error.c_str());
    }
    if(!data.source.empty()) {
        ImGui::TextWrapped("%s",utf8(data.source).c_str());
        ImGui::Text("%u x %u",data.prior.camera.width,data.prior.camera.height);
        ImGui::TextWrapped("%s",tr("Cached AI predictions. Dark pixels are invalid. Inspect the cache folder when switching projects."));
        ImGui::TextWrapped("%s",tr("Source photos may retain lens distortion; prior maps use rectified training cameras."));
        if(depth.descriptor) {
            ImGui::SetNextItemWidth(230.F);
            ImGui::InputFloat2(tr("Depth display range"),depth_range,"%.6g");
            ImGui::SameLine();
            ImGui::BeginDisabled(!std::isfinite(depth_range[0]) || !std::isfinite(depth_range[1]) ||
                depth_range[0]<0 || depth_range[1]<=depth_range[0]);
            if(ImGui::Button(tr("Apply range"))) {
                depth.upload(colorize_prior_depth(data.prior,depth_range[0],depth_range[1]));manual_range=true;
                displayed_range[0]=depth_range[0];displayed_range[1]=depth_range[1];
            }
            ImGui::EndDisabled();ImGui::SameLine();
            if(ImGui::Button(tr("Auto range"))) {
                depth_range[0]=data.depth_near;depth_range[1]=data.depth_far;
                depth.upload(data.depth);manual_range=false;
                displayed_range[0]=depth_range[0];displayed_range[1]=depth_range[1];
            }
        }
        if(ImGui::BeginTable("##prior_images",3,ImGuiTableFlags_SizingStretchSame)) {
            const auto image=[&](const char* label,gpu::PreviewTexture& texture,int kind) {
                ImGui::TableNextColumn();ImGui::TextUnformatted(tr(label));
                if(!texture.descriptor) {ImGui::TextWrapped("%s",tr("This map was not generated (its weight may have been zero)."));return;}
                float width=ImGui::GetContentRegionAvail().x;
                float height=width*texture.height/texture.width;
                const float available=std::max(80.F,ImGui::GetContentRegionAvail().y-75.F);
                if(height>available) {width*=available/height;height=available;}
                const auto origin=ImGui::GetCursorScreenPos();
                ImGui::Image(reinterpret_cast<ImTextureID>(texture.descriptor),{width,height});
                if(ImGui::IsItemHovered()) {
                    const auto mouse=ImGui::GetMousePos();
                    const auto& camera=data.prior.camera;
                    const unsigned x=std::min(camera.width-1,static_cast<unsigned>((mouse.x-origin.x)/width*camera.width));
                    const unsigned y=std::min(camera.height-1,static_cast<unsigned>((mouse.y-origin.y)/height*camera.height));
                    const std::size_t i=static_cast<std::size_t>(y)*camera.width+x,n=static_cast<std::size_t>(camera.width)*camera.height;
                    ImGui::BeginTooltip();ImGui::Text("(%u, %u)",x,y);
                    if(kind==1) ImGui::Text("Z = %.6g",data.prior.depth[i]);
                    if(kind==2) ImGui::Text("N = (%.4f, %.4f, %.4f)",data.prior.normals[i],data.prior.normals[n+i],data.prior.normals[2*n+i]);
                    ImGui::EndTooltip();
                }
            };
            image("Source image",photo,0);
            image("AI depth",depth,1);
            if(depth.descriptor) {
                ImGui::TextWrapped("%s",tr("Camera Z, scene units. Near: warm; far: cool."));
                ImGui::Text("%s: %.6g - %.6g",manual_range?tr("Manual"):"P02-P98",displayed_range[0],displayed_range[1]);
                ImGui::TextWrapped("%s",tr("Display colors may clip values outside the range. Hover to inspect raw depth."));
            }
            image("AI normals",normal,2);
            if(normal.descriptor) ImGui::TextWrapped("%s",tr("Camera normals: X right, Y down, Z forward. RGB = (N + 1) / 2."));
            ImGui::EndTable();
        }
    }
    ImGui::End();
}
} // namespace editor
