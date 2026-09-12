#pragma once

#include <array>
#include <filesystem>

namespace editor {

enum class FilePickKind {
    project, dataset, point_cloud, splat_model, mesh, alignment, video
};

bool pick_folder(const wchar_t* title, std::array<char, 1024>& destination);
bool pick_project_file(
    const wchar_t* title, std::array<char, 1024>& destination, bool save);
bool pick_dataset_file(
    const wchar_t* title, std::array<char, 1024>& destination);
bool pick_point_cloud_file(
    const wchar_t* title, std::array<char, 1024>& destination);
bool pick_splat_model_file(
    const wchar_t* title, std::array<char, 1024>& destination);
bool pick_video_file(
    const wchar_t* title, std::array<char, 1024>& destination);
bool pick_export_path(
    const wchar_t* title, std::array<char, 1024>& destination,
    FilePickKind kind, const wchar_t* default_name,
    const wchar_t* default_extension);
void reveal_in_explorer(const std::filesystem::path& path);

}  // namespace editor
