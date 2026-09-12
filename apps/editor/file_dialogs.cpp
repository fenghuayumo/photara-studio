#include "file_dialogs.hpp"

#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>

#if defined(_WIN32)
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#include <shellapi.h>
#include <shlobj.h>
#endif

namespace editor {

#if defined(_WIN32)
// Native folder picker. Failure simply leaves the text field untouched.
bool pick_folder(const wchar_t* title, std::array<char, 1024>& destination) {
    bool picked = false;
    const HRESULT initialised =
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileOpenDialog* dialog = nullptr;
    if (SUCCEEDED(CoCreateInstance(
            CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
            IID_PPV_ARGS(&dialog)))) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        dialog->SetOptions(options | FOS_PICKFOLDERS | FOS_PATHMUSTEXIST);
        dialog->SetTitle(title);
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR wide = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wide))) {
                    const int bytes = WideCharToMultiByte(
                        CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
                    if (bytes > 0 &&
                        static_cast<std::size_t>(bytes) <= destination.size()) {
                        WideCharToMultiByte(
                            CP_UTF8, 0, wide, -1, destination.data(), bytes,
                            nullptr, nullptr);
                        picked = true;
                    }
                    CoTaskMemFree(wide);
                }
                item->Release();
            }
        }
        dialog->Release();
    }
    if (SUCCEEDED(initialised)) CoUninitialize();
    return picked;
}

bool copy_wide_path(const wchar_t* wide, std::array<char, 1024>& destination) {
    const int bytes = WideCharToMultiByte(
        CP_UTF8, 0, wide, -1, nullptr, 0, nullptr, nullptr);
    if (bytes <= 0 || static_cast<std::size_t>(bytes) > destination.size())
        return false;
    WideCharToMultiByte(
        CP_UTF8, 0, wide, -1, destination.data(), bytes, nullptr, nullptr);
    return true;
}

bool pick_file(
    const wchar_t* title, std::array<char, 1024>& destination,
    const bool save, const FilePickKind kind,
    const wchar_t* default_name = nullptr,
    const wchar_t* default_extension = nullptr) {
    bool picked = false;
    const HRESULT initialised =
        CoInitializeEx(nullptr, COINIT_APARTMENTTHREADED | COINIT_DISABLE_OLE1DDE);
    IFileDialog* dialog = nullptr;
    const HRESULT created = save
        ? CoCreateInstance(
              CLSID_FileSaveDialog, nullptr, CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&dialog))
        : CoCreateInstance(
              CLSID_FileOpenDialog, nullptr, CLSCTX_INPROC_SERVER,
              IID_PPV_ARGS(&dialog));
    if (SUCCEEDED(created) && dialog) {
        DWORD options = 0;
        dialog->GetOptions(&options);
        if (save)
            dialog->SetOptions(options | FOS_OVERWRITEPROMPT);
        else
            dialog->SetOptions(options | FOS_FILEMUSTEXIST | FOS_PATHMUSTEXIST);
        dialog->SetTitle(title);
        COMDLG_FILTERSPEC project_filters[] = {
            {L"AetherScan Project (*.ascan)", L"*.ascan"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC dataset_filters[] = {
            {L"Camera datasets (*.csv;*.mvs)", L"*.csv;*.mvs"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC point_cloud_filters[] = {
            {L"Point cloud (*.ply)", L"*.ply"},
            {L"All files (*.*)", L"*.*"}};
        COMDLG_FILTERSPEC splat_model_filters[] = {
            {L"Gaussian splat (*.ply;*.sog;*.spz;*.glb)", L"*.ply;*.sog;*.spz;*.glb"},
            {L"All files (*.*)", L"*.*"}};
        if (kind == FilePickKind::splat_model && default_extension != nullptr) {
            if (std::wcscmp(default_extension, L"sog") == 0) {
                splat_model_filters[0] = {L"PlayCanvas SOG (*.sog)", L"*.sog"};
            } else if (std::wcscmp(default_extension, L"spz") == 0) {
                splat_model_filters[0] = {L"Niantic SPZ (*.spz)", L"*.spz"};
            } else if (std::wcscmp(default_extension, L"glb") == 0) {
                splat_model_filters[0] = {
                    L"Khronos Gaussian GLB (*.glb)", L"*.glb"};
            } else {
                splat_model_filters[0] = {L"Gaussian PLY (*.ply)", L"*.ply"};
            }
        }
        COMDLG_FILTERSPEC mesh_filters[] = {
            {L"Stanford PLY (*.ply)", L"*.ply"},
            {L"All files (*.*)", L"*.*"}};
        if (kind == FilePickKind::mesh && default_extension != nullptr) {
            if (std::wcscmp(default_extension, L"obj") == 0) {
                mesh_filters[0] = {L"Wavefront OBJ (*.obj)", L"*.obj"};
            } else if (std::wcscmp(default_extension, L"glb") == 0) {
                mesh_filters[0] = {L"glTF Binary (*.glb)", L"*.glb"};
            }
        }
        COMDLG_FILTERSPEC alignment_filters[] = {
            {L"AetherScan SfM (*.asfm)", L"*.asfm"},
            {L"All files (*.*)", L"*.*"}};
        if (kind == FilePickKind::alignment && default_extension != nullptr) {
            if (std::wcscmp(default_extension, L"mvs") == 0) {
                alignment_filters[0] = {L"OpenMVS scene (*.mvs)", L"*.mvs"};
            } else if (std::wcscmp(default_extension, L"json") == 0) {
                alignment_filters[0] = {
                    L"Nerfstudio / Blender (*.json)", L"*.json"};
            } else if (std::wcscmp(default_extension, L"ply") == 0) {
                alignment_filters[0] = {L"Sparse cloud (*.ply)", L"*.ply"};
            }
        }
        COMDLG_FILTERSPEC video_filters[] = {
            {L"Video files (*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.insv;*.wmv)",
             L"*.mp4;*.mov;*.mkv;*.avi;*.webm;*.m4v;*.insv;*.wmv;*.mts;*.m2ts;*.360"},
            {L"All files (*.*)", L"*.*"}};
        const COMDLG_FILTERSPEC* filters = project_filters;
        if (kind == FilePickKind::dataset) filters = dataset_filters;
        if (kind == FilePickKind::point_cloud) filters = point_cloud_filters;
        if (kind == FilePickKind::splat_model) filters = splat_model_filters;
        if (kind == FilePickKind::mesh) filters = mesh_filters;
        if (kind == FilePickKind::alignment) filters = alignment_filters;
        if (kind == FilePickKind::video) filters = video_filters;
        dialog->SetFileTypes(2, filters);
        if (save && default_extension != nullptr)
            dialog->SetDefaultExtension(default_extension);
        else if (kind == FilePickKind::project && save)
            dialog->SetDefaultExtension(L"ascan");
        if (save && default_name != nullptr)
            dialog->SetFileName(default_name);
        if (SUCCEEDED(dialog->Show(nullptr))) {
            IShellItem* item = nullptr;
            if (SUCCEEDED(dialog->GetResult(&item))) {
                PWSTR wide = nullptr;
                if (SUCCEEDED(item->GetDisplayName(SIGDN_FILESYSPATH, &wide))) {
                    picked = copy_wide_path(wide, destination);
                    CoTaskMemFree(wide);
                }
                item->Release();
            }
        }
        dialog->Release();
    }
    if (SUCCEEDED(initialised)) CoUninitialize();
    return picked;
}

bool pick_project_file(
    const wchar_t* title, std::array<char, 1024>& destination, const bool save) {
    return pick_file(title, destination, save, FilePickKind::project);
}

bool pick_dataset_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::dataset);
}

bool pick_point_cloud_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::point_cloud);
}
bool pick_splat_model_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::splat_model);
}
bool pick_video_file(
    const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_file(title, destination, false, FilePickKind::video);
}

bool pick_export_path(
    const wchar_t* title, std::array<char, 1024>& destination,
    const FilePickKind kind, const wchar_t* default_name,
    const wchar_t* default_extension) {
    return pick_file(
        title, destination, true, kind, default_name, default_extension);
}

void reveal_in_explorer(const std::filesystem::path& path) {
    std::error_code error;
    if (!std::filesystem::exists(path, error)) return;
    ShellExecuteW(nullptr, L"open", path.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
}
#else
bool pick_folder(const wchar_t*, std::array<char, 1024>&) { return false; }
bool pick_project_file(const wchar_t*, std::array<char, 1024>&, bool) {
    return false;
}
bool pick_dataset_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_point_cloud_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_splat_model_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_video_file(const wchar_t*, std::array<char, 1024>&) {
    return false;
}
bool pick_export_path(
    const wchar_t*, std::array<char, 1024>&, FilePickKind, const wchar_t*,
    const wchar_t*) {
    return false;
}
void reveal_in_explorer(const std::filesystem::path&) {}
#endif

}  // namespace editor
