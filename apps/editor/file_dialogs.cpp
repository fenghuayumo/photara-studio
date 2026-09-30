#include "file_dialogs.hpp"

#include <array>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <string>
#include <system_error>
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
#include <shlobj.h>
#else
#include <cerrno>
#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>
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
            {L"Photara Project (*.ascan)", L"*.ascan"},
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
            {L"Photara SfM (*.asfm)", L"*.asfm"},
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
namespace {

std::string wide_to_utf8(const wchar_t* text) {
    if (text == nullptr) return {};
    std::string out;
    for (const wchar_t* cursor = text; *cursor != L'\0'; ++cursor) {
        const auto code = static_cast<std::uint32_t>(*cursor);
        if (code < 0x80) {
            out.push_back(static_cast<char>(code));
        } else if (code < 0x800) {
            out.push_back(static_cast<char>(0xC0 | (code >> 6)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else if (code < 0x10000) {
            out.push_back(static_cast<char>(0xE0 | (code >> 12)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        } else {
            out.push_back(static_cast<char>(0xF0 | (code >> 18)));
            out.push_back(static_cast<char>(0x80 | ((code >> 12) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | ((code >> 6) & 0x3F)));
            out.push_back(static_cast<char>(0x80 | (code & 0x3F)));
        }
    }
    return out;
}

bool executable_on_path(const char* name) {
    const char* path_env = std::getenv("PATH");
    if (path_env == nullptr || name == nullptr || name[0] == '\0') return false;
    const std::string paths(path_env);
    std::size_t begin = 0;
    while (begin <= paths.size()) {
        const std::size_t end = paths.find(':', begin);
        const std::string dir = paths.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::filesystem::path candidate =
            std::filesystem::path(dir) / name;
        if (::access(candidate.c_str(), X_OK) == 0) return true;
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return false;
}

enum class DialogTool { none, zenity, kdialog };

DialogTool dialog_tool() {
    if (executable_on_path("zenity")) return DialogTool::zenity;
    if (executable_on_path("kdialog")) return DialogTool::kdialog;
    return DialogTool::none;
}

struct Captured {
    int status{1};
    std::string output;
};

Captured run_capture(const std::vector<std::string>& args) {
    if (args.empty()) return {};
    int pipes[2]{};
    if (::pipe(pipes) != 0) return {};
    const pid_t pid = ::fork();
    if (pid < 0) {
        ::close(pipes[0]);
        ::close(pipes[1]);
        return {};
    }
    if (pid == 0) {
        ::dup2(pipes[1], STDOUT_FILENO);
        ::close(pipes[0]);
        ::close(pipes[1]);
        const int devnull = ::open("/dev/null", O_WRONLY);
        if (devnull >= 0) {
            ::dup2(devnull, STDERR_FILENO);
            if (devnull > 2) ::close(devnull);
        }
        std::vector<char*> argv;
        argv.reserve(args.size() + 1);
        for (const std::string& arg : args)
            argv.push_back(const_cast<char*>(arg.c_str()));
        argv.push_back(nullptr);
        ::execvp(argv[0], argv.data());
        _exit(127);
    }
    ::close(pipes[1]);
    Captured captured;
    char buffer[512];
    for (;;) {
        const ssize_t read = ::read(pipes[0], buffer, sizeof(buffer));
        if (read < 0) {
            if (errno == EINTR) continue;
            break;
        }
        if (read == 0) break;
        captured.output.append(buffer, buffer + read);
    }
    ::close(pipes[0]);
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
    captured.status = WIFEXITED(status) ? WEXITSTATUS(status) : 1;
    while (!captured.output.empty() &&
           (captured.output.back() == '\n' || captured.output.back() == '\r'))
        captured.output.pop_back();
    return captured;
}

bool store_path(std::string path, std::array<char, 1024>& destination) {
    if (path.empty() || path.size() + 1 > destination.size()) return false;
    std::memcpy(destination.data(), path.data(), path.size());
    destination[path.size()] = '\0';
    return true;
}

struct Filter {
    const char* label;
    const char* pattern;
};

Filter filters_for(
    const FilePickKind kind, const wchar_t* default_extension) {
    const std::string extension = wide_to_utf8(default_extension);
    if (kind == FilePickKind::splat_model) {
        if (extension == "sog")
            return {"PlayCanvas SOG (*.sog)", "*.sog"};
        if (extension == "spz") return {"Niantic SPZ (*.spz)", "*.spz"};
        if (extension == "glb")
            return {"Khronos Gaussian GLB (*.glb)", "*.glb"};
        if (extension == "ply") return {"Gaussian PLY (*.ply)", "*.ply"};
        return {
            "Gaussian splat (*.ply *.sog *.spz *.glb)",
            "*.ply *.sog *.spz *.glb"};
    }
    if (kind == FilePickKind::mesh) {
        if (extension == "obj") return {"Wavefront OBJ (*.obj)", "*.obj"};
        if (extension == "glb") return {"glTF Binary (*.glb)", "*.glb"};
        return {"Stanford PLY (*.ply)", "*.ply"};
    }
    if (kind == FilePickKind::alignment) {
        if (extension == "mvs") return {"OpenMVS scene (*.mvs)", "*.mvs"};
        if (extension == "json")
            return {"Nerfstudio / Blender (*.json)", "*.json"};
        if (extension == "ply") return {"Sparse cloud (*.ply)", "*.ply"};
        return {"Photara SfM (*.asfm)", "*.asfm"};
    }
    switch (kind) {
        case FilePickKind::dataset:
            return {"Camera datasets (*.csv *.mvs)", "*.csv *.mvs"};
        case FilePickKind::point_cloud:
            return {"Point cloud (*.ply)", "*.ply"};
        case FilePickKind::video:
            return {
                "Video files",
                "*.mp4 *.mov *.mkv *.avi *.webm *.m4v *.insv *.wmv *.mts *.m2ts *.360"};
        case FilePickKind::project:
        default:
            return {"Photara Project (*.ascan)", "*.ascan"};
    }
}

std::string suggested_name(
    const wchar_t* default_name, const wchar_t* default_extension) {
    std::string name = wide_to_utf8(default_name);
    std::string extension = wide_to_utf8(default_extension);
    if (extension.empty()) return name;
    if (extension.front() != '.') extension.insert(extension.begin(), '.');
    if (name.size() >= extension.size() &&
        name.compare(name.size() - extension.size(), extension.size(), extension) ==
            0)
        return name;
    return name + extension;
}

bool pick_with_tool(
    const wchar_t* title, std::array<char, 1024>& destination,
    const bool directory, const bool save, const Filter* filter,
    const wchar_t* default_name, const wchar_t* default_extension) {
    const DialogTool tool = dialog_tool();
    if (tool == DialogTool::none) return false;
    const std::string title_utf8 = wide_to_utf8(title);
    const std::string suggestion =
        suggested_name(default_name, default_extension);
    std::vector<std::string> args;
    if (tool == DialogTool::zenity) {
        args = {"zenity", "--file-selection", "--title=" + title_utf8};
        if (directory) args.emplace_back("--directory");
        if (save) {
            args.emplace_back("--save");
            args.emplace_back("--confirm-overwrite");
        }
        if (!suggestion.empty())
            args.push_back("--filename=" + suggestion);
        if (filter != nullptr) {
            args.push_back(
                std::string("--file-filter=") + filter->label + " | " +
                filter->pattern);
            args.emplace_back("--file-filter=All files | *");
        }
    } else {
        if (directory) {
            args = {"kdialog", "--getexistingdirectory", "."};
        } else if (save) {
            args = {
                "kdialog", "--getsavefilename",
                suggestion.empty() ? std::string(".") : suggestion};
        } else {
            args = {"kdialog", "--getopenfilename", "."};
        }
        if (!title_utf8.empty()) {
            args.emplace_back("--title");
            args.push_back(title_utf8);
        }
        if (!directory && filter != nullptr) {
            args.push_back(
                std::string(filter->pattern) + "|" + filter->label +
                "\n*|All files");
        }
    }
    const Captured captured = run_capture(args);
    if (captured.status != 0) return false;
    std::string path = captured.output;
    if (save && !directory && default_extension != nullptr) {
        std::filesystem::path picked(path);
        if (picked.extension().empty()) {
            std::string extension = wide_to_utf8(default_extension);
            if (!extension.empty() && extension.front() != '.')
                extension.insert(extension.begin(), '.');
            path += extension;
        }
    }
    return store_path(std::move(path), destination);
}

bool pick_file(
    const wchar_t* title, std::array<char, 1024>& destination,
    const bool save, const FilePickKind kind,
    const wchar_t* default_name = nullptr,
    const wchar_t* default_extension = nullptr) {
    const Filter filter = filters_for(kind, default_extension);
    const wchar_t* extension = default_extension;
    if (save && extension == nullptr && kind == FilePickKind::project)
        extension = L"ascan";
    return pick_with_tool(
        title, destination, false, save, &filter, default_name, extension);
}

}  // namespace

bool pick_folder(const wchar_t* title, std::array<char, 1024>& destination) {
    return pick_with_tool(
        title, destination, true, false, nullptr, nullptr, nullptr);
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
    const std::string text = path.string();
    const pid_t pid = ::fork();
    if (pid < 0) return;
    if (pid == 0) {
        ::setsid();
        const int devnull = ::open("/dev/null", O_RDWR);
        if (devnull >= 0) {
            ::dup2(devnull, STDIN_FILENO);
            ::dup2(devnull, STDOUT_FILENO);
            ::dup2(devnull, STDERR_FILENO);
            if (devnull > 2) ::close(devnull);
        }
        ::execlp("xdg-open", "xdg-open", text.c_str(), nullptr);
        _exit(127);
    }
    int status = 0;
    while (::waitpid(pid, &status, 0) < 0 && errno == EINTR) {
    }
}
#endif

}  // namespace editor
