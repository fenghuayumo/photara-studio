#include "control/control_protocol.hpp"

#include "core/version.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <unistd.h>
#endif

namespace {

using nlohmann::json;
using editor::control::failure;
using editor::control::ok;

#if defined(_WIN32)
using Socket = SOCKET;
inline constexpr Socket k_invalid_socket = INVALID_SOCKET;
#else
using Socket = int;
inline constexpr Socket k_invalid_socket = -1;
#endif

constexpr int k_exit_ok = 0;
constexpr int k_exit_command = 2;
constexpr int k_exit_connection = 3;
constexpr int k_exit_usage = 4;
constexpr int k_exit_timeout = 5;

struct Options {
    bool json_output{};
    bool json_lines{};
    std::string client{"photara-ctl"};
    std::string url;
    std::string token;
    std::string project;
    std::string request_id;
    long long pid{};
    std::vector<std::string> args;
};

struct Discovery {
    long long pid{};
    std::string session_id;
    std::string url;
    std::string token;
    std::string version;
    long long started_at_ms{};
};

struct HttpResponse {
    int status{};
    std::string body;
};

void close_socket(const Socket socket) {
#if defined(_WIN32)
    closesocket(socket);
#else
    close(socket);
#endif
}

bool send_all(const Socket socket, const char* data, std::size_t size) {
    while (size > 0) {
#if defined(_WIN32)
        const int sent = send(socket, data, static_cast<int>(size), 0);
#else
        const auto sent = send(socket, data, size, 0);
#endif
        if (sent <= 0) return false;
        data += sent;
        size -= static_cast<std::size_t>(sent);
    }
    return true;
}

std::filesystem::path discovery_dir() {
    if (const char* override_dir = std::getenv("PHOTARA_CONTROL_DIR");
        override_dir != nullptr && override_dir[0] != '\0')
        return std::filesystem::path(override_dir);
#if defined(_WIN32)
    if (const char* appdata = std::getenv("APPDATA");
        appdata != nullptr && appdata[0] != '\0')
        return std::filesystem::path(appdata) / "Photara" / "control";
    if (const char* local = std::getenv("LOCALAPPDATA");
        local != nullptr && local[0] != '\0')
        return std::filesystem::path(local) / "Photara" / "control";
#else
    if (const char* config = std::getenv("XDG_CONFIG_HOME");
        config != nullptr && config[0] != '\0')
        return std::filesystem::path(config) / "Photara" / "control";
    if (const char* home = std::getenv("HOME");
        home != nullptr && home[0] != '\0')
        return std::filesystem::path(home) / ".config" / "Photara" / "control";
#endif
    return std::filesystem::current_path() / "photara-control";
}

bool parse_url(
    const std::string& url, std::string& host, std::uint16_t& port) {
    const std::string prefix = "http://";
    if (url.rfind(prefix, 0) != 0) return false;
    const std::string rest = url.substr(prefix.size());
    const std::size_t slash = rest.find('/');
    const std::string authority =
        slash == std::string::npos ? rest : rest.substr(0, slash);
    const std::size_t colon = authority.rfind(':');
    if (colon == std::string::npos) {
        host = authority;
        port = 80;
    } else {
        host = authority.substr(0, colon);
        try {
            port = static_cast<std::uint16_t>(
                std::stoi(authority.substr(colon + 1)));
        } catch (...) {
            return false;
        }
    }
    return !host.empty() && port != 0;
}

HttpResponse http_request(
    const std::string& host, const std::uint16_t port,
    const std::string& method, const std::string& path,
    const std::string& token, const std::string& body,
    const int timeout_seconds = 10) {
    HttpResponse response;
    const Socket socket = ::socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (socket == k_invalid_socket) return response;
#if defined(_WIN32)
    const DWORD timeout_ms =
        static_cast<DWORD>(std::max(1, timeout_seconds) * 1000);
    setsockopt(
        socket, SOL_SOCKET, SO_RCVTIMEO,
        reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
    setsockopt(
        socket, SOL_SOCKET, SO_SNDTIMEO,
        reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
    timeval timeout{};
    timeout.tv_sec = std::max(1, timeout_seconds);
    setsockopt(socket, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
    setsockopt(socket, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif
    sockaddr_in address{};
    address.sin_family = AF_INET;
    address.sin_port = htons(port);
    if (inet_pton(AF_INET, host.c_str(), &address.sin_addr) != 1) {
        close_socket(socket);
        return response;
    }
    if (connect(
            socket, reinterpret_cast<sockaddr*>(&address),
            sizeof(address)) != 0) {
        close_socket(socket);
        return response;
    }
    std::string request =
        method + " " + path + " HTTP/1.1\r\nHost: " + host + ":" +
        std::to_string(port) + "\r\nConnection: close\r\n";
    if (!token.empty())
        request += "Authorization: Bearer " + token + "\r\n";
    if (!body.empty()) {
        request += "Content-Type: application/json\r\nContent-Length: " +
                   std::to_string(body.size()) + "\r\n";
    }
    request += "\r\n";
    if (!body.empty()) request += body;
    if (!send_all(socket, request.data(), request.size())) {
        close_socket(socket);
        return response;
    }
    std::string raw;
    std::array<char, 8192> chunk{};
    while (true) {
#if defined(_WIN32)
        const int received =
            recv(socket, chunk.data(), static_cast<int>(chunk.size()), 0);
#else
        const auto received = recv(socket, chunk.data(), chunk.size(), 0);
#endif
        if (received <= 0) break;
        raw.append(chunk.data(), static_cast<std::size_t>(received));
    }
    close_socket(socket);
    const std::size_t header_end = raw.find("\r\n\r\n");
    if (header_end == std::string::npos) return response;
    std::istringstream header_stream(raw.substr(0, header_end));
    std::string status_line;
    std::getline(header_stream, status_line);
    std::istringstream status_stream(status_line);
    std::string version;
    status_stream >> version >> response.status;
    response.body = raw.substr(header_end + 4);
    return response;
}

bool is_number(const std::string& text) {
    if (text.empty()) return false;
    char* end = nullptr;
    std::strtod(text.c_str(), &end);
    return end != nullptr && *end == '\0';
}

json parse_value(const std::string& text) {
    if (text == "true") return true;
    if (text == "false") return false;
    if (text == "null") return nullptr;
    if (is_number(text)) {
        if (text.find_first_of(".eE") != std::string::npos)
            return std::stod(text);
        try {
            return std::stoll(text);
        } catch (...) {
            return std::stod(text);
        }
    }
    if (!text.empty() && text.front() == '[') {
        try {
            return json::parse(text);
        } catch (...) {
        }
    }
    return text;
}

void print_json(const json& value, const bool pretty) {
    std::cout << (pretty ? value.dump(2) : value.dump()) << '\n';
}

int print_failure(const json& value) {
    if (value.contains("error") && value["error"].is_object()) {
        const json& error = value["error"];
        std::cerr << "error: " << error.value("code", "INTERNAL") << ": "
                  << error.value("message", "Unknown error") << '\n';
    } else {
        std::cerr << "error: " << value.dump() << '\n';
    }
    return k_exit_command;
}

std::vector<Discovery> load_discoveries() {
    std::vector<Discovery> discoveries;
    std::error_code error;
    if (!std::filesystem::is_directory(discovery_dir(), error))
        return discoveries;
    for (const auto& entry :
         std::filesystem::directory_iterator(discovery_dir(), error)) {
        if (error || !entry.is_regular_file()) continue;
        if (entry.path().extension() != ".json") continue;
        std::ifstream input(entry.path());
        if (!input) continue;
        json value;
        try {
            input >> value;
        } catch (...) {
            continue;
        }
        Discovery discovery;
        discovery.pid = value.value("pid", 0LL);
        discovery.session_id = value.value("session_id", std::string{});
        discovery.url = value.value("url", std::string{});
        discovery.token = value.value("token", std::string{});
        discovery.version = value.value("version", std::string{});
        discovery.started_at_ms = value.value("started_at_ms", 0LL);
        if (!discovery.url.empty() && !discovery.token.empty())
            discoveries.push_back(std::move(discovery));
    }
    std::sort(
        discoveries.begin(), discoveries.end(),
        [](const Discovery& a, const Discovery& b) {
            return a.started_at_ms > b.started_at_ms;
        });
    return discoveries;
}

bool discover(
    const Options& options, Discovery& selected, std::string& error) {
    if (!options.url.empty()) {
        selected.url = options.url;
        selected.token = options.token;
        if (selected.token.empty()) {
            error = "--url requires --token";
            return false;
        }
        return true;
    }
    const auto discoveries = load_discoveries();
    if (discoveries.empty()) {
        error =
            "No running Photara Studio control found. Start Studio with "
            "--enable-agent-control.";
        return false;
    }
    for (const Discovery& candidate : discoveries) {
        if (options.pid != 0 && candidate.pid != options.pid) continue;
        std::string host;
        std::uint16_t port = 0;
        if (!parse_url(candidate.url, host, port)) continue;
        const HttpResponse response =
            http_request(host, port, "GET", "/v1/health", candidate.token, "");
        if (response.status != 200) continue;
        if (!options.project.empty()) {
            const json body{
                {"jsonrpc", "2.0"},
                {"id", 1},
                {"method", "studio.get_state"},
                {"params", json::object()},
                {"client", options.client},
            };
            const HttpResponse state =
                http_request(host, port, "POST", "/v1/rpc", candidate.token,
                             body.dump());
            if (state.status != 200) continue;
            try {
                const json parsed = json::parse(state.body);
                const std::string path =
                    parsed["result"]["project"]["path"].get<std::string>();
                const std::filesystem::path requested =
                    std::filesystem::path(options.project).lexically_normal();
                const std::filesystem::path actual =
                    std::filesystem::path(path).lexically_normal();
                if (requested != actual) continue;
            } catch (...) {
                continue;
            }
        }
        selected = candidate;
        return true;
    }
    error = "No matching running Photara Studio instance found";
    return false;
}

json rpc(
    const Discovery& discovery, const Options& options,
    const std::string& method, const json& params, std::string& error) {
    std::string host;
    std::uint16_t port = 0;
    if (!parse_url(discovery.url, host, port)) {
        error = "Invalid discovery URL";
        return failure("CONNECTION", error);
    }
    static std::uint64_t next_id = 1;
    json request{
        {"jsonrpc", "2.0"},
        {"id", next_id++},
        {"method", method},
        {"params", params},
        {"client", options.client},
    };
    if (!discovery.session_id.empty()) request["session_id"] = discovery.session_id;
    const bool read_only = method == "studio.ping" || method == "studio.get_state" ||
        method == "studio.job.status" || method == "studio.params.describe" ||
        method == "studio.inspect.logs";
    if (!read_only) {
        static const std::string invocation_id = "ctl_" + std::to_string(std::random_device{}()) + "_" +
            std::to_string(std::chrono::steady_clock::now().time_since_epoch().count());
        request["request_id"] = options.request_id.empty() ? invocation_id : options.request_id;
    }
    const HttpResponse response = http_request(
        host, port, "POST", "/v1/rpc", discovery.token, request.dump());
    if (response.status == 0) {
        error = "Cannot connect to Photara Studio";
        return failure("CONNECTION", error);
    }
    if (response.status != 200) {
        error = "HTTP " + std::to_string(response.status);
        try {
            const json parsed = json::parse(response.body);
            if (parsed.contains("error")) {
                json result = parsed;
                if (request.contains("request_id")) {
                    result["error"]["details"]["request_id"] = request["request_id"];
                    result["error"]["details"]["client"] = options.client;
                    result["error"]["details"]["session_id"] = discovery.session_id;
                }
                return result;
            }
        } catch (...) {
        }
        return failure("CONNECTION", error);
    }
    try {
        const json parsed = json::parse(response.body);
        if (parsed.contains("result")) return parsed["result"];
        if (parsed.contains("error")) return parsed;
        return failure("INTERNAL", "Malformed JSON-RPC response");
    } catch (const std::exception& parse_error) {
        error = parse_error.what();
        return failure("INTERNAL", "Malformed JSON-RPC response");
    }
}

void print_state_human(const json& result) {
    const json& project = result["project"];
    std::cout << "Studio " << result.value("version", "?")
              << "  session " << result.value("session_id", "")
              << "  revision " << result.value("revision", 0) << '\n';
    if (project["path"].is_string())
        std::cout << "Project: " << project["path"].get<std::string>() << '\n';
    else
        std::cout << "Project: (none)\n";
    if (project["images"].is_string())
        std::cout << "Images:  " << project["images"].get<std::string>() << '\n';
    const json& job = result["job"];
    std::cout << "Job: " << job.value("kind", "none")
              << "  stage=" << job.value("stage", "idle")
              << "  running=" << job.value("running", false)
              << "  preparing=" << job.value("preparing", false);
    if (job["fraction"].is_number())
        std::cout << "  progress=" << job["fraction"].get<double>();
    std::cout << '\n';
    const json& artifacts = result["artifacts"];
    std::cout << "Artifacts:";
    for (const char* name :
         {"sparse", "asfm", "mvs", "splat", "mesh", "texture"})
        if (artifacts.value(name, false)) std::cout << ' ' << name;
    std::cout << '\n';
    const json& actions = result["actions"];
    std::cout << "Available:";
    for (auto it = actions.begin(); it != actions.end(); ++it) {
        if (it.value().value("available", false))
            std::cout << ' ' << it.key();
    }
    std::cout << '\n';
    if (result.contains("message") && result["message"].contains("text"))
        std::cout << "Message: "
                  << result["message"]["text"].get<std::string>() << '\n';
}

void print_job_human(const json& job) {
    std::cout << "Job: " << job.value("kind", "none")
              << "  stage=" << job.value("stage", "idle")
              << "  running=" << job.value("running", false)
              << "  paused=" << job.value("paused", false)
              << "  preparing=" << job.value("preparing", false);
    if (job["fraction"].is_number())
        std::cout << "  progress=" << job["fraction"].get<double>();
    if (job["eta_seconds"].is_number())
        std::cout << "  eta_s=" << job["eta_seconds"].get<double>();
    std::cout << '\n';
    if (job.contains("training") && job["training"].value("valid", false)) {
        const json& training = job["training"];
        std::cout << "Training: iteration "
                  << training.value("iteration", 0) << '/'
                  << training.value("total_iterations", 0)
                  << "  gaussians=" << training.value("gaussians", 0)
                  << "  loss=" << training.value("loss", 0.0) << '\n';
    }
}

void print_schema_human(const json& result) {
    for (const json& group : result["groups"]) {
        std::cout << '[' << group.value("id", "") << "] "
                  << group.value("label", "") << '\n';
        for (const json& field : group["fields"]) {
            std::cout << "  " << field.value("name", "") << " ("
                      << field.value("type", "") << ')';
            if (field.contains("default"))
                std::cout << " default=" << field["default"];
            if (field.contains("values"))
                std::cout << " values=" << field["values"];
            std::cout << '\n';
        }
    }
}

std::pair<std::string, json> parse_assignment(const std::string& text) {
    const std::size_t equals = text.find('=');
    if (equals == std::string::npos)
        return {text, json(true)};
    return {text.substr(0, equals), parse_value(text.substr(equals + 1))};
}

Options parse_options(const int argc, char** argv) {
    Options options;
    if (const char* client = std::getenv("PHOTARA_CLIENT");
        client != nullptr && client[0] != '\0')
        options.client = client;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index] != nullptr ? argv[index] : "";
        const auto next = [&]() -> std::string {
            if (index + 1 >= argc) return {};
            return argv[++index] != nullptr ? argv[index] : "";
        };
        if (argument == "--json") {
            options.json_output = true;
        } else if (argument == "--jsonl") {
            options.json_output = true;
            options.json_lines = true;
        } else if (argument == "--client") {
            options.client = next();
        } else if (argument == "--url") {
            options.url = next();
        } else if (argument == "--token") {
            options.token = next();
        } else if (argument == "--project") {
            options.project = next();
        } else if (argument == "--request-id") {
            options.request_id = next();
        } else if (argument == "--pid") {
            try {
                options.pid = std::stoll(next());
            } catch (...) {
            }
        } else {
            options.args.push_back(argument);
        }
    }
    return options;
}

void print_usage() {
    std::cout <<
        "photara-ctl " PHOTARA_VERSION "\n"
        "Control a running Photara Studio.\n\n"
        "Usage:\n"
        "  photara-ctl doctor\n"
        "  photara-ctl state [--json]\n"
        "  photara-ctl schema [--json]\n"
        "  photara-ctl params set key=value...\n"
        "  photara-ctl params get\n"
        "  photara-ctl job start <align|train|resume_training|dense|texture|export_sfm>\n"
        "  photara-ctl job status [--op OP]\n"
        "  photara-ctl job wait [--op OP] [--timeout SECONDS]\n"
        "  photara-ctl job pause|resume|stop\n"
        "  photara-ctl inspect logs [--limit N]\n"
        "  photara-ctl capture --out PATH [--max-long-edge N]\n"
        "  photara-ctl project open <path>\n"
        "  photara-ctl project set-source <path>\n"
        "  photara-ctl project set-dataset <path>\n"
        "  photara-ctl project save [path] [--overwrite]\n"
        "  photara-ctl view set key=value...\n"
        "  photara-ctl export model --out PATH [--format ply|sog|spz|glb] [--sh-degree N] [--overwrite]\n"
        "  photara-ctl export mesh --out PATH [--format ply|obj|glb] [--include-texture] [--overwrite]\n"
        "  photara-ctl edit select_all|clear_selection|invert_selection|delete_selected|undo|redo\n"
        "  photara-ctl quality\n"
        "  photara-ctl window open <video|mask>\n"
        "  photara-ctl events [--since ID]\n"
        "  photara-ctl watch\n\n"
        "Global options:\n"
        "  --url URL --token TOKEN   Connect explicitly\n"
        "  --pid PID                 Select a Studio instance\n"
        "  --project PATH            Select by project path\n"
        "  --client NAME             Audit client name\n"
        "  --request-id ID           Deduplicate a mutation within this session\n"
        "  --json / --jsonl          Machine-readable output\n";
}

}  // namespace

int main(const int argc, char** argv) {
#if defined(_WIN32)
    SetConsoleOutputCP(CP_UTF8);
    WSADATA data{};
    if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
        std::cerr << "error: WSAStartup failed\n";
        return k_exit_connection;
    }
#endif
    const Options options = parse_options(argc, argv);
    if (options.args.empty() || options.args[0] == "help" ||
        options.args[0] == "--help" || options.args[0] == "-h") {
        print_usage();
#if defined(_WIN32)
        WSACleanup();
#endif
        return options.args.empty() ? k_exit_usage : k_exit_ok;
    }

    Discovery discovery;
    std::string discovery_error;
    if (!discover(options, discovery, discovery_error)) {
        std::cerr << "error: " << discovery_error << '\n';
#if defined(_WIN32)
        WSACleanup();
#endif
        return k_exit_connection;
    }

    const auto call = [&](const std::string& method, const json& params) {
        std::string error;
        json result = rpc(discovery, options, method, params, error);
        if (!ok(result)) {
            if (options.json_output) print_json(result, true);
            else print_failure(result);
            std::exit(k_exit_command);
        }
        return result;
    };

    const std::string& command = options.args[0];
    if (command == "doctor") {
        const json ping = call("studio.ping", json::object());
        const json state = call("studio.get_state", json::object());
        if (options.json_output)
            print_json(json{{"ping", ping}, {"state", state}}, true);
        else {
            std::cout << "Connected to Photara Studio\n";
            print_state_human(state);
        }
    } else if (command == "state") {
        const json state = call("studio.get_state", json::object());
        if (options.json_output) print_json(state, true);
        else print_state_human(state);
    } else if (command == "schema") {
        const json schema = call("studio.params.describe", json::object());
        if (options.json_output) print_json(schema, true);
        else print_schema_human(schema);
    } else if (command == "params") {
        if (options.args.size() < 2) {
            print_usage();
            return k_exit_usage;
        }
        if (options.args[1] == "get") {
            const json state = call("studio.get_state", json::object());
            if (options.json_output) print_json(state["settings"], true);
            else print_json(state["settings"], true);
        } else if (options.args[1] == "set") {
            const json patch = [&] {
                json value = json::object();
                for (std::size_t i = 2; i < options.args.size(); ++i) {
                    const auto [key, parsed] = parse_assignment(options.args[i]);
                    value[key] = parsed;
                }
                return value;
            }();
            const json result = call(
                "studio.params.patch", json{{"patch", patch}});
            if (options.json_output) print_json(result, true);
            else {
                std::cout << "Updated:";
                for (const json& name : result["changed"])
                    std::cout << ' ' << name.get<std::string>();
                std::cout << '\n';
            }
        } else {
            print_usage();
            return k_exit_usage;
        }
    } else if (command == "job") {
        if (options.args.size() < 2) {
            print_usage();
            return k_exit_usage;
        }
        const std::string& action = options.args[1];
        if (action == "start") {
            if (options.args.size() < 3) {
                print_usage();
                return k_exit_usage;
            }
            json params{{"stage", options.args[2]}};
            for (std::size_t i = 3; i < options.args.size(); ++i) {
                if (options.args[i] == "--dry-run") params["dry_run"] = true;
            }
            const json result = call("studio.job.start", params);
            if (options.json_output) print_json(result, true);
            else
                std::cout << "Started " << params["stage"]
                          << " op=" << result.value("op_id", "") << '\n';
        } else if (action == "status") {
            json params = json::object();
            if (options.args.size() == 4 && options.args[2] == "--op")
                params["op_id"] = options.args[3];
            else if (options.args.size() != 2) return k_exit_usage;
            const json result = call("studio.job.status", params);
            if (options.json_output) print_json(result, true);
            else print_job_human(result["job"]);
        } else if (action == "wait") {
            int timeout_seconds = 60;
            json params = json::object();
            for (std::size_t i = 2; i < options.args.size(); ++i) {
                if (options.args[i] == "--timeout" && i + 1 < options.args.size()) {
                    try {
                        timeout_seconds = std::stoi(options.args[++i]);
                    } catch (...) {
                        return k_exit_usage;
                    }
                } else if (options.args[i] == "--op" && i + 1 < options.args.size())
                    params["op_id"] = options.args[++i];
                else return k_exit_usage;
            }
            if (timeout_seconds < 1) return k_exit_usage;
            const auto begin = std::chrono::steady_clock::now();
            json last = json::object();
            while (true) {
                last = call("studio.job.status", params)["job"];
                // Bind to the first observed operation; never follow a newer job.
                if (!params.contains("op_id") && last.contains("op_id") && last["op_id"].is_string())
                    params["op_id"] = last["op_id"];
                const bool active = last.value("running", false) ||
                                    last.value("preparing", false) || last.value("finishing", false);
                if (!active) break;
                if (std::chrono::steady_clock::now() - begin >
                    std::chrono::seconds(std::max(1, timeout_seconds))) {
                    last["timed_out"] = true;
                    break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(500));
            }
            if (options.json_output) print_json(last, true);
            else print_job_human(last);
            if (last.value("timed_out", false)) return k_exit_timeout;
            const std::string status = last.value("status", "idle");
            if (status == "failed" || status == "cancelled" ||
                (!last["exit_code"].is_null() && last["exit_code"].get<int>() != 0) ||
                !last.value("last_error", std::string{}).empty()) return k_exit_command;
        } else if (action == "pause" || action == "resume" ||
                   action == "stop") {
            const json result = call(
                "studio.job.control", json{{"action", action}});
            if (options.json_output) print_json(result, true);
            else std::cout << "Job " << action << " requested\n";
        } else {
            print_usage();
            return k_exit_usage;
        }
    } else if (command == "inspect") {
        if (options.args.size() < 2 || options.args[1] != "logs") {
            print_usage();
            return k_exit_usage;
        }
        int limit = 200;
        for (std::size_t i = 2; i < options.args.size(); ++i) {
            if (options.args[i] == "--limit" && i + 1 < options.args.size()) {
                try {
                    limit = std::stoi(options.args[++i]);
                } catch (...) {
                }
            }
        }
        const json result = call(
            "studio.inspect.logs", json{{"limit", limit}});
        if (options.json_output) {
            print_json(result, true);
        } else {
            for (const json& line : result["lines"])
                std::cout << line.value("time", "") << " ["
                          << line.value("severity", "") << "] "
                          << line.value("message", "") << '\n';
        }
    } else if (command == "capture") {
        std::string path;
        std::uint32_t max_long_edge = 0;
        for (std::size_t i = 1; i < options.args.size(); ++i) {
            if (options.args[i] == "--out" && i + 1 < options.args.size())
                path = options.args[++i];
            else if (
                options.args[i] == "--max-long-edge" &&
                i + 1 < options.args.size()) {
                try {
                    max_long_edge = static_cast<std::uint32_t>(
                        std::stoul(options.args[++i]));
                } catch (...) {
                }
            }
        }
        if (path.empty()) {
            std::cerr << "error: capture requires --out PATH\n";
            return k_exit_usage;
        }
        const json result = call(
            "studio.view.capture",
            json{{"path", path}, {"max_long_edge", max_long_edge}});
        if (options.json_output) print_json(result, true);
        else std::cout << "Captured " << result.value("path", "") << '\n';
    } else if (command == "window") {
        if (options.args.size() < 3 || options.args[1] != "open") {
            print_usage();
            return k_exit_usage;
        }
        const json result = call(
            "studio.window.open", json{{"window", options.args[2]}});
        if (options.json_output) print_json(result, true);
        else
            std::cout << "Opened " << result.value("window", "")
                      << " window\n";
    } else if (command == "view") {
        if (options.args.size() < 3 || options.args[1] != "set") return k_exit_usage;
        json patch = json::object();
        for (std::size_t i = 2; i < options.args.size(); ++i) {
            const auto [name, value] = parse_assignment(options.args[i]);
            patch[name] = value;
        }
        const json result = call("studio.view.patch", {{"patch", patch}});
        print_json(result, options.json_output);
    } else if (command == "edit") {
        if (options.args.size() != 2) return k_exit_usage;
        const json result = call("studio.edit.apply", {{"action", options.args[1]}});
        print_json(result, options.json_output);
    } else if (command == "quality") {
        const json result = call("studio.inspect.quality", json::object());
        print_json(result, options.json_output);
    } else if (command == "export") {
        if (options.args.size() < 4 || (options.args[1] != "model" && options.args[1] != "mesh")) return k_exit_usage;
        json params = json::object();
        for (std::size_t i = 2; i < options.args.size(); ++i) {
            const auto& flag = options.args[i];
            if (flag == "--overwrite") params["overwrite"] = true;
            else if (flag == "--include-texture") params["include_texture"] = true;
            else if ((flag == "--out" || flag == "--format" || flag == "--sh-degree") &&
                     i + 1 < options.args.size()) {
                const std::string name = flag == "--out" ? "path" : flag == "--format" ? "format" : "sh_degree";
                const auto& value = options.args[++i];
                params[name] = flag == "--sh-degree" ? parse_value(value) : json(value);
            } else return k_exit_usage;
        }
        const json result = call(options.args[1] == "mesh" ? "studio.export.mesh" : "studio.export.model", params);
        print_json(result, options.json_output);
    } else if (command == "project") {
        if (options.args.size() < 2) {
            print_usage();
            return k_exit_usage;
        }
        const std::string& action = options.args[1];
        const std::string method =
            action == "open" ? "studio.project.open"
                : action == "save" ? "studio.project.save"
                : action == "set-dataset" ? "studio.project.set_dataset"
                                          : "studio.project.set_source";
        if (action != "open" && action != "set-source" && action != "set-dataset" &&
            action != "save") {
            print_usage();
            return k_exit_usage;
        }
        json params = json::object();
        for (std::size_t i = 2; i < options.args.size(); ++i) {
            if (action == "save" && options.args[i] == "--overwrite") params["overwrite"] = true;
            else if (!params.contains("path")) params["path"] = options.args[i];
            else return k_exit_usage;
        }
        if (action != "save" && !params.contains("path")) return k_exit_usage;
        const json result = call(method, params);
        if (options.json_output) print_json(result, true);
        else std::cout << "Project updated: "
                       << result.value("message", "") << '\n';
    } else if (command == "events") {
        std::uint64_t since = 0;
        for (std::size_t i = 1; i < options.args.size(); ++i) {
            if (options.args[i] == "--since" && i + 1 < options.args.size()) {
                try {
                    since = std::stoull(options.args[++i]);
                } catch (...) {
                }
            }
        }
        std::string host;
        std::uint16_t port = 0;
        if (!parse_url(discovery.url, host, port)) {
            std::cerr << "error: invalid discovery URL\n";
            return k_exit_connection;
        }
        const HttpResponse response = http_request(
            host, port, "GET",
            "/v1/events?since=" + std::to_string(since),
            discovery.token, "");
        if (response.status != 200) {
            std::cerr << "error: HTTP " << response.status << '\n';
            return k_exit_connection;
        }
        try {
            const json result = json::parse(response.body);
            print_json(result, !options.json_lines);
        } catch (const std::exception& error) {
            std::cerr << "error: invalid events response: " << error.what()
                      << '\n';
            return k_exit_connection;
        }
    } else if (command == "watch") {
        std::string host;
        std::uint16_t port = 0;
        if (!parse_url(discovery.url, host, port)) {
            std::cerr << "error: invalid discovery URL\n";
            return k_exit_connection;
        }
        std::uint64_t since = 0;
        while (true) {
            const HttpResponse response = http_request(
                host, port, "GET",
                "/v1/events?since=" + std::to_string(since),
                discovery.token, "");
            if (response.status != 200) break;
            try {
                const json result = json::parse(response.body);
                for (const json& event : result["events"]) {
                    std::cout << event.dump() << std::endl;
                    since = std::max(since, event.value("id", 0ULL));
                }
            } catch (...) {
            }
            std::this_thread::sleep_for(std::chrono::seconds(1));
        }
    } else {
        print_usage();
        return k_exit_usage;
    }

#if defined(_WIN32)
    WSACleanup();
#endif
    return k_exit_ok;
}
