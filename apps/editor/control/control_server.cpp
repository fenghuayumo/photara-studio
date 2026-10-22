#include "control/control_server.hpp"

#include "control/control_bus.hpp"
#include "control/control_paths.hpp"

#include "core/version.hpp"

#include <array>
#include <atomic>
#include <cctype>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <future>
#include <map>
#include <mutex>
#include <sstream>
#include <string>
#include <thread>
#include <utility>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#include <process.h>
#else
#include <arpa/inet.h>
#include <netinet/in.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#endif

namespace editor::control {
namespace {

#if defined(_WIN32)
using Socket = SOCKET;
inline constexpr Socket k_invalid_socket = INVALID_SOCKET;
#else
using Socket = int;
inline constexpr Socket k_invalid_socket = -1;
#endif

void close_socket(const Socket socket) {
#if defined(_WIN32)
    closesocket(socket);
#else
    close(socket);
#endif
}

std::int64_t current_pid() {
#if defined(_WIN32)
    return static_cast<std::int64_t>(_getpid());
#else
    return static_cast<std::int64_t>(getpid());
#endif
}

std::int64_t now_milliseconds() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
               std::chrono::system_clock::now().time_since_epoch())
        .count();
}

std::string lowercase(std::string text) {
    for (char& value : text)
        value = static_cast<char>(
            std::tolower(static_cast<unsigned char>(value)));
    return text;
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

std::string status_text(const int status) {
    switch (status) {
        case 200: return "OK";
        case 400: return "Bad Request";
        case 401: return "Unauthorized";
        case 404: return "Not Found";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        case 504: return "Gateway Timeout";
        default: return "Error";
    }
}

bool send_response(
    const Socket socket, const int status, const std::string& body) {
    const std::string header =
        "HTTP/1.1 " + std::to_string(status) + " " + status_text(status) +
        "\r\nContent-Type: application/json\r\nContent-Length: " +
        std::to_string(body.size()) +
        "\r\nConnection: close\r\nCache-Control: no-store\r\n\r\n";
    return send_all(socket, header.data(), header.size()) &&
           send_all(socket, body.data(), body.size());
}

void send_failure(
    const Socket socket, const int status, const std::string& code,
    const std::string& message) {
    send_response(
        socket, status,
        failure(code, message).dump());
}

bool read_until(
    const Socket socket, std::string& buffer, const std::string& marker,
    const std::size_t limit) {
    std::array<char, 4096> chunk{};
    while (buffer.find(marker) == std::string::npos) {
        if (buffer.size() > limit) return false;
#if defined(_WIN32)
        const int received = recv(
            socket, chunk.data(), static_cast<int>(chunk.size()), 0);
#else
        const auto received = recv(socket, chunk.data(), chunk.size(), 0);
#endif
        if (received <= 0) return false;
        buffer.append(chunk.data(), static_cast<std::size_t>(received));
    }
    return true;
}

std::string request_path_without_query(const std::string& target) {
    const std::size_t query = target.find('?');
    return query == std::string::npos ? target : target.substr(0, query);
}

std::string query_value(const std::string& target, const std::string& key) {
    const std::size_t query = target.find('?');
    if (query == std::string::npos) return {};
    const std::string text = target.substr(query + 1);
    std::size_t begin = 0;
    while (begin <= text.size()) {
        const std::size_t end = text.find('&', begin);
        const std::string item = text.substr(
            begin, end == std::string::npos ? std::string::npos : end - begin);
        const std::size_t equals = item.find('=');
        if (equals != std::string::npos &&
            item.substr(0, equals) == key)
            return item.substr(equals + 1);
        if (end == std::string::npos) break;
        begin = end + 1;
    }
    return {};
}

std::uint64_t parse_u64(const std::string& text, const std::uint64_t fallback) {
    try {
        std::size_t used = 0;
        const auto value = std::stoull(text, &used);
        return used == text.size() ? value : fallback;
    } catch (...) {
        return fallback;
    }
}

}  // namespace

struct Server::Impl {
    ControlBus* bus{};
    std::atomic_bool running{false};
    std::atomic<std::uint16_t> port{0};
    std::string token;
    std::string session_id;
    std::string error;
    std::filesystem::path discovery_file;
    Socket listener{k_invalid_socket};
    std::thread thread;
    std::mutex mutex;

    bool start(ControlBus& target, const std::uint16_t requested_port) {
        bus = &target;
        session_id = target.session_id().empty()
            ? make_session_id() : target.session_id();
        target.set_session_id(session_id);
        token = make_session_token();

#if defined(_WIN32)
        WSADATA data{};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            error = "WSAStartup failed";
            return false;
        }
#endif

        listener = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listener == k_invalid_socket) {
            error = "Cannot create control socket";
            return false;
        }
        int reuse = 1;
        setsockopt(
            listener, SOL_SOCKET, SO_REUSEADDR,
            reinterpret_cast<const char*>(&reuse), sizeof(reuse));

        sockaddr_in address{};
        address.sin_family = AF_INET;
        address.sin_port = htons(requested_port);
        inet_pton(AF_INET, "127.0.0.1", &address.sin_addr);
        if (bind(
                listener, reinterpret_cast<sockaddr*>(&address),
                sizeof(address)) != 0) {
            error = "Cannot bind control socket to 127.0.0.1";
            close_socket(listener);
            listener = k_invalid_socket;
            return false;
        }
        if (listen(listener, 16) != 0) {
            error = "Cannot listen on control socket";
            close_socket(listener);
            listener = k_invalid_socket;
            return false;
        }

        sockaddr_in actual{};
#if defined(_WIN32)
        int length = sizeof(actual);
#else
        socklen_t length = sizeof(actual);
#endif
        if (getsockname(
                listener, reinterpret_cast<sockaddr*>(&actual),
                &length) != 0) {
            error = "Cannot query control port";
            close_socket(listener);
            listener = k_invalid_socket;
            return false;
        }
        port.store(ntohs(actual.sin_port));

        std::error_code directory_error;
        std::filesystem::create_directories(
            discovery_dir(), directory_error);
        if (directory_error) {
            error = "Cannot create control discovery directory";
            close_socket(listener);
            listener = k_invalid_socket;
            return false;
        }
        discovery_file =
            discovery_dir() /
            (std::to_string(current_pid()) + ".json");
        const json discovery{
            {"pid", current_pid()},
            {"session_id", session_id},
            {"url", "http://127.0.0.1:" + std::to_string(port.load())},
            {"token", token},
            {"version", PHOTARA_VERSION},
            {"started_at_ms", now_milliseconds()},
        };
        {
            std::ofstream output(discovery_file, std::ios::trunc);
            if (!output) {
                error = "Cannot write control discovery file";
                close_socket(listener);
                listener = k_invalid_socket;
                return false;
            }
            output << discovery.dump(2);
        }
        running.store(true);
        thread = std::thread([this] { accept_loop(); });
        return true;
    }

    void accept_loop() {
        while (running.load()) {
            sockaddr_in remote{};
#if defined(_WIN32)
            int length = sizeof(remote);
#else
            socklen_t length = sizeof(remote);
#endif
            const Socket client = accept(
                listener, reinterpret_cast<sockaddr*>(&remote), &length);
            if (client == k_invalid_socket) {
                if (!running.load()) break;
                continue;
            }
            handle_client(client);
            close_socket(client);
        }
    }

    void handle_client(const Socket client) {
#if defined(_WIN32)
        const DWORD timeout_ms = 5000;
        setsockopt(
            client, SOL_SOCKET, SO_RCVTIMEO,
            reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
        setsockopt(
            client, SOL_SOCKET, SO_SNDTIMEO,
            reinterpret_cast<const char*>(&timeout_ms), sizeof(timeout_ms));
#else
        timeval timeout{};
        timeout.tv_sec = 5;
        setsockopt(client, SOL_SOCKET, SO_RCVTIMEO, &timeout, sizeof(timeout));
        setsockopt(client, SOL_SOCKET, SO_SNDTIMEO, &timeout, sizeof(timeout));
#endif

        std::string buffer;
        if (!read_until(client, buffer, "\r\n\r\n", 64 * 1024)) {
            send_failure(
                client, 400, "INVALID_REQUEST", "Incomplete HTTP request");
            return;
        }
        const std::size_t header_end = buffer.find("\r\n\r\n") + 4;
        const std::string header_text = buffer.substr(0, header_end);
        std::istringstream header_stream(header_text);
        std::string request_line;
        std::getline(header_stream, request_line);
        std::istringstream request_stream(request_line);
        std::string method;
        std::string target;
        std::string version;
        request_stream >> method >> target >> version;
        if (method.empty() || target.empty()) {
            send_failure(
                client, 400, "INVALID_REQUEST", "Malformed request line");
            return;
        }

        std::map<std::string, std::string> headers;
        std::string line;
        while (std::getline(header_stream, line)) {
            if (!line.empty() && line.back() == '\r') line.pop_back();
            const std::size_t colon = line.find(':');
            if (colon == std::string::npos) continue;
            std::string name = lowercase(line.substr(0, colon));
            std::size_t begin = colon + 1;
            while (begin < line.size() &&
                   std::isspace(static_cast<unsigned char>(line[begin])))
                ++begin;
            headers[std::move(name)] = line.substr(begin);
        }

        const std::string authorization =
            headers.contains("authorization")
                ? headers["authorization"] : std::string{};
        if (authorization != "Bearer " + token) {
            send_failure(
                client, 401, "PERMISSION_DENIED",
                "Missing or invalid control token");
            return;
        }

        const std::string path = request_path_without_query(target);
        if (method == "GET" && path == "/v1/health") {
            send_response(
                client, 200,
                success({
                    {"version", PHOTARA_VERSION},
                    {"session_id", session_id},
                    {"pid", current_pid()},
                    {"revision", bus->revision()},
                }).dump());
            return;
        }
        if (method == "GET" && path == "/v1/events") {
            const std::uint64_t since =
                parse_u64(query_value(target, "since"), 0);
            send_response(client, 200, bus->events_since(since).dump());
            return;
        }
        if (method != "POST" || path != "/v1/rpc") {
            send_failure(client, 404, "NOT_FOUND", "Unknown endpoint");
            return;
        }

        std::size_t content_length = 0;
        if (headers.contains("content-length")) {
            try {
                content_length = static_cast<std::size_t>(
                    std::stoull(headers["content-length"]));
            } catch (...) {
                send_failure(
                    client, 400, "INVALID_REQUEST",
                    "Invalid Content-Length");
                return;
            }
        }
        if (content_length > 1024 * 1024) {
            send_failure(
                client, 400, "INVALID_REQUEST", "Request body is too large");
            return;
        }
        while (buffer.size() - header_end < content_length) {
            std::array<char, 4096> chunk{};
#if defined(_WIN32)
            const int received = recv(
                client, chunk.data(), static_cast<int>(chunk.size()), 0);
#else
            const auto received = recv(client, chunk.data(), chunk.size(), 0);
#endif
            if (received <= 0) {
                send_failure(
                    client, 400, "INVALID_REQUEST", "Incomplete request body");
                return;
            }
            buffer.append(chunk.data(), static_cast<std::size_t>(received));
        }
        const std::string body =
            buffer.substr(header_end, content_length);
        json request;
        try {
            request = json::parse(body);
        } catch (const std::exception& parse_error) {
            send_failure(
                client, 400, "INVALID_REQUEST",
                std::string("Invalid JSON: ") + parse_error.what());
            return;
        }
        if (!request.is_object() || !request.contains("method") ||
            !request["method"].is_string()) {
            send_failure(
                client, 400, "INVALID_REQUEST", "Missing method");
            return;
        }
        const std::string control_method =
            request["method"].get<std::string>();
        const json params =
            request.contains("params") && request["params"].is_object()
                ? request["params"] : json::object();
        const std::string client_name =
            request.value("client", std::string("unknown"));
        const json id =
            request.contains("id") ? request["id"] : json(nullptr);
        const std::string request_id =
            request.value("request_id", std::string{});

        std::future<json> future =
            bus->submit(control_method, params, client_name, request_id);
        if (future.wait_for(std::chrono::seconds(5)) !=
            std::future_status::ready) {
            send_failure(
                client, 504, "TIMEOUT",
                "Studio did not process the command in time");
            return;
        }
        json result;
        try {
            result = future.get();
        } catch (const std::exception& error) {
            send_failure(
                client, 500, "INTERNAL",
                std::string("Command did not complete: ") + error.what());
            return;
        }
        send_response(
            client, 200,
            json{
                {"jsonrpc", "2.0"},
                {"id", id},
                {"result", std::move(result)},
            }.dump());
    }

    void stop() {
        if (!running.exchange(false)) return;
        if (listener != k_invalid_socket) {
#if defined(_WIN32)
            shutdown(listener, SD_BOTH);
#else
            shutdown(listener, SHUT_RDWR);
#endif
            close_socket(listener);
            listener = k_invalid_socket;
        }
        if (thread.joinable()) thread.join();
        std::error_code error_code;
        if (!discovery_file.empty())
            std::filesystem::remove(discovery_file, error_code);
#if defined(_WIN32)
        WSACleanup();
#endif
    }
};

Server::Server() : impl_(std::make_unique<Impl>()) {}

Server::~Server() { stop(); }

bool Server::start(ControlBus& bus, const std::uint16_t port) {
    return impl_->start(bus, port);
}

void Server::stop() { impl_->stop(); }

bool Server::running() const noexcept { return impl_->running.load(); }

std::uint16_t Server::port() const noexcept { return impl_->port.load(); }

const std::string& Server::error() const noexcept { return impl_->error; }

}  // namespace editor::control
