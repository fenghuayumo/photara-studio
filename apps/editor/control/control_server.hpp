#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace editor::control {

class ControlBus;

class Server {
public:
    Server();
    Server(const Server&) = delete;
    Server& operator=(const Server&) = delete;
    ~Server();

    bool start(ControlBus& bus, std::uint16_t port = 0);
    void stop();

    [[nodiscard]] bool running() const noexcept;
    [[nodiscard]] std::uint16_t port() const noexcept;
    [[nodiscard]] const std::string& error() const noexcept;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace editor::control
