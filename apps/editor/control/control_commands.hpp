#pragma once

#include "control/control_protocol.hpp"

#include <string>

namespace editor {
struct App;
}

namespace editor::control {

class ControlBus;

json execute_command(
    App& app, ControlBus& bus, const std::string& method,
    const json& params, const std::string& client);

}  // namespace editor::control
