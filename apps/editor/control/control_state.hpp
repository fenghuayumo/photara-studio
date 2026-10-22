#pragma once

#include "control/control_protocol.hpp"

#include <string>

namespace editor {
struct App;
struct ProjectSettings;
}

namespace editor::control {

class ControlBus;

json snapshot_state(const App& app, const ControlBus& bus);
json snapshot_job(const App& app, const ControlBus& bus);
json describe_schema();
std::string state_revision_key(const App& app);

void normalize_settings(ProjectSettings& settings);
json apply_parameter_patch(App& app, const json& patch);

}  // namespace editor::control
