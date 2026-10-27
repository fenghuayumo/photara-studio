# Photara Studio Agent control scheme

> Status: Design Draft
> Target version: 0.6.x or later
> Goal: Let external or built-in AI agents drive the currently running Photara Studio in a verifiable, revocable, and auditable manner.

> Implementation progress: The current source code has been implemented in Phase 0/MVP - Studio local control service,
> `photara-ctl`, status snapshot, parameter modification, task start/pause/resume/stop, log reading and
> splat/training preview screenshot. MCP bridge, recipes, editing, and multi-client leases were planned for later phases of this document.

### Current incremental implementation (2026-10-23)

- Parameter and view patches are validated as a whole before being applied; a failed patch makes no partial changes.
- `job status/wait --op OP` queries the specified task, and the memory retains the latest 128 completed tasks.
  The final states are succeeded, failed, and cancelled; the history is not retained after restarting.
- CLI returns 0 for success, 2 for failure/cancellation, 3 for connection failure, 4 for usage error, and 5 for timeout.
- The request can carry `request_id`, and will be deduplicated by client and request ID. If the parameters are inconsistent, CONFLICT will be returned.
  The maximum deduplication cache is 1024 entries, only for the current Studio session. CLI write commands automatically generate IDs;
  Use `--request-id ID` to retry across CLI calls. An HTTP timeout simply means the result is unknown;
  You should try again using the same instance, client, ID, and parameters. You cannot blindly resend by changing the ID.
- Commands queued for more than 5 seconds without being started return EXPIRED; commands that have been started may complete after HTTP timeout.
- `project save [path] [--overwrite]` Save or save as .ascan; currently only supports saving when idle.
- `view set yaw=1 pitch=0.3 distance=6 target=[0,0,0]` Modify camera;
  Support fov_degrees, ortho_height, projection, mode, channel, workspace.
  The unit of yaw/pitch is radians; projection uses perspective/orthographic/fisheye/panorama.
- `project set-dataset PATH` mounts external data sets (COLMAP/RealityCapture/OpenMVS directory),
   Use imported camera and skip Align, you can directly `job start train`; cooperate with Blender export plug-in
   (photara-blender) implements rendering → import pose → 3DGS training closed loop.
- `export model --out PATH --format ply --sh-degree 3` supports ply/sog/spz/glb.
  Saving/exporting requires absolute paths, either in the project, cache, temporary directory, or PHOTARA_CONTROL_ALLOWED_ROOTS.
  Explicitly specified existing files are not overwritten by default; use --overwrite to replace them; saving the current project without specifying a path will update the original file.

Verification entrance: `photara.studio.control`, `photara.studio.control_cli` CTest.
Real Studio Verification:

```powershell
python scripts/agent_control_smoke.py --studio build/photara/Release/photara_studio.exe --ctl build/photara/Release/photara-ctl.exe
# Add --images IMAGE_DIRECTORY to exercise the complete align/train/capture/save/export workflow.
```

The above is the actual implementation scope, and subsequent chapters of this article still contain target designs that have not yet been achieved.

### Subsequent increments

Control API has been added `studio.edit.apply` (select all, clear/inverse selection, delete selection, undo, redo),
`studio.export.mesh` (PLY/OBJ geometry, GLB optional texture) and `studio.inspect.quality`.
Editing requires that the splat viewport has been loaded and the undo stack of the UI is reused; execution is refused when busy or the gesture is in progress.
Multi-file export of textured OBJs remains in the UI for now.

The optional stdio MCP bridge and align/train recipe are located at `tools/photara_mcp/`,
For installation, startup and limitations, see README.md in this directory. Recipe is advanced by the client repeatedly advance,
Stop when session/project changes; return needs_review when quality metrics are missing or not up to standard.
There is currently no backend recipe persistence, multi-client tenancy, or perceived quality scoring.

## 1. Conclusion

The recommended route is:

1. **first build the semantic control plane inside Studio, and do not use UI click/OCR automation as the main path.**
2. **control commands uniformly go through the main thread command queue**, and the background thread is only responsible for network, JSON and event distribution.
3. ** provides both MCP server and `photara-ctl` CLI** to the outside world. MCP for clients that support MCP,
   CLI is provided to any coding agent that can execute terminal commands.
4. **also provides a deterministic recipe (workflow)**. The agent is responsible for selecting recipes, passing parameters, and viewing results, instead of letting the model play freely at every step.
5. **closes the control port** by default, explicitly opens it, and adds permissions, confirmations, path whitelists and audit logs.

In one sentence:

> Transform Studio from "a GUI that can only be clicked by humans" to "a local service with a stable semantic API",
> People, external agents, and automated tests all become clients of this API.

## 2. Constraints brought by the current code to the design

The following facts directly determine the plan and cannot be bypassed:

| Current Situation | Location | Design Implications |
|---|---|---|
| `App` is a single state root, including project, task, GPU preview, and editing status | `apps/editor/app.hpp` | The control layer must make a state snapshot around `App`, and cannot create another set of states |
| The main loop is GLFW + Dear ImGui immediate mode | `apps/editor/main.cpp` | All UI/Vulkan objects can only be touched in the main thread; commands must be delivered to the main loop for execution |
| The reconstruction task is run through a child process, and the log is redirected to the file | `ProcessJob`, `pipeline.cpp` | Do not reimplement task scheduling; reuse existing tasks and log parsing |
| `RunMonitor` has parsed the log into stage, progress, and training indicators | `pipeline.hpp` / `pipeline.cpp` | The control plane can directly consume structured progress without the need for an agent to read the log text |
| `ProjectSettings` is a strongly typed parameter set, and the path field is a UTF-8 char array | `pipeline.hpp` | requires stable JSON schema and path conversion, and cannot directly reflect the C++ structure |
| `.ascan` is the project's authoritative format, read and write through `photara::project` | `app_commands.cpp`, `photara/project` | The agent should call the open/save command, do not directly change `.ascan` |
| Artifact path and "existence" are deduced by `ProjectLayout` and `refresh_artifacts()` | `pipeline.hpp`, `app_commands.cpp` | State snapshots should expose artifact flags instead of letting the agent guess the path |
| There are already precondition checks such as `start_align`, `start_train`, `start_dense` | `app_commands.cpp` | These checks are returned in a structured manner to avoid repeated trial and error by the agent |
| `SplatEdit` has selected, deleted, undo/redo | `apps/editor/splat_edit.*` | Editing capabilities can be gradually exposed, but do not expose all mouse tools at the beginning |
| Existing smoke mode can start tasks without user operation | `main.cpp` | Can be reused as `--control-smoke` integration test |
| Currently there is no JSON, HTTP, IPC, MCP infrastructure | Full warehouse | Need to add a small control module and dependencies |
| Studio build capabilities vary due to different CMake switches | `PHOTARA_BUILD_STUDIO`, `PHOTARA_ENABLE_*` | API must return capabilities, agent cannot assume that there must be CUDA/SAM/Texture |

The two most important ones:

- **cannot directly call `start_align()`, `start_train()` or modify `App` from the network thread.**
- **cannot let LLM splice the CLI command string by itself.** All parameters must go through typed schema, range checking and path policy.

## 3. Target architecture

```text
┌─────────────────────────────────────────────────────────┐
│ Codex / Claude Code / DeepSeek client / zcode / self-research Agent │
│ planner + skill/playbook + multimodal observation                     │
└───────────────┬───────────────────────┬─────────────────┘
                │ MCP (stdio)            │ shell
┌───────────────▼──────────────┐  ┌─────▼─────────────────┐
│ photara-mcp bridge (Python)   │  │ photara-ctl (native)  │
│ Tools schema / progress / Reconnect  │  │ --json / doctor / job │
└───────────────┬──────────────┘  └─────┬─────────────────┘
                └───────────┬────────────┘
                            │ HTTP + JSON / SSE
                            │ 127.0.0.1 + bearer token
┌──────────────────▼───────────────────┐
│ Studio Control Server                 │
│ Authentication / JSON-RPC / event ring buffer    │
└──────────────────┬───────────────────┘
                   │ MainThreadExecutor
┌──────────────────▼───────────────────┐
│ CommandBus / StudioSession            │
│ get_state / params / job / recipe      │
│ capture / edit / export                │
└──────────────────┬───────────────────┘
                   │ Reuse existing functions
┌──────────────────▼───────────────────┐
│ App + app_commands + pipeline          │
│ ProcessJob → photara.exe               │
│ Project / Artifacts / RunMonitor       │
│ SplatEdit / GPU preview                │
└──────────────────────────────────────┘
```

Optional GUI-less path:

```text
Agent → MCP → photara_agentd(none Vulkan window)
             → same set CommandBus/recipe
             → photara.exe or photara Python module
```

### 3.1 Why not focus on UI automation?

UI automation (screenshot + mouse click) can be made into fallback, but it is not suitable as the main path:

- Slow and affected by window occlusion, DPI, language, and theme;
- There is no way to reliably know the internal status, such as "whether training has entered restoring progress";
- It is difficult to recover after making a mistake, and it is easy to click the wrong button;
- Unable to do structured verification, auditing and unit testing;
- Each step consumes visual tokens.

Semantic control surfaces can:

- Directly return stage, progress, error code, and artifact status;
- Perform precondition checks before command execution;
- Reuse existing undo/redo and task control;
- Use JSON contracts for regression testing.

## 4. Three-layer interface design

### 4.1 The first level: Atomic commands

Stable, fine-grained, and strongly typed. Used for debugging, testing and advanced agents.

Example:

- `studio.get_state`
- `studio.params.describe`
- `studio.params.patch`
- `studio.project.open`
- `studio.job.start`
- `studio.job.control`
- `studio.view.capture`
- `studio.edit.delete_selected`
- `studio.export.run`

### 4.2 Second layer: Recipe/Workflow

Deterministic state machines encapsulate common tasks into retryable, pauseable, and observable processes.

Example:

- `recipe.reconstruct_3dgs`
- `recipe.align_only`
- `recipe.mesh_from_model`
- `recipe.mesh_from_mvs`
- `recipe.texture_mesh`
- `recipe.export_bundle`

Recipe is code, not a script for model generation. The agent only selects the recipe and overrides a few parameters.

### 4.3 The third layer: Agent skill / playbook

Operation manual for LLM, containing:

- When to use which recipe;
- Quality threshold;
- Recovery strategies after failure;
- When do you need to ask the user;
- Which operations are destructive.

For example:

| Observations | Suggested Actions |
|---|---|
| alignment registration rate is low | switch hierarchical SfM, check camera model, reduce max_features, check mask |
| Training loss does not decrease for a long time | Check data/exposure, restore checkpoint, adjust learning rate or strategy |
| Insufficient video memory | Reduce max_resolution, densification_cap, sh_degree, or switch Vulkan backend |
| Calling texture without mesh | Returns `PRECONDITION_FAILED`, it is recommended to extract mesh first |
| Existing external SfM dataset | Skip align, go directly to train |

## 5. Core module

It is recommended to add a new directory:

```text
apps/editor/control/
├── control_protocol.hpp      # command/event/error code/schema version
├── control_bus.hpp/.cpp      # main thread executor,revision, idempotent
├── control_state.cpp         # App -> JSON status snapshot
├── control_commands.cpp      # command -> Existing editor function
├── control_server.hpp/.cpp   # HTTP + SSE + token
└── control_events.hpp/.cpp   # event ring buffer
```

### 5.1 MainThreadExecutor

The control server runs in a background thread, and commands cannot be executed directly. Use a thread-safe queue:

```cpp
struct CommandRequest {
    std::string request_id;
    std::string method;
    nlohmann::json params;
    std::promise<CommandResult> reply;
};

class ControlBus {
public:
    // Called by the background thread and delivered to the main thread.
    std::future<CommandResult> post(CommandRequest request);

    // Main loop call.
    void pump_main_thread(App& app);
    // in present Called later, the process will be rebuilt/release GPU Resource commands.
    void pump_post_present(App& app);

private:
    std::mutex mutex_;
    std::deque<CommandRequest> main_queue_;
    std::deque<CommandRequest> post_present_queue_;
};
```

`pump_main_thread` only executes short commands:

- read status;
- Modify parameters;
- Start the task;
- Pause/Resume/Stop;
- Submit a capture request.

`pump_post_present` handles things that in existing code must wait for the draw list of the current frame to be submitted before doing it:

- Stop；
- Recreate shared preview;
- Switch items;
- Delete data set;
- Release the splat preview descriptor.

This is consistent with the current practice of `main.cpp` and `e484d54 fix(studio): defer dataset removal until frame submission`.

### 5.2 State Snapshot

Don't let the agent read C++ structures. Define stable DTO:

```json
{
  "schema_version": 1,
  "session_id": "2b1f...",
  "revision": 57,
  "project": {
    "path": "D:/data/chuan.ascan",
    "name": "chuan",
    "dirty": false,
    "images": "D:/data/chuan/images",
    "dataset_source": null
  },
  "capabilities": {
    "cuda": true,
    "vulkan_training": true,
    "sam": true,
    "texture": true,
    "onnx": false
  },
  "artifacts": {
    "sparse": true,
    "asfm": true,
    "mvs": false,
    "splat": true,
    "mesh": false,
    "texture": false
  },
  "job": {
    "kind": "train",
    "op_id": "op_0007",
    "stage": "training",
    "fraction": 0.42,
    "eta_seconds": 812.5,
    "paused": false,
    "exit_code": null,
    "training": {
      "iteration": 12600,
      "total_iterations": 30000,
      "gaussians": 1843921,
      "loss": 0.0312,
      "rgb_loss": 0.0181,
      "depth_loss": 0.0021,
      "vram_used": 10485760000,
      "vram_total": 25769803776
    }
  },
  "actions": {
    "align": {"available": false, "reason": "job_running"},
    "train": {"available": false, "reason": "job_running"},
    "dense": {"available": false, "reason": "no_alignment"},
    "texture": {"available": false, "reason": "no_mesh"},
    "export": {"available": true, "reason": null}
  },
  "view": {
    "workspace": "scene_3d",
    "mode": "splat",
    "channel": "color",
    "camera": {
      "target": [0.0, 1.2, 0.0],
      "yaw": 0.7,
      "pitch": -0.2,
      "distance": 4.5
    }
  },
  "selection": {
    "gaussians": 1843921,
    "selected": 12034
  },
  "message": {
    "text": "Training 42%",
    "severity": "accent"
  }
}
```

Design points:

- `revision` monotonically increasing, used for optimistic concurrency control.
- `actions` directly reuses existing judgments such as `alignment_ready()`, `has_reconstruction_result()`, and `can_export_*()`.
- `artifact` The result from `refresh_artifacts()` will not be rescanned.
- `capabilities` comes from compile macros and runtime device queries.
- The first version does not need to expose all fields, first cover projects, parameters, tasks, products, and views.

### 5.3 Revision and Concurrency

The user may change parameters in the UI at the same time. `params.patch` supports:

```json
{
  "if_revision": 57,
  "patch": {
    "iterations": 30000,
    "strategy": "adc_igs"
  }
}
```

- If revision does not match, `CONFLICT` is returned, along with the latest state;
- The agent can choose to retry or ask the user first;
- A successful patch returns `changed` and normalized `normalized`;
- All values are uniformly normalized equivalent to `sanitize()`, and the values given by LLM are not trusted.

In implementation, `control_revision` can be added to `App`. The first version can perform fingerprint comparison on the exposed fields at the end of each frame, and increment it automatically when it changes; later it will be replaced by explicit `mark_state_changed()`.

### 5.4 Event flow

Event recommendations are pushed through SSE, event type:

| Events | Content |
|---|---|
| `state.changed` | revision, changed field path, merge |
| `job.started` | kind、op_id |
| `job.progress` | stage、fraction、ETA、training stats |
| `job.log` | severity、time、message |
| `job.finished` | exit_code、error、artifacts |
| `artifact.changed` | New/invalid products |
| `project.saved` | path、revision |
| `capture.ready` | File path, width and height |
| `permission.required` | Operations that require user confirmation |
| `agent.command` | For audit purposes, record who executed what |

SSE example:

```text
id: 58
event: job.progress
data: {"kind":"train","op_id":"op_0007","stage":"training","fraction":0.43,"eta_seconds":790,"revision":58}
```

The event ring only retains the most recent N entries (for example, 2000). After the client is disconnected, it can be filled with `Last-Event-ID`. Only the summary of the log is pushed, and the complete log can still be read from the `inspect.logs` page.

### 5.5 Transport layer

Recommended first edition:

- **Windows/Linux use 127.0.0.1 HTTP**;
- JSON-RPC 2.0 requests;
- SSE push events;
- Generate random tokens at startup;
- Write the port, token, pid, session_id, and project path into the discovery file;
- Only local connections are allowed.

Discovery file example:

```json
{
  "pid": 12345,
  "session_id": "2b1f...",
  "url": "http://127.0.0.1:48123",
  "token": "....",
  "project": "D:/data/chuan.ascan",
  "started_at": "2026-10-18T12:00:00+08:00"
}
```

Location:

- Windows：`%APPDATA%/Photara/control/<pid>.json`
- Linux: `$XDG_RUNTIME_DIR/photara/control/<pid>.json` or `$XDG_CONFIG_HOME/Photara/control/`

Alternative transmission:

- Windows named pipe + Unix domain socket: fewer ports exposed, but Python MCP bridge requires additional dependencies;
- File command queue: The simplest implementation, but the delay and event push experience are poor, so it is not recommended as the main solution.

HTTP scheme must pay attention to:

- Bind `127.0.0.1`, but cannot bind `0.0.0.0`;
- Check `Authorization: Bearer <token>`;
- Verify `Origin` to prevent browser web pages from being called through localhost CSRF;
- The token file permissions are set to be readable only by the current user;
- The release version is not started by default, only command line parameters or preferences are explicitly enabled.

### 5.6 MCP bridge

> MCP is an optional enhanced entrance, and `photara-ctl` is the universal entrance. Both share the same set of JSON-RPC contracts,
> Do not implement a separate set of business logic for MCP.

MCP server is recommended to be placed in a separate process, for example:

```text
tools/photara_mcp/
├── server.py
├── client.py
├── tools.py
└── skills/photara.md
```

Reason:

- GUI programs are not suitable to use stdio MCP as their own life cycle;
- Python’s MCP SDK, JSON, and HTTP clients are all mature;
- MCP server can return clear errors when Studio closes or crashes;
- Can support multiple Studio instances simultaneously.

MCP bridge is responsible for:

- Discovery Studio (scan for discovery files or explicit `--url`);
- Translate MCP tool calls into control commands;
- Translate SSE events into MCP progress notification;
- Long tasks return op_id and do not block the MCP main loop;
- Disconnection and reconnection;
- Require secondary confirmation of dangerous operations.

It is recommended that the tool list be limited to around 12 to avoid tool explosion:

| Tools | Function |
|---|---|
| `studio_get_state` | Projects, parameters, tasks, products, available actions |
| `studio_get_schema` | Parameter metadata, range, enumeration, default value |
| `studio_patch_parameters` | Modify parameters in batches, support dry-run |
| `studio_project` | new/open/save/save-as/import images |
| `studio_start_job` | align/train/resume/dense/texture/export |
| `studio_control_job` | pause/resume/stop |
| `studio_wait` | Waiting with an upper limit and returning to the current progress |
| `studio_inspect` | Logs, indicators, products, errors |
| `studio_capture` | Specify camera and resolution screenshot |
| `studio_set_view` | workspace/mode/channel/camera |
| `studio_edit` | Select, delete, undo/redo |
| `studio_run_recipe` | Execute deterministic workflow |

It is not recommended to expose:

- `run_shell`
- `run_cli_command`
- `write_file`
- `set_any_field`

### 5.7 Multi-client access (Codex / Claude Code / DeepSeek / zcode)

There is a premise that needs to be made clear first:

> The model itself cannot directly operate native applications. The client must be able to turn the model's actions into "executing local tools".

Local tools can be:

- Terminal commands;
- MCP tools;
- Custom HTTP tools;
- Plugin SDK.

If a client can only chat and cannot execute any local tools, then no matter whether the underlying layer is replaced by DeepSeek or
Zcode can only give operation suggestions and cannot actually drive Studio. The design should clearly distinguish between these two things.

#### Compatibility Matrix

| Client | Recommended access method | Description |
|---|---|---|
| Codex | MCP + `photara-ctl` | The best experience is when MCP is supported; if it is not supported, just use shell |
| Claude Code | MCP + `photara-ctl` | Called via `.mcp.json` or terminal |
| DeepSeek class client | `photara-ctl` priority | Depends on whether the client supports shell/custom tools |
| zcode / other IDE agent | `photara-ctl` or local HTTP API | as long as terminal or custom requests are supported |
| Self-developed agent / Python script | Local HTTP API | Directly call `/v1/rpc` |

Conclusion:

- **MCP is an enhanced entrance, not the only entrance. **-**`photara-ctl` is the universal entry. **-** local HTTP API is the bottom entry point. **
- The three share the same set of commands and status contracts, and do not maintain multiple sets of logic.

#### Why CLI is a universal entry point

Codex, Claude Code, DeepSeek, and zcode have different product forms, but most coding agents can:

- Execute the command in the project directory;
- Read stdout/stderr;
- Determine success or failure based on the exit code;
- Use JSON output as next step context.

Therefore `photara-ctl` can cover the widest range of clients:

```text
Codex / Claude Code / DeepSeek / zcode
              │ shell
              ▼
        photara-ctl.exe
              │ local HTTP + token
              ▼
        Studio Control Server
```

The MCP path simply wraps the same command into a tool:

```text
Codex / Claude Code / MCP Compatible client
              │ MCP stdio
              ▼
        photara-mcp
              │ local HTTP + token
              ▼
        Studio Control Server
```

#### `photara-ctl` Design

It is recommended to make it a native executable built with Studio to avoid requiring users to install additional Python:

```text
photara-ctl.exe      # Windows
photara-ctl          # Linux
```

Basic behavior:

- Automatically scan and discover files and connect to the running Studio;
- Support `--json`, all commands have machine-readable output;
- Human-readable output is concise by default and suitable for direct viewing;
- Express the error category through exit code;
- Support `--project` filtering multiple Studio instances;
- Read the token in the discovery file and do not put the token on the command line;
- Do not execute arbitrary shells, only send whitelisted commands.

Suggested command:

```text
photara-ctl doctor
photara-ctl state --json
photara-ctl schema --json
photara-ctl params set iterations=30000 strategy=adc_igs
photara-ctl job start train --json
photara-ctl job status --op op_0007 --json
photara-ctl job wait --op op_0007 --timeout 30 --json
photara-ctl job pause
photara-ctl job resume
photara-ctl job stop
photara-ctl watch --jsonl
photara-ctl capture --out D:/data/agent/view.png
photara-ctl recipe run reconstruct_3dgs \
    --images D:/data/chuan \
    --quality standard \
    --export D:/data/chuan.sog \
    --json
```

`doctor` is used to quickly judge:

- Whether Studio is running;
- Whether the versions match;
- Is the token correct?
- What is the current project;
- Whether there are tasks running;
- What permissions the current client has.

`schema` outputs complete parameter metadata, and the agent does not need to memorize CLI flags.

#### Prompt words for coding agent

There is no need to write a set of plug-ins for each client. Just put a short rule in the project description or system prompt:

```text
you can pass photara-ctl control running Photara Studio.
Before starting, execute:
    photara-ctl state --json
Execute when parameters are required:
    photara-ctl schema --json
Do not splice directly photara.exe command.
Do not modify it directly .ascan or cache file.
For long tasks job start + job wait, don't block indefinitely.
Destructive operations ask the user first.
```

Codex can be placed at `AGENTS.md`, Claude Code can be placed at `CLAUDE.md`,
Other clients are placed in their system prompt / project rules.

#### MCP configuration example

Codex style:

```toml
[mcp_servers.photara]
command = "photara-mcp"
args = ["--auto-discover"]
```

Claude Code style:

```json
{
  "mcpServers": {
    "photara": {
      "command": "photara-mcp",
      "args": ["--auto-discover"]
    }
  }
}
```

Clients that do not support MCP do not need these configurations, just ensure that `photara-ctl` is in `PATH`.

#### Multi-client concurrency

Multiple agents operating the same Studio at the same time will cause conflicts. For example, Codex is changing parameters.
Claude Code starts training at the same time. It is recommended to add "execute lease":

```text
photara-ctl session acquire --client codex --ttl 30m
photara-ctl session renew
photara-ctl session release
```

Rules:

- All clients can connect read-only;
- Only one client holds an execution lease at a time;
- Only clients holding leases can modify parameters, start/stop tasks, and save projects;
- Other clients receive `LEASE_REQUIRED` or `BUSY`;
- Studio UI displays currently connected clients and lease holders;
- User can force lease revocation on UI.

The first version can first use the existing "single task lock" to limit concurrency; multi-client leases are placed in Phase 2.

#### Typical usage process

The user opens the project in Studio and then says in Codex / Claude Code:

```text
put D:\data\chuan Rebuild into 3DGS,Standard quality, final export SOG.
```

Agent actually executes:

```text
photara-ctl state --json
photara-ctl project open D:/data/chuan.ascan --json
photara-ctl schema --json
photara-ctl recipe run reconstruct_3dgs \
    --images D:/data/chuan/images \
    --quality standard \
    --json
photara-ctl job wait --op op_0007 --timeout 30 --json
photara-ctl capture --out D:/data/chuan/agent_view.png
photara-ctl export run --format sog --out D:/data/chuan.sog --json
```

Users can see parameters, progress, previews and logs in real time in Studio; they can click Stop at any time.

#### Minimum requirements for client access

To declare "support for a certain client" on a product, at least the following must be met:

1. The client can execute local commands or call MCP/HTTP tools;
2. The client can read stdout or tool return value;
3. The client has project-level prompt words or tool descriptions and knows the existence of `photara-ctl`;
4. The client supports multiple rounds of tool calls, because the reconstruction task requires "observation → waiting → judgment".

Clients that only have a "chat window" but no tool execution are not supported.

### 5.8 No GUI mode

In the second stage, you can add:

```text
photara_studio.exe --headless --control-url ...
```

Or a cleaner standalone program:

```text
photara_agentd.exe
```

It reuses:

- `ProjectSettings`
- `ProjectLayout`
- `build_*_command`
- `ProcessJob`
- `RunMonitor`

But no Vulkan window is created and ImGui is not executed. This allows you to use the same set of command APIs in server, CI, and batch processing scenarios.

What the GUI shares with headless is the **CommandBus + recipe layer **, not the entire `App`. GPU preview, camera and editing status in `App` are only available in GUI mode.

## 6. Command and status contract

### 6.1 Request format

```http
POST /v1/rpc
Authorization: Bearer <token>
Content-Type: application/json
```

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "studio.params.patch",
  "params": {
    "if_revision": 57,
    "dry_run": false,
    "patch": {
      "iterations": 30000,
      "strategy": "adc_igs",
      "max_resolution": 1920
    }
  }
}
```

### 6.2 Successful response

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "result": {
    "ok": true,
    "revision": 58,
    "changed": ["iterations", "strategy", "max_resolution"],
    "normalized": {
      "iterations": 30000,
      "strategy": "adc_igs",
      "max_resolution": 1920
    }
  }
}
```

### 6.3 Domain error

```json
{
  "jsonrpc": "2.0",
  "id": 18,
  "result": {
    "ok": false,
    "error": {
      "code": "PRECONDITION_FAILED",
      "message": "Cannot bake texture before a mesh exists",
      "retryable": false,
      "details": {
        "action": "texture",
        "reason": "no_mesh",
        "suggested_action": "studio_start_job",
        "suggested_args": {"stage": "dense", "mesh_source": "gaussians"}
      }
    },
    "revision": 58
  }
}
```

Error code suggestions:

| code | meaning |
|---|---|
| `INVALID_REQUEST` | JSON/schema error |
| `UNSUPPORTED` | Current build does not support this capability |
| `PRECONDITION_FAILED` | The precondition is not met |
| `BUSY` | Tasks are already running |
| `CONFLICT` | revision mismatch or project being modified concurrently |
| `NOT_FOUND` | Project/file/product does not exist |
| `PERMISSION_DENIED` | Path or permission policy denied |
| `CONFIRMATION_REQUIRED` | Destructive operation requires confirmation |
| `CANCELLED` | User or agent cancel |
| `INTERNAL` | Uncategorized internal error |

### 6.4 Task command

Start the task:

```json
{
  "method": "studio.job.start",
  "params": {
    "stage": "train",
    "resume": false,
    "dry_run": false
  }
}
```

Return:

```json
{
  "ok": true,
  "op_id": "op_0007",
  "kind": "train",
  "revision": 59
}
```

Task startup is asynchronous. Progress is pushed through events, or with:

```json
{
  "method": "studio.wait",
  "params": {
    "op_id": "op_0007",
    "timeout_ms": 30000
  }
}
```

`studio.wait` Waits up to the specified time and then returns to the current state. Don't let the MCP tool block indefinitely.

Control tasks:

```json
{
  "method": "studio.job.control",
  "params": {
    "op_id": "op_0007",
    "action": "pause"
  }
}
```

`action`：`pause`、`resume`、`stop`。

### 6.5 Parameter metadata

`studio.params.describe` returns:

```json
{
  "groups": [
    {
      "id": "training",
      "label": "Training",
      "fields": [
        {
          "name": "iterations",
          "type": "int",
          "min": 1,
          "max": 1000000,
          "default": 30000,
          "affects": ["train"],
          "description": "Total training iterations"
        },
        {
          "name": "strategy",
          "type": "enum",
          "values": ["adc_igs", "adc_plus", "dense_adaptive"],
          "default": "adc_igs",
          "affects": ["train"]
        }
      ]
    }
  ]
}
```

The first version recommended exposing only high-impact fields:

- alignment: `camera_model`, `sfm_mode`, `max_features`, `reuse_cache`, SAM related;
- training：`iterations`、`strategy`、`training_mode`、`sh_degree`、`densification_cap`、`max_resolution`、`progressive_resolution`、`geometry_from_iter`、`normal_field`、AI prior weights、`ppisp_layout`；
- video：`fps`、`max_frames`、`sharp_window`、`quality`、`scale`、`rotate`；
- mesh：`build_mesh`、`mesh_source`、`mesh_method`、quality、simplify、remesh；
- texture：`texture_quality`、`atlas_resolution`、`texture_delight`、`texture_optimize`；
- export：`splat_format`。

The full field can be added later, but it must be ensured that the field name returned by the API has a clear mapping with `ProjectSettings`.

### 6.6 Screenshots/Observations

```json
{
  "method": "studio.view.capture",
  "params": {
    "width": 1280,
    "height": 720,
    "source": "viewport",
    "show_ui": false,
    "camera": {
      "mode": "orbit",
      "target": [0.0, 1.0, 0.0],
      "yaw": 0.7,
      "pitch": -0.2,
      "distance": 4.5
    },
    "path": "D:/data/agent/capture_001.png"
  }
}
```

Return:

```json
{
  "ok": true,
  "path": "D:/data/agent/capture_001.png",
  "width": 1280,
  "height": 720,
  "revision": 60
}
```

The first version can:

- read back from the swap chain after `present`;
- Temporarily hide ImGui or only crop the 3D viewport area;
- Reuse `PreviewTexture::download_rgb()` or add offscreen capture.

In the future, it is recommended to create ** independent offscreen renderer**, so that the image can be stably produced when the window is minimized, blocked, and under different DPI. Currently, `ExternalPreview::consume_without_present()` has proven that it can continue to consume training previews when minimized, and the ideas can be reused.

## 7. Threads, lifecycle and resource rules

Hard rules:

1. `App`, ImGui, and Vulkan resources are only accessed in the main thread.
2. The background thread only does socket, JSON parse, and event caching.
3. Command execution must be a short operation; long operations are converted into existing sub-process tasks.
4. Commands to release/rebuild GPU resources must be executed after present.
5. Screenshot readback must be performed between rendering frames and wait for fence.
6. Only one reconstruction task is allowed at the same time; `studio.job.start` returns `BUSY` when busy.

Task state machine:

```text
idle
  ├─ align ─────────────┐
  ├─ train ─────────────┤
  ├─ dense ─────────────┤
  ├─ texture ───────────┤
  └─ export_sfm ────────┤
                        ▼
                    finished
                    ├─ success
                    ├─ failed
                    └─ stopped
```

`align` may automatically load SfM after success; `dense` may go to 3DGS extraction or MVS according to `mesh_source`; these branches already exist, and the control layer should not copy the logic.

Studio exits or crashes:

- Default strategy: The task continues to run, and observation is resumed through artifacts/checkpoint after the agent reconnects;
- Optional policy: `--kill-jobs-on-disconnect`, for strict resource control;
- MCP bridge must be able to detect `session_id` changes and cannot regard the old op_id as the task of the new session.

## 8. Security and Permissions

It is recommended to define four permission modes:

| Mode | Allow |
|---|---|
| `observe` | Read status, logs, screenshots |
| `edit` | Modify parameters, views, selections |
| `execute` | Start/pause/stop tasks, save projects |
| `destructive` | Delete results, clear cache, overwrite export, merge models |

Implementation:

- Default `observe`;
- User authorization in preferences or startup parameters;
- destructive operation requirement `confirm_token`;
- The confirmation request pops up through the UI, and the agent cannot forge the confirmation by itself;
- `destructiveHint` of MCP tool is consistent with the permission mode.

Path strategy:

- The input path must exist and be readable;
- The output path must be in the project directory, cache directory or export directory explicitly authorized by the user;
- Deny UNC/network paths and system directories by default;
- Do not use shell splicing;
- All paths are converted via existing `path_from_utf8_field()` / `store_utf8_path_field()`;
- Watch out for long paths and UTF-8/UTF-16 conversion on Windows.

Audit:

- Each command is written to JSONL, including timestamp, client, method, parameter summary, results, and time consumption;
- Write to the session directory or project cache;
- UI console can optionally display `agent.command` summary;
- Destructive operations must be traceable in audit logs.

Emergency stop:

- The UI Stop button is always available;
- Added "Disconnect agent" or "Disable agent control" switch;
- Tasks will not be killed by default when disconnected, but new commands will be rejected immediately.

## 9. Agent behavior design

### 9.1 Recommended high-level workflow

`recipe.reconstruct_3dgs`：

```text
1. preflight
   - Check images/video/dataset
   - Check disk space, video memory,ffmpeg
   - Return warnings/errors
2. alignment
   - If available alignment And no recalculation is required, skip
   - external dataset then skip
3. alignment quality gate
   - Registration rate, reprojection error, number of valid points,warnings
   - Return if not up to standard structured failure,agent Decide whether to try again
4. training
   - Choose a strategy/parameters
   - start train, subscribe progress
5. evaluation
   - loss,gaussian count,VRAM, screenshot
6. optional export
   - PLY/SOG/SPZ/GLB/USDC
```

`recipe.mesh_from_model`：

```text
preflight -> need splat -> extract mesh -> validate -> optional simplify/remesh
```

`recipe.mesh_from_mvs`：

```text
preflight -> need alignment -> dense -> mesh -> validate
```

`recipe.texture_mesh`：

```text
preflight -> need mesh -> texture bake -> validate output files
```

### 9.2 Quality threshold

Recipe returns structured metrics such as:

```json
{
  "ok": true,
  "metrics": {
    "registered_images": 118,
    "total_images": 120,
    "registration_rate": 0.983,
    "mean_reprojection_error_px": 0.71,
    "training_final_loss": 0.027,
    "gaussians": 1843921,
    "peak_vram_bytes": 10485760000
  },
  "warnings": []
}
```

The threshold should not be hard-coded in the prompt, but should be placed in the recipe code or configurable strategy. The agent is only responsible for interpreting and deciding the next step.

### 9.3 Failure recovery

| Errors | AutoRecovery Suggestions |
|---|---|
| `BUSY` | Wait for the current op or stop first |
| `PRECONDITION_FAILED` | Press `suggested_action` to perform pre-steps |
| `OUT_OF_MEMORY` | Lower the resolution/density limit and try again |
| `ALIGNMENT_LOW_COVERAGE` | Toggle hierarchical, check image/mask, try again |
| `TRAINING_DIVERGED` | Check data, restore checkpoint, reduce learning rate |
| `DISK_FULL` | Clean the cache or change the output directory without automatically deleting user files |

### 9.4 When must you ask the user

- Overwrite existing projects or export files;
- Delete results and clear cache;
- Open another project and discard unsaved changes;
- Estimated run time/disk/video memory exceeds threshold;
- Quality threshold failure and automatic recovery strategy may change user intent.

## 10. List of code changes

### 10.1 New

| Documentation | Content |
|---|---|
| `apps/editor/control/control_protocol.hpp` | schema version, error code, command/event DTO |
| `apps/editor/control/control_bus.*` | Main thread queue, revision, idempotent |
| `apps/editor/control/control_state.cpp` | `App` -> JSON snapshot, actions/capabilities |
| `apps/editor/control/control_commands.cpp` | Command dispatch to existing editor function |
| `apps/editor/control/control_server.*` | HTTP, SSE, token, discovery file |
| `apps/editor/control/control_events.*` | event ring buffer |
| `apps/ctl/photara_ctl.cpp` | Universal CLI client (native, no Python dependencies) |
| `tools/photara_mcp/server.py` | MCP server |
| `tools/photara_mcp/client.py` | HTTP/SSE client |
| `tools/photara_mcp/tools.py` | Tool definition |
| `tools/photara_mcp/skills/photara.md` | agent playbook |
| `tools/photara_agent/AGENTS.md` | Codex Project Rules Template |
| `tools/photara_agent/CLAUDE.md` | Claude Code Project Rules Template |
| `tools/photara_agent/system-prompt.txt` | General coding agent prompt words |
| `schemas/studio-control.schema.json` | Status/command schema (optional but recommended) |
| `docs/AGENT_CONTROL.md` | This document |

### 10.2 Modification

| Documentation | Changes |
|---|---|
| `apps/editor/app.hpp` | Add `control_revision`, possible control status |
| `apps/editor/main.cpp` | Parse `--enable-agent-control`, initialize server, pump queue, present and then pump |
| `apps/editor/app_commands.cpp` | Change the key precondition to return the structured reason; add the entrance to open/save/set the source by path |
| `apps/editor/pipeline.hpp` | Expose necessary command construction/layout information; add parameter metadata interface |
| `photara/CMakeLists.txt` | Added control target/sources, `photara-ctl` executable file, `nlohmann_json`, `httplib`, Windows `ws2_32` |
| `CMakeLists.txt` | Optional `PHOTARA_ENABLE_AGENT_CONTROL` switch |
| `scripts/generate_api_reference.py` or new script | Generate/verify control schema document |

### 10.3 It is not recommended to do it in the first version

- Fully reconstruct `App` to MVVM;
- Change all inspector controls to schema drivers;
- Directly integrate LLM SDK in C++;
- Expose all `SplatEdit` mouse tools;
- Let the agent directly edit `.ascan` or cache files;
- The listening port is enabled by default.

### 10.4 Notes on CMake and test targets

Several test targets in `photara/CMakeLists.txt` will copy the source list of `photara_studio`.
For example `photara_gaussian_session_test`:

```cmake
get_target_property(_session_sources photara_studio SOURCES)
add_executable(photara_gaussian_session_test
    tests/gaussian_session_test.cpp ${_session_sources})
```

If the control server, HTTP, and JSON source files are directly inserted into `photara_studio`, these tests will also be forced to
Compile and link the control layer, increasing dependencies and build time. Suggestions:

- Put the parts not related to `App` (protocol, event ring, HTTP server, JSON codec)
  Independent static library `photara_studio_control`;
- `photara_studio` links to this library;
- `control_commands.cpp` can stay in the source list of `photara_studio` because it depends on `App`.
  Or put it into a `photara_studio_control_app` library that depends on the editor;
- The test target links the control library on demand instead of copying all source files indiscriminately;
- Remember to link `ws2_32` (HTTP/SSE server) on Windows, no additional libraries are required on Linux;
- If using header-only `httplib`, you still need to handle the thread and socket compilation options.

In the first version, you can also not split the static library first, but you must explicitly check all copies of the studio source list when implementing it.
Test the target to avoid CI failure due to new network dependencies.

## 11. Phased roadmap

### Phase 0: Protocol and Skeleton (3-5 days)

Delivery:

- `control_protocol.hpp`；
- `control_bus`；
- `--enable-agent-control`；
- `studio.ping`、`studio.get_state`；
- `photara-ctl` native CLI: `doctor`, `state`, `schema`;
- Unit testing: JSON encoding and decoding, revision, error codes.

Acceptance:

- After Studio is started, the status can be read through local HTTP;
- When the flag is turned off, there is no port and no file found;
- State snapshots are consistent with UI display.

### Phase 1: Read-only + Mission Control MVP (1-2 weeks)

Delivery:

- `params.describe`, `params.patch` (high impact fields);
- `job.start` / `pause` / `resume` / `stop`；
- `RunMonitor` progress and training indicator events;
- `inspect.logs`、`inspect.artifacts`；
- `studio.view.capture`；
- `photara-ctl`：`params`、`job start/status/wait/pause/resume/stop`、`capture`；
- MCP bridge：`studio_get_state`、`studio_start_job`、`studio_control_job`、`studio_wait`、`studio_capture`。

Acceptance:

- External scripts can be done on `PHOTARA_SMOKE_IMAGES` align + train;
- You can receive stage/progress/loss during the training process;
- stop is executed safely after present;
- Screenshotting works when the window is minimized (at least it doesn't crash).

### Phase 2: Parameters, Recipe, Quality Assessment (2-3 weeks)

Delivery:

- `recipe.reconstruct_3dgs`、`recipe.mesh_from_model`、`recipe.texture_mesh`；
- alignment/training quality indicators;
- structured `suggested_action`;
- dry-run preflight (disk/video memory/precondition);
- Permission mode and audit log.

Acceptance:

- The agent can use a recipe to complete common rebuilds;
- Invalid commands will not be repeatedly tried when the preconditions are not met;
- Clear reasons will be returned when the quality is not up to standard.

### Phase 3: Editing, exporting, productization (3-5 weeks)

Delivery:

- `edit.select_*`、`edit.delete_selected`、undo/redo；
-Multiple model merge/visibility/transform;
-Commandization of export panel;
- headless `photara_agentd`；
- Optional in-app agent panel;
- Multi-instance discovery and reconnection;
- More complete schema/codegen and regression testing.

Acceptance:

- The agent can complete the closed loop of "alignment → training → inspection → editing → export";
- Destructive operations require confirmation;
- Task status can be restored after Studio crashes.

Overall estimate:

- Read-only + Mission Control MVP: **1-2 weeks**;
- Version for daily use: **4-6 weeks **;
- Productization (permissions, editing, headless, testing): **8-12 weeks **.

It depends on whether a single person develops, whether a cross-platform is needed, and whether `App` needs to be further split.

## 12. Key technology trade-offs

### 12.1 Why use the main thread queue instead of letting the command thread lock

The coupling between `App` and Vulkan/ImGui is too deep, making it unrealistic to lock all fields and prone to deadlock. The main thread queue converges the concurrency boundary to one place:

- The background thread only parses and queues;
- Main thread executes sequentially;
- Resource life cycle is consistent with frame boundaries;
- Tests can call `pump_main_thread()` synchronously.

### 12.2 Why HTTP instead of directly named pipe

- Python MCP bridge and debugging client are the easiest to connect;
- Browser/script/CI can be used;
- Native loopback + token is enough;
- Add named pipe as an optional transport when stricter isolation is required.

### 12.3 Why MCP is not put into the GUI process

- The GUI process life cycle is controlled by the user;
- stdio MCP requires exclusive use of stdin/stdout;
- An independent bridge makes it easier to reconnect, multiple instances, log and upgrade;
- Studio does not need to link to MCP SDK.

### 12.4 Why is headless still needed?

- Batch jobs, servers, and CI should not launch Vulkan windows;
- Python API/CLI can already run core algorithms;
- But headless cannot replace the real-time preview and editing of the GUI, so the two coexist.

## 13. Suggested first PR

Don’t make your first PR “big and comprehensive”. It is recommended to make a vertical slice:

```text
--enable-agent-control
  -> start loopback HTTP server
  -> studio.ping
  -> studio.get_state
  -> studio.params.patch(3-5 fields)
  -> studio.job.start（train）
  -> studio.job.control（stop）
  -> SSE job.progress
  -> studio.view.capture
```

Paired with a Python script:

```text
scripts/agent_control_smoke.py
  1. Read discovery file
  2. get_state
  3. patch iterations=100
  4. start train
  5. Subscribe progress
  6. stop
  7. capture
  8. Assert status and products
```

This PR verifies four most critical things:

1. Whether the main thread execution model is established;
2. Whether the existing `start_train`/`ProcessJob`/`RunMonitor` can be stably driven externally;
3. Whether the safety stop does not return after present;
4. Whether the screenshot link is available.

If these four things are verified, the subsequent parameters, recipes, editing and MCP are all incremental work.

## 14. Final recommendations

Sort by priority:

1. **first build the semantic state and command bus**, do not build the chat UI first.
2. ** first covers align/train/stop/capture**, which is the minimum set of agent closed loops.
3. **uses recipe to solidify the successful path**. Do not let LLM freely combine dozens of underlying fields.
4. **`photara-ctl` serves as the general entrance **, ensuring Codex, Claude Code, DeepSeek, and zcode
   As long as you can execute commands in the terminal, you can access.
5. The **MCP independent process serves as the enhanced entry point **, giving clients that support MCP a better tool experience.
6. **Permissions, confirmation, and auditing The interface** has been left in the first version, otherwise it will be difficult to make up for it later.
7. **UI automation is only used as fallback** for verification or extreme scenarios, and is not used as the main path of the product.
