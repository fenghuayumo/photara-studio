# Photara Studio Agent Control

## 1. Current Implementation

### 1.1 Call Flow and Code Entry Points

```text
External agent / script
  ├─ photara-ctl (native CLI)
  └─ stdio MCP bridge (Python) → photara-ctl
                    ↓
          127.0.0.1 HTTP + Bearer token
                    ↓
        Studio Control Server → ControlBus
                    ↓
          Studio main thread / post-present queue
                    ↓
       App / app_commands / ProcessJob / RunMonitor
                    ↓
          State, job results, events, captures and exports
```

The control layer reuses Studio's project, job, preview and editing state, including its existing reconstruction scheduling.
The network thread handles HTTP, JSON and command submission. Commands that access `App`, ImGui or Vulkan execute on the main thread.

| Code entry point | Responsibility |
|---|---|
| `apps/editor/main.cpp` | Parse startup arguments, start/stop the server and pump commands in the main loop and after present |
| `apps/editor/control/control_protocol.hpp` | Schema version 1 and success/failure result helpers |
| `apps/editor/control/control_server.*` | Loopback HTTP, authentication, discovery files and request waiting |
| `apps/editor/control/control_bus.*` | Main-thread queues, request deduplication, revisions, job history and event cache |
| `apps/editor/control/control_state.*` | State snapshots, parameter descriptions, patch validation and application |
| `apps/editor/control/control_commands.*` | Command dispatch, preconditions and output path checks |
| `apps/editor/control/control_paths.*` | Discovery directory, session IDs and token generation |
| `apps/ctl/photara_ctl.cpp` | Instance discovery, HTTP client, CLI commands and exit codes |
| `tools/photara_mcp/` | Stdio MCP tools, CLI adapter and in-memory reconstruction recipe |

### 1.2 Enabling Control and Selecting an Instance

Build Studio and `photara-ctl` with `PHOTARA_ENABLE_AGENT_CONTROL` enabled.
The runtime listener is disabled by default. Enable it explicitly:

```powershell
& ./build/photara/Release/photara_studio.exe --enable-agent-control
# Optional fixed port: append --agent-control-port PORT. The default is an OS-assigned port.
```

The server binds only to `127.0.0.1`. On startup, it writes `<PID>.json` containing `pid`, `session_id`,
`url`, `token`, `version` and `started_at_ms`. Normal shutdown removes the file.

The discovery directory is resolved in this order:

- `PHOTARA_CONTROL_DIR`, when set. Studio and the CLI should use the same value.
- Windows: `%APPDATA%/Photara/control`, then `%LOCALAPPDATA%/Photara/control`.
- Other platforms: `$XDG_CONFIG_HOME/Photara/control`, then `$HOME/.config/Photara/control`.
- If none of these variables is available, `photara-control` under the current directory.

```powershell
& ./build/photara/Release/photara-ctl.exe --pid 12345 doctor
& ./build/photara/Release/photara-ctl.exe --pid 12345 state --json
& ./build/photara/Release/photara-ctl.exe --pid 12345 schema --json
```

The CLI probes each candidate's health endpoint. Without an explicit selection, it prefers the most recently started reachable instance.
Automation should pin `--pid`. It can also filter by `--project PATH` or connect explicitly with `--url URL --token TOKEN`.
Discovery files contain credentials and should stay out of logs and version control.

### 1.3 Available Commands

| Function | Studio method | CLI |
|---|---|---|
| Health and state | `studio.ping`, `studio.get_state` | `doctor`, `state` |
| Describe, read and update parameters | `studio.params.describe`, `studio.params.patch`; reads use state | `schema`, `params get`, `params set key=value...` |
| Open a project or set input | `studio.project.open`, `studio.project.set_source`, `studio.project.set_dataset` | `project open/set-source/set-dataset PATH` |
| Save a project | `studio.project.save` | `project save [PATH] [--overwrite]` |
| Start and inspect jobs | `studio.job.start`, `studio.job.status` | `job start STAGE`, `job status/wait [--op OP]` |
| Pause, resume and stop jobs | `studio.job.control` | `job pause/resume/stop` |
| Read logs | `studio.inspect.logs` | `inspect logs [--limit N]` |
| Check quality | `studio.inspect.quality` | `quality` |
| Update the view and capture previews | `studio.view.patch`, `studio.view.capture` | `view set key=value...`, `capture --out PATH` |
| Edit | `studio.edit.apply` | `edit ACTION` |
| Export models and meshes | `studio.export.model`, `studio.export.mesh` | `export model/mesh --out PATH` |
| Open auxiliary windows | `studio.window.open` | `window open video/mask` |
| Read and poll events | HTTP `/v1/events` | `events [--since ID]`, `watch` |

Supported job stages are `align`, `train`, `resume_training`, `dense`, `texture` and `export_sfm`.
`job start STAGE --dry-run` checks action availability in the state snapshot without starting a job.
It does not perform a full disk-space, VRAM or dependency check.

State snapshots include `schema_version`, `session_id`, `revision`, `project`, `capabilities`,
`artifacts`, `job`, `actions`, `view`, `selection`, `settings` and `message`.
Callers should use the actual `capabilities` and `actions` to choose the next step. SAM, texture support and training backends vary by build.

### 1.4 Input, Training, Inspection and Export

The example below pins an instance by PID. Replace the input/output paths with local paths.
If the output directory is outside the project, cache and system temporary directories, allow it before starting Studio.

```powershell
# Set before launching Studio. Separate multiple directories with semicolons.
$env:PHOTARA_CONTROL_ALLOWED_ROOTS = 'D:/agent-output'
& ./build/photara/Release/photara_studio.exe --enable-agent-control

# In another terminal, replace 12345 with the Studio process ID.
$ctl = (Resolve-Path './build/photara/Release/photara-ctl.exe').Path
& $ctl --pid 12345 project set-source D:/data/scene/images
& $ctl --pid 12345 params set iterations=30000
& $ctl --pid 12345 job start align --json
# Replace OP_ALIGN with the returned op_id. After a timeout, keep waiting for the same ID.
& $ctl --pid 12345 job wait --op OP_ALIGN --timeout 60 --json
& $ctl --pid 12345 job start train --json
# Replace OP_TRAIN with the op_id returned by job start train.
& $ctl --pid 12345 job wait --op OP_TRAIN --timeout 60 --json
& $ctl --pid 12345 quality --json
& $ctl --pid 12345 capture --out D:/agent-output/preview.png --max-long-edge 1280
& $ctl --pid 12345 project save D:/agent-output/scene.ascan
& $ctl --pid 12345 export model --out D:/agent-output/model.ply --format ply --sh-degree 3
```

Confirm that each job succeeds before proceeding. The scene or preview may still be loading before the first `train`;
check the latest state and handle `BUSY`. A `job wait` timeout does not stop the Studio job.

For an existing COLMAP, RealityCapture or OpenMVS dataset, use `project set-dataset PATH`,
then start `train` directly, skipping `align`. This is also the entry point for training from Blender camera/data exports.

Parameter patches are validated on a copy and applied only after all fields pass. Invalid fields or values cause no partial parameter changes.
Use `schema` for parameter names, types and enums. Reconstruction parameter changes are rejected while a job is running or training preparation is active.
View patches are also validated as a whole. Supported fields are target, yaw, pitch, distance, fov_degrees,
ortho_height, projection, mode, channel and workspace. Yaw/pitch use radians.
Accepted projection values are `perspective/orthographic/fisheye/panorama`.

### 1.5 Jobs, Queues and Request Retries

- `job start` returns an `op_id`. Bind subsequent queries and waits to that ID to avoid following a new job started by another client.
- Job states include `preparing`, `running`, `paused` and `finishing`. Terminal states are `succeeded`, `failed` and `cancelled`.
- The latest 128 completed jobs are retained in memory. Unknown or evicted IDs return `NOT_FOUND`; history is lost on restart.
- The main-thread and post-present queues hold at most 256 commands in total. A command that has not begun executing within 5 seconds returns `EXPIRED`.
- Job start, project open/set-source, save, edit, export and stop commands execute through the post-present queue.
- HTTP waits up to 5 seconds for a command result. `TIMEOUT` means the outcome is unknown; a command that has already started may still complete.
- Requests are deduplicated by `client + request_id`, with the method and parameters compared as well. Reusing an ID with different content returns `CONFLICT`.
- The deduplication cache holds at most 1024 entries within the current Studio session. Completed entries may be evicted; idempotency is not permanent.

The CLI automatically generates request IDs for write commands. For retries across CLI invocations, preserve the instance, client, request ID and arguments:

```powershell
& $ctl --pid 12345 --client agent --request-id train-001 job start train --json
# Repeat this exact command only to retrieve the result of the same request.
```

If the session has changed or the request may have left the cache, inspect the actual state before resubmitting a write with a new ID.
A failed `session_id` check returns `SESSION_CHANGED`. Currently, `revision` marks state changes;
conditional writes using `if_revision` and multi-client leases are not implemented.

CLI exit codes are 0 for success, 2 for command failure/job failure or cancellation, 3 for discovery or initial connection failure,
4 for usage errors, and 5 for `job wait` timeout. Connection errors during an RPC call may also return 2 through the command-failure path.
Automation should inspect JSON `error.code` as well as the process exit code.

### 1.6 HTTP and Events

All endpoints require `Authorization: Bearer TOKEN`:

| Endpoint | Behavior |
|---|---|
| `GET /v1/health` | Return version, session, PID and revision |
| `POST /v1/rpc` | Accept a command and wait for the queued result |
| `GET /v1/events?since=ID` | Return event JSON after the supplied cursor |

Example RPC request:

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "session_id": "CURRENT_SESSION",
  "client": "agent",
  "request_id": "params-001",
  "method": "studio.params.patch",
  "params": {"patch": {"iterations": 30000}}
}
```

Executed commands return their application result in `result`, with `ok` indicating success. Failures include
`error.code`, `error.message` and `error.details`. Authentication, malformed HTTP requests, session validation and
HTTP wait timeouts can return a top-level error with a non-200 status.

Events use an in-memory ring buffer of up to 2000 entries, including `job.started`, `job.progress`,
`job.finished` and `state.changed`. Responses include `events`, `next` and `revision`.
The transport uses JSON polling. `watch` polls once per second and emits one JSON object per event; SSE is not implemented.

### 1.7 Save, Edit, Capture and Export Constraints

Save, capture and export destinations must be absolute paths under the project directory, cache directory,
system temporary directory or a root listed in `PHOTARA_CONTROL_ALLOWED_ROOTS`. Directory containment is checked using normalized paths.

| Operation | Current constraints |
|---|---|
| Save `.ascan` | Requires idle state; the first save needs a path; omitting the path updates the current project file |
| Save/export overwrite | An explicit existing destination is rejected unless `--overwrite` is supplied; saving the current project without a path updates the original file |
| Edit | Supports `select_all`, `clear_selection`, `invert_selection`, `delete_selected`, `undo` and `redo`; requires a loaded splat viewport, rejects busy/active-gesture states and shares the UI undo stack |
| Capture | Reads RGB from the splat renderer or training preview and writes PNG; `max_long_edge` is limited to 0–8192, with 0 meaning unlimited; no available preview returns `UNSUPPORTED` |
| Model export | Supports `ply/sog/sog-xt/spz/glb`, with `sh_degree` in 0–3; check the returned `completed` field and observe `job.splat_export` for asynchronous exports |
| Mesh export | PLY/OBJ geometry and GLB with optional `--include-texture`; textured OBJ sidecar export still requires the UI |

Capture does not currently use the save/export check that requires explicit overwrite for an existing file. Callers should use unique filenames.
Capture reads preview pixels; it has no interface for a full UI screenshot, arbitrary output dimensions, a temporary camera or independent offscreen rendering.
Switching inputs with unsaved changes returns `CONFIRMATION_REQUIRED`; save first or resolve the changes in Studio.

### 1.8 MCP and Recipe Execution

The MCP bridge uses Python 3.10+ and the official MCP SDK `mcp>=1.28,<2`. It invokes the native CLI through subprocesses.
See [tools/photara_mcp/README.md](../tools/photara_mcp/README.md) for setup and startup.
Set `PHOTARA_CTL` to the CLI executable and `PHOTARA_STUDIO_PID` to pin the instance. Stdio is reserved for MCP messages.

Available tools cover state, parameter updates, job start/control/wait, capture, quality checks, editing, export,
and `studio_recipe_start` / `studio_recipe_advance`. MCP tools expose a subset of the native CLI commands and options.

The current recipe flow is:

1. `start(iterations)` checks busy state and active recipes, updates the iteration count, records the session and project, and returns a `run_id`.
2. The client repeatedly calls `advance(run_id)` to start or observe align, then start or observe train.
3. A change to the session or project snapshot marks the recipe as failed.
4. After both stages succeed, quality checks run. All checks passing produces `succeeded`; missing or failing metrics produce `needs_review`.

Default native quality thresholds are registered-image ratio ≥ 0.8, mean reprojection error ≤ 2 px and training loss ≤ 0.1.
RPC parameters `min_registered_ratio`, `max_mean_reprojection` and `max_training_loss` override these thresholds.
Metrics come from the currently loaded scene and latest training monitor. Unknown metrics do not pass, and loss is not a perceptual quality score.

## 2. Problems

These limitations are established from the current source; they are not a record of reproduced runtime failures.

| Problem | Current impact |
|---|---|
| Incomplete permissions and auditing | A client holding the token can call write methods; no observe/edit/execute permission tiers, general confirmation token or persistent JSONL command audit |
| No coordination between writers | No leases, command ownership isolation or conditional writes; pause/stop targets the current active job, so clients must avoid contention |
| In-memory state and request records | Job history, events and deduplication records are lost on Studio restart; old op_ids cannot be used in a new session |
| No event cursor gap indication | Events beyond the 2000-entry retention window disappear; clients must reread full state after a long disconnection |
| Serial HTTP connection handling | One slow request can block other clients; concurrent-load behavior is not guaranteed |
| Limited dry-run coverage | Action availability does not guarantee that files, GPU, dependencies and resources will satisfy actual startup requirements |
| Capture depends on an existing preview | No independent offscreen-rendering guarantee; capture is unavailable without preview pixels and lacks a consistent overwrite policy |
| Inconsistent export job tracking | Model export may complete asynchronously through `splat_export`, without a unified export op_id workflow |
| Limited recipe input branches | The fixed align → train sequence does not skip align for external datasets or existing alignment, and has no mesh/texture/export stages |
| No recipe persistence or automatic advancement | Up to 128 plans are retained without eviction; clients must keep calling advance; bridge restart loses plans while started Studio jobs require separate tracking |
| Full project snapshot comparison in recipes | Equality checks include the entire project object, including dirty state; normal state changes may prematurely fail a plan |
| Limited MCP write recovery | Each mutation call generates a new request ID, with no tool parameter for retrying a failed call using its original ID |
| Limited quality metrics | Training metrics may be unknown after reopening a project; no perceptual quality score, unified resource preflight or automatic recovery strategy |

## 3. Follow-up Work

These items remain open, ordered by gaps in the current workflow.

- [ ] Fix recipe input identification: compare stable project/input identities so dirty-state changes do not interrupt execution.
- [ ] Add recipe branches for external datasets and existing alignment, with regression coverage.
- [ ] Preserve request ID, client and session in MCP error results, and provide a recovery path for the same write request.
- [ ] Unify overwrite policies and completion reporting across save, capture, model export and mesh export; provide explicit tracking for asynchronous exports.
- [ ] Define expired/gapped event cursor behavior and support full state resynchronization.
- [ ] Add coordination between writers, conditional writes, permission tiers and persistent command auditing.
- [ ] Extend resource/dependency preflight to distinguish action availability from actual execution readiness.
- [ ] Complete recipe lifecycle support: cancellation, history eviction, persistence and reassociation after bridge restart.
- [ ] Extend quality metrics and mesh/texture/export workflows while preserving review requirements for unknown metrics.
- [ ] Add integration coverage for slow connections, instance contention, restart/reconnection, capture availability and asynchronous exports.

## 4. Verification

The following tests and scripts are available. Runtime behavior must be verified against the corresponding build and input data.

```powershell
# Build the relevant targets first. Adjust the build directory to the local configuration.
ctest --test-dir build -C Release -R '^photara\.studio\.control(_cli)?$' --output-on-failure

# Recipe unit tests using a test client.
python -m unittest discover -s tools/photara_mcp -p test_recipes.py

# Launch real Studio to check protocol, atomic patches, deduplication, view updates and error handling.
python scripts/agent_control_smoke.py --studio build/photara/Release/photara_studio.exe --ctl build/photara/Release/photara-ctl.exe
# Append --images IMAGE_DIRECTORY to cover align/train/capture/save/export.
```

- `photara.studio.control`: C++ regression coverage for the control bus, parameters, commands and quality checks.
- `photara.studio.control_cli`: native CLI interaction with a test HTTP server.
- `tools/photara_mcp/test_recipes.py`: recipe advancement and failure/quality-review branches.
- `scripts/agent_control_smoke.py`: control-service smoke testing with a real GUI process; full reconstruction requires image input.
