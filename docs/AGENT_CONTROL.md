# Photara Studio Agent 控制方案

> 状态：设计草案
> 目标版本：0.6.x 之后
> 目标：让外部或内置 AI agent 以可验证、可撤销、可审计的方式驱动当前运行的 Photara Studio。

> 实现进度：`agent_dev` 分支已落地 Phase 0/MVP——Studio 本地控制服务、
> `photara-ctl`、状态快照、参数修改、任务启动/暂停/恢复/停止、日志读取和
> splat/training 预览截图。MCP bridge、recipe、编辑和多客户端租约仍按本文档后续实现。

## 1. 结论

推荐路线是：

1. **先做 Studio 内部的语义控制面（control plane），不要以 UI 点击/OCR 自动化为主路径。**
2. **控制命令统一走主线程命令队列**，后台线程只负责网络、JSON 和事件分发。
3. **对外同时提供 MCP server 和 `photara-ctl` CLI**。MCP 给支持 MCP 的客户端，
   CLI 给所有能执行终端命令的 coding agent。
4. **同时提供确定性 recipe（工作流）**，agent 负责选 recipe、传参数、看结果，而不是每一步都让模型自由发挥。
5. **默认关闭控制端口**，显式开启，并加入权限、确认、路径白名单和审计日志。

一句话概括：

> 把 Studio 从“只能被人点击的 GUI”改造成“有一个稳定语义 API 的本地服务”，
> 人、外部 agent、自动化测试都成为这个 API 的客户端。

## 2. 当前代码给设计带来的约束

以下事实直接决定了方案，不能绕开：

| 现状 | 位置 | 设计含义 |
|---|---|---|
| `App` 是单一状态根，包含项目、任务、GPU 预览、编辑状态 | `apps/editor/app.hpp` | 控制层必须围绕 `App` 做状态快照，不能另建一套状态 |
| 主循环是 GLFW + Dear ImGui immediate mode | `apps/editor/main.cpp` | 所有 UI/Vulkan 对象只能在主线程碰；命令要投递到主循环执行 |
| 重建任务通过子进程运行，日志重定向到文件 | `ProcessJob`、`pipeline.cpp` | 不要重新实现任务调度；复用现有任务和日志解析 |
| `RunMonitor` 已把日志解析成 stage、进度、训练指标 | `pipeline.hpp` / `pipeline.cpp` | 控制面可以直接消费结构化进度，不需要 agent 读日志文本 |
| `ProjectSettings` 是强类型参数集合，路径字段是 UTF-8 char 数组 | `pipeline.hpp` | 需要稳定的 JSON schema 和路径转换，不能直接反射 C++ 结构 |
| `.ascan` 是项目权威格式，通过 `photara::project` 读写 | `app_commands.cpp`、`photara/project` | agent 应调用 open/save 命令，不要直接改 `.ascan` |
| 产物路径和“是否存在”由 `ProjectLayout` 和 `refresh_artifacts()` 推导 | `pipeline.hpp`、`app_commands.cpp` | 状态快照应暴露 artifact flags，而不是让 agent 猜路径 |
| 已有 `start_align`、`start_train`、`start_dense` 等前置条件检查 | `app_commands.cpp` | 把这些检查结构化返回，避免 agent 反复试错 |
| `SplatEdit` 已有选择、删除、undo/redo | `apps/editor/splat_edit.*` | 编辑能力可以逐步暴露，但不要一开始暴露所有鼠标工具 |
| 已有 smoke 模式可以在无用户操作下启动任务 | `main.cpp` | 可以复用为 `--control-smoke` 集成测试 |
| 当前没有 JSON、HTTP、IPC、MCP 基础设施 | 全仓库 | 需要新增一个小型 control 模块和依赖 |
| Studio 构建能力因 CMake 开关不同而不同 | `PHOTARA_BUILD_STUDIO`、`PHOTARA_ENABLE_*` | API 必须返回 capabilities，agent 不能假设一定有 CUDA/SAM/Texture |

最重要的两条：

- **不能从网络线程直接调用 `start_align()`、`start_train()` 或修改 `App`。**
- **不能让 LLM 自己拼接 CLI 命令字符串。** 所有参数必须经过类型化 schema、范围校验和路径策略。

## 3. 目标架构

```text
┌─────────────────────────────────────────────────────────┐
│ Codex / Claude Code / DeepSeek 客户端 / zcode / 自研 Agent │
│ planner + skill/playbook + 多模态观察                     │
└───────────────┬───────────────────────┬─────────────────┘
                │ MCP (stdio)            │ shell
┌───────────────▼──────────────┐  ┌─────▼─────────────────┐
│ photara-mcp bridge (Python)   │  │ photara-ctl (native)  │
│ 工具 schema / progress / 重连  │  │ --json / doctor / job │
└───────────────┬──────────────┘  └─────┬─────────────────┘
                └───────────┬────────────┘
                            │ HTTP + JSON / SSE
                            │ 127.0.0.1 + bearer token
┌──────────────────▼───────────────────┐
│ Studio Control Server                 │
│ 鉴权 / JSON-RPC / event ring buffer    │
└──────────────────┬───────────────────┘
                   │ MainThreadExecutor
┌──────────────────▼───────────────────┐
│ CommandBus / StudioSession            │
│ get_state / params / job / recipe      │
│ capture / edit / export                │
└──────────────────┬───────────────────┘
                   │ 复用现有函数
┌──────────────────▼───────────────────┐
│ App + app_commands + pipeline          │
│ ProcessJob → photara.exe               │
│ Project / Artifacts / RunMonitor       │
│ SplatEdit / GPU preview                │
└──────────────────────────────────────┘
```

可选的无 GUI 路径：

```text
Agent → MCP → photara_agentd（无 Vulkan 窗口）
             → 同一套 CommandBus/recipe
             → photara.exe 或 photara Python 模块
```

### 3.1 为什么不是 UI 自动化为主

UI 自动化（截图 + 鼠标点击）可以做成 fallback，但不适合作为主路径：

- 慢，且受窗口遮挡、DPI、语言、主题影响；
- 无法可靠知道内部状态，例如“训练是否已经进入 restoring progress”；
- 出错后难以恢复，容易点错按钮；
- 不能做结构化校验、审计和单元测试；
- 每一步都消耗视觉 token。

语义控制面则可以：

- 直接返回 stage、进度、错误码、artifact 状态；
- 在命令执行前做前置条件检查；
- 复用现有 undo/redo 和任务控制；
- 用 JSON 契约做回归测试。

## 4. 三层接口设计

### 4.1 第一层：原子命令

稳定、细粒度、强类型。用于调试、测试和高级 agent。

示例：

- `studio.get_state`
- `studio.params.describe`
- `studio.params.patch`
- `studio.project.open`
- `studio.job.start`
- `studio.job.control`
- `studio.view.capture`
- `studio.edit.delete_selected`
- `studio.export.run`

### 4.2 第二层：Recipe / 工作流

确定性状态机，把常见任务封装成可重试、可暂停、可观测的流程。

示例：

- `recipe.reconstruct_3dgs`
- `recipe.align_only`
- `recipe.mesh_from_model`
- `recipe.mesh_from_mvs`
- `recipe.texture_mesh`
- `recipe.export_bundle`

Recipe 是代码，不是模型生成的脚本。Agent 只选择 recipe 并覆盖少量参数。

### 4.3 第三层：Agent skill / playbook

给 LLM 的操作手册，包含：

- 何时用哪个 recipe；
- 质量门限；
- 失败后的恢复策略；
- 什么时候需要询问用户；
- 哪些操作有破坏性。

例如：

| 观察 | 建议动作 |
|---|---|
| alignment 注册率低 | 切换 hierarchical SfM、检查 camera model、减少 max_features、检查 mask |
| 训练 loss 长时间不降 | 检查数据/曝光、恢复 checkpoint、调整学习率或策略 |
| 显存不足 | 降低 max_resolution、densification_cap、sh_degree，或切换 Vulkan backend |
| 没有 mesh 却调用 texture | 返回 `PRECONDITION_FAILED`，先建议 extract mesh |
| 已有外部 SfM dataset | 跳过 align，直接 train |

## 5. 核心模块

建议新增目录：

```text
apps/editor/control/
├── control_protocol.hpp      # 命令/事件/错误码/schema version
├── control_bus.hpp/.cpp      # 主线程执行器、revision、幂等
├── control_state.cpp         # App -> JSON 状态快照
├── control_commands.cpp      # 命令 -> 现有 editor 函数
├── control_server.hpp/.cpp   # HTTP + SSE + token
└── control_events.hpp/.cpp   # 事件 ring buffer
```

### 5.1 MainThreadExecutor

控制服务器运行在后台线程，命令不能直接执行。用一个线程安全队列：

```cpp
struct CommandRequest {
    std::string request_id;
    std::string method;
    nlohmann::json params;
    std::promise<CommandResult> reply;
};

class ControlBus {
public:
    // 后台线程调用，投递到主线程。
    std::future<CommandResult> post(CommandRequest request);

    // 主循环调用。
    void pump_main_thread(App& app);
    // 在 present 之后调用，处理会重建/释放 GPU 资源的命令。
    void pump_post_present(App& app);

private:
    std::mutex mutex_;
    std::deque<CommandRequest> main_queue_;
    std::deque<CommandRequest> post_present_queue_;
};
```

`pump_main_thread` 只执行短命令：

- 读状态；
- 修改参数；
- 启动任务；
- 暂停/恢复/停止；
- 提交 capture 请求。

`pump_post_present` 处理现有代码里必须等当前帧 draw list 提交后才能做的事：

- Stop；
- 重新创建 shared preview；
- 切换项目；
- 删除数据集；
- 释放 splat preview descriptor。

这与当前 `main.cpp` 和 `e484d54 fix(studio): defer dataset removal until frame submission` 的做法保持一致。

### 5.2 State Snapshot

不要让 agent 读 C++ 结构。定义稳定的 DTO：

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

设计要点：

- `revision` 单调递增，用于乐观并发控制。
- `actions` 直接复用 `alignment_ready()`、`has_reconstruction_result()`、`can_export_*()` 等现有判断。
- `artifact` 来自 `refresh_artifacts()` 的结果，不重新扫盘。
- `capabilities` 来自编译宏和运行时设备查询。
- 第一版不必暴露全部字段，先覆盖项目、参数、任务、产物、视图。

### 5.3 Revision 与并发

用户可能同时在 UI 里改参数。`params.patch` 支持：

```json
{
  "if_revision": 57,
  "patch": {
    "iterations": 30000,
    "strategy": "adc_igs"
  }
}
```

- revision 不匹配返回 `CONFLICT`，并附带最新 state；
- agent 可以选择重试或先询问用户；
- 成功的 patch 返回 `changed` 和归一化后的 `normalized`；
- 所有值经过与 `sanitize()` 等价的统一规范化，不信任 LLM 给的数值。

实现上可以给 `App` 增加 `control_revision`。第一版可以在每帧末对暴露字段做指纹比较，变化时自增；后续再替换为显式 `mark_state_changed()`。

### 5.4 事件流

事件建议通过 SSE 推送，事件类型：

| 事件 | 内容 |
|---|---|
| `state.changed` | revision、变化的字段路径，做合并 |
| `job.started` | kind、op_id |
| `job.progress` | stage、fraction、ETA、training stats |
| `job.log` | severity、time、message |
| `job.finished` | exit_code、error、artifacts |
| `artifact.changed` | 新增/失效产物 |
| `project.saved` | path、revision |
| `capture.ready` | 文件路径、宽高 |
| `permission.required` | 需要用户确认的操作 |
| `agent.command` | 审计用，记录谁执行了什么 |

SSE 示例：

```text
id: 58
event: job.progress
data: {"kind":"train","op_id":"op_0007","stage":"training","fraction":0.43,"eta_seconds":790,"revision":58}
```

事件环只保留最近 N 条（例如 2000 条），客户端断线后可以用 `Last-Event-ID` 补齐。日志只推送摘要，完整日志仍可从 `inspect.logs` 分页读取。

### 5.5 传输层

推荐第一版：

- **Windows/Linux 都用 127.0.0.1 HTTP**；
- JSON-RPC 2.0 请求；
- SSE 推送事件；
- 启动时生成随机 token；
- 把端口、token、pid、session_id、项目路径写入发现文件；
- 只允许本机连接。

发现文件示例：

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

位置：

- Windows：`%APPDATA%/Photara/control/<pid>.json`
- Linux：`$XDG_RUNTIME_DIR/photara/control/<pid>.json` 或 `$XDG_CONFIG_HOME/Photara/control/`

备选传输：

- Windows named pipe + Unix domain socket：更少端口暴露，但 Python MCP bridge 需要额外依赖；
- 文件命令队列：实现最简单，但延迟和事件推送体验差，不建议作为主方案。

HTTP 方案必须注意：

- 绑定 `127.0.0.1`，不能绑定 `0.0.0.0`；
- 校验 `Authorization: Bearer <token>`；
- 校验 `Origin`，避免浏览器网页通过 localhost CSRF 调用；
- token 文件权限设为仅当前用户可读；
- 发布版默认不启动，只有命令行参数或偏好设置显式开启。

### 5.6 MCP bridge

> MCP 是可选增强入口，`photara-ctl` 才是通用入口。两者共用同一套 JSON-RPC 契约，
> 不要为 MCP 单独实现一套业务逻辑。

MCP server 建议放在独立进程，例如：

```text
tools/photara_mcp/
├── server.py
├── client.py
├── tools.py
└── skills/photara.md
```

原因：

- GUI 程序不适合把 stdio MCP 作为自己的生命周期；
- Python 的 MCP SDK、JSON、HTTP 客户端都成熟；
- Studio 关闭或崩溃时，MCP server 可以返回清晰错误；
- 可以同时支持多个 Studio 实例。

MCP bridge 负责：

- 发现 Studio（扫描发现文件或显式 `--url`）；
- 把 MCP 工具调用翻译成控制命令；
- 把 SSE 事件翻译成 MCP progress notification；
- 长任务返回 op_id，不阻塞 MCP 主循环；
- 断线重连；
- 对危险操作要求二次确认。

工具列表建议控制在 12 个左右，避免 tool explosion：

| 工具 | 作用 |
|---|---|
| `studio_get_state` | 项目、参数、任务、产物、可用动作 |
| `studio_get_schema` | 参数元数据、范围、枚举、默认值 |
| `studio_patch_parameters` | 批量修改参数，支持 dry-run |
| `studio_project` | new/open/save/save-as/import images |
| `studio_start_job` | align/train/resume/dense/texture/export |
| `studio_control_job` | pause/resume/stop |
| `studio_wait` | 有上限的等待，返回当前进度 |
| `studio_inspect` | 日志、指标、产物、错误 |
| `studio_capture` | 指定相机和分辨率截图 |
| `studio_set_view` | workspace/mode/channel/camera |
| `studio_edit` | 选择、删除、undo/redo |
| `studio_run_recipe` | 执行确定性工作流 |

不建议暴露：

- `run_shell`
- `run_cli_command`
- `write_file`
- `set_any_field`

### 5.7 多客户端接入（Codex / Claude Code / DeepSeek / zcode）

这里有一个前提需要先说清楚：

> 模型本身不能直接操作本机应用。客户端必须能把模型的动作变成“执行本地工具”。

本地工具可以是：

- 终端命令；
- MCP 工具；
- 自定义 HTTP 工具；
- 插件 SDK。

如果某个客户端只能聊天，不能执行任何本地工具，那么无论底层换成 DeepSeek 还是
zcode，都只能给出操作建议，不能真正驱动 Studio。设计上要明确区分这两件事。

#### 兼容矩阵

| 客户端 | 推荐接入方式 | 说明 |
|---|---|---|
| Codex | MCP + `photara-ctl` | 支持 MCP 时体验最好；不支持时直接走 shell |
| Claude Code | MCP + `photara-ctl` | 通过 `.mcp.json` 或终端调用 |
| DeepSeek 类客户端 | `photara-ctl` 优先 | 取决于客户端是否支持 shell / 自定义工具 |
| zcode / 其他 IDE agent | `photara-ctl` 或本地 HTTP API | 只要支持终端或自定义请求即可 |
| 自研 agent / Python 脚本 | 本地 HTTP API | 直接调 `/v1/rpc` |

结论：

- **MCP 是增强入口，不是唯一入口。**
- **`photara-ctl` 是通用入口。**
- **本地 HTTP API 是底层入口。**
- 三者共用同一套命令和状态契约，不维护多套逻辑。

#### 为什么 CLI 是通用入口

Codex、Claude Code、DeepSeek、zcode 的产品形态不同，但大多数 coding agent 都能：

- 在项目目录执行命令；
- 读取 stdout/stderr；
- 根据退出码判断成败；
- 把 JSON 输出作为下一步上下文。

因此 `photara-ctl` 可以覆盖最广的客户端：

```text
Codex / Claude Code / DeepSeek / zcode
              │ shell
              ▼
        photara-ctl.exe
              │ local HTTP + token
              ▼
        Studio Control Server
```

MCP 路径只是把同样的命令包装成工具：

```text
Codex / Claude Code / MCP 兼容客户端
              │ MCP stdio
              ▼
        photara-mcp
              │ local HTTP + token
              ▼
        Studio Control Server
```

#### `photara-ctl` 设计

建议把它做成随 Studio 一起构建的原生可执行文件，避免要求用户额外安装 Python：

```text
photara-ctl.exe      # Windows
photara-ctl          # Linux
```

基本行为：

- 自动扫描发现文件，连接正在运行的 Studio；
- 支持 `--json`，所有命令都有机器可读输出；
- 人类可读输出默认简洁，适合直接看；
- 通过退出码表达错误类别；
- 支持 `--project` 过滤多个 Studio 实例；
- 读取发现文件里的 token，不把 token 放在命令行；
- 不执行任意 shell，只发送白名单命令。

建议命令：

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

`doctor` 用来快速判断：

- Studio 是否在运行；
- 版本是否匹配；
- token 是否正确；
- 当前项目是什么；
- 是否有任务在运行；
- 当前客户端有哪些权限。

`schema` 输出完整的参数元数据，agent 不需要死记 CLI flag。

#### 给 coding agent 的提示词

不需要给每个客户端写一套插件。只要在项目说明或系统提示里放一段简短规则：

```text
你可以通过 photara-ctl 控制正在运行的 Photara Studio。
开始前先执行：
    photara-ctl state --json
需要参数时执行：
    photara-ctl schema --json
不要直接拼接 photara.exe 命令。
不要直接修改 .ascan 或 cache 文件。
长任务用 job start + job wait，不要无限阻塞。
破坏性操作先询问用户。
```

Codex 可以放在 `AGENTS.md`，Claude Code 可以放在 `CLAUDE.md`，
其他客户端放在它们的 system prompt / project rules 中。

#### MCP 配置示例

Codex 风格：

```toml
[mcp_servers.photara]
command = "photara-mcp"
args = ["--auto-discover"]
```

Claude Code 风格：

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

不支持 MCP 的客户端不需要这些配置，直接确保 `photara-ctl` 在 `PATH` 中即可。

#### 多客户端并发

多个 agent 同时操作同一个 Studio 会产生冲突，例如 Codex 在改参数，
Claude Code 同时启动训练。建议增加“执行租约”：

```text
photara-ctl session acquire --client codex --ttl 30m
photara-ctl session renew
photara-ctl session release
```

规则：

- 所有客户端都可以只读连接；
- 同一时间只有一个客户端持有执行租约；
- 持有租约的客户端才能修改参数、启动/停止任务、保存项目；
- 其他客户端收到 `LEASE_REQUIRED` 或 `BUSY`；
- Studio UI 显示当前连接的客户端和租约持有者；
- 用户可以在 UI 上强制收回租约。

第一版可以先用现有的“单任务锁”限制并发；多客户端租约放在 Phase 2。

#### 典型使用流程

用户在 Studio 中打开项目，然后在 Codex / Claude Code 里说：

```text
把 D:\data\chuan 重建成 3DGS，Standard 质量，最后导出 SOG。
```

Agent 实际执行：

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

用户可以在 Studio 里实时看到参数、进度、预览和日志；随时可以点 Stop。

#### 客户端接入的最低要求

要在产品上宣称“支持某个客户端”，至少要满足：

1. 客户端能执行本地命令，或能调用 MCP/HTTP 工具；
2. 客户端能读取 stdout 或工具返回值；
3. 客户端有项目级提示词或工具描述，知道 `photara-ctl` 的存在；
4. 客户端支持多轮工具调用，因为重建任务需要“观察 → 等待 → 再判断”。

只有“聊天窗口”而没有工具执行的客户端，不在支持范围内。

### 5.8 无 GUI 模式

第二阶段可以增加：

```text
photara_studio.exe --headless --control-url ...
```

或者更干净的独立程序：

```text
photara_agentd.exe
```

它复用：

- `ProjectSettings`
- `ProjectLayout`
- `build_*_command`
- `ProcessJob`
- `RunMonitor`

但不创建 Vulkan 窗口，不执行 ImGui。这样可以在服务器、CI、批处理场景中使用同一套命令 API。

GUI 与 headless 共享的是 **CommandBus + recipe 层**，不是整个 `App`。`App` 中的 GPU 预览、相机和编辑状态只在 GUI 模式可用。

## 6. 命令与状态契约

### 6.1 请求格式

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

### 6.2 成功响应

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

### 6.3 领域错误

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

错误码建议：

| code | 含义 |
|---|---|
| `INVALID_REQUEST` | JSON/schema 错误 |
| `UNSUPPORTED` | 当前构建不支持该能力 |
| `PRECONDITION_FAILED` | 前置条件不满足 |
| `BUSY` | 已有任务在运行 |
| `CONFLICT` | revision 不匹配或项目被并发修改 |
| `NOT_FOUND` | 项目/文件/产物不存在 |
| `PERMISSION_DENIED` | 路径或权限策略拒绝 |
| `CONFIRMATION_REQUIRED` | 破坏性操作需要确认 |
| `CANCELLED` | 用户或 agent 取消 |
| `INTERNAL` | 未分类内部错误 |

### 6.4 任务命令

启动任务：

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

返回：

```json
{
  "ok": true,
  "op_id": "op_0007",
  "kind": "train",
  "revision": 59
}
```

任务启动是异步的。进度通过事件推送，或者用：

```json
{
  "method": "studio.wait",
  "params": {
    "op_id": "op_0007",
    "timeout_ms": 30000
  }
}
```

`studio.wait` 最多等待指定时间，然后返回当前状态。不要让 MCP tool 无限阻塞。

控制任务：

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

### 6.5 参数元数据

`studio.params.describe` 返回：

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

第一版建议只暴露高影响字段：

- alignment：`camera_model`、`sfm_mode`、`max_features`、`reuse_cache`、SAM 相关；
- training：`iterations`、`strategy`、`training_mode`、`sh_degree`、`densification_cap`、`max_resolution`、`progressive_resolution`、`geometry_from_iter`、`normal_field`、AI prior weights、`ppisp_layout`；
- video：`fps`、`max_frames`、`sharp_window`、`quality`、`scale`、`rotate`；
- mesh：`build_mesh`、`mesh_source`、`mesh_method`、quality、simplify、remesh；
- texture：`texture_quality`、`atlas_resolution`、`texture_delight`、`texture_optimize`；
- export：`splat_format`。

全量字段可以后续补，但必须保证 API 返回的字段名与 `ProjectSettings` 有明确映射。

### 6.6 截图/观察

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

返回：

```json
{
  "ok": true,
  "path": "D:/data/agent/capture_001.png",
  "width": 1280,
  "height": 720,
  "revision": 60
}
```

第一版可以：

- 在 `present` 之后从交换链读回；
- 临时隐藏 ImGui 或只截 3D viewport 区域；
- 复用 `PreviewTexture::download_rgb()` 或新增 offscreen capture。

后续建议做 **独立 offscreen renderer**，这样窗口最小化、被遮挡、不同 DPI 下都能稳定出图。当前 `ExternalPreview::consume_without_present()` 已经证明可以在最小化时继续消费训练预览，思路可以复用。

## 7. 线程、生命周期和资源规则

硬规则：

1. `App`、ImGui、Vulkan resource 只在主线程访问。
2. 后台线程只做 socket、JSON parse、事件缓存。
3. 命令执行必须是短操作；长操作转成现有子进程任务。
4. 释放/重建 GPU 资源的命令必须在 present 之后执行。
5. 截图读回必须在渲染帧之间执行，并等待 fence。
6. 同一时间只允许一个重建任务；`studio.job.start` 在忙时返回 `BUSY`。

任务状态机：

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

`align` 成功后可能自动加载 SfM；`dense` 可能根据 `mesh_source` 转到 3DGS extraction 或 MVS；这些分支已经存在，控制层不要复制逻辑。

Studio 退出或崩溃：

- 默认策略：任务继续运行，agent 重连后通过 artifacts/checkpoint 恢复观察；
- 可选策略：`--kill-jobs-on-disconnect`，用于严格资源控制；
- MCP bridge 必须能检测 `session_id` 变化，不能把旧 op_id 当成新 session 的任务。

## 8. 安全与权限

建议定义四种权限模式：

| 模式 | 允许 |
|---|---|
| `observe` | 读状态、日志、截图 |
| `edit` | 修改参数、视图、选择 |
| `execute` | 启动/暂停/停止任务、保存项目 |
| `destructive` | 删除结果、清缓存、覆盖导出、合并模型 |

实现：

- 默认 `observe`；
- 用户在偏好设置或启动参数中授权；
- destructive 操作要求 `confirm_token`；
- 确认请求通过 UI 弹出，agent 不能自己伪造确认；
- MCP tool 的 `destructiveHint` 与权限模式一致。

路径策略：

- 输入路径必须存在且可读；
- 输出路径必须在项目目录、cache 目录或用户显式授权的导出目录下；
- 默认拒绝 UNC/网络路径和系统目录；
- 不使用 shell 拼接；
- 所有路径通过现有 `path_from_utf8_field()` / `store_utf8_path_field()` 转换；
- Windows 上注意长路径和 UTF-8/UTF-16 转换。

审计：

- 每次命令写入 JSONL，包括 timestamp、client、method、参数摘要、结果、耗时；
- 写到 session 目录或项目 cache；
- UI 控制台可选择显示 `agent.command` 摘要；
- 破坏性操作必须可以在审计日志里追溯。

紧急停止：

- UI 的 Stop 按钮始终可用；
- 增加“断开 agent”或“禁用 agent 控制”开关；
- 断开时默认不杀任务，但立即拒绝新命令。

## 9. Agent 行为设计

### 9.1 推荐的高层工作流

`recipe.reconstruct_3dgs`：

```text
1. preflight
   - 检查 images/video/dataset
   - 检查磁盘空间、显存、ffmpeg
   - 返回 warnings/errors
2. alignment
   - 如已有可用 alignment 且未要求重算，跳过
   - 外部 dataset 则跳过
3. alignment quality gate
   - 注册率、重投影误差、有效点数、warnings
   - 不达标则返回 structured failure，agent 决定是否重试
4. training
   - 选择策略/参数
   - 启动 train，订阅 progress
5. evaluation
   - loss、gaussian count、VRAM、截图
6. optional export
   - PLY/SOG/SPZ/GLB/USDC
```

`recipe.mesh_from_model`：

```text
preflight -> 需要 splat -> extract mesh -> validate -> optional simplify/remesh
```

`recipe.mesh_from_mvs`：

```text
preflight -> 需要 alignment -> dense -> mesh -> validate
```

`recipe.texture_mesh`：

```text
preflight -> 需要 mesh -> texture bake -> validate output files
```

### 9.2 质量门限

Recipe 返回结构化指标，例如：

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

门限不要写死在 prompt 里，而应放在 recipe 代码或可配置策略中。Agent 只负责解释和决定下一步。

### 9.3 失败恢复

| 错误 | 自动恢复建议 |
|---|---|
| `BUSY` | 等待当前 op 或先 stop |
| `PRECONDITION_FAILED` | 按 `suggested_action` 执行前置步骤 |
| `OUT_OF_MEMORY` | 降低分辨率/密度上限，重试 |
| `ALIGNMENT_LOW_COVERAGE` | 切换 hierarchical、检查图像/掩码，重试 |
| `TRAINING_DIVERGED` | 检查数据、恢复 checkpoint、降低学习率 |
| `DISK_FULL` | 清理 cache 或更换输出目录，不自动删用户文件 |

### 9.4 什么时候必须问用户

- 覆盖已有项目或导出文件；
- 删除结果、清 cache；
- 打开另一个项目并丢弃未保存修改；
- 预计运行时间/磁盘/显存超过阈值；
- 质量门限失败且自动恢复策略可能改变用户意图。

## 10. 代码改动清单

### 10.1 新增

| 文件 | 内容 |
|---|---|
| `apps/editor/control/control_protocol.hpp` | schema version、错误码、命令/事件 DTO |
| `apps/editor/control/control_bus.*` | 主线程队列、revision、幂等 |
| `apps/editor/control/control_state.cpp` | `App` -> JSON 快照、actions/capabilities |
| `apps/editor/control/control_commands.cpp` | 命令分发到现有 editor 函数 |
| `apps/editor/control/control_server.*` | HTTP、SSE、token、发现文件 |
| `apps/editor/control/control_events.*` | 事件 ring buffer |
| `apps/ctl/photara_ctl.cpp` | 通用 CLI 客户端（原生，无 Python 依赖） |
| `tools/photara_mcp/server.py` | MCP server |
| `tools/photara_mcp/client.py` | HTTP/SSE 客户端 |
| `tools/photara_mcp/tools.py` | 工具定义 |
| `tools/photara_mcp/skills/photara.md` | agent playbook |
| `tools/photara_agent/AGENTS.md` | Codex 项目规则模板 |
| `tools/photara_agent/CLAUDE.md` | Claude Code 项目规则模板 |
| `tools/photara_agent/system-prompt.txt` | 通用 coding agent 提示词 |
| `schemas/studio-control.schema.json` | 状态/命令 schema（可选但推荐） |
| `docs/AGENT_CONTROL.md` | 本文档 |

### 10.2 修改

| 文件 | 改动 |
|---|---|
| `apps/editor/app.hpp` | 增加 `control_revision`、可能的 control 状态 |
| `apps/editor/main.cpp` | 解析 `--enable-agent-control`，初始化 server，pump 队列，present 后 pump |
| `apps/editor/app_commands.cpp` | 把关键前置条件改成可返回结构化原因；增加按路径打开/保存/设置源的入口 |
| `apps/editor/pipeline.hpp` | 公开必要的命令构建/布局信息；增加参数元数据接口 |
| `photara/CMakeLists.txt` | 新增 control target/sources、`photara-ctl` 可执行文件、`nlohmann_json`、`httplib`、Windows `ws2_32` |
| `CMakeLists.txt` | 可选 `PHOTARA_ENABLE_AGENT_CONTROL` 开关 |
| `scripts/generate_api_reference.py` 或新脚本 | 生成/校验 control schema 文档 |

### 10.3 不建议第一版就做

- 全量重构 `App` 为 MVVM；
- 把所有 inspector 控件改成 schema 驱动；
- 在 C++ 里直接集成 LLM SDK；
- 暴露所有 `SplatEdit` 鼠标工具；
- 让 agent 直接编辑 `.ascan` 或缓存文件；
- 默认开启监听端口。

### 10.4 CMake 与测试目标的注意事项

`photara/CMakeLists.txt` 中有几个测试目标会复制 `photara_studio` 的 source list，
例如 `photara_gaussian_session_test`：

```cmake
get_target_property(_session_sources photara_studio SOURCES)
add_executable(photara_gaussian_session_test
    tests/gaussian_session_test.cpp ${_session_sources})
```

如果直接把 control server、HTTP、JSON 源文件塞进 `photara_studio`，这些测试也会被迫
编译并链接控制层，增加依赖和构建时间。建议：

- 把与 `App` 无关的部分（protocol、event ring、HTTP server、JSON 编解码）放进
  独立静态库 `photara_studio_control`；
- `photara_studio` 链接该库；
- `control_commands.cpp` 因为依赖 `App`，可以留在 `photara_studio` 的 source list，
  或者放进一个依赖 editor 的 `photara_studio_control_app` 库；
- 测试目标按需链接 control 库，而不是无差别复制全部源文件；
- 在 Windows 上记得链接 `ws2_32`（HTTP/SSE 服务端），Linux 上不需要额外库；
- 如果使用 header-only 的 `httplib`，仍然需要处理线程和 socket 的编译选项。

第一版也可以先不做静态库拆分，但要在实现时明确检查所有复制 studio source list 的
测试目标，避免 CI 因为新增网络依赖而失败。

## 11. 分阶段路线图

### Phase 0：协议和骨架（3-5 天）

交付：

- `control_protocol.hpp`；
- `control_bus`；
- `--enable-agent-control`；
- `studio.ping`、`studio.get_state`；
- `photara-ctl` 原生 CLI：`doctor`、`state`、`schema`；
- 单元测试：JSON 编解码、revision、错误码。

验收：

- Studio 启动后可以通过本地 HTTP 读取状态；
- 关闭 flag 时没有端口、没有发现文件；
- 状态快照与 UI 显示一致。

### Phase 1：只读 + 任务控制 MVP（1-2 周）

交付：

- `params.describe`、`params.patch`（高影响字段）；
- `job.start` / `pause` / `resume` / `stop`；
- `RunMonitor` 进度和训练指标事件；
- `inspect.logs`、`inspect.artifacts`；
- `studio.view.capture`；
- `photara-ctl`：`params`、`job start/status/wait/pause/resume/stop`、`capture`；
- MCP bridge：`studio_get_state`、`studio_start_job`、`studio_control_job`、`studio_wait`、`studio_capture`。

验收：

- 外部脚本可以在 `PHOTARA_SMOKE_IMAGES` 上完成 align + train；
- 训练过程中能收到 stage/进度/loss；
- stop 在 present 后安全执行；
- 截图能在窗口最小化时工作（至少不崩溃）。

### Phase 2：参数、Recipe、质量评估（2-3 周）

交付：

- `recipe.reconstruct_3dgs`、`recipe.mesh_from_model`、`recipe.texture_mesh`；
- alignment/training 质量指标；
- 结构化 `suggested_action`；
- dry-run preflight（磁盘/显存/前置条件）；
- 权限模式和审计日志。

验收：

- agent 可以用一个 recipe 完成常见重建；
- 前置条件不满足时不会反复尝试无效命令；
- 质量不达标时返回明确原因。

### Phase 3：编辑、导出、产品化（3-5 周）

交付：

- `edit.select_*`、`edit.delete_selected`、undo/redo；
- 多模型合并/可见性/变换；
- export 面板的命令化；
- headless `photara_agentd`；
- 可选 in-app agent 面板；
- 多实例发现和重连；
- 更完整的 schema/codegen 和回归测试。

验收：

- agent 可以完成“对齐 → 训练 → 检查 → 编辑 → 导出”的闭环；
- 破坏性操作需要确认；
- Studio 崩溃后可以恢复任务状态。

总体估算：

- 只读 + 任务控制 MVP：**1-2 周**；
- 可日常使用的版本：**4-6 周**；
- 产品化（权限、编辑、headless、测试）：**8-12 周**。

取决于是否单人开发、是否需要 cross-platform、以及是否要把 `App` 进一步拆分。

## 12. 关键技术取舍

### 12.1 为什么用主线程队列，而不是让命令线程加锁

`App` 和 Vulkan/ImGui 的耦合太深，给所有字段加锁不现实，也容易死锁。主线程队列把并发边界收敛到一个地方：

- 后台线程只解析和排队；
- 主线程顺序执行；
- 资源生命周期与帧边界一致；
- 测试可以同步调用 `pump_main_thread()`。

### 12.2 为什么 HTTP 而不是直接 named pipe

- Python MCP bridge 和调试客户端最容易接；
- 浏览器/脚本/CI 都能用；
- 本机 loopback + token 足够；
- 需要更严格隔离时再加 named pipe 作为可选 transport。

### 12.3 为什么 MCP 不放进 GUI 进程

- GUI 进程生命周期由用户控制；
- stdio MCP 需要独占 stdin/stdout；
- 独立 bridge 更容易做重连、多实例、日志和升级；
- Studio 不需要链接 MCP SDK。

### 12.4 为什么还需要 headless

- 批量任务、服务器、CI 不应该启动 Vulkan 窗口；
- Python API/CLI 已经能跑核心算法；
- 但 headless 不能替代 GUI 的实时预览和编辑，所以两者共存。

## 13. 建议的第一个 PR

第一个 PR 不要做“大而全”。建议做一个垂直切片：

```text
--enable-agent-control
  -> 启动 loopback HTTP server
  -> studio.ping
  -> studio.get_state
  -> studio.params.patch（3-5 个字段）
  -> studio.job.start（train）
  -> studio.job.control（stop）
  -> SSE job.progress
  -> studio.view.capture
```

配套一个 Python 脚本：

```text
scripts/agent_control_smoke.py
  1. 读取发现文件
  2. get_state
  3. patch iterations=100
  4. start train
  5. 订阅 progress
  6. stop
  7. capture
  8. 断言状态和产物
```

这个 PR 能验证四件最关键的事：

1. 主线程执行模型是否成立；
2. 现有 `start_train`/`ProcessJob`/`RunMonitor` 能否被外部稳定驱动；
3. present 后安全停止是否没有回归；
4. 截图链路是否可用。

如果这四件事验证通过，后面的参数、recipe、编辑和 MCP 都是增量工作。

## 14. 最终建议

按优先级排序：

1. **先做语义状态和命令总线**，不要先做聊天 UI。
2. **先覆盖 align/train/stop/capture**，这是 agent 闭环的最小集合。
3. **用 recipe 固化成功路径**，不要让 LLM 自由组合几十个底层字段。
4. **`photara-ctl` 作为通用入口**，保证 Codex、Claude Code、DeepSeek、zcode
   只要能在终端执行命令就能接入。
5. **MCP 独立进程作为增强入口**，给支持 MCP 的客户端更好的工具体验。
6. **权限、确认、审计从第一版就留接口**，否则后期很难补。
7. **UI 自动化只作为 fallback**，用于验证或极端场景，不作为产品主路径。
