"""Hand the finished dataset over to Photara.

Both paths below are implemented against the real photara-studio source
(apps/editor/control, photara/src/tools/reconstruct.cpp):

1. Studio Agent Control — for a running ``photara_studio.exe``.
   Studio must be started with ``--enable-agent-control``. It then publishes a
   discovery file ``<control-dir>/<pid>.json`` with pid, session_id, url,
   token, version and started_at_ms. Commands are sent as
   ``POST {url}/v1/rpc`` with ``Authorization: Bearer {token}``:

   - ``studio.project.set_dataset {path}`` — attach the exported COLMAP
     dataset with imported cameras (Align/SfM is skipped),
   - ``studio.job.start {stage: "train"}`` — start 3DGS training.

2. CLI training — headless via ``photara.exe``:
   ``--splat-dataset <dir>`` on an external dataset (without ``--dense``)
   implies Gaussian-splat training; ``--output <name>.sog`` names the result.
"""

import json
import os
import shlex
import shutil
import subprocess
import threading
import time
import urllib.error
import urllib.request
import uuid


DEFAULT_STUDIO_ARGS = "--enable-agent-control"
DEFAULT_CLI_ARGS = '--splat-dataset "{dataset}" --output "{dataset}/{name}.sog"'


class BridgeError(RuntimeError):
    def __init__(self, message, code="INTERNAL"):
        super().__init__(message)
        self.code = code


class BridgeResult:
    def __init__(self, ok, message):
        self.ok = ok
        self.message = message


# --------------------------------------------------------------------------
# discovery
# --------------------------------------------------------------------------
def control_discovery_dirs():
    dirs = []
    override = os.environ.get("PHOTARA_CONTROL_DIR", "")
    if override:
        dirs.append(override)
    if os.name == "nt":
        appdata = os.environ.get("APPDATA", "")
        local = os.environ.get("LOCALAPPDATA", "")
        if appdata:
            dirs.append(os.path.join(appdata, "Photara", "control"))
        if local:
            dirs.append(os.path.join(local, "Photara", "control"))
    else:
        config = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
        dirs.append(os.path.join(config, "Photara", "control"))
    seen = set()
    unique = []
    for d in dirs:
        key = os.path.normcase(os.path.abspath(d))
        if key not in seen:
            seen.add(key)
            unique.append(d)
    return unique


def _read_discovery_entries():
    entries = []
    for directory in control_discovery_dirs():
        if not os.path.isdir(directory):
            continue
        try:
            names = os.listdir(directory)
        except OSError:
            continue
        for name in names:
            if not name.endswith(".json"):
                continue
            try:
                with open(os.path.join(directory, name), encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data.get("url"), str) and isinstance(data.get("token"), str):
                    entries.append(data)
            except (OSError, ValueError):
                continue
    entries.sort(key=lambda e: e.get("started_at_ms", 0), reverse=True)
    return entries


def _http_get(url, token, timeout):
    req = urllib.request.Request(url, headers={"Authorization": "Bearer " + token})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.status


def _endpoint_alive(endpoint, timeout):
    try:
        url = endpoint["url"].rstrip("/") + "/v1/health"
        return _http_get(url, endpoint["token"], timeout) == 200
    except Exception:
        return False


def discover_studio(timeout=1.0):
    """Return the newest live Studio endpoint dict, or None."""
    for endpoint in _read_discovery_entries():
        if _endpoint_alive(endpoint, timeout):
            return endpoint
    return None


def studio_process_running():
    """Best-effort check for a Studio process without control enabled."""
    try:
        if os.name == "nt":
            out = subprocess.run(
                ["tasklist", "/FI", "IMAGENAME eq photara_studio.exe"],
                capture_output=True, text=True, timeout=5,
            ).stdout
            return "photara_studio.exe" in out.lower()
        out = subprocess.run(
            ["pgrep", "-f", "photara_studio"], capture_output=True, text=True, timeout=5
        ).stdout
        return bool(out.strip())
    except Exception:
        return False


# --------------------------------------------------------------------------
# rpc
# --------------------------------------------------------------------------
def rpc(endpoint, method, params, timeout=8.0, request_id=None):
    url = endpoint["url"].rstrip("/") + "/v1/rpc"
    payload = json.dumps(
        {
            "client": "photara-blender",
            "request_id": request_id or uuid.uuid4().hex,
            "method": method,
            "params": params,
        }
    ).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": "Bearer " + endpoint["token"],
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            body = json.loads(resp.read().decode("utf-8", errors="replace"))
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode("utf-8", errors="replace"))
            error = data.get("error", {})
            raise BridgeError(
                "{}: {}".format(error.get("code", e.code), error.get("message", "")),
                code=str(error.get("code", e.code)),
            ) from e
        except (ValueError, OSError):
            raise BridgeError("Studio RPC 返回 HTTP {}".format(e.code)) from e
    except urllib.error.URLError as e:
        raise BridgeError("无法连接 Studio RPC：{}".format(e.reason), code="UNAVAILABLE") from e
    except OSError as e:
        raise BridgeError("无法连接 Studio RPC：{}".format(e), code="UNAVAILABLE") from e
    result = body.get("result") if isinstance(body, dict) else None
    if result is None:
        error = body.get("error", {}) if isinstance(body, dict) else {}
        raise BridgeError(
            "{}: {}".format(error.get("code", "INTERNAL"), error.get("message", "RPC 无结果")),
            code=str(error.get("code", "INTERNAL")),
        )
    return result


def handover_to_studio(props, dataset_dir, endpoint=None):
    """Attach the dataset in a running Studio and optionally start training."""
    if endpoint is None:
        endpoint = discover_studio(timeout=float(props.studio_timeout))
    if endpoint is None:
        hint = (
            "Studio 正在运行但未开启 Agent Control，请关闭后用"
            " --enable-agent-control 重新启动"
            if studio_process_running()
            else "未发现运行中的 Photara Studio（需要 --enable-agent-control 启动）"
        )
        raise BridgeError(hint, code="UNAVAILABLE")

    images_dir = os.path.join(dataset_dir, "images")
    # Select the freshly exported format explicitly; old COLMAP files may remain.
    source = dataset_dir if getattr(props, "export_colmap", True) else os.path.join(dataset_dir, "transforms.json")
    try:
        result = rpc(endpoint, "studio.project.set_dataset", {"path": source})
        message = "已连接 Studio（session {}）：{}".format(
            str(endpoint.get("session_id", "?"))[:8],
            result.get("message") or "外部数据集已挂载",
        )
    except BridgeError as e:
        if e.code == "UNSUPPORTED":
            # Older Studio without set_dataset: fall back to the image-source
            # flow and tell the user to upgrade / finish in the GUI.
            rpc(endpoint, "studio.project.set_source", {"path": images_dir})
            raise BridgeError(
                "已挂载图像目录，但当前 Studio 版本缺少 project.set_dataset（导入相机直训）；"
                "请升级 Studio 后重试，或在 Studio 中手动选择外部数据集",
                code="UNSUPPORTED",
            ) from e
        raise

    if props.auto_start_training:
        try:
            started = rpc(endpoint, "studio.job.start", {"stage": "train"})
            message += "；3DGS 训练已启动（op {}）".format(started.get("op_id", "?"))
        except BridgeError as e:
            if e.code == "BUSY":
                message += "；Studio 正忙，请在任务完成后手动开始 Train 3DGS"
            else:
                raise
    return message


# --------------------------------------------------------------------------
# executables
# --------------------------------------------------------------------------
def find_photara_executable(kind="studio", user_path=""):
    """Locate photara_studio.exe / photara.exe: explicit path, PATH, installs."""
    if user_path and os.path.isfile(user_path):
        return user_path
    if kind == "studio":
        names = ("photara_studio", "photara-studio", "Photara Studio")
    else:
        names = ("photara",)
    for name in names:
        found = shutil.which(name)
        if found:
            return found

    candidates = []
    if os.name == "nt":
        local = os.environ.get("LOCALAPPDATA", "")
        program_files = (
            os.environ.get("ProgramFiles", ""),
            os.environ.get("ProgramFiles(x86)", ""),
        )
        if local:
            for root in (os.path.join(local, "Programs"), local):
                candidates += [
                    os.path.join(root, "Photara Studio", "photara_studio.exe"),
                    os.path.join(root, "Photara", "photara_studio.exe"),
                    os.path.join(root, "Photara", "photara.exe"),
                ]
        for pf in program_files:
            if pf:
                candidates += [
                    os.path.join(pf, "Photara Studio", "photara_studio.exe"),
                    os.path.join(pf, "Photara Studio", "photara.exe"),
                    os.path.join(pf, "Photara", "photara_studio.exe"),
                    os.path.join(pf, "Photara", "photara.exe"),
                ]
    else:
        home = os.path.expanduser("~")
        candidates += [
            os.path.join(home, ".local", "bin", "photara_studio"),
            os.path.join(home, ".local", "bin", "photara"),
            "/usr/local/bin/photara_studio",
            "/usr/local/bin/photara",
            "/opt/photara/photara_studio",
            "/opt/photara/photara",
        ]
    for path in candidates:
        base = os.path.basename(path).lower() if path else ""
        if kind == "studio" and "photara_studio" not in base:
            continue
        if path and os.path.isfile(path):
            return path
    return None


def parse_launch_args(template, replacements):
    """Split a user arg template, keeping Windows backslashes intact."""
    parts = shlex.split(template or "", posix=False)
    out = []
    for p in parts:
        for key, value in replacements.items():
            p = p.replace("{" + key + "}", value)
        if len(p) >= 2 and p[0] == p[-1] and p[0] in ("'", '"'):
            p = p[1:-1]
        out.append(p)
    return [a for a in out if a]


def _popen_detached(argv, stdout=None):
    kwargs = {"close_fds": True}
    if stdout is not None:
        kwargs["stdout"] = stdout
        kwargs["stderr"] = subprocess.STDOUT
    if os.name == "nt":
        kwargs["creationflags"] = (
            subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
        )
    return subprocess.Popen(argv, **kwargs)


def launch_studio_and_train(props, dataset_dir):
    """Start a new Studio with control enabled, then hand the dataset over."""
    exe = find_photara_executable("studio", props.studio_exe)
    if not exe:
        raise BridgeError(
            "未找到 photara_studio.exe，请在面板中手动指定 Studio Executable",
            code="NOT_FOUND",
        )
    before = {(e.get("pid"), e.get("session_id")) for e in _read_discovery_entries()}
    argv = [exe] + parse_launch_args(
        props.studio_args or DEFAULT_STUDIO_ARGS,
        {"dataset": os.path.abspath(dataset_dir)},
    )
    try:
        _popen_detached(argv)
    except OSError as e:
        raise BridgeError("启动 Studio 失败（{}）：{}".format(exe, e)) from e

    deadline = time.monotonic() + float(props.launch_wait)
    while time.monotonic() < deadline:
        for endpoint in _read_discovery_entries():
            key = (endpoint.get("pid"), endpoint.get("session_id"))
            if key in before:
                continue
            if _endpoint_alive(endpoint, min(1.0, float(props.studio_timeout))):
                return handover_to_studio(props, dataset_dir, endpoint=endpoint)
        time.sleep(0.5)
    raise BridgeError(
        "Studio 已启动，但控制服务在 {} 秒内未就绪；待 Studio 启动完成后，"
        "在面板中点击 Test Connection 重新交付即可".format(props.launch_wait),
        code="TIMEOUT",
    )


def run_cli_training(props, dataset_dir):
    exe = find_photara_executable("cli", props.cli_exe)
    if not exe:
        raise BridgeError(
            "未找到 photara.exe（CLI），请在面板中手动指定 CLI Executable",
            code="NOT_FOUND",
        )
    dataset_dir = os.path.abspath(dataset_dir)
    args = props.cli_args or DEFAULT_CLI_ARGS
    argv = [exe] + parse_launch_args(
        args, {"dataset": dataset_dir, "name": props.dataset_name}
    )
    if not getattr(props, "export_colmap", True) and "--splat-dataset" in argv:
        index = argv.index("--splat-dataset") + 1
        argv[index] = os.path.join(dataset_dir, "transforms.json")
    log_path = os.path.join(dataset_dir, "photara_train.log")
    try:
        with open(log_path, "ab") as log:
            _popen_detached(argv, stdout=log)
    except OSError as e:
        raise BridgeError("启动 CLI 训练失败：{}".format(e)) from e
    return "已启动 CLI 3DGS 训练：{}（日志：{}）".format(" ".join(argv), log_path)


def run_bridge(props, dataset_dir):
    """Blocking handover according to props.handover_mode. Never raises."""
    mode = props.handover_mode
    if mode == "OFF":
        return None
    try:
        if mode == "CLI":
            return BridgeResult(True, run_cli_training(props, dataset_dir))
        if mode == "STUDIO":
            return BridgeResult(True, handover_to_studio(props, dataset_dir))
        # AUTO: connect to a running Studio, otherwise launch one.
        endpoint = discover_studio(timeout=float(props.studio_timeout))
        if endpoint is not None:
            return BridgeResult(True, handover_to_studio(props, dataset_dir, endpoint))
        if studio_process_running():
            return BridgeResult(
                False,
                "Studio 正在运行但未开启 Agent Control；请用 --enable-agent-control"
                " 重启 Studio 后重试",
            )
        return BridgeResult(True, launch_studio_and_train(props, dataset_dir))
    except BridgeError as e:
        return BridgeResult(False, "Photara 交付失败：{}".format(e))


def submit_async(props_snapshot, dataset_dir, on_done):
    """Run the bridge in a background thread, then call ``on_done(result)``."""

    def worker():
        result = run_bridge(props_snapshot, dataset_dir)
        try:
            on_done(result)
        except Exception:
            pass

    threading.Thread(target=worker, daemon=True).start()


class PropsSnapshot:
    """Plain copy of the bridge-relevant properties (thread-safe)."""

    def __init__(self, props):
        self.dataset_name = props.dataset_name
        self.export_colmap = props.export_colmap
        self.handover_mode = props.handover_mode
        self.auto_start_training = props.auto_start_training
        self.studio_timeout = props.studio_timeout
        self.studio_exe = props.studio_exe
        self.studio_args = props.studio_args
        self.cli_exe = props.cli_exe
        self.cli_args = props.cli_args
        self.launch_wait = props.launch_wait
