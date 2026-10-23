# Photara MCP bridge

Requires Python 3.10+ and the native `photara-ctl` built with Studio.
The bridge pins the official MCP SDK v1 compatibility line (`mcp>=1.28,<2`).
Business operations use the native CLI and the existing Studio control API.

```powershell
python -m venv .venv
.venv/Scripts/python -m pip install -r tools/photara_mcp/requirements.txt
$env:PHOTARA_CTL = 'D:/path/to/photara-ctl.exe'
$env:PHOTARA_STUDIO_PID = '12345'
.venv/Scripts/python tools/photara_mcp/server.py
```

Start Studio with `--enable-agent-control`. Set `PHOTARA_STUDIO_PID` to pin
the intended instance. Configure an MCP host to launch the Python executable
with the absolute server.py path as its argument and these environment values.
The stdio stream is reserved for MCP protocol messages.

Tools cover state, parameters, task start/control/wait, capture, edit,
model/mesh export, quality metrics and a reconstruction recipe.
`studio_recipe_start` prepares a plan for the currently selected input;
call `studio_recipe_advance` repeatedly until its status is terminal.
Each advance starts or observes at most one stage. Plans run align then train,
and end with quality checks. Missing metrics produce `needs_review`.
The bridge keeps plans in memory; restarting it loses recipe tracking,
while a Studio job already started continues running. Use its `op_id` to inspect it.
Do not launch another recipe or change the project during an active plan.

Quality thresholds currently default to registration ratio >= 0.8,
mean reprojection <= 2 pixels and training loss <= 0.1. These are diagnostic
defaults, not universal quality guarantees. Native `studio.inspect.quality`
accepts explicit thresholds. Measurements come from the loaded scene and latest
training monitor, so exported/reopened projects can have unknown training metrics.

CLI equivalents:

```text
photara-ctl edit select_all
photara-ctl edit delete_selected
photara-ctl edit undo
photara-ctl edit redo
photara-ctl export mesh --out D:/project/mesh.glb --format glb --include-texture
photara-ctl quality
```

Editing requires a fully loaded splat viewport. Selection and deletion share
the UI undo stack. Mesh export supports PLY/OBJ geometry and GLB with optional
texture. Textured OBJ sidecar export remains a UI operation.
