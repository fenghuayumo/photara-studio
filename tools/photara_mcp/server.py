"""Optional stdio MCP adapter. Business commands stay in Studio/photara-ctl."""
import asyncio
import json
from mcp.server.fastmcp import FastMCP
from client import StudioClient
from recipes import ReconstructionRecipe

mcp = FastMCP('Photara Studio')
client = StudioClient()
recipe = ReconstructionRecipe(client)
recipe_lock = asyncio.Lock()


@mcp.tool()
async def studio_get_state() -> dict:
    """Read Studio state, capabilities, actions and task progress."""
    return await client.call('state')


@mcp.tool()
async def studio_start_job(stage: str) -> dict:
    """Start align/train/resume_training/dense/texture/export_sfm; return op_id."""
    if stage not in ('align', 'train', 'resume_training', 'dense', 'texture', 'export_sfm'):
        raise ValueError('Unsupported stage')
    return await client.call('job', 'start', stage, mutation=True)


@mcp.tool()
async def studio_wait(op_id: str, timeout: int = 30) -> dict:
    """Bounded wait for this exact operation. Repeat if timed_out is true."""
    return await client.wait(op_id, timeout)


@mcp.tool()
async def studio_control_job(action: str) -> dict:
    """Pause/resume/stop the active task."""
    if action not in ('pause', 'resume', 'stop'):
        raise ValueError('Unsupported action')
    return await client.call('job', action, mutation=True)


@mcp.tool()
async def studio_patch_params(patch: dict) -> dict:
    """Atomically validate and apply reconstruction parameters while idle."""
    return await client.call('params', 'set', *[f'{k}={json.dumps(v)}' for k, v in patch.items()], mutation=True)


@mcp.tool()
async def studio_capture(path: str) -> dict:
    """Capture the currently rendered preview to an allowed absolute PNG path."""
    return await client.call('capture', '--out', path, mutation=True)


@mcp.tool()
async def studio_quality() -> dict:
    """Inspect loaded-scene alignment and latest training metrics; unknown is not passed."""
    return await client.call('quality')


@mcp.tool()
async def studio_edit(action: str) -> dict:
    """Select all/clear/invert, delete_selected, undo or redo. Deletion is reversible."""
    return await client.call('edit', action, mutation=True)


@mcp.tool()
async def studio_export(kind: str, path: str, format: str = 'ply', overwrite: bool = False) -> dict:
    """Export model or mesh to an allowed absolute path; overwrite requires explicit intent."""
    if kind not in ('model', 'mesh'):
        raise ValueError('kind must be model or mesh')
    return await client.call('export', kind, '--out', path, '--format', format,
        *(['--overwrite'] if overwrite else []), mutation=True)


@mcp.tool()
async def studio_recipe_start(iterations: int = 30000) -> dict:
    """Prepare an align/train plan for the selected input; return run_id."""
    async with recipe_lock:
        return await recipe.start(iterations)


@mcp.tool()
async def studio_recipe_advance(run_id: str) -> dict:
    """Start/observe one recipe step. Repeat until succeeded, failed or needs_review."""
    async with recipe_lock:
        return await recipe.advance(run_id)


if __name__ == '__main__':
    mcp.run(transport='stdio')
