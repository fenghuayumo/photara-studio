"""CLI transport shared by recipes and the optional MCP bridge."""
import asyncio
import json
import os
import uuid


class StudioClient:
    def __init__(self, executable=None, pid=None):
        self.executable = executable or os.environ.get('PHOTARA_CTL', 'photara-ctl')
        self.pid = pid or os.environ.get('PHOTARA_STUDIO_PID')

    async def call(self, *arguments, mutation=False):
        command = [self.executable, '--json', '--client', 'photara-mcp']
        if self.pid:
            command += ['--pid', str(self.pid)]
        if mutation:
            command += ['--request-id', uuid.uuid4().hex]
        process = await asyncio.create_subprocess_exec(*command, *map(str, arguments),
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        try:
            stdout, stderr = await asyncio.wait_for(process.communicate(), timeout=45)
        except (TimeoutError, asyncio.CancelledError):
            if process.returncode is None:
                process.kill()
            await process.wait()
            raise
        try:
            result = json.loads(stdout.decode('utf-8'))
        except (ValueError, UnicodeError) as error:
            raise RuntimeError(f'CLI returned invalid JSON (exit {process.returncode}): {stderr.decode("utf-8", errors="replace")}') from error
        if process.returncode not in (0, 5):
            raise RuntimeError(json.dumps(result, ensure_ascii=False))
        return result

    async def wait(self, op_id, timeout=30):
        if not 1 <= timeout <= 30:
            raise ValueError('timeout must be in [1, 30] seconds')
        return await self.call('job', 'wait', '--op', op_id, '--timeout', timeout)
