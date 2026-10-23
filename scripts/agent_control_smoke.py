"""Launch an isolated Studio and verify its real control API.

Use --images (or PHOTARA_SMOKE_IMAGES) to additionally align, train, capture,
save and export a dataset. All outputs go into a temporary directory.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--studio', type=Path, required=True)
    parser.add_argument('--ctl', type=Path, required=True)
    parser.add_argument('--images', type=Path, default=os.environ.get('PHOTARA_SMOKE_IMAGES'))
    parser.add_argument('--iterations', type=int, default=20)
    parser.add_argument('--job-timeout', type=int, default=600)
    args = parser.parse_args()
    studio, ctl = args.studio.resolve(), args.ctl.resolve()
    with tempfile.TemporaryDirectory(prefix='photara_control_smoke_') as temporary:
        root = Path(temporary)
        discovery = root / 'control'
        env = dict(os.environ, APPDATA=str(root), XDG_CONFIG_HOME=str(root),
                   PHOTARA_CONTROL_DIR=str(discovery), PHOTARA_CONTROL_ALLOWED_ROOTS=str(root))
        process = subprocess.Popen([str(studio), '--enable-agent-control'], cwd=studio.parent, env=env)
        try:
            deadline = time.monotonic() + 30
            while not list(discovery.glob('*.json')):
                if process.poll() is not None:
                    raise RuntimeError(f'Studio exited during startup: {process.returncode}')
                if time.monotonic() > deadline:
                    raise TimeoutError('Studio did not publish its control endpoint')
                time.sleep(.1)
            endpoint = json.loads(next(discovery.glob('*.json')).read_text())

            def cli(*arguments, codes=(0,)):
                reply = subprocess.run([str(ctl), '--pid', str(process.pid), '--json', *arguments],
                    env=env, capture_output=True, text=True, encoding='utf-8', timeout=45)
                if reply.returncode not in codes:
                    raise RuntimeError(f'{arguments}: exit={reply.returncode} {reply.stdout} {reply.stderr}')
                return json.loads(reply.stdout)

            def rpc(payload, token=None):
                request = urllib.request.Request(endpoint['url'] + '/v1/rpc',
                    data=json.dumps(payload).encode(), headers={'Authorization': 'Bearer ' + (token or endpoint['token'])})
                with urllib.request.urlopen(request, timeout=10) as response:
                    return json.load(response)['result']

            cli('doctor')
            cli('schema')
            before = cli('params', 'get')['iterations']
            invalid = cli('params', 'set', 'iterations=17', 'strategy=invalid', codes=(2,))
            assert invalid['error']['code'] == 'INVALID_REQUEST'
            assert cli('params', 'get')['iterations'] == before, 'failed patch changed parameters'
            cli('--request-id', 'view-1', 'view', 'set', 'yaw=1', 'target=[1,2,3]')
            cli('view', 'set', 'yaw=2')
            cli('--request-id', 'view-1', 'view', 'set', 'yaw=1', 'target=[1,2,3]')
            assert cli('state')['view']['camera']['yaw'] == 2, 'retry executed again'
            try:
                rpc({'method': 'studio.ping'}, token='invalid')
                raise AssertionError('invalid token accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 401
            try:
                rpc({'method': 'studio.ping', 'client': 42})
                raise AssertionError('invalid client accepted')
            except urllib.error.HTTPError as error:
                assert error.code == 400
            cli('doctor')  # Malformed requests must not kill the server thread.
            unknown = cli('job', 'wait', '--op', 'missing', codes=(2,))
            assert unknown['error']['code'] == 'NOT_FOUND'
            dry_run = cli('job', 'start', 'export_sfm', '--dry-run', codes=(2,))
            assert dry_run['error']['code'] == 'PRECONDITION_FAILED'

            if args.images:
                cli('project', 'set-source', str(args.images.resolve()))
                cli('params', 'set', f'iterations={args.iterations}', 'preview_interval=1',
                    'sam_masks=false', 'ai_priors=false', 'build_mesh=false')
                for stage in ('align', 'train'):
                    operation = cli('job', 'start', stage)['op_id']
                    deadline = time.monotonic() + args.job_timeout
                    while True:
                        result = cli('job', 'wait', '--op', operation, '--timeout', '30', codes=(0, 5))
                        if not result.get('timed_out'):
                            assert result['status'] == 'succeeded', result
                            break
                        if time.monotonic() > deadline:
                            raise TimeoutError(f'{stage} exceeded the job timeout')
                deadline = time.monotonic() + 20
                while True:
                    capture = cli('capture', '--out', str(root / 'view.png'), codes=(0, 2))
                    if capture.get('ok'):
                        break
                    if time.monotonic() > deadline:
                        raise RuntimeError(f'No preview after training: {capture}')
                    time.sleep(.2)
                cli('project', 'save', str(root / 'result.ascan'))
                cli('export', 'model', '--out', str(root / 'result.ply'))
                assert (root / 'result.ply').stat().st_size > 0
                assert (root / 'result.ascan').stat().st_size > 0
            print('Studio live control smoke passed' + (' (align/train/capture/save/export)' if args.images else ' (protocol/parameters/retries/view)'))
        finally:
            process.terminate()
            process.wait(timeout=15)


if __name__ == '__main__':
    main()
