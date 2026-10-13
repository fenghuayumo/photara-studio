"""Exercise checkpoint recovery across actual process termination and restart."""
import argparse
import os
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import zlib


def png(path):
    def chunk(kind, data):
        return struct.pack('>I', len(data)) + kind + data + struct.pack('>I', zlib.crc32(kind + data))
    pixels = b''.join(b'\0' + bytes([128]) * (32 * 3) for _ in range(32))
    path.write_bytes(b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', 32, 32, 8, 2, 0, 0, 0))
                    + chunk(b'IDAT', zlib.compress(pixels)) + chunk(b'IEND', b''))


def run(executable):
    root = Path(tempfile.mkdtemp(prefix='photara-process-resume-'))
    child = None
    try:
        images = root / 'images'; images.mkdir()
        sparse = root / 'sparse' / '0'; sparse.mkdir(parents=True)
        png(images / 'frame.png')
        (sparse / 'cameras.txt').write_text('1 PINHOLE 32 32 20 20 15.5 15.5\n')
        (sparse / 'images.txt').write_text(''.join(
            f'{i+1} 1 0 0 0 {-i * 0.1} 0 0 1 frame.png\n13.5 15.5 1 17.5 15.5 2\n'
            for i in range(3)))
        (sparse / 'points3D.txt').write_text(
            '1 -0.2 0 2 128 128 128 0.01 1 0 2 0 3 0\n'
            '2 0.2 0 2 128 128 128 0.01 1 1 2 1 3 1\n')
        control = root / 'control'; control.write_text('process-save\n')
        output = root / 'project.ascan'
        checkpoint = output
        args = [str(executable), '--images', str(images), '--splat', '--splat-dataset', str(root), '--output', str(output),
                '--splat-checkpoint-control', str(control),
                '--splat-iterations', '12', '--splat-log-interval', '1', '--splat-use-mask=false',
                '--splat-focus-region=false', '--splat-max-resolution', '32',
                '--splat-sh-degree', '0', '--splat-eval-checkpoints=false']
        kwargs = {'creationflags': subprocess.CREATE_NO_WINDOW} if os.name == 'nt' else {}
        with (root / 'initial.log').open('w') as log:
            child = subprocess.Popen(args, stdout=log, stderr=subprocess.STDOUT, **kwargs)
            deadline = time.monotonic() + 45
            reply = Path(str(control) + '.reply')
            while time.monotonic() < deadline:
                if reply.exists():
                    response = reply.read_text().splitlines()
                    if len(response) >= 2 and response[0] == 'process-save':
                        assert response[1] == 'saved 1', response
                        break
                if child.poll() is not None:
                    raise AssertionError(f'Training exited before saving: {child.returncode}')
                time.sleep(0.02)
            else:
                raise AssertionError('Training save request timed out')
            assert checkpoint.exists(), 'No checkpoint was published'
            assert not Path(str(output) + '.training.ascan').exists(), 'Automatic save created a visible project sidecar'
            # A second process must not overwrite the held training state.
            contender = subprocess.run(args + ['--splat-resume'], capture_output=True, text=True,
                                       timeout=45, **kwargs)
            (root / 'contender.log').write_text(contender.stdout + contender.stderr)
            assert contender.returncode != 0 and 'in use by another process' in contender.stdout + contender.stderr, \
                'Concurrent training did not refuse the active checkpoint'
            child.kill(); child.wait(timeout=5); child = None
        control.unlink(missing_ok=True)
        # A process crash must release its OS lease and leave the snapshot readable.
        restarted = subprocess.run(args + ['--splat-resume'], capture_output=True, text=True,
                                   timeout=45, **kwargs)
        (root / 'resumed.log').write_text(restarted.stdout + restarted.stderr)
        assert restarted.returncode == 0, 'Restart failed after process termination'
        assert 'splat_checkpoint_resumed iteration=1' in restarted.stdout, 'Saved progress was not loaded'
        assert 'splat iteration=2/12' in restarted.stdout, 'Restart did not continue at the next iteration'
        assert 'splat iteration=1/12' not in restarted.stdout, 'Restart began training from scratch'
        assert output.exists(), 'Completed project was not published'
        assert not (root / '.photara-recovery').exists(), 'Native project save created a recovery directory'
        assert not Path(str(output) + '.previous').exists(), 'Native project save created a separate backup'
        # A native container contains its completed model, not pending training chunks.
        with output.open('rb') as project:
            header = project.read(24)
            count = struct.unpack('<I', header[16:20])[0]
            kinds = [struct.unpack('<IIQQQ', project.read(32))[0] for _ in range(count)]
        assert 3 in kinds and not set(kinds).intersection({6, 7, 8, 9}), 'Completed project retained pending training state'
        print('Process crash/restart and concurrent-writer recovery tests passed')
        shutil.rmtree(root)
    except BaseException:
        print(f'Process test artifacts: {root}')
        for name in ('initial.log', 'contender.log', 'resumed.log'):
            path = root / name
            if path.exists():
                print(f'{name}:\n{path.read_text(errors="replace")[-5000:]}')
        raise
    finally:
        if child is not None and child.poll() is None:
            child.kill(); child.wait(timeout=5)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('executable', type=Path)
    run(parser.parse_args().executable.resolve())
