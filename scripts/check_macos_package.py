#!/usr/bin/env python3
"""Check a packaged app after relocation, including LaunchServices compatibility."""
import argparse
import os
from pathlib import Path
import plistlib
import re
import subprocess
import tempfile

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('app', type=Path)
a = p.parse_args()
app = a.app.resolve()
info = plistlib.loads((app / 'Contents/Info.plist').read_bytes())
def version(value):
    return tuple((list(map(int, value.split('.'))) + [0, 0])[:3])
host = subprocess.check_output(['sw_vers', '-productVersion'], text=True).strip()
if version(info['LSMinimumSystemVersion']) > version(host):
    raise SystemExit(f'App requires macOS {info["LSMinimumSystemVersion"]}; host is {host}')
subprocess.run(['codesign', '--verify', '--deep', '--strict', str(app)], check=True)
with tempfile.TemporaryDirectory(prefix='photara-relocation-') as tmp:
    moved = Path(tmp) / 'Moved Studio.app'
    subprocess.run(['ditto', str(app), str(moved)], check=True)
    env = {'HOME': os.environ['HOME'], 'PATH': '/usr/bin:/bin:/usr/sbin:/sbin',
           'DYLD_PRINT_LIBRARIES': '1'}
    for binary, args in [('photara', ['--help']), ('ffmpeg', ['-version']),
                         ('PhotaraLauncher', ['--project-transition-smoke'])]:
        result = subprocess.run([str(moved / 'Contents/MacOS' / binary), *args],
                                env=env, cwd=tmp, capture_output=True, text=True, timeout=90)
        if result.returncode:
            raise RuntimeError(f'{binary} failed ({result.returncode}): {result.stderr[-4000:]}')
        loaded = [line for line in result.stderr.splitlines() if line.startswith('dyld[')]
        for line in loaded:
            match = re.search(r'(/.*)$', line)
            if match:
                path = match[1]
                if not path.startswith(('/usr/lib/', '/System/Library/', str(moved),
                                        str(moved.resolve()), '/Library/Apple/',
                                        '/System/Volumes/Preboot/Cryptexes/OS/')):
                    raise RuntimeError(f'External runtime dependency: {line}')
        if binary == 'PhotaraLauncher' and not any('libMoltenVK.dylib' in x for x in loaded):
            raise RuntimeError('Bundled Vulkan driver was not loaded')
        print(f'{binary}: passed after relocation with clean environment')
# Exercise normal app launch through LaunchServices, not the inner executable.
# This intentionally opens the original application and leaves it available for visual QA.
subprocess.run(['open', '-n', '-g', str(app)], check=True)
print(f'LaunchServices accepted app on macOS {host}; confirm the window visually.')
