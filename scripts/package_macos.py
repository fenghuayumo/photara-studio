#!/usr/bin/env python3
"""Bundle an existing macOS arm64 Release build and produce a relocatable DMG."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import plistlib
import re
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)
p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--build-dir', type=Path, default=Path('build'))
p.add_argument('--sdk', type=Path, default=os.environ.get('VULKAN_SDK'))
p.add_argument('--ffmpeg', type=Path, default=shutil.which('ffmpeg'))
p.add_argument('--output-dir', type=Path, default=Path('build/release'))
p.add_argument('--deployment-target', default='27.0',
               help='Maximum permitted minimum macOS version (default: 27.0)')
p.add_argument('--dependency-prefix', type=Path, default=Path('build/macos-deps/prefix'),
               help='Prefix containing runtime dependencies rebuilt for the target OS')
a = p.parse_args()
if not a.sdk or not a.ffmpeg:
    p.error('--sdk / VULKAN_SDK and --ffmpeg / ffmpeg in PATH are required')
build, sdk, output = a.build_dir.resolve(), a.sdk.resolve(), a.output_dir.resolve()
output.mkdir(parents=True, exist_ok=True)
version = re.search(r'#define PHOTARA_VERSION "([^"]+)"',
    (build / 'photara/include/core/version.hpp').read_text())[1]
stage = output / 'macos-arm64'
if stage.exists(): shutil.rmtree(stage)
stage.mkdir()
app = stage / 'Photara Studio.app'
shutil.copytree(build / 'Photara Studio.app', app, symlinks=True)
c = app / 'Contents'
shutil.rmtree(c / '_CodeSignature', ignore_errors=True)
frameworks = c / 'Frameworks'
frameworks.mkdir(exist_ok=True)
resources = c / 'Resources'
binaries = c / 'MacOS'

def run(*args):
    return subprocess.check_output([str(x) for x in args], text=True)

def deps(path):
    return [line.strip().split(' (compatibility')[0]
            for line in run('otool', '-arch', 'arm64', '-L', path).splitlines()[1:]]

def rpaths(path):
    return re.findall(r'cmd LC_RPATH\n\s+cmdsize \d+\n\s+path (.*?) \(offset', run('otool', '-arch', 'arm64', '-l', path))

def system(path):
    return path.startswith(('/usr/lib/', '/System/Library/'))

# Resolve original load commands before changing them; recurse through all dylibs.
search = [build / 'photara', build / 'ggml/src',
          build / '_deps/photara_poselib-build/PoseLib', sdk / 'lib']
collected = {}
manifest = []
dependency_prefix = a.dependency_prefix.resolve()

def version_tuple(value):
    return tuple((list(map(int, value.split('.'))) + [0, 0])[:3])

def collect(source, destination, executable=False):
    replacement = dependency_prefix / 'lib' / source.name
    if not executable and replacement.is_file():
        source = replacement
    source = source.resolve()
    if source in collected: return collected[source]
    if destination.exists() and not executable:
        raise RuntimeError(f'Dylib filename collision: {destination}')
    collected[source] = destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    destination.chmod(0o755)
    run('lipo', '-verify_arch', 'arm64', destination)
    paths = rpaths(source)
    def expand(s):
        return Path(s.replace('@loader_path', str(source.parent)).replace('@executable_path', str(source.parent)))
    changes = []
    # otool lists a dylib's own LC_ID_DYLIB first; it is not a dependency and
    # need not resolve through an rpath (rebuilt runtimes may have none).
    dependencies = deps(source)
    for dep in dependencies if executable else dependencies[1:]:
        if system(dep): continue
        if dep.startswith('@rpath/'):
            tail = dep[len('@rpath/'):]
            candidates = [expand(r) / tail for r in paths] + [d / tail for d in search]
        elif dep.startswith('@loader_path/'):
            candidates = [expand(dep)]
        else:
            candidates = [Path(dep)]
        found = next((x for x in candidates if x.is_file()), None)
        if found is None: raise RuntimeError(f'Unresolved dependency {dep} in {source}')
        if found.resolve() == source: continue  # dylib install name
        bundled = collect(found, frameworks / found.name)
        relative = '@loader_path/' + os.path.relpath(bundled, destination.parent)
        changes += ['-change', dep, relative]
    for r in paths: changes += ['-delete_rpath', r]
    if not executable: changes += ['-id', '@rpath/' + destination.name]
    if changes: run('install_name_tool', *changes, destination)
    manifest.append({'source': str(source), 'bundled': str(destination.relative_to(app))})
    return destination

for name, source in [('PhotaraStudio', build / 'photara/photara_studio'),
                     ('photara', build / 'photara/photara'), ('ffmpeg', a.ffmpeg)]:
    collect(source, binaries / name, True)
collect(sdk / 'lib/libMoltenVK.dylib', frameworks / 'libMoltenVK.dylib')
icd = resources / 'vulkan/icd.d'
icd.mkdir(parents=True, exist_ok=True)
metadata = json.loads((sdk / 'share/vulkan/icd.d/MoltenVK_icd.json').read_text())
metadata['ICD']['library_path'] = '../../../Frameworks/libMoltenVK.dylib'
(icd / 'MoltenVK_icd.json').write_text(json.dumps(metadata, indent=2) + '\n')
shutil.copy2(ROOT / 'apps/icons/icon.png', resources / 'icon.png')
run('clang', '-arch', 'arm64', '-mmacosx-version-min=' + a.deployment_target,
    'scripts/macos_launcher.c', '-o', binaries / 'PhotaraLauncher')
info = plistlib.loads((c / 'Info.plist').read_bytes())
info.pop('LSEnvironment', None)
info.update(CFBundleExecutable='PhotaraLauncher', CFBundleIdentifier='org.photara.studio',
            CFBundleShortVersionString=version, CFBundleVersion=version)
# Advertise the highest minimum OS required by any included Mach-O image.
minimums = []
for image in [*collected.values(), binaries / 'PhotaraLauncher']:
    commands = run('otool', '-arch', 'arm64', '-l', image)
    versions = re.findall(r'\bminos ([\d.]+)', commands)
    versions += re.findall(r'cmd LC_VERSION_MIN_MACOSX\n\s+cmdsize \d+\n\s+version ([\d.]+)', commands)
    if not versions:
        raise RuntimeError(f'Cannot determine minimum macOS version: {image}')
    for required in versions:
        if version_tuple(required) > version_tuple(a.deployment_target):
            raise RuntimeError(f'{image.name} requires macOS {required}, above target '
                               f'{a.deployment_target}; rebuild this dependency first')
    minimums += versions
minimum = a.deployment_target
info['LSMinimumSystemVersion'] = a.deployment_target
(c / 'Info.plist').write_bytes(plistlib.dumps(info))
licenses = resources / 'licenses'
licenses.mkdir(exist_ok=True)
for name in ('LICENSE', 'NOTICE', 'THIRD_PARTY_NOTICES.md'):
    shutil.copy2(ROOT / name, resources / name)
if (build / 'photara/licenses').exists():
    shutil.copytree(build / 'photara/licenses', licenses, dirs_exist_ok=True)
if (dependency_prefix / 'share/licenses').exists():
    shutil.copytree(dependency_prefix / 'share/licenses', licenses / 'rebuilt-runtime', dirs_exist_ok=True)
# Preserve licenses, package receipts and SBOMs from each bundled Homebrew formula.
formula_roots = set()
for source in collected:
    parts = source.parts
    if 'Cellar' in parts:
        i = parts.index('Cellar')
        formula_roots.add(Path(*parts[:i + 3]))
for formula in sorted(formula_roots):
    target = licenses / (formula.parent.name + '-' + formula.name)
    target.mkdir(exist_ok=True)
    for entry in formula.iterdir():
        if entry.is_file() and (re.search(r'licen[sc]e|copying|notice|copyright|receipt|sbom', entry.name, re.I)):
            shutil.copy2(entry, target / entry.name)
    if (formula / '.brew').is_dir():
        shutil.copytree(formula / '.brew', target / 'brew-formula', dirs_exist_ok=True)
for source in (sdk.parent / 'Licenses', sdk / 'share', sdk / 'lib'):
    for entry in source.glob('*'):
        if entry.is_file() and re.search(r'license|copyright|notice', entry.name, re.I):
            shutil.copy2(entry, licenses / entry.name)
# Validate that every non-system dependency resolves inside the finished bundle.
for image in collected.values():
    for dep in deps(image):
        if system(dep) or dep == '@rpath/' + image.name: continue
        assert dep.startswith('@loader_path/'), (image, dep)
        resolved = (image.parent / dep[len('@loader_path/'):]).resolve()
        assert resolved.is_relative_to(app.resolve()) and resolved.is_file(), (image, dep)
    assert not rpaths(image), image
for image in collected.values():
    run('codesign', '--force', '--sign', '-', image)
run('codesign', '--force', '--sign', '-', app)
run('codesign', '--verify', '--deep', '--strict', app)
readme = f'''Photara Studio {version} — macOS arm64

Requires Apple Silicon and macOS {minimum} or later.
Drag Photara Studio.app to Applications, then open it.
This build is ad-hoc signed and is not notarized by Apple. If macOS blocks
opening it, review the source of your download, then use System Settings >
Privacy & Security > Open Anyway to allow this specific application.

Vulkan loader, MoltenVK, FFmpeg and dynamic library dependencies are bundled.
Optional AI model weights are downloaded separately by features that need them.
Texture baking and USDC export are disabled in this build.
Licenses and dependency notices are inside the app under Contents/Resources.
Project source: https://github.com/fenghuayumo/photara-studio
'''
(stage / 'README.txt').write_text(readme)
(stage / 'Applications').symlink_to('/Applications')
(output / 'macos-arm64-dependencies.json').write_text(json.dumps(manifest, indent=2) + '\n')
dmg = output / f'Photara-Studio-{version}-macOS-arm64.dmg'
run('hdiutil', 'create', '-ov', '-format', 'UDZO', '-volname', 'Photara Studio', '-srcfolder', stage, dmg)
run('hdiutil', 'verify', dmg)
(output / (dmg.name + '.sha256')).write_text(hashlib.sha256(dmg.read_bytes()).hexdigest() + '  ' + dmg.name + '\n')
print(f'Created {dmg} ({dmg.stat().st_size / 1024**2:.1f} MiB); macOS >= {minimum}')
