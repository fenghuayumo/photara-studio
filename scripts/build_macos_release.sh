#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VULKAN_SDK:?Set VULKAN_SDK to the macOS Vulkan SDK directory}"
: "${PHOTARA_DEPS_PREFIX:?Set PHOTARA_DEPS_PREFIX to the dependency installation prefix}"
: "${PHOTARA_HNSWLIB_INCLUDE:?Set PHOTARA_HNSWLIB_INCLUDE to the directory containing hnswlib/}"
export PATH="$VULKAN_SDK/bin:$PHOTARA_DEPS_PREFIX/bin:$PATH"
cmake -S . -B build-studio-release -G Ninja \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PHOTARA_DEPS_PREFIX" \
  -DPHOTARA_HNSWLIB_INCLUDE="$PHOTARA_HNSWLIB_INCLUDE" \
  -DPHOTARA_ENABLE_CUDA=OFF -DPHOTARA_ENABLE_SPLAT=ON \
  -DPHOTARA_ENABLE_VULKAN_COMPUTE=ON -DPHOTARA_BUILD_STUDIO=ON \
  -DPHOTARA_ENABLE_TEXTURE=OFF -DPHOTARA_ENABLE_USDC=OFF \
  -DPHOTARA_INFERENCE_VULKAN=OFF -DBUILD_TESTING=OFF
cmake --build build-studio-release --target photara_studio --parallel 8
python3 - "$VULKAN_SDK" <<'PY'
import pathlib, plistlib, shutil, sys
base = pathlib.Path('build-studio-release')
app = base / 'Photara Studio.app/Contents'
bin_dir = app / 'MacOS'
bin_dir.mkdir(parents=True, exist_ok=True)
shutil.copy2(base / 'photara/photara_studio', bin_dir / 'PhotaraStudio')
cli = bin_dir / 'photara'
if not cli.is_symlink(): cli.symlink_to('../../../photara/photara')
info = dict(CFBundleName='Photara Studio', CFBundleIdentifier='org.photara.studio.local',
    CFBundleExecutable='PhotaraStudio', CFBundlePackageType='APPL', NSHighResolutionCapable=True,
    LSEnvironment={'VK_ICD_FILENAMES': str(pathlib.Path(sys.argv[1]).resolve() / 'share/vulkan/icd.d/MoltenVK_icd.json')})
with (app / 'Info.plist').open('wb') as f: plistlib.dump(info, f)
PY
