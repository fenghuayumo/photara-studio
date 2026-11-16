#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
: "${VULKAN_SDK:?Set VULKAN_SDK to the macOS Vulkan SDK directory}"
: "${PHOTARA_DEPS_PREFIX:?Set PHOTARA_DEPS_PREFIX to the dependency installation prefix}"
: "${PHOTARA_HNSWLIB_INCLUDE:?Set PHOTARA_HNSWLIB_INCLUDE to the directory containing hnswlib/}"
export PATH="$VULKAN_SDK/bin:$PHOTARA_DEPS_PREFIX/bin:$PATH"
build_dir="${PHOTARA_BUILD_DIR:-build}"
extra_args=()
cpu_target="${PHOTARA_CPU_TARGET:-apple-m2}"
if [[ -n "$cpu_target" ]]; then
  extra_args+=("-DCMAKE_C_FLAGS=-mcpu=$cpu_target -Werror=unguarded-availability"
    "-DCMAKE_CXX_FLAGS=-mcpu=$cpu_target -Werror=unguarded-availability"
    -DPHOTARA_ENABLE_NATIVE_ARCH=OFF -DGGML_NATIVE=OFF)
  # CPU feature probes must be rerun when switching from a host-native build.
  for feature in DOTPROD SVE MATMUL_INT8 FMA FP16_VECTOR_ARITHMETIC SME; do
    extra_args+=(-U "HAVE_$feature")
  done
fi
cmake -S . -B "$build_dir" -G Ninja "${extra_args[@]}" \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH="$PHOTARA_DEPS_PREFIX" \
  -DCMAKE_OSX_DEPLOYMENT_TARGET="${MACOSX_DEPLOYMENT_TARGET:-27.0}" \
  -DPHOTARA_HNSWLIB_INCLUDE="$PHOTARA_HNSWLIB_INCLUDE" \
  -DPHOTARA_ENABLE_CUDA=OFF -DPHOTARA_ENABLE_SPLAT=ON \
  -DPHOTARA_ENABLE_VULKAN_COMPUTE=ON -DPHOTARA_BUILD_STUDIO=ON \
  -DPHOTARA_ENABLE_TEXTURE=OFF -DPHOTARA_ENABLE_USDC=OFF \
  -DPHOTARA_INFERENCE_VULKAN=OFF -DBUILD_TESTING=OFF
cmake --build "$build_dir" --target photara_studio --parallel 8
python3 - "$VULKAN_SDK" "$build_dir" <<'PY'
import pathlib, plistlib, shutil, subprocess, sys, tempfile
base = pathlib.Path(sys.argv[2])
app = base / 'Photara Studio.app/Contents'
bin_dir = app / 'MacOS'
bin_dir.mkdir(parents=True, exist_ok=True)
resources = app / 'Resources'
resources.mkdir(parents=True, exist_ok=True)
icon = pathlib.Path('apps/icons/icon.png').resolve()
with tempfile.TemporaryDirectory(prefix='photara-icon-') as tmp:
    iconset = pathlib.Path(tmp) / 'PhotaraStudio.iconset'
    iconset.mkdir()
    for size in (16, 32, 128, 256, 512):
        for scale in (1, 2):
            pixels = str(size * scale)
            suffix = '@2x' if scale == 2 else ''
            subprocess.run(['sips', '-z', pixels, pixels, str(icon), '--out',
                str(iconset / f'icon_{size}x{size}{suffix}.png')],
                check=True, stdout=subprocess.DEVNULL)
    subprocess.run(['iconutil', '-c', 'icns', str(iconset), '-o',
        str(resources / 'PhotaraStudio.icns')], check=True)
shutil.copy2(base / 'photara/photara_studio', bin_dir / 'PhotaraStudio')
cli = bin_dir / 'photara'
if cli.is_symlink(): cli.unlink()
shutil.copy2(base / 'photara/photara', cli)
info = dict(CFBundleName='Photara Studio', CFBundleIdentifier='org.photara.studio.local',
    CFBundleExecutable='PhotaraStudio', CFBundlePackageType='APPL',
    CFBundleIconFile='PhotaraStudio.icns', NSHighResolutionCapable=True,
    LSEnvironment={'VK_ICD_FILENAMES': str(pathlib.Path(sys.argv[1]).resolve() / 'share/vulkan/icd.d/MoltenVK_icd.json')})
with (app / 'Info.plist').open('wb') as f: plistlib.dump(info, f)
PY
# Refresh the local ad-hoc signature after changing the executable and resources.
codesign --force --sign - "$build_dir/Photara Studio.app"
touch "$build_dir/Photara Studio.app"
