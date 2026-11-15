#!/bin/bash
# Rebuild the two runtime dependencies whose Homebrew bottles require macOS 27.2.
set -euo pipefail
cd "$(dirname "$0")/.."
if [[ -n "${PHOTARA_DEPS_PREFIX:-}" ]]; then
    export PATH="$PHOTARA_DEPS_PREFIX/bin:$PATH"
fi
target="${MACOSX_DEPLOYMENT_TARGET:-27.0}"
base="$PWD/build/macos-deps"
prefix="$base/prefix"
mkdir -p "$base/sources" "$prefix/share/licenses"
fetch() {
    local name="$1" url="$2" checksum="$3"
    if [[ ! -f "$base/sources/$name" ]]; then
        curl -fL --retry 3 "$url" -o "$base/sources/$name"
    fi
    echo "$checksum  $base/sources/$name" | shasum -a 256 -c -
}
fetch svt-av1-4.3.0.tar.bz2 \
    https://gitlab.com/AOMediaCodec/SVT-AV1/-/archive/v4.3.0/SVT-AV1-v4.3.0.tar.bz2 \
    9223ce06e6172ad41357ed23b3b00f036e03794b86b0e20ea9ccb3f648b8955d
fetch llvm-project-23.1.5.src.tar.xz \
    https://github.com/llvm/llvm-project/releases/download/llvmorg-23.1.5/llvm-project-23.1.5.src.tar.xz \
    fb50b1eac93ddeb3aa8dab4cfe324dff74f4a9c2bdee6160c9e36188f8d06a6e
if [[ ! -d "$base/sources/SVT-AV1-v4.3.0" ]]; then
    tar -xf "$base/sources/svt-av1-4.3.0.tar.bz2" -C "$base/sources"
fi
if [[ ! -d "$base/sources/llvm-project-23.1.5.src/third-party/unittest" ]]; then
    tar -xf "$base/sources/llvm-project-23.1.5.src.tar.xz" -C "$base/sources" \
        llvm-project-23.1.5.src/cmake llvm-project-23.1.5.src/llvm \
        llvm-project-23.1.5.src/runtimes llvm-project-23.1.5.src/openmp \
        llvm-project-23.1.5.src/third-party
fi
cmake -S "$base/sources/SVT-AV1-v4.3.0" -B "$base/svt-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$target" -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DBUILD_APPS=OFF -DBUILD_SHARED_LIBS=ON
cmake --build "$base/svt-build" --parallel 8
cmake --install "$base/svt-build"
cmake -S "$base/sources/llvm-project-23.1.5.src/runtimes" -B "$base/omp-build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=arm64 \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$target" -DCMAKE_INSTALL_PREFIX="$prefix" \
    -DLLVM_ENABLE_RUNTIMES=openmp -DLIBOMP_INSTALL_ALIASES=OFF \
    -DOPENMP_ENABLE_OMPT_TOOLS=OFF -DOPENMP_ENABLE_LIBOMPTARGET=OFF
cmake --build "$base/omp-build" --parallel 8
cmake --install "$base/omp-build"
cp "$base/sources/SVT-AV1-v4.3.0/LICENSE-BSD2.md" "$prefix/share/licenses/SVT-AV1-LICENSE-BSD2.md"
cp "$base/sources/SVT-AV1-v4.3.0/LICENSE.md" "$prefix/share/licenses/SVT-AV1-LICENSE.md"
cp "$base/sources/llvm-project-23.1.5.src/openmp/LICENSE.TXT" "$prefix/share/licenses/OpenMP-LICENSE.txt"
printf '%s\n' "Built arm64 runtime dependencies for macOS $target in $prefix"
