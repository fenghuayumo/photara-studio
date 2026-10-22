#!/usr/bin/env bash
# Packages the Release build into a portable Linux directory and tarball.
# Runtime libraries are collected automatically by walking the dynamic
# dependency table (ldd) of photara_studio / photara, mirroring
# package_windows.ps1. Libraries provided by the C library, the GPU driver,
# or the CUDA driver are never bundled; README.txt documents the end-user
# requirements instead.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
repository_root="$(cd "$script_dir/.." && pwd)"

runtime_dir=""
output_root=""
version=""
include_cli=1
clean=0
build_deb=1

usage() {
    cat <<'EOF'
Usage: package_linux.sh [--runtime-dir DIR] [--output-root DIR]
                        [--version X.Y.Z] [--no-cli] [--clean] [--no-deb]

  --runtime-dir DIR   Directory holding the photara_studio binary
                      (default: <repo>/build/photara).
  --output-root DIR   Where to place the package (default: <repo>/dist).
  --version X.Y.Z     Package version string (default: read from VERSION).
  --no-cli            Exclude the photara command-line tool.
  --clean             Replace an existing staging directory for the version.
  --no-deb            Skip building the .deb installer package.
  --ffmpeg PATH       FFmpeg binary to bundle beside the application
                      (default: PATH lookup, /opt/ffmpeg/ffmpeg, then
                      ~/.local/bin/ffmpeg).
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --runtime-dir) runtime_dir="$2"; shift 2 ;;
        --output-root) output_root="$2"; shift 2 ;;
        --version)     version="$2"; shift 2 ;;
        --no-cli)      include_cli=0; shift ;;
        --clean)       clean=1; shift ;;
        --no-deb)      build_deb=0; shift ;;
        --ffmpeg)      ffmpeg_arg="$2"; shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "$runtime_dir" ]]; then runtime_dir="$repository_root/build/photara"; fi
if [[ -z "$output_root" ]]; then output_root="$repository_root/dist"; fi
if [[ -z "$version" ]]; then
    version="$(tr -d '[:space:]' < "$repository_root/VERSION")"
fi
if [[ ! "$version" =~ ^[0-9]+\.[0-9]+\.[0-9]+([-._][0-9A-Za-z.]+)?$ ]]; then
    echo "Version must be a release-like value (for example 0.6.0): $version" >&2
    exit 1
fi

for required in photara_studio; do
    if [[ ! -f "$runtime_dir/$required" ]]; then
        echo "Release output is missing $required : $runtime_dir" >&2
        exit 1
    fi
done
if (( include_cli )) && [[ ! -f "$runtime_dir/photara" ]]; then
    echo "Release output is missing photara : $runtime_dir (use --no-cli to skip)" >&2
    exit 1
fi

if ! command -v patchelf >/dev/null 2>&1; then
    echo "patchelf was not found. Install it (for example: sudo apt install patchelf)" >&2
    exit 1
fi

# Shared libraries provided by glibc, the GPU driver, or the CUDA driver.
# They are never bundled; README.txt documents the end-user requirements.
is_system_lib() {
    case "$1" in
        linux-vdso.so.1 | ld-linux-x86-64.so.2 | ld-linux.so.2 | ld64.so.2) return 0 ;;
        libc.so.6 | libm.so.6 | libpthread.so.0 | libdl.so.2 | librt.so.1) return 0 ;;
        libresolv.so.2 | libnsl.so.1 | libnsl.so.2 | libutil.so.1) return 0 ;;
        libanl.so.1 | libthread_db.so.1 | libBrokenLocale.so.1) return 0 ;;
        libmvec.so.1 | libcrypt.so.1 | libSegFault.so) return 0 ;;
        # CUDA driver API lives in the NVIDIA driver, like nvcuda.dll.
        libcuda.so.1) return 0 ;;
        *) return 1 ;;
    esac
}

# CUDA runtimes (cudart, ...) live in the toolkit installation, not next to
# the binaries. Mirror Find-CudaRuntimeDirs from package_windows.ps1.
cuda_dirs=()
if [[ -n "${CUDA_PATH:-}" && -d "$CUDA_PATH/lib64" ]]; then
    cuda_dirs+=("$(readlink -f "$CUDA_PATH/lib64")")
fi
cmake_cuda_dir="$(sed -n 's/^CMAKE_CUDA_COMPILER:FILEPATH=//p' \
    "$repository_root/build/CMakeCache.txt" 2>/dev/null | head -1)"
if [[ -n "$cmake_cuda_dir" ]]; then
    candidate="$(readlink -f "$(dirname "$cmake_cuda_dir")/../lib")"
    [[ -d "$candidate" ]] && cuda_dirs+=("$candidate")
fi
for candidate in /usr/local/cuda-*/lib64 /usr/local/cuda/lib64; do
    [[ -d "$candidate" ]] && cuda_dirs+=("$(readlink -f "$candidate")")
done
cuda_dirs=($(printf '%s\n' "${cuda_dirs[@]}" | awk '!seen[$0]++'))

# Walk the import table (DT_NEEDED) of every staged binary, copying each
# non-system dependency into lib/ and recursing into it. Dependencies are
# resolved from the CUDA runtime directories, the directory of the object
# that needs them, and the ldconfig cache — never from build-tree RPATHs.
stage="$output_root/Photara-Studio-$version-linux64"
lib_dir="$stage/lib"

# System library cache: soname -> path, like the loader's ld.so.cache.
declare -A ldconfig_index=()
while IFS=$'\t' read -r soname path; do
    [[ -n "$soname" && -n "$path" && -z "${ldconfig_index[$soname]:-}" ]] && \
        ldconfig_index[$soname]="$path"
done < <(ldconfig -p 2>/dev/null | awk -F' => ' '
    $1 ~ /\.so/ { n=$1; sub(/ *\([^)]*\) *$/, "", n); print n "\t" $2 }
')

declare -A staged=()
declare -a search_dirs=()

collect_dependencies() {
    local -a queue=("$@")
    local dep_path name candidate rpath_dir origin
    local unresolved=0

    for dep_path in "$@"; do
        search_dirs+=("$(dirname "$dep_path")")
        # The loader consults the object's own RPATH/RUNPATH first; mirror
        # that so libraries reached only through a build prefix resolve.
        origin="$(dirname "$dep_path")"
        while IFS= read -r rpath_dir; do
            rpath_dir="${rpath_dir//\$ORIGIN/$origin}"
            rpath_dir="${rpath_dir//\$\{ORIGIN\}/$origin}"
            [[ -d "$rpath_dir" ]] && search_dirs+=("$rpath_dir")
        done < <(readelf -d "$dep_path" 2>/dev/null |
                 grep -E '\((RPATH|RUNPATH)\)' |
                 sed -E 's/.*\[(.*)\].*/\1/' | tr ':;' '\n\n' |
                 awk 'NF' | sed 's/^ *//; s/ *$//')
    done

    while (( ${#queue[@]} > 0 )); do
        dep_path="${queue[0]}"; queue=("${queue[@]:1}")
        while IFS= read -r name; do
            [[ -z "$name" ]] && continue
            is_system_lib "$name" && continue
            [[ -n "${staged[$name]:-}" ]] && continue
            candidate=""
            for candidate_dir in \
                "$(dirname "$dep_path")" "$lib_dir" "${cuda_dirs[@]}" "${search_dirs[@]}"; do
                if [[ -e "$candidate_dir/$name" ]]; then
                    candidate="$(readlink -f "$candidate_dir/$name")"
                    break
                fi
            done
            if [[ -z "$candidate" && -n "${ldconfig_index[$name]:-}" ]]; then
                candidate="${ldconfig_index[$name]}"
            fi
            if [[ -z "$candidate" ]]; then
                echo "  unresolved: $name (needed by $(basename "$dep_path"))" >&2
                unresolved=1
                continue
            fi
            staged[$name]="$candidate"
            cp -L "$candidate" "$lib_dir/$name"
            search_dirs+=("$(dirname "$candidate")")
            queue+=("$candidate")
        done < <(readelf -d "$dep_path" 2>/dev/null |
                 sed -n 's/.*(NEEDED).*\[\(.*\)\].*/\1/p')
    done

    return "$unresolved"
}

if [[ -d "$stage" ]]; then
    if (( ! clean )); then
        echo "Staging directory already exists: $stage. Re-run with --clean to replace it." >&2
        exit 1
    fi
    rm -rf "$stage"
fi
mkdir -p "$stage/lib"

echo "Staging binaries into $stage"
cp "$runtime_dir/photara_studio" "$stage"
if (( include_cli )); then cp "$runtime_dir/photara" "$stage"; fi
if [[ -f "$runtime_dir/icon.png" ]]; then cp "$runtime_dir/icon.png" "$stage"; fi

# Search path for ldd: the CMake output directory plus the CUDA runtime
# directories, so every import resolves before it is collected.
search_path="${cuda_dirs[*]:-}"
if [[ -n "$search_path" ]]; then
    search_path=":$search_path"
fi
echo "CUDA runtime directories: ${cuda_dirs[*]:-none}"

roots=("$runtime_dir/photara_studio")
(( include_cli )) && roots+=("$runtime_dir/photara")
if ! collect_dependencies "${roots[@]}"; then
    echo "ldd reported dependencies that are neither bundled nor system libraries." >&2
    exit 1
fi
lib_count=$(find "$lib_dir" -type f | wc -l)
echo "Collected $lib_count runtime librar(y/ies) via ldd dependency walk."

# FFmpeg is searched next to the application at runtime. Bundle it when it
# is available, exactly like the Windows package does.
if [[ -z "${ffmpeg_arg:-}" ]]; then
    for candidate in /opt/ffmpeg/ffmpeg "$HOME/.local/bin/ffmpeg"; do
        if [[ -x "$candidate" ]]; then ffmpeg_arg="$candidate"; break; fi
    done
fi
bundled_ffmpeg=0
if [[ -n "${ffmpeg_arg:-}" ]]; then
    if [[ ! -x "$ffmpeg_arg" ]]; then
        echo "ffmpeg was not found at: $ffmpeg_arg" >&2
        exit 1
    fi
    cp -L "$ffmpeg_arg" "$stage/ffmpeg"
    collect_dependencies "$stage/ffmpeg" || true
    lib_count=$(find "$lib_dir" -type f | wc -l)
    bundled_ffmpeg=1
    echo "Bundled ffmpeg ($("$stage/ffmpeg" -version 2>/dev/null | head -1 | cut -d' ' -f1-3)); $lib_count librar(y/ies) total."
else
    echo "Warning: ffmpeg was not found; video input will require ffmpeg on the target machine." >&2
fi

# Rewrite runpaths so the package is self-contained: binaries find lib/
# beside themselves and each library finds its siblings. $ORIGIN survives
# relocation of the extracted directory.
patchelf --set-rpath '$ORIGIN/lib' "$stage/photara_studio"
if (( include_cli )); then patchelf --set-rpath '$ORIGIN/lib' "$stage/photara"; fi
# Static ffmpeg builds have no DT_NEEDED entries and no .dynamic section;
# only dynamic binaries need a rewritten runpath.
if file "$stage/ffmpeg" | grep -q "dynamically linked"; then
    patchelf --set-rpath '$ORIGIN/lib' "$stage/ffmpeg"
fi
find "$lib_dir" -maxdepth 1 -type f -name '*.so*' -print0 |
    xargs -0 -r -n 1 patchelf --set-rpath '$ORIGIN'

cat > "$stage/photara_studio.sh" <<'LAUNCHER'
#!/usr/bin/env bash
# Launcher for the portable Photara Studio package.
here="$(cd "$(dirname "$(readlink -f "$0")")" && pwd)"
export LD_LIBRARY_PATH="$here/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
exec "$here/photara_studio" "$@"
LAUNCHER
chmod +x "$stage/photara_studio.sh"

for document in LICENSE NOTICE THIRD_PARTY_NOTICES.md; do
    cp "$repository_root/$document" "$stage"
done
mkdir -p "$stage/licenses"
cp "$repository_root/docs/LICENSE-Intrinsic.md" "$stage/licenses/"

# Report the oldest glibc the package can run on, so the README is accurate.
max_glibc="$( (objdump -T "$stage/photara_studio" 2>/dev/null;
               (( include_cli )) && objdump -T "$stage/photara" 2>/dev/null) |
               grep -o 'GLIBC_[0-9.]*' | sort -uV | tail -1 | sed 's/GLIBC_//')"

if (( bundled_ffmpeg )); then
    ffmpeg_note="FFmpeg is bundled beside the application for video input \
(the static build is GNU GPL v3 software; see https://ffmpeg.org for its source)."
else
    ffmpeg_note="FFmpeg is not bundled; install it on the target machine for video input."
fi

cat > "$stage/README.txt" <<EOF
Photara Studio $version (Linux x86_64)

Run ./photara_studio.sh (or ./photara_studio directly). $ffmpeg_note

Requirements: 64-bit Linux with glibc $max_glibc or newer (Ubuntu 24.04+,
Debian 13, Fedora 40+), a Vulkan-capable graphics driver, and (for CUDA /
3DGS features) a compatible NVIDIA driver. The C library, graphics drivers,
Vulkan ICDs, and the CUDA driver are provided by the operating system / GPU
vendor and are intentionally not bundled.

SAM 3 model weights are not included. Download them from Studio only after
accepting Meta's model license. Intrinsic delight models are not included;
see licenses/LICENSE-Intrinsic.md before redistribution.

See LICENSE, NOTICE, and THIRD_PARTY_NOTICES.md for distribution terms.
EOF

package="$output_root/Photara-Studio-$version-linux64.tar.gz"
if [[ -e "$package" ]]; then rm -f "$package"; fi
echo "Creating $package"
tar -C "$output_root" -czf "$package" "$(basename "$stage")"
echo "Portable package: $package"

# Build the .deb installer from the portable directory, mirroring the Inno
# Setup step in package_windows.ps1. Everything installs under
# /opt/photara-studio with desktop-entry and CLI symlinks.
if (( build_deb )); then
    if ! command -v dpkg-deb >/dev/null 2>&1; then
        echo "dpkg-deb was not found; skipping the .deb package (use a Debian-based build host or --no-deb)." >&2
    else
        deb_root="$output_root/deb-staging"
        install_dir="$deb_root/opt/photara-studio"
        if [[ -d "$deb_root" ]]; then rm -rf "$deb_root"; fi
        mkdir -p "$deb_root/DEBIAN" "$install_dir" \
            "$deb_root/usr/bin" \
            "$deb_root/usr/share/applications" \
            "$deb_root/usr/share/icons/hicolor/256x256/apps"
        cp -a "$stage/." "$install_dir/"

        ln -s /opt/photara-studio/photara_studio.sh "$deb_root/usr/bin/photara-studio"
        if (( include_cli )); then
            ln -s /opt/photara-studio/photara "$deb_root/usr/bin/photara"
        fi
        cp "$repository_root/apps/icons/icon.png" \
            "$deb_root/usr/share/icons/hicolor/256x256/apps/photara-studio.png"
        cat > "$deb_root/usr/share/applications/photara-studio.desktop" <<'DESKTOP'
[Desktop Entry]
Type=Application
Name=Photara Studio
GenericName=3D Reconstruction Editor
Comment=Structure from motion, dense reconstruction, and 3D Gaussian splatting
Exec=/opt/photara-studio/photara_studio.sh
Icon=photara-studio
Categories=Graphics;3DGraphics;Science;
Terminal=false
StartupWMClass=photara-studio
Keywords=photogrammetry;sfm;gaussian splatting;reconstruction;
DESKTOP

        installed_size="$(du -sk "$install_dir" | cut -f1)"
        cat > "$deb_root/DEBIAN/control" <<EOF
Package: photara-studio
Version: $version
Architecture: amd64
Maintainer: Photara <photara@example.invalid>
Installed-Size: $installed_size
Depends: libc6 (>= $max_glibc)
Recommends: mesa-vulkan-drivers
Suggests: ffmpeg
Section: graphics
Priority: optional
Description: Structure from motion, dense reconstruction, and 3D Gaussian splatting
 Photara Studio is a Vulkan-accelerated reconstruction editor. Runtime
 libraries ship inside /opt/photara-studio/lib; the C library, the Vulkan
 loader's installable client drivers, and the CUDA driver come from the
 operating system and GPU vendor.
EOF

        deb_package="$output_root/photara-studio_${version}_amd64.deb"
        dpkg-deb --root-owner-group --build "$deb_root" "$deb_package" >&2
        rm -rf "$deb_root"
        echo "Installer package: $deb_package"
    fi
fi
