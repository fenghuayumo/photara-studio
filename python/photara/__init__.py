"""Python experiment interface for the native Photara pipeline."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import sys

_dll_directories = []


def _candidate_native_directories() -> list[Path]:
    candidates: list[Path] = []
    configured = os.environ.get("PHOTARA_NATIVE_DIR")
    if configured:
        candidates.append(Path(configured))
    repository = Path(__file__).resolve().parents[2]
    candidates.extend(
        [
            repository / "build-codex-verify" / "photara" / "Release",
            repository / "build" / "photara" / "Release",
            repository / "build" / "photara",
        ]
    )
    return candidates


def _load_native():
    errors: list[str] = []
    for directory in _candidate_native_directories():
        if not directory.is_dir():
            continue
        if hasattr(os, "add_dll_directory"):
            _dll_directories.append(os.add_dll_directory(str(directory)))
        cuda_path = os.environ.get("CUDA_PATH")
        if cuda_path and hasattr(os, "add_dll_directory"):
            cuda_bin = Path(cuda_path) / "bin"
            if cuda_bin.is_dir():
                _dll_directories.append(os.add_dll_directory(str(cuda_bin)))
        sys.path.insert(0, str(directory))
        try:
            return importlib.import_module("photara_native")
        except ImportError as error:
            errors.append(f"{directory}: {error}")
    detail = "\n".join(errors) if errors else "No native build directory found."
    raise ImportError(
        "Unable to load photara_native. Build with "
        "-DPHOTARA_BUILD_PYTHON=ON or set PHOTARA_NATIVE_DIR.\n"
        + detail
    )


_native = _load_native()
__doc__ = _native.__doc__
__all__ = [name for name in dir(_native) if not name.startswith("_")]
globals().update({name: getattr(_native, name) for name in __all__})
