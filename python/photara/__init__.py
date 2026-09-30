"""Photara Python API. Loads the compiled ``photara`` extension."""

from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import sys


def _read_version() -> str:
    path = Path(__file__).resolve().parents[2] / "VERSION"
    try:
        return path.read_text(encoding="utf-8").strip()
    except OSError:
        return "0.0.0"


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


def _is_photara_extension(path: Path) -> bool:
    if not path.is_file():
        return False
    suffix = path.suffix.lower()
    if suffix not in {".pyd", ".so"}:
        return False
    name = path.name
    if name.startswith("photara_"):
        return False
    return name == f"photara{suffix}" or name.startswith("photara.")


def _add_dll_directories(directory: Path) -> None:
    if not hasattr(os, "add_dll_directory"):
        return
    _dll_directories.append(os.add_dll_directory(str(directory)))
    cuda_path = os.environ.get("CUDA_PATH")
    if not cuda_path:
        return
    cuda_bin = Path(cuda_path) / "bin"
    if cuda_bin.is_dir():
        _dll_directories.append(os.add_dll_directory(str(cuda_bin)))


def _load_native():
    errors: list[str] = []
    for directory in _candidate_native_directories():
        if not directory.is_dir():
            continue
        matches = sorted(
            path for path in directory.iterdir() if _is_photara_extension(path)
        )
        if not matches:
            errors.append(f"{directory}: no photara extension")
            continue
        _add_dll_directories(directory)
        path = matches[0]
        spec = importlib.util.spec_from_file_location("photara", path)
        if spec is None or spec.loader is None:
            errors.append(f"{path}: unable to create module spec")
            continue
        module = importlib.util.module_from_spec(spec)
        if not getattr(module, "__version__", None):
            module.__version__ = _read_version()
        sys.modules["photara"] = module
        try:
            spec.loader.exec_module(module)
            return module
        except ImportError as error:
            errors.append(f"{path}: {error}")
    detail = "\n".join(errors) if errors else "No native build directory found."
    raise ImportError(
        "Unable to load the photara extension. Build with "
        "-DPHOTARA_BUILD_PYTHON=ON or set PHOTARA_NATIVE_DIR.\n"
        + detail
    )


_load_native()
