"""Extract LightUSD build inputs without locale-dependent archive handling."""
import shutil
import sys
import tarfile
from pathlib import Path, PurePosixPath

archive, destination, prefix = sys.argv[1:]
root = Path(destination).resolve()
with tarfile.open(archive, "r:gz") as source:
    for member in source:
        path = PurePosixPath(member.name)
        if not member.isfile() or ".." in path.parts or len(path.parts) < 2:
            continue
        if path.parts[0] != prefix:
            continue
        relative = path.parts[1:]
        if not (relative[0] in ("src", "cmake") or
                (len(relative) == 1 and
                 (relative[0] == "CMakeLists.txt" or relative[0].startswith("LICENSE")))):
            continue
        output = root.joinpath(*path.parts)
        output.parent.mkdir(parents=True, exist_ok=True)
        with source.extractfile(member) as inp, output.open("wb") as out:
            shutil.copyfileobj(inp, out)
