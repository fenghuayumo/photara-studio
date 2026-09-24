"""Compile the 3DGUT preview shaders and emit a SPIR-V header."""

import pathlib
import struct
import subprocess
import sys


def main() -> None:
    glslang = sys.argv[1]
    output = pathlib.Path(sys.argv[2])
    include = sys.argv[3]
    shaders = [pathlib.Path(path) for path in sys.argv[4:]]
    work = output.parent
    work.mkdir(parents=True, exist_ok=True)
    lines = [
        "#pragma once",
        "#include <cstdint>",
        "namespace splat_render::gut_spv {",
    ]
    for shader in shaders:
        spv_path = work / (shader.name + ".spv")
        result = subprocess.run(
            [
                glslang,
                "-V",
                "--target-env",
                "vulkan1.2",
                "-I" + include,
                "-o",
                str(spv_path),
                str(shader),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            sys.stderr.write(result.stdout)
            sys.stderr.write(result.stderr)
            raise SystemExit(result.returncode)
        data = spv_path.read_bytes()
        if len(data) % 4 != 0:
            raise SystemExit(f"{shader} SPIR-V length is not a multiple of 4")
        words = struct.unpack("<" + "I" * (len(data) // 4), data)
        name = shader.name.replace(".", "_")
        lines.append(
            f"inline constexpr std::uint32_t {name}[] = {{")
        row: list[str] = []
        for word in words:
            row.append(f"0x{word:08x}u")
            if len(row) == 8:
                lines.append("    " + ", ".join(row) + ",")
                row = []
        if row:
            lines.append("    " + ", ".join(row) + ",")
        lines.append("};")
        lines.append("")
    lines.append("}  // namespace splat_render::gut_spv")
    lines.append("")
    output.write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
