"""Format C++/CUDA with clang-format; normalize basic HLSL whitespace only.

Usage: python scripts/format_code.py third_party/splat_drender
"""

import argparse
from pathlib import Path
import re
import subprocess


EXTENSIONS = {".c", ".cc", ".cpp", ".cxx", ".h", ".hh", ".hpp", ".hxx",
              ".cu", ".cuh", ".hlsl", ".hlsli"}

# Keep literals intact and distinguish adjacent operators (e.g. '+ +'/'++').
TOKENS = re.compile(
    r'(?P<comment>//[^\n]*|/\*[\s\S]*?\*/)'
    r'|(?P<space>\s+)'
    r'|(?P<literal>"(?:\\[\s\S]|[^"\\])*"|\'(?:\\[\s\S]|[^\'\\])*\')'
    r'|(?P<word>[A-Za-z_]\w*)'
    r'|(?P<number>(?:\d|\.\d)(?:[eEpP][+-]|[\w.])*)'
    r'|(?P<operator>>>=|<<=|>>=|\+\+|--|->|&&|\|\||==|!=|<=|>=|<<|>>|'
    r'\+=|-=|\*=|/=|%=|&=|\|=|\^=|##|::|.)'
)


def validate_hlsl(original, updated):
    """Refuse token, directive, or line-sensitive changes before writing."""
    def tokens(text):
        return [m[0] for m in TOKENS.finditer(text)
                if m.lastgroup not in {"comment", "space"}]

    if tokens(original) != tokens(updated):
        raise ValueError("HLSL formatting changed code tokens; refusing to write")
    # Whitespace and physical lines can be significant in macros (#/##,
    # function-like defines, continuations). Be conservative: preserve them.
    def directives(text):
        lines = text.replace("\r\n", "\n").splitlines(keepends=True)
        result = []
        continuation = False
        for line in lines:
            if continuation or line.lstrip().startswith("#"):
                result.append(line)
                continuation = line.rstrip("\n").endswith("\\")
        return result

    if directives(original) != directives(updated):
        raise ValueError("HLSL formatting changed a preprocessor directive; refusing to write")
    if "__LINE__" in tokens(original) and original.replace("\r\n", "\n") != updated:
        raise ValueError("HLSL uses __LINE__; refusing to change line layout")
    if re.search(r"\\\r?\n", original):
        # This also covers continued strings/comments outside directives.
        if original.replace("\r\n", "\n") != updated:
            raise ValueError("HLSL contains line continuations; refusing layout changes")


def normalize_hlsl(text):
    # Preserve line layout, attributes, include order and alignment. Expand
    # leading tabs to four-column tab stops without changing interior tabs.
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"(?m)^[ \t]+", lambda m: m[0].expandtabs(4), text)
    return text + "\n" if text and not text.endswith("\n") else text


def format_hlsl(text, path, formatter):
    """Format bodies while shielding HLSL declarations from clang-format."""
    text = normalize_hlsl(text)
    marker = "photara-hlsl-declaration"
    if marker in text:
        raise ValueError(f"Reserved formatter marker in {path}")
    lines = text.splitlines(keepends=True)
    protected = []
    index = 0
    while index < len(lines):
        line = lines[index]
        if line.lstrip().startswith("#"):
            protected.append(f"// clang-format off: {marker}\n")
            while index < len(lines):
                line = lines[index]
                protected.append(line)
                index += 1
                if not line.rstrip("\n").endswith("\\"):
                    break
            protected.append(f"// clang-format on: {marker}\n")
            continue
        if re.match(r"\s*(?:\[numthreads\b|\[\[vk::|void\s+main\s*\()", line):
            declaration = []
            while index < len(lines):
                line = lines[index]
                # Normalize attribute argument spacing without reflowing it.
                if "[numthreads" in line:
                    line = re.sub(r"\[numthreads\(([^\n\]]*)\)\]",
                                  lambda m: "[numthreads(" + re.sub(r",[ \t]*", ", ", m[1]) + ")]",
                                  line)
                declaration.append(line)
                index += 1
                if ";" in line or "{" in line:
                    break
            protected.append(f"// clang-format off: {marker}\n")
            protected.extend(declaration)
            protected.append(f"// clang-format on: {marker}\n")
        else:
            protected.append(line)
            index += 1
    result = subprocess.run(
        [formatter, "--style=file", f"--assume-filename={path}"],
        input="".join(protected), encoding="utf-8", capture_output=True, check=True,
    ).stdout
    return "".join(line for line in result.splitlines(keepends=True)
                   if marker not in line)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--clang-format", default="clang-format")
    parser.add_argument("--hlsl-style", action="store_true",
                        help="Format HLSL bodies while preserving shader declarations")
    args = parser.parse_args()
    files = set()
    for path in args.paths:
        if not path.exists():
            parser.error(f"Path does not exist: {path}")
        candidates = path.rglob("*") if path.is_dir() else [path]
        files.update(p.resolve() for p in candidates
                     if p.is_file() and p.suffix.lower() in EXTENSIONS)
    for path in sorted(files):
        if path.suffix.lower() in {".hlsl", ".hlsli"}:
            with path.open(encoding="utf-8", newline="") as stream:
                original = stream.read()
            updated = (format_hlsl(original, path, args.clang_format)
                       if args.hlsl_style else normalize_hlsl(original))
            validate_hlsl(original, updated)
            if updated != original:
                with path.open("w", encoding="utf-8", newline="") as stream:
                    stream.write(updated)
        else:
            subprocess.run([args.clang_format, "--style=file", "-i", str(path)], check=True)
    print(f"Formatted {len(files)} files.")


if __name__ == "__main__":
    main()
