"""Small deterministic files shared by the Python binding integration tests."""
from pathlib import Path
import os
import struct
import sys
import zlib


def import_photara(native_directory):
    # Exercise the source package loader, including its DLL search paths.
    os.environ["PHOTARA_NATIVE_DIR"] = str(Path(native_directory).resolve())
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "python"))
    import photara
    return photara


def write_png(path):
    def chunk(kind, data):
        return (struct.pack(">I", len(data)) + kind + data
                + struct.pack(">I", zlib.crc32(kind + data)))

    rows = b"".join(b"\0" + bytes((x * 7 + y * 11 + c * 31) % 256
                                for x in range(32) for c in range(3))
                    for y in range(32))
    path.write_bytes(b"\x89PNG\r\n\x1a\n"
                     + chunk(b"IHDR", struct.pack(">IIBBBBB", 32, 32, 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(rows)) + chunk(b"IEND", b""))


def write_dataset(root):
    images = root / "images"
    sparse = root / "sparse" / "0"
    images.mkdir(parents=True)
    sparse.mkdir(parents=True)
    (sparse / "cameras.txt").write_text("1 PINHOLE 32 32 20 20 15.5 15.5\n")
    points = [(x / 10, y / 10, 2.0) for x in (-3, -1, 1, 3) for y in (-3, -1, 1, 3)]
    records = []
    for i in range(3):
        write_png(images / f"frame{i}.png")
        records.append(f"{i + 1} 1 0 0 0 {-i * .1} 0 0 1 frame{i}.png\n")
        records.append(" ".join(f"{20 * (x - i * .1) / z + 15.5} "
                                f"{20 * y / z + 15.5} {j + 1}"
                                for j, (x, y, z) in enumerate(points)) + "\n")
    (sparse / "images.txt").write_text("".join(records))
    (sparse / "points3D.txt").write_text("".join(
        f"{j + 1} {x} {y} {z} 128 128 128 .01 1 {j} 2 {j} 3 {j}\n"
        for j, (x, y, z) in enumerate(points)))
    return sparse, images


def write_gaussians(path):
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 2\n"
    header += "".join(f"property float {field}\n" for field in fields) + "end_header\n"
    rows = [[x, 0, 2, 0, 0, 0, 0, -3, -3, -3, 1, 0, 0, 0] for x in (-.2, .2)]
    path.write_bytes(header.encode() + b"".join(struct.pack("<14f", *row) for row in rows))


def write_mesh(path):
    path.write_text("ply\nformat ascii 1.0\nelement vertex 3\n"
                    "property float x\nproperty float y\nproperty float z\n"
                    "element face 1\nproperty list uchar int vertex_indices\nend_header\n"
                    "0 0 2\n1 0 2\n0 1 2\n3 0 1 2\n")
