"""Exercise both Python PAM seed paths using the native octahedral field fixture."""
import argparse
import json
import math
from pathlib import Path
import struct

from python_api_fixture import import_photara, write_png


def run(native_directory, output):
    p = import_photara(native_directory)
    output.mkdir(parents=True, exist_ok=True)
    write_png(output / "frame.png")
    # OpenGL camera-to-world converted by the dataset loader to a +Z camera.
    (output / "transforms.json").write_text(json.dumps({
        "fl_x": 25, "fl_y": 25, "cx": 15.5, "cy": 15.5, "w": 32, "h": 32,
        "frames": [{"file_path": "frame.png", "transform_matrix": [
            [1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, -4], [0, 0, 0, 1]]}]}))
    shell = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    faces = [(0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
             (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5)]
    fields = ["x", "y", "z", "f_dc_0", "f_dc_1", "f_dc_2", "opacity",
              "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3",
              "gaussian_features_0", "gaussian_features_1", "gaussian_features_2",
              "gaussian_features_3", "filter_3d"]
    header = "ply\nformat binary_little_endian 1.0\nelement vertex 6\n"
    header += "".join(f"property float {field}\n" for field in fields) + "end_header\n"
    rows = [list(v) + [0, 0, 0, 8] + [math.log(.65)] * 3 + [1, 0, 0, 0]
            + list(v) + [3, .01] for v in shell]
    (output / "model.ply").write_bytes(header.encode() + b"".join(
        struct.pack("<19f", *row) for row in rows))
    (output / "seed.ply").write_text(
        "ply\nformat ascii 1.0\nelement vertex 6\nproperty float x\nproperty float y\n"
        "property float z\nelement face 8\nproperty list uchar int vertex_indices\nend_header\n"
        + "".join(" ".join(map(str, v)) + "\n" for v in shell)
        + "".join("3 " + " ".join(map(str, f)) + "\n" for f in faces))
    request = p.DatasetRequest()
    request.source = output / "transforms.json"
    request.random_initial_point_count = 16
    scene = p.load_dataset(request)
    model = p.load_3dgs(output / "model.ply", device=p.Device.CUDA)
    options = p.PamOptions()
    options.max_points = 64
    options.scene_crop = False
    options.oversampling_factor = 8
    options.occupancy_chunk_size = 257
    options.refinement_steps = 2
    options.vector_field_neighbors = 6
    options.points_per_tetrahedron = 1
    options.occupancy_iso_value = .05
    options.vacancy_threshold = 1
    report = {}
    for name, seed in [("automatic", None), ("external_ply", p.load_mesh(output / "seed.ply"))]:
        result = p.extract_pam(model, scene, seed_mesh=seed, pam_options=options)
        assert result.vertex_count > 0 and result.face_count > 0, name
        assert result.surface_point_count == options.max_points, name
        assert result.tetrahedron_count > 0 and result.occupied_tetrahedron_count > 0, name
        result.save(output / f"{name}.ply")
        result.save_surface_cloud(output / f"{name}-surface.ply")
        loaded = p.load_mesh(output / f"{name}.ply")
        assert (loaded.vertex_count, loaded.face_count) == (result.vertex_count, result.face_count)
        report[name] = {key: getattr(result, key) for key in ["vertex_count", "face_count",
            "surface_point_count", "tetrahedron_count", "occupied_tetrahedron_count"]}
    (output / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("native_directory")
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    run(args.native_directory, args.output.resolve())
