#!/usr/bin/env python3
"""Render a PLY triangle mesh as a labeled three-view PNG contact sheet."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image, ImageDraw, ImageFont


def _unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1.0e-12:
        raise ValueError("cannot normalize a zero-length vector")
    return vector / norm


def _pca_frame(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centered = vertices - np.median(vertices, axis=0)
    covariance = np.cov(centered, rowvar=False)
    _, eigenvectors = np.linalg.eigh(covariance)
    right = _unit(eigenvectors[:, -1])
    up = _unit(eigenvectors[:, -2])
    front = _unit(np.cross(right, up))
    up = _unit(np.cross(front, right))
    return right, up, front


def _colmap_views(
    model_path: Path,
    indices: list[int],
    width: int,
    height: int,
) -> list[tuple[str, np.ndarray, np.ndarray, o3d.camera.PinholeCameraParameters]]:
    import pycolmap

    reconstruction = pycolmap.Reconstruction(str(model_path))
    images = sorted(reconstruction.images.values(), key=lambda image: image.name)
    result = []
    for index in indices:
        image = images[index]
        camera = reconstruction.cameras[image.camera_id]
        calibration = camera.calibration_matrix()
        scale_x = width / camera.width
        scale_y = height / camera.height
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            calibration[0, 0] * scale_x,
            calibration[1, 1] * scale_y,
            calibration[0, 2] * scale_x,
            calibration[1, 2] * scale_y,
        )
        world_to_camera = np.eye(4)
        world_to_camera[:3, :4] = image.cam_from_world().matrix()
        parameters = o3d.camera.PinholeCameraParameters()
        parameters.intrinsic = intrinsic
        parameters.extrinsic = world_to_camera
        rotation = world_to_camera[:3, :3]
        direction = _unit(rotation.T @ np.array([0.0, 0.0, 1.0]))
        camera_up = _unit(rotation.T @ np.array([0.0, -1.0, 0.0]))
        result.append((f"View {index} · {image.name}", direction, camera_up, parameters))
    return result


def render_views(
    mesh_path: Path,
    output_path: Path,
    width: int,
    height: int,
    colmap_model: Path | None = None,
    colmap_indices: list[int] | None = None,
) -> None:
    mesh = o3d.io.read_triangle_mesh(str(mesh_path), enable_post_processing=False)
    if mesh.is_empty() or not mesh.has_triangles():
        raise ValueError(f"mesh has no triangles: {mesh_path}")
    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()

    vertices = np.asarray(mesh.vertices)
    center = 0.5 * (vertices.min(axis=0) + vertices.max(axis=0))
    radius = 0.5 * float(np.linalg.norm(vertices.max(axis=0) - vertices.min(axis=0)))
    right, up, front = _pca_frame(vertices)
    if colmap_model is not None:
        views = _colmap_views(colmap_model, colmap_indices or [0], width, height)
    else:
        views = [
            ("Front", front, up, None),
            ("Three-quarter", _unit(front + 0.75 * right + 0.18 * up), up, None),
            ("Side", right, up, None),
        ]

    del radius  # Open3D's view controller fits the geometry before applying zoom.
    normals = np.asarray(mesh.vertex_normals)
    visualizer = o3d.visualization.Visualizer()
    if not visualizer.create_window(
        window_name="AetherScan mesh renderer",
        width=width,
        height=height,
        visible=False,
    ):
        raise RuntimeError("Open3D could not create a hidden rendering window")
    visualizer.add_geometry(mesh)
    render_option = visualizer.get_render_option()
    render_option.background_color = np.array([0.025, 0.032, 0.045])
    render_option.light_on = True
    render_option.mesh_show_back_face = True
    render_option.mesh_color_option = o3d.visualization.MeshColorOption.Color
    render_option.mesh_shade_option = o3d.visualization.MeshShadeOption.Color

    rendered: list[Image.Image] = []
    for label, direction, nominal_up, camera_parameters in views:
        camera_up = nominal_up - direction * float(np.dot(nominal_up, direction))
        if np.linalg.norm(camera_up) < 1.0e-4:
            camera_up = front
        camera_up = _unit(camera_up)
        intensity = (
            0.22
            + 0.58 * np.abs(normals @ direction)
            + 0.20 * np.abs(normals @ camera_up)
        )
        base_color = np.array([0.38, 0.72, 0.98])
        colors = np.clip(intensity[:, None] * base_color[None, :], 0.0, 1.0)
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
        visualizer.update_geometry(mesh)
        view = visualizer.get_view_control()
        if camera_parameters is not None:
            view.convert_from_pinhole_camera_parameters(
                camera_parameters,
                allow_arbitrary=True,
            )
        else:
            view.set_lookat(center)
            view.set_front(direction)
            view.set_up(camera_up)
            view.set_zoom(0.72)
        visualizer.poll_events()
        visualizer.update_renderer()
        pixels = np.asarray(visualizer.capture_screen_float_buffer(do_render=True))
        image = Image.fromarray(np.clip(pixels * 255.0, 0, 255).astype(np.uint8)).convert("RGB")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=20)
        box = draw.textbbox((0, 0), label, font=font)
        draw.rounded_rectangle(
            (18, 18, 38 + box[2] - box[0], 34 + box[3] - box[1]),
            radius=7,
            fill=(10, 16, 25, 215),
        )
        draw.text((28, 24), label, font=font, fill=(240, 247, 255))
        rendered.append(image)

    visualizer.destroy_window()

    sheet = Image.new("RGB", (width * len(rendered), height), (6, 8, 12))
    for index, image in enumerate(rendered):
        sheet.paste(image, (index * width, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output_path, quality=95)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mesh", type=Path, help="input PLY triangle mesh")
    parser.add_argument("output", type=Path, help="output PNG contact sheet")
    parser.add_argument("--width", type=int, default=720, help="width of each view")
    parser.add_argument("--height", type=int, default=720, help="height of each view")
    parser.add_argument("--colmap-model", type=Path, help="optional COLMAP sparse model")
    parser.add_argument(
        "--colmap-index",
        type=int,
        action="append",
        dest="colmap_indices",
        help="zero-based sorted COLMAP image index; repeat for multiple views",
    )
    args = parser.parse_args()
    render_views(
        args.mesh.resolve(),
        args.output.resolve(),
        args.width,
        args.height,
        args.colmap_model.resolve() if args.colmap_model else None,
        args.colmap_indices,
    )


if __name__ == "__main__":
    main()
