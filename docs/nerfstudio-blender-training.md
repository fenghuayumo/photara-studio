# Blender / Nerfstudio 3DGS training

Photara imports a dataset directory containing `transforms.json`, or a JSON file directly:

```powershell
photara.exe --splat-dataset "D:/data/scene/transforms.json" --output "D:/data/scene/result.sog"
```

No SfM step or `--images` argument is required. `--images` can override image lookup.
Studio's external dataset file picker accepts JSON; Agent Control `studio.project.set_dataset`
accepts a directory or file, followed by `studio.job.start {stage:"train"}`.
When both COLMAP and JSON exist in a directory, auto detection prefers COLMAP. Select the
JSON file explicitly to use Nerfstudio metadata.

Supported cameras: perspective PINHOLE, SIMPLE_PINHOLE, and OPENCV (k1/k2/p1/p2).
Other camera models and nonzero k3/k4 are rejected. Intrinsics may be global or per frame;
`camera_angle_x/y` is accepted for legacy Blender datasets. Extensionless image paths
are resolved using common image extensions. Declared image dimensions are used to scale
intrinsics when the selected image has a different resolution.

`transform_matrix` is a rigid 4x4 OpenGL camera-to-world pose. The reader flips camera Y/Z
to OpenCV, preserving world coordinates and scale. It validates poses before training.
An optional `ply_file_path` is resolved relative to the JSON. An explicitly supplied
`--dense-ply` overrides it; otherwise camera-only data uses deterministic random initialization.

The companion photara-blender plugin exports per-frame intrinsics in JSON and distinct
COLMAP cameras for changing intrinsics. For static 3DGS, animate the camera around a
static scene, use opaque RGB PNG, and disable render borders and motion blur.

Validation: `photara_nerfstudio_dataset_test` checks loading, OpenGL conversion, frame
intrinsics, angle fallback, resized images, random initialization, and invalid-camera rejection.
