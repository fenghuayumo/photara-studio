# GGGS reference rasterizer notice

This directory is a source-level import of the CUDA GGGS rasterizer core used
by the local `pygsplat/dvsplat_utils/fastergs` implementation. The PyTorch and
PyBind wrappers were intentionally not imported; Photara wraps the raw CUDA
entry points with TinyTensor.

The imported source headers state that the code is available only for
non-commercial research and evaluation use under an upstream `LICENSE.md`.
That license file was not present in the provided local FasterGS directory.
Consequently this backend is suitable for prototyping and parity work only.
Before distributing or commercializing Photara, obtain the complete
upstream license/permission or replace this directory with an independently
licensed implementation behind the same `splat::Rasterizer` API.
