# Studio AI geometry priors

In **Train 3DGS → Loss & constraints → Geometry**, enable **AI depth & normal
priors**. The feature is available in Standard and High quality modes. Fast
mode keeps RGB-only training and skips prior generation and loading.

Choose either source:

- **Generate with MoGe**: the first training run downloads the fixed MoGe-2
  ViT-B Normal F32 GGUF (419,001,600 bytes, about 400 MiB) from the project's
  GitHub release. No file selection, Python or local conversion is required.
  Download progress, cancellation and retry are available; interrupted files
  can resume. The downloaded file must pass exact size, SHA-256 and native
  GGUF structure checks before it replaces the cached model. Training resumes
  automatically after a training-triggered download succeeds. **Download
  model** also allows preloading without starting training.
  The model is shared across projects in `%LOCALAPPDATA%/Photara/models`
  (Linux: `$XDG_CACHE_HOME/Photara/models`, or `~/.cache/Photara/models`).
  Subsequent use checks a verification receipt against the expected hash,
  current size and modification timestamp, without rehashing on the UI thread.
  Existing unmarked files are verified on a worker before reuse, including
  offline reuse. Maps are then prepared before starting the optimizer.
  The default token budget is 1200. Leave the prior-map cache folder blank to
  use `<task cache>/ai_priors`, or select a folder.
- **Existing aligned cache**: select a folder containing `view_*.prior` maps
  prepared for the same image files, camera poses, intrinsics and full training
  resolution. Legacy PHPRIOR1 caches and optional-map PHPRIOR2 caches are
  supported. This mode does not load MoGe.

Depth and normal weights are independent (initial UI values: 0.01 each; the
feature is off by default). Setting one to zero avoids loading that map. Both
zero skips all prior work. Normal-only preparation does not require sparse
depth alignment. Depth preparation requires at least 16 robust observed sparse
points per training view. Unsupported or insufficient inputs fail explicitly.

AI priors do not enable DN, normal-field or multi-view losses. Their existing
controls remain independent. AI priors share **Geometry start step** with DN
and multi-view losses (`--splat-geometry-from-iter`, default 7000), including
when DN is disabled. Before that step they contribute no loss or gradients;
at that step the configured weights take effect without a ramp. Gaussian mesh
extraction retains its existing
geometry policy. Priors use pinhole training views: fisheye images are rectified
when AI priors are active; equirectangular panoramas are unsupported.

Preparation runs in the training child process, using the finalized training
cameras, image loader and evaluation split. Its view progress appears before
the optimizer timer starts. The MoGe session is released before 3DGS training.
Compatible caches are checked before loading the model. Reuse checks model
path/size/modification time, token budget, preprocessing options, sparse scale
support, and each cached camera and source image. Stale or missing maps are
regenerated. Interrupted generations are not marked as reusable. Held-out
views are never predicted or supervised.

Source selection, model/cache paths, token budget and independent weights are
saved in `.ascan` settings version 13. Versions through 12 load with AI priors
disabled. Generated maps remain in the cache rather than being embedded in
the project archive.

Equivalent CLI parameters:

```text
--splat-prior-dir <cache> --splat-prior-model <model.gguf>
--splat-prior-tokens 1200 --splat-lambda-d 0.01 --splat-lambda-n 0.01
--splat-undistort=true
```

Omit `--splat-prior-model` to consume an existing cache without inference.

## Distributed model identity

- Download: https://github.com/fenghuayumo/photara-studio/releases/download/v0.5.0/moge-2-vitb-normal-f32.gguf
- SHA-256: `b515fd1ac1a187f46871aabd346870eaad22d2f20d2848637758c5c6e674b1c2`
- Original model: `Ruicheng/moge-2-vitb-normal`
- Original revision: `ca5f0e07ff01d3e5a364c1d954ed12ee1814b368`
- Original checkpoint SHA-256: `16b8110e86d5dc5a849db120ca96ef3a223fd30b0c9146d1d81db504073da5f6`
- Conversion: `python/models/moge/convert_weights.py`, F32, tensor values preserved.
- License: upstream model card labels the weights MIT. MoGe code is MIT;
  upstream DINOv2 code is Apache-2.0. See `licenses/MoGe-LICENSE.txt` and
  `THIRD_PARTY_NOTICES.md`. Unlike SAM 3, MoGe download does not require
  accepting Meta's SAM license.

`photara.studio.model_download` tests SHA-256 vectors, resumed transfers,
offline reuse, changed-file receipts, checksum and size failures, repair,
transport failure and cancellation using local fixtures. No network requests
are made by this test. An explicit real-release check is available via:

```text
photara_model_download_test --remote <output.gguf>
```
