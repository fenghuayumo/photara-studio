# Mask models on GGML / Vulkan

Studio's reconstruction inspector and mask editor share the same engine/model
selection and download queue. New sessions default to Grounding DINO + SAM 2.1
Base Plus and the Vulkan backend. Existing projects preserve their saved engine.

| Engine | Segmentation checkpoints | Text detection |
| --- | --- | --- |
| SAM 3 | q4_0, f16 | Built into SAM 3 |
| Language | SAM 2 and SAM 2.1: Tiny, Small, Base Plus, Large (f16 or q8_0) | Grounding DINO Tiny or Base |
| Language | EdgeTAM f16, q8_0, q4_0 | Grounding DINO Tiny or Base |

Grounding DINO reads the original IDEA-Research F32 safetensors checkpoints,
converts matrix weights to GGML f16, and runs its Swin/BERT encoder, multimodal
encoder, deformable attention and box decoder through the shared GGML runtime.
It does not require ONNX Runtime, PyTorch, or Python. SAM and EdgeTAM use the
bundled sam3 GGML implementation. Explicit Vulkan requests never switch the
Grounding DINO graph to CPU. The language path accepts `auto`, `cpu`, or `vulkan`.

Enter one or more English subject phrases, separated by semicolons. Each detected
box prompts a separate visual segmentation call; their masks are unioned.
Negative phrases subtract regions before the keep/remove polarity is applied.
White pixels in the saved PNG mean keep. Source image dimensions are preserved.
The language engine detects again on every image; SAM 3 alone supports the
existing text tracking option. Preview and batch generation use the same API.

Select **Download missing models** to fetch the selected segmentation checkpoint,
detector and BERT vocabulary sequentially. Changing the selection does not enqueue
additional downloads. Cancel/failure stops the queue. Model files are stored in
the existing Photara model cache; custom paths and catalog ids are also accepted
by the CLI. Known downloads must reach their complete catalog size before they
are treated as ready. Every model family, including Grounding DINO and its
vocabulary, requires explicit agreement before downloading or using local files.
Studio shows a separate, initially unchecked confirmation for each unaccepted
family. Receipts are stored in the current user's configuration directory under
`Photara/model-agreements`, with the family, terms URL and agreement revision.
Legacy SAM 3 marker files alone do not satisfy this check. Receipts are never
stored in projects or portable application directories. MoGe downloads also wait
for explicit confirmation. Model terms are separate from the application licence.

The CLI never downloads weights. After reviewing the provider links displayed by
Studio (or in the missing-agreement error), users can explicitly record consent
with `--accept-model-licenses sam2,grounding-dino` (or `sam3`, `edgetam`, `moge`).
The mask library and MoGe prior-preparation entry point enforce agreements before
loading weights. No test or
build step accepts terms on behalf of the current user.

Release scripts stage only application binaries and notices, reject weight files
and agreement/cache records before archiving, and disable the old Intrinsic
weight-bundling option. The installer also excludes model/cache/receipt patterns.

```powershell
photara --images images --sam-engine language --sam-model sam2.1-tiny `
  --detector-model grounding-dino-tiny --sam-backend vulkan --sam-text "cat"

photara --images images --sam-engine language --sam-model edgetam-f16 `
  --sam-backend vulkan --sam-text "person; car" --sam-neg-text "bicycle"
```

Environment overrides remain available: `PHOTARA_SAM_MODEL`,
`PHOTARA_TEXT_DETECTOR_MODEL`, and `PHOTARA_TEXT_DETECTOR_VOCAB`.

## Validation

Real-checkpoint smokes now require prior acceptance by the user. Automated agreement
checks use isolated temporary fixture directories and never write to user settings.

`photara.sam.model_cache` checks catalog uniqueness, checkpoint discovery and
partial-file rejection. `photara.project.io` checks model/detector/engine settings
and legacy-project migration. The manual `photara_sam_masks_smoke` target takes
`model image output_dir [language|sam3]`, uses Vulkan without downloading weights,
and checks output dimensions, progress and a nonempty/nonfull mask of the cat
fixture in `third_party/sam3/tests/cat.jpg`. `photara_grounding_dino_smoke` takes
`weights vocab image.ppm [cpu]` and requires a cat detection. Real-checkpoint
smokes are intentionally not part of automatic ctest runs.

The reference project's BiRefNet uses its own Vulkan operators and is not a
GGML model adapter in this change. The model selector lists implemented models.
