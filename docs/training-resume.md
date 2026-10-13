# Training recovery

Photara Studio saves a complete 3DGS training checkpoint after the first iteration,
every 120 seconds, and at the last iteration. Checkpoint I/O runs at an iteration
boundary and temporarily adds to that iteration's time. A failed automatic save
is logged and training continues; the previous successful checkpoint is retained.

**Save Project** / **Ctrl+S** during training asks the trainer to finish its current
iteration, save its state and hold at that boundary. The editor embeds the checkpoint
in the `.ascan` project, then lets training continue. A paused training process is
briefly resumed to serve the request and is paused again afterwards.

Closing the window or choosing Exit opens **Save and Exit / Exit Without Saving /
Cancel**. Save and Exit waits for the trainer's acknowledgement and a successful
project save before terminating the task. On failure the window stays open. Cancel
releases a pending save request; Exit Without Saving retains any earlier automatic
checkpoints, but does not promise to preserve the latest iterations.

Train 3DGS always starts a new training run directly, including after Stop. It replaces
the old pending checkpoint. Recovery is a separate **Resume Training** action in the
toolbar, Reconstruction menu and inspector. Reopening a project exposes its saved
progress without opening a recovery dialog. Automatic
checkpoints and their previous good generation are stored internally under
`.photara-recovery/<project filename>/training.ascan` and `training.ascan.previous`.
The recovery directory is hidden in Windows Explorer. The only visible project
file is `<project>.ascan`; internal recovery files are independent of the cache
folder and are not deleted by cache retention. Opening an older project migrates
its visible `.training.ascan` sidecars into recovery storage without rewriting
training data or overwriting existing generations. A manual project save embeds the
checkpoint so that reopening the `.ascan` can restore a missing sidecar.

The checkpoint preserves model tensors, FP32 and quantized SH Adam moments,
normal-feature update counts, PPISP/bilateral-grid parameters and Adam moments,
densification statistics, camera order and cursor, CPU/device random-generator state,
focus bounds, geometry/learning-rate bounds and adaptive multi-view scheduler state.
Preview images, decoded image caches, gradients and other scratch buffers are rebuilt.

Resume validates the training-state ABI, backend, numerical training settings,
cameras, initial cloud and image/mask contents. A mismatch produces an error; it does
not silently start over. Changing the total iteration count also requires a new run.
A corrupt current checkpoint falls back to the previous valid generation. Checkpoints
use the versioned `.ascan` container with per-chunk checksums, atomic replacement and
filesystem flushes. An OS-released process lease prevents concurrent training writers.
Training completion publishes the final products and clears the pending checkpoint.

Crash recovery returns to the last successful automatic checkpoint, not necessarily
the exact iteration at which the process crashed. The original images/dataset must
still be available. SFm alignment continues to use its existing stage caches;
mesh extraction and texture baking do not gain iteration-level resume from this change.
During preprocessing, Save and Exit may wait until 3DGS reaches its first iteration;
the dialog can be cancelled without terminating the task.

## CLI

```powershell
photara --images IMAGES --splat-dataset DATASET --output project.ascan --splat
photara --images IMAGES --splat-dataset DATASET --output project.ascan --splat --splat-resume
```

The same Resume Training action resumes an OS-paused process while training is
running, or loads a persisted checkpoint while idle. Train 3DGS starts a new run.
Storage and UI changes preserve checkpoint compatibility; incompatible changes
in training-state semantics require an explicit ABI/schema change.

Use the same numerical training arguments for both commands. Optional controls:

- `--splat-checkpoint PATH`: override the checkpoint file (default: internal recovery directory beside output).
- `--splat-checkpoint-seconds N`: periodic automatic save cadence (default 120;
  zero disables periodic saves, while first/last iteration saves remain enabled).
- `--splat-checkpoint-control PATH`: editor integration request sidecar; not needed
  for ordinary CLI runs.

`photara_splat_checkpoint_test` verifies interrupted/continuous training equivalence,
correct resumed iteration, topology changes, quantized optimizer state, appearance
correction, progressive resolution, checksum fallback, project embedding, invalidation,
write failures and cooperative save/hold/cancel. Run it with `--vulkan` for the Vulkan
backend. `training_resume_process_test.py` also kills an actual CLI process and
verifies that a fresh process resumes the saved iteration and that a concurrent
writer is rejected. The corresponding CTest names are `photara.splat.checkpoint` and
`photara.splat.checkpoint_vulkan`; the process test is `photara.splat.process_resume`.
