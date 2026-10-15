# Training recovery

Train 3DGS prepares its task cache and starts a new run without saving a project.
Only Save Project / Ctrl+S, Save As, or an explicitly chosen Save and Exit writes
an `.ascan` file. A clean window closes directly; unsaved changes prompt the user
to save, discard, or cancel.

## Storage

Training recovery uses a small `.checkpoint` manifest and an independent binary
PLY model in the task cache. The manifest contains the PLY path, iteration, total
iterations, Gaussian count, input/settings fingerprint, and focus bounds.
Optimizer moments, random-generator state, appearance-correction state and
refinement statistics are not stored.

Automatic checkpoints run at iteration boundaries every 120 seconds by default,
on interruption, and at the final iteration. There is no forced first-iteration
save. Preferences > Storage > Training autosave changes the periodic interval;
zero disables periodic saves. A failed save is logged and training continues.
The new PLY is flushed before the manifest is atomically replaced; a failed
publication retains the previous successful manifest and model. A process lease
prevents concurrent writers.

Ctrl+S during training asks the trainer to save and hold at a completed iteration.
Studio embeds that model and small progress metadata in the user-selected project,
then releases the trainer. Saved projects also contain compact SfM data: cameras,
poses, triangulated points, valid inlier observations, and only referenced XY
features. Unused features and feature scale/orientation/response are omitted.
Original working SfM caches are not compacted.

A saved `.ascan` contains the model and SfM data needed to restore the result even
after deleting the task cache. Opening it recreates an independent PLY checkpoint
when resuming; the saved project itself is not modified. Original images and any
other training inputs must still be available to continue optimization.

## Resume semantics

Resume Training continues at the next iteration using the saved PLY model.
Optimizer, sampling, appearance-correction and refinement state are initialized
again: this is model-based continuation, not numerically identical interrupted
training. Inputs, cameras and numerical training settings are fingerprinted; a
mismatch requires a new training run.

Scale regularization now uses `scale_regularization_weight` (CLI:
`--splat-scale-reg`) on the mean actual Gaussian scale, with a default of zero.
The old `log_scale_regularization_weight` Python name,
`--splat-log-scale-reg` CLI flag and separate `shape_scale_reg` log-scale penalty
have been removed. Projects
through settings v10 ignore its stored weight rather than converting it to an
actual-scale weight. The training settings fingerprint has changed; old
checkpoints require a new training run.

Retired full-tensor checkpoints, embedded optimizer checkpoints and external
legacy recovery files are not read, migrated, or used as a fallback. Start a new
run for those formats; opening a project does not delete old recovery files.

Training completion publishes the final result and clears pending task recovery
artifacts. Crash recovery returns to the last successful checkpoint. SfM stage
caches, mesh extraction and texture baking retain their own existing behavior.

## CLI

```powershell
photara --images IMAGES --splat-dataset DATASET --output result.ply --splat
photara --images IMAGES --splat-dataset DATASET --output result.ply --splat --splat-resume
```

Use the same numerical training arguments for both commands.

- `--splat-checkpoint PATH`: independent manifest path; defaults to the output path
  with its extension replaced by `.checkpoint`. It must not be the project/output
  file itself.
- `--splat-checkpoint-seconds N`: periodic save interval (default 120; zero disables
  periodic saves without disabling interruption/final saves).
- `--splat-checkpoint-control PATH`: editor save/hold acknowledgement channel.

CLI output explicitly requested as `.ascan` still produces a completed project;
automatic training checkpoints remain separate PLY references.

`photara_splat_checkpoint_test` covers PLY continuation, next-iteration progress,
portable model snapshots, retired-format rejection, fingerprint invalidation,
single-writer leases, publication failures, and cooperative save/hold/cancel.
Run with `--vulkan` for the Vulkan backend. The process test kills an actual CLI
trainer and verifies recovery and concurrent-writer rejection.
