# SfM Improvement Notes

Key findings for future analysis and improvements. See [SFM_ARCHITECTURE.md](SFM_ARCHITECTURE.md) for the algorithm structure.

## Established Findings

- CUDA block solves, recovery caching, and parallel track splitting provide measurable gains and have numerical regression coverage. Berlin's split peel decreased from about 29.9 s to 1.5 s; further optimization should focus on remaining positioning, BA, and recovery costs.
- Bicycle, bonsai, flowers, and garden registered all images. The warm-start approach reduced mapping time by about 36%–60% without noticeable quality degradation. Flowers had fewer than 50000 positioning observations and fell back to CPU, so its speedup cannot be attributed to CUDA.
- With sufficient budgets, CPU/CUDA pose errors were very close on Berlin and Alameda. Differences under default budgets mainly reflected optimization progress rather than CUDA precision.
- Multi-cluster coverage loss, incorrect intrinsic merging, severe rotation conflicts, and panorama rear-hemisphere statistics have been fixed. Berlin's rotation outlier decreased from about 55° to 1.1°. Retain the relevant regressions even though these are no longer open defects.
- Keep the default cluster capacity of 2048, warm-start threshold of 512, and CPU fallback for now. Lowering the warm-start threshold is promising, but a few datasets do not justify changing the general default.
- Frontend defaults and Hierarchical parameter forwarding are unified. Studio applies its alignment backend to positioning and BA, reuses compatible caches for new projects, and displays persisted position risk. These changes make execution and uncertainty visible; they do not repair weakly constrained geometry.

## Improvement Priorities

| Issue | Evidence | Next Steps |
|---|---|---|
| Short CPU budgets | Timeout, stall, and IRLS convergence are now distinguished. Passes without effective updates or genuine convergence no longer change weights; retries receive larger slices only when the remaining budget permits. | Continue validating quality across hardware and CPU fallback paths. Budget exhaustion does not imply acceptable accuracy. |
| Optimization objective versus pose accuracy | More iterations and lower residuals did not always improve agreement with the reference on Berlin or Alameda. | Analyze weak constraints, incorrect matches, and robust weights; avoid using a single residual metric as the quality objective. |
| Alameda local camera-center drift | Maximum center deviation was about 3.19 reference steps. Strong-connection and parallax audits now flag the worst cameras, but PnP from stable-only depths did not pass their repair validation. | Position-unverified status is now persisted. Continue investigating cross-branch constraints; neither a risk flag nor low RMS demonstrates a successful repair. |
| Tail quality in multiple clusters and weak branches | Coverage was restored, but indoor multi-cluster center P95 remained about 1.78 steps; the office scene had a maximum rotation difference of about 4° relative to Global. | Continue checking cross-cluster merging, shared intrinsics, and weak-branch recovery. Registration rate and mean error alone are insufficient. |
| Large scenes and resource fallback | Most experiments used the same RTX 5090. Default multi-cluster runs above 2048 images and low-VRAM comparisons are still missing. | Add real large-scene, target-hardware, and backend-failure fallback tests. |

In the earlier Alameda comparison, center P95 under default CPU/CUDA budgets was about 0.100/0.127 reference steps; CPU with sufficient budget also reached about 0.127 steps. CUDA numerical consistency and position accuracy in the weak segment therefore require separate analysis. Flowers' `_DSC9142.JPG` retained a rotation difference of about 1.11°, a tail issue shared by the default and warm-start paths.

## Comparison Guidelines

- Reuse identical tracks and bypass reconstruction-cache shortcuts. Use reference models only for post-reconstruction evaluation, never as reconstruction input.
- Compare registered image sets, rotation/center P95, worst cameras, and reprojection errors. Neither `valid` nor the exit code establishes accuracy; the reference model is not absolute ground truth either.
- Measure mapping, complete alignment including final recovery/audits, and cold-frontend end-to-end time separately. Exclude runs with GPU contention from performance conclusions.
- Confirm iteration limits, tolerances, and actual timeouts when comparing backend accuracy. A time parameter of `0` disables only the shared budget; the existing 45 s limit per bearing Solve remains.
- Use `PHOTARA_LOG_LEVEL=debug` to inspect per-pass costs, termination reasons, and `accepted_updates`. Record the actual backend; requesting CUDA does not prove that GPU execution occurred.
