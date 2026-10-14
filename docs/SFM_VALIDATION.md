# SfM Improvement Notes

Key findings for future analysis and improvements. See [SFM_ARCHITECTURE.md](SFM_ARCHITECTURE.md) for the algorithm structure.

The complete fixed-track regression and reference-free Alameda recovery probes
are documented in [SFM_FULL_BASELINE.md](SFM_FULL_BASELINE.md). That baseline
includes final recovery, BA and audits; the mapping-only comparisons below have
a different scope.

## Established Findings

- CUDA block solves, recovery caching, and parallel track splitting provide measurable gains and have numerical regression coverage. Berlin's split peel decreased from about 29.9 s to 1.5 s; further optimization should focus on remaining positioning, BA, and recovery costs.
- Bicycle, bonsai, flowers, and garden registered all images. The warm-start approach reduced mapping time by about 36%–60% without noticeable quality degradation. Flowers had fewer than 50000 positioning observations and fell back to CPU, so its speedup cannot be attributed to CUDA.
- With sufficient budgets, CPU/CUDA pose errors were very close on Berlin and Alameda. Differences under default budgets mainly reflected optimization progress rather than CUDA precision.
- Multi-cluster coverage loss, incorrect intrinsic merging, severe rotation conflicts, and panorama rear-hemisphere statistics have been fixed. Berlin's rotation outlier decreased from about 55° to 1.1°. Retain the relevant regressions even though these are no longer open defects.
- Keep the default cluster capacity of 2048, warm-start threshold of 512, and CPU fallback for now. Lowering the warm-start threshold is promising, but a few datasets do not justify changing the general default. Warm-start comparisons remain in development tools; CLI and Studio expose no new tuning switch. Internal geometry checks now screen bearing refinement for regressions against its initializer; passing this screen does not establish new position-accuracy evidence.
- Frontend defaults and Hierarchical parameter forwarding are unified. Studio applies its alignment backend to positioning and BA, reuses compatible caches for new projects, and displays persisted position risk. These changes make execution and uncertainty visible; they do not repair weakly constrained geometry.

## Improvement Priorities

| Issue | Evidence | Next Steps |
|---|---|---|
| Short CPU budgets | Timeout, stall, and IRLS convergence are now distinguished. Passes without effective updates or genuine convergence no longer change weights; retries receive larger slices only when the remaining budget permits. | Continue validating quality across hardware and CPU fallback paths. Budget exhaustion does not imply acceptable accuracy. |
| Optimization objective versus pose accuracy | More iterations and lower residuals did not always improve agreement with the reference on Berlin or Alameda. | Analyze weak constraints, incorrect matches, and robust weights; avoid using a single residual metric as the quality objective. |
| Alameda local camera-center drift | Maximum center deviation remains about 3.19 reference steps. The nine-frame independent submap reconstructs but fails stable-map similarity consensus (216/723 fit inliers). Layered stable-only resection can worsen rotation despite improving pixel residuals; an independent pair-rotation guard now rejects a demonstrated harmful proposal. | Keep layered propagation offline and retain the geometric gates. Improve independent cross-segment evidence; neither a risk flag nor low RMS demonstrates a successful repair. |
| Tail quality in multiple clusters and weak branches | Coverage was restored, but indoor multi-cluster center P95 remained about 1.78 steps; the office scene had a maximum rotation difference of about 4° relative to Global. | Continue checking cross-cluster merging, shared intrinsics, and weak-branch recovery. Registration rate and mean error alone are insufficient. |
| Large scenes and resource fallback | Most experiments used the same RTX 5090. Default multi-cluster runs above 2048 images and low-VRAM comparisons are still missing. | Add real large-scene, target-hardware, and backend-failure fallback tests. |

In the earlier Alameda comparison, center P95 under default CPU/CUDA budgets was about 0.100/0.127 reference steps; CPU with sufficient budget also reached about 0.127 steps. CUDA numerical consistency and position accuracy in the weak segment therefore require separate analysis. Flowers' `_DSC9142.JPG` retained a rotation difference of about 1.11°, a tail issue shared by the default and warm-start paths.

## Comparison Guidelines

Targeted warm-start screening validation (2026-10-14) reused fixed tracks and
kept the product threshold at 512. Development-only small-scene runs retained
all 292 bonsai and 173 flowers cameras. Center P95 against the reference was
0.033725/0.027146 steps, versus 0.033850/0.027285 for the default path; rotation
P95 was 0.243809/0.218198 degrees versus 0.243718/0.218024. Berlin's 1511-camera
default warm-start run had center P95 0.110038 steps and rotation P95 0.408634
degrees. Flowers/Berlin logged 2/11 direction changes without enough local
held-out evidence, while their independently observed errors improved. Missing
evidence is not a proven regression or an accuracy certificate: forcing full
fallback solely for missing coverage gave Berlin center P95 11.544147 steps in
the intermediate screening experiment. These are mapping-only comparisons,
excluding final reconstruction recovery/audits; build/test activity overlapped
some runs, so timings do not establish a general speedup. Keep 512 until broader
coverage and weak-branch validation justify expansion.

- Reuse identical tracks and bypass reconstruction-cache shortcuts. Use reference models only for post-reconstruction evaluation, never as reconstruction input.
- Compare registered image sets, rotation/center P95, worst cameras, and reprojection errors. Neither `valid` nor the exit code establishes accuracy; the reference model is not absolute ground truth either.
- Measure mapping, complete alignment including final recovery/audits, and cold-frontend end-to-end time separately. Exclude runs with GPU contention from performance conclusions.
- Confirm iteration limits, tolerances, and actual timeouts when comparing backend accuracy. A time parameter of `0` disables only the shared budget; the existing 45 s limit per bearing Solve remains.
- Use `PHOTARA_LOG_LEVEL=debug` to inspect per-pass costs, termination reasons, and `accepted_updates`. Record the actual backend; requesting CUDA does not prove that GPU execution occurred.
