# Logging and progress reporting

`core/logging.hpp` provides the process-wide `Logger`, `StageScope`, and
`ProgressReporter` used by the reconstruction pipeline.

## CLI behavior

The `aetherscan` executable writes logs to both the console and a timestamped
file next to the requested output file:

```text
aetherscan-YYYYMMDD-HHMMSS-mmm.log
```

Console verbosity defaults to `info`. Set `AETHERSCAN_LOG_LEVEL` to one of:

```text
error | warning | info | debug | trace | off
```

PowerShell example:

```powershell
$env:AETHERSCAN_LOG_LEVEL = "debug"
.\aetherscan.exe images 900 incremental scene.mvs
```

The log file always records through `trace`, independently of console
verbosity. BA iteration details are emitted at `debug`; stage boundaries,
progress, BA summaries, and final reconstruction statistics use `info`.

## Progress format

Periodic reports contain stable machine-readable fields:

```text
progress: match image pairs completed=179/321 percent=55.8 items/s=14.71 elapsed_s=12.17 eta_s=9.65
```

Worker threads only call `ProgressReporter::advance()`, which is an atomic
increment. A dedicated reporter thread performs formatting and logger I/O, so
parallel feature, matching, geometry, track, and subscene workers do not
contend on the output mutex.

## Library use

Applications embedding AetherScan can configure logging before invoking SfM:

```cpp
#include "core/logging.hpp"

using namespace aetherscan::core;
Logger::instance().configure("logs", "my-product", LogLevel::info, LogLevel::trace);
```

Without explicit configuration, the library prints warnings and errors to the
console and does not create a file.
