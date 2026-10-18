# Photara Studio control

Use `photara-ctl` only when the user asks you to operate a running Photara
Studio. Do not use it for ordinary source-code edits.

Start with:

```text
photara-ctl state --json
```

Discover parameters and ranges with:

```text
photara-ctl schema --json
```

Rules:

- Never construct raw `photara.exe` command lines for Studio control.
- Never edit `.ascan` files or the runtime cache directly.
- Use `job start ... --json`, then `job wait` or `job status`.
- Use `capture --out ...` when you need a visual check.
- Ask the user before overwriting projects, deleting results, or clearing caches.
- If a command returns `PRECONDITION_FAILED`, follow its `details.reason`
  instead of retrying the same command.

Typical workflow:

```text
photara-ctl project open <project.ascan> --json
photara-ctl params set strategy=adc_igs iterations=30000 --json
photara-ctl job start train --json
photara-ctl job wait --timeout 30 --json
photara-ctl capture --out <project-dir>/agent_view.png
```
