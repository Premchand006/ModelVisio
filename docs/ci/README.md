# ModelVisio in CI

`@modelvisio/cli` (`packages/cli`) runs the same analysis as the app from the command line:
compiler-compat pre-flight, size and latency budgets, a diff between two versions of a model,
and a model registry kept as a JSON file in your repo. The composite action at
`.github/actions/modelvisio-check` wraps it for GitHub Actions.

## The action

Copy [`modelvisio.yml`](modelvisio.yml) into `.github/workflows/` of the repo that holds your
models and adjust the paths and budgets. The action:

- builds the CLI from the ModelVisio checkout GitHub downloads with the action (Node and a pinned
  pnpm via `npx`, about a minute on a cold runner),
- runs `modelvisio check` on every file matched by `models` (spaces or newlines between globs,
  `**` supported; paths with spaces aren't),
- if `baseline-ref` is set, fetches each model at that ref with `git show` and runs
  `modelvisio diff` against it (use `fetch-depth: 0` on checkout; Git LFS pointers are detected
  and skipped with a warning unless LFS objects are fetched),
- appends a markdown report to the job summary, optionally writes JUnit XML (`junit-path`),
  and optionally adds the models to a registry file (`registry`).

The step fails when a check fails at the `fail-on` severity, when a budget is exceeded, or,
with `fail-on-regression: "true"`, when the diff reports a regression. Outputs: `passed`
(`"true"`/`"false"`) and `report` (path of the markdown report).

| Input | Default | Meaning |
|---|---|---|
| `models` | (required) | Files or globs to check |
| `targets` | `coral,rk3588,kneron,hailo` | Compiler targets or device ids (`modelvisio targets`) |
| `device` | each target's device | Device for the latency budget |
| `max-latency-ms` | none | Latency budget (roofline estimate) |
| `max-size-mb` | none | File size budget |
| `fail-on` | `error` | `warn` also fails on warnings |
| `baseline-ref` | none | Ref to diff against, e.g. `${{ github.event.pull_request.base.sha }}` |
| `fail-on-regression` | `false` | Fail when the diff finds a regression |
| `latency-tolerance` | `5` | Percent latency increase that counts as a regression |
| `junit-path` | none | Also write JUnit XML here |
| `registry` | none | Add the models to this registry JSON (written, not committed) |
| `node-version` | `22` | Node.js used to run the CLI |

## The CLI

Build it once (`pnpm --filter @modelvisio/cli build`), then run `node packages/cli/dist/modelvisio.cjs`.
The bundle is one self-contained file that runs on Windows, Linux and macOS with Node 18 or
newer. The examples below write `modelvisio` for that command.

```sh
modelvisio inspect model.onnx [--json]
modelvisio check model.onnx --target coral,rk3588 --device orin_nano \
  --max-latency-ms 15 --max-size-mb 25 --fail-on error --format markdown
modelvisio diff old.onnx new.onnx --fail-on-regression --latency-tolerance 10
modelvisio registry add model.onnx --tags detector --notes "pruned 30%"
modelvisio registry export --format markdown
modelvisio targets
```

- `--format text|json|markdown|junit` (`check`), `text|json|markdown` (`inspect`, `diff`).
- `--output FILE` writes the report to a file instead of stdout. `--summary FILE` also appends
  the markdown report to a file, e.g. `--summary "$GITHUB_STEP_SUMMARY"`.
- Exit codes: `0` ok, `1` check failed or regression found, `2` usage or input error.

### What counts as a failure

- **Compat:** per target, the op-level issues attached to layers (the same list the Compiler
  tab shows) plus device-level blockers from the hardware scorer for that target's reference
  device: the model doesn't fit in device memory (error), an unsupported op with no CPU fallback
  (error), spilling out of on-chip memory or ops falling back to the CPU (warning).
- **Budgets:** file size over `--max-size-mb`, or estimated latency over `--max-latency-ms`
  (errors). If latency can't be estimated (a format whose parser has no FLOP counts, such as
  PyTorch `.pt` today, or a metadata-only format), the budget is reported as `unknown`, which
  is a warning.
- **Regressions (diff):** more compat errors or warnings on a target, a model that no longer
  fits or compiles on a reference device, estimated latency up by more than the tolerance, and
  (with `--size-tolerance`) file-size growth.

Latency is the roofline estimate from the model graph (see the main README), not an on-device
measurement. Treat budgets on it as a guard against large changes rather than a timing SLA.

### Registry file

`modelvisio registry add` keeps `modelvisio-registry.json` (schema `modelvisio.registry/v1`),
the same format the app's Registry panel exports and imports. Each version stores the file
name, size, SHA-256, format, stats, a per-layer fingerprint, the compat matrix and hardware
estimates. It does not store the model itself. Adding a file whose SHA-256 is already
registered for that model is a no-op. The file's `exportedAt` is the newest change, so
rewriting it without changes gives no git diff. Set `SOURCE_DATE_EPOCH` for reproducible
timestamps.
