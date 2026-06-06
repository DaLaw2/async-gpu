# split-devpath synthesis

Dev builds now skip ptxas entirely (PTX only, <1s incremental).
The `--prod` flag restores full cubin compilation (30+ min).

Host loader and `#[gpu_test]` macro already had cubin-optional
fallback via `unwrap_or_default()` + empty-slice PTX JIT path.
No Rust code changes were needed — only build script gating.

Trade-off: runtime PTX JIT is slow for large kernels (~10 min
for the 6 MB test PTX). Acceptable for dev; use `--prod` for
benchmarks or repeated test runs.
