# split-devpath.1: PTX JIT dev path — skip ptxas in default build

**Status**: DONE
**Kind**: experiment

## Summary

Implemented the PTX-only dev build path. Default `build-kernels.sh` and
`build-kernel-test.sh` now skip ptxas entirely, producing only PTX files.
The `--prod` flag restores the full PTX+cubin pipeline.

## Findings

### 1. Build scripts updated

- **build-kernels.sh**: ptxas discovery, SM detection, and Step 3 (cubin
  compilation) are now gated behind `BUILD_MODE == "prod"`. Dev mode prints
  a skip message and the final summary omits cubin file sizes.
- **build-kernel-test.sh**: Same treatment — ptxas search and Step 4
  (cubin pre-compilation) only run with `--prod`.

### 2. build.rs — no changes needed

The `gpu-host/build.rs` only compiles PTX via `cargo build --release` for
each kernel crate and copies the `.ptx` files. It never invokes ptxas.
No modification required.

### 3. Host loader already handles missing cubin

The entire PTX JIT fallback chain is already in place:

- `load_module_cubin_or_ptx(ptx, &[])` — empty cubin slice goes straight
  to PTX JIT via `cuModuleLoadData` (gpu.rs line 56).
- `run_zero_param_with_cubin(ptx, &[], ...)` — calls the above.
- `run_zero_param_with_config(...)` — calls `run_zero_param_with_cubin`
  with `&[]`.
- `GpuStdModule::load(...)` → `load_with_print(...)` → `load_with_cubin(..., &[], ...)`
  — same empty cubin path.

### 4. #[gpu_test] macro — already cubin-optional

The macro generates:
```rust
let cubin = std::fs::read(&cubin_path).unwrap_or_default();
```
When `.cubin` files don't exist, `unwrap_or_default()` returns `Vec::new()`,
which becomes `&[]` passed to `run_zero_param_with_cubin`. PTX JIT fallback
activates automatically.

### 5. Timing

Dev build (all 4 crates, incremental, no ptxas): **< 1s** with cached PTX.
Clean build: ~30s for PTX compilation across all 4 crates.
Previously: 30+ minutes with ptxas.

The trade-off: runtime PTX JIT for the large test kernel (~6 MB) takes
10+ minutes on first load. This is acceptable for dev iteration since most
dev workflows don't need every kernel test to run repeatedly.

## Files changed

- `scripts/build-kernels.sh` — gate ptxas behind `--prod`
- `scripts/build-kernel-test.sh` — gate ptxas behind `--prod`
