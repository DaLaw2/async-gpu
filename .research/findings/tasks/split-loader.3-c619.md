# split-loader.3: Build script — parallel per-crate PTX + cubin

**Status**: DONE
**Kind**: experiment
**Cycle**: 619

## Summary

Created `scripts/build-kernels.sh`, a unified build script that compiles all 4
kernel crates (core, compute, io, test) to PTX sequentially, then runs ptxas in
parallel for all cubins.

## What was done

### 1. Created `scripts/build-kernels.sh`

New unified build script with:

- **Sequential PTX builds**: Shared dependencies compile once across the 4 crates
- **Parallel ptxas**: All cubin compilations run simultaneously via background jobs
- **Selective builds**: Optional crate filter (`./build-kernels.sh core test`)
- **SM auto-detection**: Reads compute capability from nvidia-smi, defaults to sm_75
- **Backward-compat aliases**: kernel_test.ptx → kernel.ptx + kernel_std.ptx,
  kernel_test.cubin → kernel_std.cubin
- **Proper error handling**: Validates arguments, tracks ptxas failures per-crate,
  reports which builds succeeded/failed

### 2. Kept `build-kernel-test.sh` as-is

The single-crate script remains for quick test-kernel-only iteration.

### 3. Updated `scripts/setup.sh`

Changed the `--std` mode kernel build step (line ~571) from calling
`build-kernel-test.sh` to `build-kernels.sh`, and updated the cache-hit check
to verify all 4 cubin files exist (not just kernel_std.cubin).

### 4. Verification

- `bash -n build-kernels.sh` — syntax check passes
- Argument validation tested (rejects unknown crates, accepts valid ones)
- PTX build verified working for gpu-kernel-core with the nightly toolchain
- Script trace (`bash -x`) confirms correct flow: toolchain detection, ptxas
  discovery, SM detection, sequential PTX build, parallel cubin dispatch

## Files changed

- `scripts/build-kernels.sh` — NEW (unified multi-crate build script)
- `scripts/setup.sh` — MODIFIED (uses build-kernels.sh, checks all cubins)

## Design decisions

1. **Sequential PTX, parallel ptxas**: PTX builds share Cargo's dependency cache,
   so sequential avoids redundant recompilation of gpu-kernel-common etc. ptxas
   is CPU-bound per-crate with no shared state, so parallelism is safe.

2. **Short crate names as arguments**: `./build-kernels.sh core test` instead of
   `gpu-kernel-core gpu-kernel-test` — less typing, the prefix is implicit.

3. **SM auto-detection**: Falls back to sm_75 (Turing) if nvidia-smi is not
   available, matching the existing build-kernel-test.sh hardcoded sm_75.
