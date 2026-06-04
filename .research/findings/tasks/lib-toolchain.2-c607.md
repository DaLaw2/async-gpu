# lib-toolchain.2 — Polish scripts/setup.sh

## Changes Made

### 1. Added minimal PTX smoke test function (`smoke_test_ptx`)
- Compiles a trivial `#![no_std]` kernel with `extern "ptx-kernel"` ABI to PTX
- Uses a temp directory with automatic cleanup (trap RETURN)
- Verifies output contains `.visible .entry` (valid PTX kernel entry point)
- Required `#![feature(abi_ptx)]` since the ptx-kernel ABI is unstable
- Used in both `--check` mode (step 4/5) and setup modes (step 4/N after toolchain install)

### 2. Expanded `--check` mode from 4 to 5 steps
- Step 4/5: PTX smoke test (compile trivial kernel) — fast, targeted verification
- Step 5/5: Full crate build + example run — comprehensive validation
- Smoke test failure blocks the full build (avoids wasting time on a known-broken setup)
- Added `WARNINGS` counter separate from `ISSUES` for non-blocking warnings

### 3. Added smoke test step to setup modes (quick/std/full)
- After toolchain install + PTX build, runs `smoke_test_ptx` as validation
- Step counts updated: quick=4, std=7, full=8
- On smoke test failure, provides actionable remediation advice

### 4. Improved error handling
- PTX kernel build failures now capture output to a temp log file and show last 10 lines
- Smoke test failures display the last 5 lines of compiler output
- hello-gpu example missing now shows an info message instead of silently skipping

### 5. Better progress messages with time estimates
- Each step now includes approximate duration: `(~5s)`, `(~1 min if not cached)`, `(~30s)`, `(~10-15 min)`
- Helps users set expectations, especially for long builds

### 6. Fixed banner box alignment
- All completion banners (`quick`, `std`, `full`) and the initial banner now have consistent box-drawing alignment
- Reduced box width to 40 chars for consistency

### 7. Improved summary output
- `--check` mode now distinguishes "all passed" vs "OK with N warnings" vs "N issues found"
- Warnings are explicitly labeled as non-blocking
- Summary references the marker symbols used in output

### 8. File sizes shown in human-readable format
- kernel_std.ptx and kernel_std.cubin sizes now use `numfmt --to=iec` (e.g., "8.6M", "142M")
- Falls back to raw byte count if numfmt unavailable

## Test Results (`--check` mode)

All 5 steps pass on the development machine:

```
[1/5] Environment          — 7/7 checks pass (rustup, nightly, components, target, GPU, ptxas)
[2/5] Patched std status   — 4/4 checks pass (patched-std dir, sysroot patched, PTX, cubin)
[3/5] Patched compiler     — patched rustc found at stage1
[4/5] PTX smoke test       — minimal PTX compilation succeeded
[5/5] Full crate build     — gpu-kernel-std build + hello-gpu example both pass
```

Exit code: 0 ("Setup looks good! All checks passed.")

## Issues Found and Resolved

1. **Missing `#![feature(abi_ptx)]`**: The initial smoke test kernel used `extern "ptx-kernel"` without the required feature gate. Fixed by adding the feature attribute to the minimal kernel source.

2. **Build output swallowed on failure**: The original script piped build output to `/dev/null` in check mode, making failures opaque. Fixed by capturing to a temp log and displaying the tail on failure.

3. **No validation after setup**: The setup modes (quick/std/full) installed the toolchain but never verified it actually works. Added the smoke test as an explicit post-install verification step.
