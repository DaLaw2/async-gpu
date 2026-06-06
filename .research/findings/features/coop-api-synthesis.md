# Theme Synthesis: coop-api — Cooperative API Ergonomics

## Problem
Closure captures across warps fail (per-warp local memory). Users needed
3+ global atomics + unsafe to pass data into `cooperative()`.

## Solution: Three cooperative APIs

1. **`cooperative_map(src, dst, len, fn)`** — data-parallel transform. Zero
   global atomics. `fn(&CoopMapArgs)` enforces no-capture at compile time.

2. **`cooperative_reduce(src, len, fn) -> u64`** — multi-warp reduction.
   Each warp returns a partial via `WARP_RESULT`; warp 0 collects and sums.

3. **`cooperative_map_with_params(src, dst, len, [u64;4], fn)`** — map with
   extra parameters (scalars, dimensions, strides) via `CoopMapExtArgs`.

## Impact
- `unified_io_compute` (North Star demo) rewritten: 3 global atomics eliminated
- Call-site boilerplate: ~15 lines of atomics → ~8 lines of pure compute
- Type-safe: function pointers prevent accidental closure captures

## Verified
All 5 cooperative tests pass on GTX 1660 (sm_75), 4 warps. CI lint clean.
