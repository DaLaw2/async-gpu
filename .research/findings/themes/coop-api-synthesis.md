# Theme Synthesis: coop-api — Cooperative API Ergonomics

## Key Finding

Closure captures fail across warps because captured references point to per-warp
local memory (PTX `.local`). Heap data (Vec, etc.) is already in global memory —
the bottleneck is getting the *pointer* to other warps without closure captures.

## Solution: `cooperative_map(src, dst, len, fn)`

New API passes data via an explicit global argument block instead of closures.
Takes `fn(&CoopMapArgs)` (function pointer, not closure) so the "no captures"
constraint is enforced at compile time. Each warp receives `CoopMapArgs` with
`{src, dst, len, warp_id, n_warps}` — all partitioning info in one struct.

## Impact

- Eliminates 3 global atomic statics + 6 atomic ops + `unsafe` per cooperative call
- Call-site goes from ~15 lines of boilerplate to ~10 lines of pure compute logic
- Type-safe: accidentally capturing a local is a compile error, not GPU crash

## Next Steps

- coop-api.2: Apply cooperative_map to `unified_io_compute` (the North Star demo)
- Consider generic `cooperative_map_typed<T, U>()` that casts pointers internally
- Explore whether the trampoline overhead is measurable vs `cooperative()`
