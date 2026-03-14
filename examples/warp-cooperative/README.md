# warp-cooperative — MIR Pass Verification

Tests the `#[warp_cooperative]` custom rustc MIR pass directly, verifying that async state machines work correctly with NVIDIA's SIMT execution model.

**Requires patched rustc** — this example will not build on stock nightly.

## What It Demonstrates

- `#[warp_cooperative]` attribute on `async fn`
- `bar.warp.sync` barriers inserted at every `.await` yield point
- `shfl.sync.idx` discriminant broadcast from lane 0 to all lanes
- Simulated I/O pipeline with 6 `.await` points (no actual hostcall)

## Kernels

### test_simple_warp

`simple_add(x) -> x + 1` — no `.await`, single-state coroutine. Only `bar.warp.sync` is inserted (no shfl needed since there is only one state).

Expected: `output[tid] = tid + 1`

### test_multi_await

`multi_await(x)` — two `.await` points creating a 3-state coroutine. The MIR pass inserts `shfl.sync.idx` to broadcast the discriminant from lane 0 so all lanes resume at the same state.

Expected: `output[tid] = 2 * tid + 12`

### test_async_pipeline

`async_pipeline(tid)` — 6 `.await` points simulating open/write/close/open/read/close. Each `.await` yields the warp. All lanes should converge at each yield point.

Expected: `output[tid] = 29029` for all lanes

## Running

```bash
# Build kernel (requires patched rustc)
cd kernel && cargo +patched build --release

# Run host tests
cd host && cargo run --release
```

## Expected Output

```
=== Warp-Cooperative Async Kernel Test ===

--- Test 1: test_simple_warp ---
  test_simple_warp: PASSED

--- Test 2: test_multi_await ---
  test_multi_await: PASSED

--- Test 3: test_async_pipeline ---
  test_async_pipeline: PASSED

=== All tests complete ===
```

## Key PTX to Inspect

- `bar.warp.sync` — warp barrier at every yield/return point
- `shfl.sync.idx` — discriminant broadcast in multi-state coroutines (test_multi_await, test_async_pipeline)
- State machine structure: `switchInt` on discriminant with branches for each `.await` point
