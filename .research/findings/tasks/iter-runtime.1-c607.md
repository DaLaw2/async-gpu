# iter-runtime.1 — Warp-parallel input partitioning + output buffer management

## Status: DONE

## 1. Warp-striped partitioning analysis

### Correctness: VERIFIED

The current warp-striped access pattern in `par_iter.rs` is correct and optimal:

```
warp 0: indices 0, N, 2N, 3N, ...
warp 1: indices 1, N+1, 2N+1, ...
warp 2: indices 2, N+2, 2N+2, ...
```

Where N = `n_warps`. This is the standard round-robin / interleaved
partitioning used in GPU programming.

### Memory coalescing assessment

**Reads (`get_unchecked`)**: Each warp reads `ptr.add(i)` where `i` strides
by `n_warps`. Within a single warp, only lane 0 executes the closure body
(per the `spawn_all` trampoline which guards with `if lid == 0`). This
means each warp issues a single scalar load per iteration, not a 32-wide
coalesced load. This is expected and correct for the async-gpu execution
model where each warp is a single logical thread.

**Writes (`collect_into`)**: Same pattern — `write_volatile(ptr.add(i), elem)`
issues one scalar store per warp per iteration. Correct for the 1-warp =
1-thread model.

**Why warp-striped is correct here**: In this project's model, warps are
logical threads (not CUDA threads). The "coalescing" question is about
inter-warp access patterns, not intra-warp. Warp-striped gives good
spatial locality across warps: if 4 warps process a 128-element array,
warp 0 touches [0,4,8,...], warp 1 touches [1,5,9,...] — neighboring
warps touch neighboring addresses. This is the optimal pattern for the
execution model.

### Alternative: block partitioning

Block partitioning (warp 0 gets [0..32], warp 1 gets [32..64], ...) would
also work but has two drawbacks:
1. Load imbalance when `len % n_warps != 0` — one warp gets extra work
2. Less uniform memory access timing — warps finish at different times

The current warp-striped approach distributes work evenly (max 1 element
difference between warps) and is the standard choice.

### Fold correctness

The two-level fold hierarchy is correct:
1. Each warp reduces its partition sequentially in registers (no atomics)
2. Each warp writes its partial result to `WARP_RESULT[wid]` via atomic store
3. After `block_scope` joins all warps, warp 0 reads all partials and
   combines them sequentially

The `transmute` via `copy_nonoverlapping` to u64 bits is safe because the
assert enforces `size_of::<Item>() <= 8`. The Acquire/Release ordering on
`WARP_RESULT` stores/loads ensures cross-warp visibility.

## 2. Par_iter demo kernel

Created `crates/kernel/gpu-kernel-std/src/par_iter_demo.rs` with four
kernel entry points:

| Kernel | Chain | Formula |
|--------|-------|---------|
| `par_iter_map_collect` | `map + collect_into` | `output[i] = input[i] * 2.0 + 1.0` |
| `par_iter_map_sum` | `map + sum` | `sum(input[i]^2)` |
| `par_iter_enumerate_collect` | `enumerate + map + collect_into` | `output[i] = input[i] + i as f32` |
| `par_iter_zip_collect` | `zip + map + collect_into` | `output[i] = a[i] + b[i]` |

All kernels use:
- `thread::gpu_main` for warp pool setup
- `init_shared_mem_allocator(512)` for block_scope
- Warp-parallel execution via `block_scope` + `spawn_all` (inside par_iter terminals)
- Standard launch config: 1 block x 128 threads (4 warps)

The module is registered in `lib.rs` alongside other kernel modules.

## 3. Compilation verification

### nvptx64 build: PASS

```
cd crates/kernel/gpu-kernel-std && cargo +nightly-2026-06-03 build --release
```

Compiles successfully. The par_iter chains are monomorphized and fused at
compile time — zero intermediate buffers, zero heap allocation.

### CI lint: PASS

```
bash scripts/ci-lint.sh
```

All checks pass: fmt, clippy, doc, host checks, PTX kernel builds.

## Files modified

- `crates/kernel/gpu-kernel-std/src/par_iter_demo.rs` — NEW: four demo kernels
- `crates/kernel/gpu-kernel-std/src/lib.rs` — added `mod par_iter_demo`
- `crates/async-gpu/src/lib.rs` — fixed pre-existing fmt issue (import ordering)
