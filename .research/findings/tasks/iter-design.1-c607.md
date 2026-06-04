# iter-design.1 — Rust Iterator/Rayon → GPU Mapping

## Status: done
## Summary: Systematic analysis of which Rust Iterator and Rayon ParallelIterator operations map to GPU execution, and which are fundamentally incompatible. The project's existing `cooperative_map`, `cooperative_reduce`, and `BlockScope::spawn_all()` already implement the core GPU-side primitives needed for `map`, `reduce`, and `collect`. The recommended strategy is a library-level `GpuParallelIterator` trait that builds on these primitives, with a future MIR pass for chain fusion as an optimization (not MVP). Key constraint: closures must be `Copy + Send` with no heap allocation, no I/O, and no dynamic dispatch — the existing `spawn_all` closure mechanism proves this works.

## 1. Iterator Trait Operations on GPU

### GPU-Friendly (natural mapping)

| Operation | GPU Mapping | Existing Infrastructure |
|-----------|-------------|------------------------|
| `map(f)` | 1:1 element-wise transform. Each warp processes `elements[wid..].step_by(n_warps)`. | `cooperative_map`, `BlockScope::spawn_all` |
| `zip(a, b)` | Paired indexing: `(a[i], b[i])`. Trivial when both are indexed. | Straightforward pointer arithmetic |
| `enumerate()` | `(global_idx, element)` — index is `warp_id * stride + lane`. | `CoopMapArgs::warp_id` already provides this |
| `take(n)` / `skip(n)` | Subrange via offset arithmetic: `src + skip`, `len - skip`. Zero cost. | Pointer + length adjustment |
| `inspect(f)` | Side-effect per element, same as map but ignores return. Trivial. | Same as map |
| `cloned()` / `copied()` | No-op on GPU (everything is `Copy`, no `Clone` with heap allocation). | Already the case |
| `chain(a, b)` | Concatenation. Static when both lengths known at compile time: first warp batch processes `a`, second processes `b`. | Split cooperative dispatch |
| `step_by(n)` | Stride adjustment in the element loop. | Modify `i += n_warps * step` |

### GPU-Compatible with Effort (need specific GPU primitives)

| Operation | GPU Mapping | Complexity |
|-----------|-------------|------------|
| `filter(pred)` | Warp ballot (`__ballot_sync(mask, pred(x))`) gives a bitmask of which lanes pass. Then prefix sum (`__popc`) computes output positions. Compaction kernel writes survivors to contiguous output. | Medium — needs warp ballot + prefix sum + atomic output counter |
| `fold(init, f)` / `reduce(f)` | Warp-level: butterfly shuffle reduction (5 steps for 32 lanes). Block-level: shared memory reduction tree. Grid-level: `cooperative_reduce` already exists. | Low — `warp::reduce_sum_f32`, `cooperative_reduce` already implemented |
| `collect()` | Output buffer allocation. For `map` (1:1), output size = input size. For `filter`, need atomic counter for dynamic output size. | Low for map chains, medium for filter chains |
| `for_each(f)` | Same as map but no output buffer. Fire-and-forget per element. | Low — simplification of map |
| `sum()` / `product()` / `min()` / `max()` | Special cases of reduce. Warp shuffle reductions exist for all of these. | Low — `warp::reduce_sum_f32`, `reduce_max_f32` exist |
| `count()` | After filter: atomic counter. Without filter: just return `len`. | Low |
| `any(pred)` / `all(pred)` | Warp vote: `__ballot_sync(mask, pred(x))`. `any` = result != 0, `all` = result == full_mask. | Low — single PTX instruction |
| `position(pred)` / `find(pred)` | Warp ballot + `__ffs` (find first set). Returns min index where predicate is true. | Low-medium |
| `unzip()` | Scatter to two output buffers. Each element `(a, b)` writes `a` to buf_a and `b` to buf_b. | Low — two output pointers |

### GPU-Hostile (fundamentally sequential or impractical)

| Operation | Why It Fails | Recommendation |
|-----------|-------------|----------------|
| `flat_map(f)` | Each element can produce 0..N outputs — output size is data-dependent per element. Requires per-element prefix sum over all output counts, then scatter. Extremely expensive on GPU (global memory barrier between count pass and write pass). | Exclude from MVP. Could support with two-pass kernel (count → allocate → write) as a future extension. |
| `scan()` (prefix scan) | Inherently sequential: `acc[i] = f(acc[i-1], x[i])`. GPU prefix scan exists (Blelloch algorithm) but is complex and requires shared memory + multiple sync barriers. | Exclude from MVP. Valuable later — prefix scan is foundational for `filter` compaction. |
| `windows(n)` / `chunks(n)` | Overlapping reads. Each element needs access to neighbors. On GPU this means shared memory tiling with halo regions — a well-known pattern (stencil computation) but complex to auto-generate. | Exclude from MVP. Manual implementation via `spawn_all` is straightforward. |
| `take_while(pred)` / `skip_while(pred)` | Data-dependent boundary: must scan sequentially to find the cutoff point, then process the remainder in parallel. Two-phase: sequential scan + parallel map. | Exclude from MVP |
| `cycle()` / `repeat()` | Infinite iterators. GPU needs a finite element count for grid launch config. | Exclude. User must specify a length. |
| `peekable()` | Requires mutable shared state across iterations. Fundamentally single-threaded. | Exclude entirely. |

### Verdict

The MVP should support: **map, filter, fold/reduce, collect, zip, enumerate, take, skip, for_each, sum/min/max, any/all**. This covers the vast majority of data-parallel use cases. `flat_map` and `scan` are valuable but too complex for MVP.

## 2. Rayon ParallelIterator Analysis

### How Rayon Works

Rayon's `par_iter()` returns a `ParallelIterator` trait object. Key traits:

- **`ParallelIterator`**: Unindexed parallel iteration. Consumer-driven: the `drive_unindexed` method splits work recursively (divide-and-conquer) and dispatches to a thread pool via work-stealing. Supports `map`, `filter`, `for_each`, `reduce`, `collect`.

- **`IndexedParallelIterator`**: Adds random-access semantics. Knows its length, can split at arbitrary positions. Supports `zip`, `enumerate`, `chunks`, `position`. This is the important one for GPU.

- **Work-stealing**: Rayon splits the input range recursively until each task is "small enough", then dispatches to CPU threads. The sequential fallback threshold is tunable.

### What Maps to GPU, What Doesn't

| Rayon Concept | GPU Equivalent | Notes |
|---------------|---------------|-------|
| `par_iter()` on `&[T]` | `GpuSlice<T>::par_iter()` | GPU data must already be in device memory |
| Work-stealing scheduler | Warp-striped partitioning | GPU uses static partitioning (thread `i` processes elements `i, i+n_threads, i+2*n_threads, ...`). No work-stealing needed — GPU has thousands of threads, load imbalance averages out. |
| `IndexedParallelIterator` | Primary GPU iterator | GPU excels at indexed access. This is the core trait. |
| `ParallelIterator` (unindexed) | Not directly needed | GPU parallelism is always indexed (global thread ID → element index). Unindexed iteration adds complexity without benefit. |
| `into_par_iter()` | `GpuVec<T>::into_par_iter()` | Consumes the GPU buffer, avoids copy |
| `par_bridge()` | Not applicable | Bridge from sequential to parallel — irrelevant on GPU (everything is parallel) |
| `par_chunks()` | Block-level partitioning | Each block gets a chunk. Already the natural GPU execution model. |
| Rayon's `join(a, b)` | `BlockScope::spawn()` + join | Fork-join within a kernel — already implemented |

### Key Differences: Rayon vs GPU par_iter

1. **Data location**: Rayon operates on CPU memory. GPU par_iter operates on device memory. The iterator must wrap a `GpuSlice<T>` (device pointer + length), not a `&[T]`.

2. **No work-stealing**: GPU threads are statically assigned to elements. Work-stealing is unnecessary (and impossible — GPU threads cannot steal work from other SMs). Static round-robin partitioning (`element[wid..].step_by(n_warps)`) is the standard GPU pattern and is already used by `cooperative_map` and `spawn_all`.

3. **Indexed is king**: `IndexedParallelIterator` is the only trait that matters for GPU. The GPU execution model is fundamentally indexed — each thread knows its global ID and uses it to index into data. `ParallelIterator` (unindexed) adds no value.

4. **Consumer model differs**: Rayon uses a recursive split + consumer callback pattern. GPU uses a flat dispatch: launch kernel with `N` threads, each processes its elements. No recursion, no callbacks.

5. **Collect semantics**: Rayon's `collect()` uses a producer-consumer protocol that builds the output in parallel segments, then concatenates. GPU `collect()` for 1:1 operations (map) writes directly to output[global_id]. For compacting operations (filter), uses atomic output counter.

### What Changes: `rayon::par_iter()` → `gpu::par_iter()`

```rust
// Rayon (CPU):
let result: Vec<f32> = data.par_iter()
    .map(|x| x * 2.0 + 1.0)
    .filter(|x| *x > 5.0)
    .collect();

// GPU equivalent (proposed API):
let result: GpuVec<f32> = gpu_data.par_iter()
    .map(|x| x * 2.0 + 1.0)
    .filter(|x| *x > 5.0)
    .collect();  // result stays on GPU
let host_result: Vec<f32> = result.to_host();  // explicit transfer
```

Key API differences:
- Input is `GpuSlice<T>` or `GpuVec<T>`, not `&[T]`
- Output of `collect()` is `GpuVec<T>` (stays on GPU), not `Vec<T>`
- Explicit `.to_host()` / `.to_gpu()` for data transfer
- Closures restricted: `Copy + Send`, no heap, no I/O, no trait objects

## 3. Closure Capture on GPU

### Current State

The existing `cooperative_map` uses **function pointers** (no captures):
```rust
pub fn cooperative_map(src: *const u8, dst: *mut u8, len: usize, f: fn(&CoopMapArgs))
```

`BlockScope::spawn_all` uses **closures** (captures allowed):
```rust
pub fn spawn_all<F>(&mut self, f: F)
where F: Fn(u32, u32) + Send + Sync + 'scope
```

The `spawn_all` mechanism already proves closures work on GPU:
- The closure is monomorphized at compile time (no dynamic dispatch)
- The closure bytes are `copy_nonoverlapping`'d to each warp's scratch buffer
- Each warp reads its own copy from its scratch buffer
- The closure must fit in `SCRATCH_SIZE` (256 bytes)

### What Can Be Captured

**Allowed** (GPU-safe captures):
- Scalar values (`f32`, `u32`, `i64`, etc.)
- Raw pointers (`*const T`, `*mut T`) — wrapped in `SendPtr` for `Send`
- Fixed-size arrays (`[f32; 4]`)
- `Copy` structs with no heap fields
- References to scope-allocated shared/global memory (`&'scope [T]`)

**Prohibited** (GPU-unsafe captures):
- `Box<T>`, `Vec<T>`, `String` — heap allocation (no global allocator on GPU)
- `&dyn Trait` — dynamic dispatch requires vtables which aren't available in GPU global memory
- `Arc<T>`, `Rc<T>` — reference counting requires heap allocation
- `Mutex`, `RwLock` — std sync primitives (use GPU-specific `sync::Mutex`)
- File handles, network handles — no OS resources on GPU

### How Closures Work in the Iterator Context

For `par_iter().map(|x| x * 2.0 + scale)`, the closure captures `scale: f32`:

1. **Compile time**: rustc monomorphizes the closure into a concrete type `[closure@map::<closure>]` with layout `{ scale: f32 }` (4 bytes).

2. **Launch time**: The closure is serialized into kernel arguments (passed to the kernel as a parameter, or embedded in the scratch buffer).

3. **Execution**: Each warp reads the closure's captured data and applies the function body per element.

Since `spawn_all` already handles arbitrary `Fn(u32, u32) + Send + Sync` closures up to 256 bytes, the iterator's `map`/`filter` closures (which are typically tiny — a few scalars) fit comfortably.

### Restrictions for par_iter Closures

The iterator API should enforce:
```rust
trait GpuParallelIterator {
    fn map<F>(self, f: F) -> Map<Self, F>
    where F: Fn(Self::Item) -> T + Copy + Send + Sync;
    //       ^^^ Copy ensures no heap allocation
    //           Send + Sync for cross-warp safety
}
```

`Copy` is the key bound: it excludes `Box`, `Vec`, `String`, closures that capture `&mut` (which are `FnMut`, not `Fn`), and anything with `Drop`. This is more restrictive than Rayon (which requires `Send` only) but necessary for GPU safety.

## 4. Memory Model

### Data Location

| Stage | Memory | Access Pattern |
|-------|--------|----------------|
| Input data | GPU global memory (device allocation via `cuMemAlloc`) | Coalesced reads: threads 0..31 read consecutive addresses |
| Output data | GPU global memory (device allocation) | Coalesced writes: threads 0..31 write consecutive addresses |
| Closure captures | Kernel arguments (constant memory) or warp scratch buffers | Broadcast: all threads read the same captured values |
| Intermediate reduce accumulators | Registers (per-thread) → warp shuffle → shared memory (per-block) → global memory (final) | Hierarchical reduction |
| Filter output counter | GPU global memory, atomic | Single atomic counter for all blocks |

### Intermediate Buffers

**Fused chains need no intermediates**:
```rust
// This should compile to ONE kernel with NO intermediate buffer:
data.par_iter()
    .map(|x| x * 2.0)      // fused: compute in register
    .map(|x| x + 1.0)       // fused: compute in register
    .collect()               // write to output
```

Each element flows through the entire chain in a single thread, register-to-register. The chain is just function composition: `f(g(x))` compiles to `x * 2.0 + 1.0`.

**Filter breaks fusion**:
```rust
data.par_iter()
    .map(|x| x * 2.0)       // fused with filter (applied before predicate)
    .filter(|x| *x > 5.0)   // compaction: needs ballot + prefix sum + atomic counter
    .map(|x| x + 1.0)       // can fuse with the compaction write
    .collect()               // write to compacted output
```

`filter` is a synchronization point: it needs to count how many elements pass (to size the output), then write survivors to contiguous positions. The maps before and after filter can fuse with the filter's read and write respectively, so no extra buffers are needed even here — the entire chain is still one kernel.

**Reduce produces a single scalar**:
```rust
data.par_iter()
    .map(|x| x * x)         // fused: compute in register
    .sum::<f32>()            // warp reduce → block reduce → grid reduce
```

No intermediate buffer — each warp reduces its partition to one value in registers, then a block-level reduction (shared memory) combines warp results, and a grid-level atomic combines block results.

### Memory Management for `collect()`

Two cases:

1. **Known output size** (map, zip, enumerate — 1:1):
   - Allocate output buffer with same size as input
   - Each thread writes directly to `output[global_id]`
   - No coordination needed

2. **Unknown output size** (filter):
   - Pass 1 (optional optimization): count how many pass the predicate
   - OR: use atomic counter: each surviving element atomically increments a global counter and writes to `output[counter_value]`
   - The atomic approach is simpler and works well when the filter selectivity is moderate (not too many atomics)
   - For high selectivity (few survivors), warp-level compaction with `__ballot_sync` + `__popc` reduces atomic pressure: each warp writes a contiguous chunk.

### Host-Device Transfer

The iterator chain operates entirely on GPU memory. Transfer points are explicit:
```rust
// Upload to GPU
let gpu_data: GpuVec<f32> = data.to_gpu()?;

// Compute on GPU (no transfers)
let gpu_result: GpuVec<f32> = gpu_data.par_iter().map(|x| x * 2.0).collect();

// Download to CPU
let host_result: Vec<f32> = gpu_result.to_host()?;
```

This mirrors Rayon's model where data stays in the "execution domain" (CPU threads for Rayon, GPU for par_iter) and explicit transfers happen at boundaries.

## 5. Implementation Strategy

### Recommendation: Library-first, MIR fusion later

**Phase 1 (MVP): Pure library approach**

Build `GpuParallelIterator` trait on top of existing `cooperative_map` / `spawn_all` / `cooperative_reduce`:

```rust
// Library trait (kernel-side, in gpu-runtime)
pub trait GpuParallelIterator: Sized {
    type Item;

    fn map<F, T>(self, f: F) -> GpuMap<Self, F>
    where F: Fn(Self::Item) -> T + Copy + Send + Sync;

    fn filter<F>(self, pred: F) -> GpuFilter<Self, F>
    where F: Fn(&Self::Item) -> bool + Copy + Send + Sync;

    fn fold<T, ID, F>(self, identity: ID, fold_op: F) -> T
    where ID: Fn() -> T + Copy + Send + Sync,
          F: Fn(T, Self::Item) -> T + Copy + Send + Sync;

    fn collect(self) -> GpuVec<Self::Item>;
    fn for_each<F>(self, f: F) where F: Fn(Self::Item) + Copy + Send + Sync;
    fn sum(self) -> Self::Item where Self::Item: core::ops::Add<Output = Self::Item> + Default;
}
```

The chain is evaluated lazily: `map(f).map(g)` builds a `GpuMap<GpuMap<...>>` type that, when `collect()` is called, emits a single `spawn_all` call where each warp applies `g(f(x))` per element.

**Why library-first works**: The Rust compiler already monomorphizes the closure chain at compile time. `data.par_iter().map(|x| x * 2.0).map(|x| x + 1.0).collect()` becomes a single concrete type with a single `fn apply(&self, x: f32) -> f32` that the compiler inlines into `x * 2.0 + 1.0`. No MIR pass needed for basic fusion — the Rust optimizer already does it.

**Phase 2 (Optimization): MIR pass for cross-boundary fusion**

A MIR pass would only be needed for:
- Fusing across function call boundaries (where inlining doesn't happen)
- Fusing filter compaction with surrounding maps (requires ballot intrinsics the optimizer won't discover)
- Detecting `par_iter().map().collect()` patterns from OUTSIDE the kernel and auto-generating the kernel launch

This is the hybrid approach: the library defines the API, the MIR pass is an optimizer that kicks in for hot paths.

### How This Composes with Existing Infrastructure

The iterator chain ultimately calls one of:
- `cooperative_map(src, dst, len, compiled_chain_fn)` — for map chains
- `cooperative_reduce(src, len, compiled_reduce_fn)` — for fold/reduce
- `BlockScope::spawn_all(compiled_closure)` — for scope-bounded iterator chains

The `spawn_all` path is preferred because it supports closure captures (the iterator chain's captured values) and integrates with scope-based memory management.

### Host-Side Integration

The host-side `gpu::par_iter()` API would:
1. Accept a `&[T]` or `Vec<T>` from the user
2. Upload to GPU device memory (or use mapped/pinned memory)
3. Launch a generic "iterator kernel" with the chain descriptor
4. Download the result

```rust
// Host-side API (in async-gpu/gpu-host)
pub fn par_iter<T: GpuRepr>(data: &[T]) -> GpuParIter<T> {
    // Uploads data to GPU, returns a lazy iterator builder
}
```

## 6. MVP Definition

### Minimum Viable par_iter

**Scope**: Library-only, no MIR pass. Kernel-side trait + host-side launch wrapper.

**Operations**:
1. `map(|x| expr)` — element-wise transform
2. `collect()` — materialize into `GpuVec<T>`
3. `fold(init, |acc, x| expr)` / `sum()` — reduction to scalar
4. `for_each(|x| expr)` — side-effect per element
5. `enumerate()` — `(index, element)` pairs
6. `zip(other)` — pair two equal-length iterators

**Not in MVP** (Phase 2):
- `filter()` — needs warp ballot compaction
- `flat_map()` — needs two-pass kernel
- `scan()` — needs Blelloch prefix sum
- MIR pass — optimizer, not required for correctness

**Closure restrictions**:
- `Copy + Send + Sync`
- No heap allocation (no `Box`, `Vec`, `String`)
- No I/O (no `println!`, no `File`)
- Captures must fit in 256 bytes (warp scratch buffer)

**Type constraints**:
- `T: Copy + Send + Sync` for element types
- No `Drop` types (GPU has no destructors)
- `f32`, `f64`, `u32`, `i32`, `u64`, `i64`, fixed-size structs

**Litmus test**: `data.par_iter().map(|x| x * 2.0 + 1.0).collect()` produces correct f32 results on GPU, using `spawn_all` under the hood, with the user never writing `cooperative_map`, `thread::gpu_main`, or `extern "gpu-kernel"`.

**Implementation plan**:
1. `GpuParallelIterator` trait in `gpu-runtime` (kernel-side)
2. `GpuMap`, `GpuFold`, `GpuEnumerate`, `GpuZip` adapter types
3. `collect()` implementation via `spawn_all` dispatch
4. Host-side `gpu::par_iter()` in `gpu-host` that launches the kernel
5. Integration test: 1M f32 elements, verify correctness vs CPU

**Estimated risk**: LOW. All GPU primitives exist (`spawn_all`, `cooperative_map`, `cooperative_reduce`). The iterator is a type-level composition layer on top of proven infrastructure. The main engineering work is the host-side launch wrapper that auto-generates the kernel launch config from the chain descriptor.

## Files Changed: none
