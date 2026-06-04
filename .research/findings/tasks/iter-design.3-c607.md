# iter-design.3 — MIR Pass Strategy: Iterator Chain → Single Fused GPU Kernel

## Status: done
## Summary

This design documents the MIR pass strategy for compiling iterator chains into fused GPU kernels, while establishing that **the MVP needs no MIR pass at all**. Library-level composition via Rust's monomorphization already fuses chained maps into a single inlined closure. The MIR pass is a Phase 3 optimization for cross-boundary fusion and host-side `gpu::par_iter()` auto-kernel-generation. The document covers: (1) how library-level fusion works today, (2) what a MIR pass would add, (3) the host-side `par_iter` story, and (4) recommended phasing.

---

## 1. Library-Level Fusion Without a MIR Pass

### 1.1 How Monomorphization Fuses Iterator Chains

When a user writes:

```rust
data.par_iter()
    .map(|x| x * 2.0)
    .map(|x| x + 1.0)
    .collect_into(output);
```

Rust monomorphizes the chain into a concrete type:

```
GpuMap<GpuMap<GpuParIter<f32>, [closure@*2.0]>, [closure@+1.0]>
```

The terminal `collect_into` calls `default_collect_into`, which calls `get_unchecked(i)` on the outer `GpuMap`. This method is:

```rust
unsafe fn get_unchecked(&self, i: usize) -> f32 {
    (self.f)(self.inner.get_unchecked(i))
    // = (+1.0)( (*2.0)( read_volatile(ptr.add(i)) ) )
}
```

After inlining (which Rust/LLVM always does for small closures), this becomes:

```rust
let x = core::ptr::read_volatile(ptr.add(i));
let result = (x * 2.0) + 1.0;
```

**Zero intermediate buffers. Zero MIR pass. The Rust compiler does the fusion automatically through its generic instantiation + inlining pipeline.**

### 1.2 What Happens at the LLVM Level

The monomorphized `get_unchecked` chain compiles to a sequence of LLVM IR instructions that operate entirely in registers:

```llvm
; GpuMap<GpuMap<GpuParIter<f32>, closure1>, closure2>::get_unchecked(i)
%ptr_i = getelementptr float, ptr %base, i64 %i
%x = load volatile float, ptr %ptr_i
%mul = fmul float %x, 2.0          ; closure1: x * 2.0
%add = fadd float %mul, 1.0        ; closure2: x + 1.0
; result in %add — pure register, no memory traffic
```

LLVM's optimizer may further transform this (e.g., `x * 2.0 + 1.0` → `fma(x, 2.0, 1.0)` on architectures with FMA). On nvptx64, this emits a single `fma.rn.f32` instruction.

### 1.3 The Execution Path (Kernel-Side)

The full execution path for `collect_into` with a fused chain:

```
collect_into(output)
  └─ default_collect_into(self, output)
       └─ block_scope(|scope| {
              scope.spawn_all(|wid, n_warps| {
                  // The closure captures:
                  //   - iter: GpuMap<GpuMap<GpuParIter<f32>, C1>, C2>  (~24 bytes)
                  //   - output: SendPtrMut<f32>                        (~8 bytes)
                  //   - len: usize                                      (~8 bytes)
                  // Total: ~40 bytes, well within SCRATCH_SIZE (256 bytes)

                  let mut i = wid as usize;
                  while i < len {
                      // This single call inlines the full chain:
                      //   read_volatile → closure1 → closure2
                      let elem = unsafe { iter.get_unchecked(i) };
                      unsafe { write_volatile(out_ptr.add(i), elem); }
                      i += n_warps as usize;
                  }
              });
          });
```

Each warp processes its stripe of elements. Within each element, the entire iterator chain executes in registers. The `spawn_all` mechanism copies the closure (including the full chain descriptor) to each warp's scratch buffer.

### 1.4 Verification: Fusion Depth Limits

**How deep can the chain go?** There is no depth limit from Rust's type system — `GpuMap<GpuMap<GpuMap<...>>>` nests arbitrarily. The practical limits are:

1. **Closure capture size**: Each `GpuMap` adds `size_of::<F>()` to the chain. For zero-capture closures (e.g., `|x| x * 2.0`), `F` is a ZST (zero-sized type), so the entire `GpuMap` wrapper is the same size as the inner iterator. Even with captures, each scalar adds only 4-8 bytes. The 256-byte scratch buffer can accommodate chains with dozens of captured values.

2. **LLVM inlining depth**: LLVM defaults to aggressive inlining for small functions. A chain of 10 `map` calls with trivial closures will always inline fully. For chains of 50+ maps with complex bodies, LLVM may decline to inline the outermost layers, resulting in function calls instead of fused register ops. This is unlikely in practice — real iterator chains are 2-5 operations.

3. **PTX register pressure**: Each fused operation adds register usage. On SM 7.5, each thread block has 65536 registers (1024 per thread at 64 threads/block). Even a 10-deep chain uses < 50 registers, well within limits.

**Verdict**: Library-level fusion handles all practical cases. No depth-related limitations will be hit in real code.

### 1.5 What Library-Level Fusion Handles (Complete List)

| Pattern | Fuses? | Mechanism |
|---------|--------|-----------|
| `.map(f).map(g)` | YES | `get_unchecked` inlining: `g(f(x))` |
| `.map(f).map(g).map(h)` | YES | Same: `h(g(f(x)))` |
| `.enumerate().map(f)` | YES | `f((i, x))` inlined |
| `.zip(other).map(f)` | YES | `f((a[i], b[i]))` inlined |
| `.map(f).sum()` | YES | `fold_op(acc, f(x))` inlined |
| `.map(f).fold(init, op)` | YES | Same |
| `.map(f).for_each(g)` | YES | `g(f(x))` inlined |
| `.map(f).collect_into(out)` | YES | `out[i] = f(x)` inlined |
| Cross-function `map` body | SOMETIMES | Depends on LLVM inlining heuristics |
| `map` calling `extern "C"` fn | NO | Opaque call, no inlining |

---

## 2. What a MIR Pass Would Add Beyond Library-Level

### 2.1 Three Capabilities a MIR Pass Would Unlock

**Capability A: Host-side auto-kernel-generation from `gpu::par_iter(data)`**

The user writes on the HOST:

```rust
let result = gpu::par_iter(&host_data)
    .map(|x| x * 2.0 + 1.0)
    .collect();
```

This cannot work with library-level composition alone. The closures `|x| x * 2.0 + 1.0` are HOST code — Rust compiles them for the host target (x86_64). To execute on GPU, something must:

1. Recognize the `gpu::par_iter().map().collect()` pattern
2. Extract the closure bodies
3. Emit a GPU kernel that applies the closures element-wise
4. Replace the host-side chain with: upload data → launch kernel → download results

This is fundamentally a compiler transformation. The library can provide the API types, but the code generation requires either a MIR pass, a procedural macro, or runtime JIT (NVRTC).

**Capability B: Cross-function-boundary fusion**

```rust
fn double(x: f32) -> f32 { x * 2.0 }
fn add_one(x: f32) -> f32 { x + 1.0 }

// Inside a kernel:
data.par_iter()
    .map(double)
    .map(add_one)
    .collect_into(output);
```

With library-level composition, this fuses IF LLVM inlines `double` and `add_one`. For small functions, this always happens. For large or recursive functions, LLVM may choose not to inline, resulting in function calls per element (expensive on GPU due to call overhead).

A MIR pass could force-inline the map bodies regardless of LLVM's heuristics, guaranteeing fusion.

**Capability C: Semantic fusion of GPU-specific patterns**

```rust
data.par_iter()
    .filter(|x| *x > 0.0)
    .map(|x| x * 2.0)
    .collect_into(output);
```

The `filter` operation requires warp ballot intrinsics (`__ballot_sync`, `__popc`) for efficient compaction. LLVM will never discover these — it does not know about GPU warp-level operations. A MIR pass could:

1. Recognize `filter` followed by `map`
2. Emit fused ballot + compaction + map code
3. Apply the map transform during the compaction write (avoiding an intermediate buffer)

### 2.2 What a MIR Pass Would NOT Add

The MIR pass does NOT help with:

- **Map chain fusion**: Already handled by monomorphization + inlining. A MIR pass would be redundant.
- **Fold/reduce optimization**: Warp shuffle reduction is implemented in the library's `default_fold`. No compiler help needed.
- **Memory coalescing**: This is a launch config concern (thread indexing), not a code transformation concern.
- **Vectorization (float4)**: LLVM's SLP vectorizer handles this, or we emit explicit vectorized code in the library.

### 2.3 Architecture of a Hypothetical MIR Pass

If built, the MIR pass (`IteratorFusionTransform`) would:

**Location**: `compiler/rustc_mir_transform/src/iterator_fusion.rs`, registered in `compiler/rustc_mir_transform/src/lib.rs` alongside `WarpCooperativeTransform`.

**Activation**: Only on `nvptx64` target, only for functions containing `GpuParallelIterator` trait method calls.

**Detection phase**:

```
1. Walk the MIR for the current function body
2. Find call sites to methods on types implementing GpuParallelIterator:
   - GpuParIter::map, GpuParIter::filter, GpuParIter::fold, etc.
3. Build a chain graph: source → adapter₁ → adapter₂ → ... → terminal
4. Verify the chain is fusable (no side-effecting adapters, all closures are Copy)
```

**Transformation phase**:

```
1. For each fusable chain:
   a. Extract the closure MIR bodies for each adapter
   b. Compose them into a single fused closure body:
      - map(f).map(g) → single closure: |x| g(f(x))
      - map(f).filter(p) → fused ballot+apply closure
   c. Replace the chain with a direct spawn_all call containing the fused body
   d. Inline the composed closure into the spawn_all body
```

**MIR-level closure composition example**:

Before (conceptual MIR):
```
_chain = GpuParIter::new(ptr, len)
_chain2 = GpuMap::new(_chain, closure_f)  // f = |x| x * 2.0
_chain3 = GpuMap::new(_chain2, closure_g) // g = |x| x + 1.0
GpuMap::collect_into(_chain3, output)
```

After MIR pass:
```
// All adapter construction eliminated
// Replaced with direct spawn_all:
block_scope(|scope| {
    scope.spawn_all(|wid, n_warps| {
        let mut i = wid as usize;
        while i < len {
            let x = read_volatile(ptr.add(i));
            let r = x * 2.0 + 1.0;  // f and g composed inline
            write_volatile(output.add(i), r);
            i += n_warps as usize;
        }
    });
});
```

**Important**: This is exactly what the library already does via monomorphization and LLVM inlining. The MIR pass would only help when LLVM's inliner fails (rare) or for non-library patterns (filter with ballot intrinsics, host-side kernel generation).

### 2.4 Relationship to the Existing WarpCooperativeTransform

| Aspect | WarpCooperativeTransform | IteratorFusionTransform (hypothetical) |
|--------|--------------------------|---------------------------------------|
| Trigger | Coroutine bodies (async fn) | GpuParallelIterator trait method calls |
| Phase | After StateTransform | After monomorphization, before codegen |
| Scope | Narrow: insert barriers + shuffle | Broad: rewrite entire iterator chains |
| Complexity | ~400 LOC, pattern-match + insert ASM | ~2000+ LOC, closure extraction + composition |
| Risk | Low (well-understood coroutine structure) | High (must understand arbitrary closure MIR) |
| Current need | Required (async/await correctness) | Not needed (library handles fusion) |

**Key insight**: `WarpCooperativeTransform` is a correctness pass — without it, async/await breaks on GPU. `IteratorFusionTransform` would be a performance optimization — the library works correctly without it.

---

## 3. Host-Side `gpu::par_iter()` Story

### 3.1 The Problem

The kernel-side `GpuParallelIterator` (from iter-design.2) runs INSIDE a kernel. The user must still write:

```rust
// Kernel (runs on GPU):
#[no_mangle]
pub unsafe extern "gpu-kernel" fn my_kernel(input: *const f32, output: *mut f32, len: usize) {
    thread::gpu_main(|| {
        let data = GpuSlice::from_raw_parts(input, len);
        let out = GpuSliceMut::from_raw_parts(output, len);
        data.par_iter().map(|x| x * 2.0).collect_into(out);
    });
}

// Host (runs on CPU):
fn main() {
    let data = vec![1.0f32; 1_000_000];
    // Must manually: upload, launch kernel, download
    let result = gpu::run_with_output::<f32>("my_kernel", 1_000_000)?;
}
```

The North Star is eliminating the kernel boilerplate:

```rust
// Host-only (user's ideal API):
fn main() {
    let data = vec![1.0f32; 1_000_000];
    let result: Vec<f32> = gpu::par_iter(&data)
        .map(|x| x * 2.0)
        .collect();
}
```

### 3.2 Three Approaches to Host-Side `par_iter`

**Approach A: MIR Pass Kernel Extraction**

A MIR pass would detect `gpu::par_iter(&data).map(closure).collect()` on the host side, extract the closure body, emit a GPU kernel containing the iterator chain, and replace the host-side code with upload → launch → download.

- **Pros**: Seamless syntax, no user-visible boilerplate.
- **Cons**: Extremely complex MIR transformation. Must cross the host/device compilation boundary. The closure MIR is in host-target IR; the kernel must emit nvptx64-target IR. This is essentially a split-compilation model.
- **Feasibility**: HIGH complexity. Would require changes to the build pipeline, not just a MIR pass. The kernel and host are compiled in separate cargo invocations (host workspace vs GPU workspace). A single MIR pass cannot span both.

**Approach B: Procedural Macro + Pre-compiled Generic Kernel**

A procedural macro `#[gpu_par_iter]` transforms the closure at the syntax level:

```rust
#[gpu_par_iter]
fn double_add(data: &[f32]) -> Vec<f32> {
    data.par_iter().map(|x| x * 2.0 + 1.0).collect()
}
```

The macro:
1. Extracts the closure body `|x| x * 2.0 + 1.0`
2. Generates a GPU kernel source string containing that closure
3. At runtime, JIT-compiles the kernel via NVRTC
4. Replaces the function body with upload → launch → download

- **Pros**: Works within Rust's existing compilation model. No rustc patches needed.
- **Cons**: Limited to closures expressible as CUDA C (no Rust trait calls, no complex pattern matching). Requires NVRTC at runtime. Macro hygiene concerns.
- **Feasibility**: MEDIUM complexity. Similar to the tape-level NVRTC codegen in fusion-analysis.2.

**Approach C: Generic Kernel Trampoline (Library-Only)**

Pre-compile a generic kernel that accepts a function pointer and applies it element-wise. The host-side `gpu::par_iter()` uploads the data and the closure bytes, launches the generic kernel, and downloads results.

```rust
// Pre-compiled GPU kernel (in gpu-runtime):
#[no_mangle]
pub unsafe extern "gpu-kernel" fn par_iter_map_f32(
    input: *const f32,
    output: *mut f32,
    len: usize,
    // Closure bytes are in a separate GPU buffer
    closure_buf: *const u8,
    closure_fn: fn(*const u8, f32) -> f32,  // trampoline
) {
    thread::gpu_main(|| {
        let data = GpuSlice::from_raw_parts(input, len);
        let out = GpuSliceMut::from_raw_parts(output, len);
        // Apply closure via trampoline
        block_scope(|scope| {
            scope.spawn_all(|wid, n_warps| {
                let mut i = wid as usize;
                while i < len {
                    let x = core::ptr::read_volatile(input.add(i));
                    let r = closure_fn(closure_buf, x);
                    core::ptr::write_volatile(output.add(i), r);
                    i += n_warps as usize;
                }
            });
        });
    });
}
```

- **Pros**: Pure library, no compiler changes, no NVRTC.
- **Cons**: Function pointer call per element (no inlining, ~2-5x slower than fused). Closure must be serializable to a byte buffer and deserializable on GPU. Complex calling convention.
- **Feasibility**: LOW complexity, but POOR performance. Function pointer overhead negates GPU parallelism benefits for simple ops.

**Approach D: `gpu::run` Wrapper + User-Written Kernel (Recommended for Phase 2)**

Instead of hiding the kernel entirely, provide a thin wrapper that reduces boilerplate to one line while keeping the kernel explicit:

```rust
// Kernel (user writes, compiled to GPU):
#[no_mangle]
pub unsafe extern "gpu-kernel" fn scale_add(
    input: *const f32, output: *mut f32, len: usize,
) {
    thread::gpu_main(|| {
        let data = GpuSlice::from_raw_parts(input, len);
        let out = GpuSliceMut::from_raw_parts(output, len);
        data.par_iter().map(|x| x * 2.0 + 1.0).collect_into(out);
    });
}

// Host (user writes):
fn main() {
    let data = vec![1.0f32; 1_000_000];
    let result = gpu::run_data("scale_add", &data)?;
    // gpu::run_data handles: upload, launch with correct grid config, download
}
```

Where `gpu::run_data` is a new host-side API that:
1. Uploads `&[T]` to GPU global memory
2. Allocates output buffer (same size)
3. Launches the named kernel with `(input_ptr, output_ptr, len)` signature
4. Downloads the output

- **Pros**: Minimal new infrastructure. Kernel compilation is the existing pipeline. Full inlining and fusion. User understands what runs where.
- **Cons**: User still writes a kernel function. Not as magical as `gpu::par_iter(&data)`.
- **Feasibility**: LOW complexity, HIGH performance. Ships immediately.

### 3.3 Recommended Host-Side Phasing

| Phase | API | Kernel Authoring | Code Generation | Performance |
|-------|-----|-----------------|-----------------|-------------|
| Phase 1 (now) | Kernel-side `GpuParallelIterator` | User writes `extern "gpu-kernel"` | None (library) | Optimal (monomorphized) |
| Phase 2 | `gpu::run_data("kernel", &data)` | User writes kernel, host wrapper reduces boilerplate | None | Optimal |
| Phase 3 | `#[gpu_par_iter] fn f(data) { ... }` | Macro generates kernel source, NVRTC compiles | Proc macro + NVRTC | Near-optimal (NVRTC code quality) |
| Phase 4 (future) | `gpu::par_iter(&data).map(f).collect()` | Fully automatic | MIR pass or proc macro | Optimal (compiler-generated) |

### 3.4 `gpu::run_data` API Design (Phase 2)

```rust
// In crates/core/gpu-host/src/gpu.rs

/// Launch a kernel that processes input data and produces output data.
///
/// Kernel signature must be:
///   `extern "gpu-kernel" fn(input: *const T, output: *mut T, len: usize)`
///
/// Handles: host-to-device upload, kernel launch with appropriate grid config,
/// device-to-host download.
pub fn run_data<T>(
    kernel_name: &'static str,
    input: &[T],
) -> Result<Vec<T>>
where
    T: cudarc::driver::DeviceRepr + cudarc::driver::ValidAsZeroBits + Clone,
{
    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    let func = get_kernel(&dev, kernel_name)?;
    let session = HostcallSession::start(64)?;

    // Upload input
    let gpu_input = dev.htod_sync_copy(input)
        .map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("htod: {e}"),
        })?;

    // Allocate output (same size as input for 1:1 maps)
    let mut gpu_output = dev.alloc_zeros::<T>(input.len())
        .map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("alloc: {e}"),
        })?;

    let len = input.len();

    // Launch with 4 warps (128 threads) for spawn_all support
    let config = LaunchConfig {
        grid_dim: (1, 1, 1),
        block_dim: (128, 1, 1),
        shared_mem_bytes: 0,
    };

    unsafe {
        func.launch(config, (
            session.dev_ptr(),
            &gpu_input,
            &mut gpu_output,
            len,
        )).map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("launch: {e}"),
        })?;
    }

    dev.synchronize().map_err(|e| GpuHostError::Verification {
        test: kernel_name,
        detail: format!("sync: {e}"),
    })?;

    let result = dev.dtoh_sync_copy(&gpu_output)
        .map_err(|e| GpuHostError::Verification {
            test: kernel_name,
            detail: format!("dtoh: {e}"),
        })?;

    session.shutdown();
    Ok(result)
}
```

---

## 4. Relationship to Auto-Fusion (fusion-analysis)

### 4.1 Two Different Fusion Problems

| Concern | GPU Iterator Fusion | Auto-Fusion (fusion-analysis) |
|---------|---------------------|------------------------------|
| Level | Kernel-side, within a single kernel | Host-side, across kernel launches |
| What fuses | Iterator adapter chain (map → map → map) | Tensor op sequence (matmul → bias_add → gelu) |
| Mechanism | Rust monomorphization + LLVM inlining | Tape-level pattern matching + NVRTC codegen |
| Granularity | Per-element (scalar operations) | Per-tensor (entire buffer operations) |
| When it happens | Compile time (always) | Runtime (trace-then-replay) |
| MIR pass needed? | No (library handles it) | No (tape-level handles it) |

### 4.2 Where They Overlap

Both solve the same fundamental problem: eliminating intermediate memory traffic between composed operations. The difference is scale:

- **Iterator fusion**: Element-level. `map(f).map(g)` eliminates register → memory → register round-trip per element. This is already handled by Rust's optimizer.

- **Tensor fusion**: Kernel-level. `gelu(bias_add(matmul(x)))` eliminates GPU-global-memory → read-back → re-launch → GPU-global-memory round-trip per operation. This requires runtime infrastructure (tape, NVRTC, cache).

### 4.3 Convergence Point

In Phase 4 (future), both systems converge: host-side `gpu::par_iter()` with auto-kernel-generation IS tensor-level fusion expressed as iterator chains. The user writes:

```rust
let result = gpu::par_iter(&data)
    .map(|x| x * scale + bias)  // bias_add
    .map(gelu)                   // activation
    .collect();
```

This is semantically identical to:

```rust
let result = ops::gelu(ops::bias_add(data, scale, bias));
```

The fusion optimizer (whether MIR pass or NVRTC codegen) should produce the same fused kernel for both. The iterator syntax is just a more ergonomic way to express the same computation graph.

### 4.4 Shared MIR Infrastructure

If both gpu-iterator and auto-fusion eventually need MIR passes, they should share infrastructure:

- **Pattern detection**: Both need to identify chains of composable operations in MIR
- **Closure extraction**: Both need to extract function bodies and compose them
- **nvptx64 target awareness**: Both activate only on GPU targets

The `WarpCooperativeTransform` provides the template: target-gated, narrowly scoped, registered in `rustc_mir_transform/src/lib.rs`. A shared `GpuFusionAnalysis` module could serve both.

However, this is premature to build. The library-level and tape-level approaches handle all current needs without touching the compiler.

---

## 5. Recommended Phasing

### Phase 1 (Now): Kernel-Side `GpuParallelIterator` — Library Only

**What**: Implement the `GpuParallelIterator` trait, adapter types, and terminal operations as designed in iter-design.2. All code lives in `crates/core/gpu-runtime/src/par_iter.rs`.

**Fusion**: Handled automatically by Rust's monomorphization. `.map(f).map(g)` compiles to `g(f(x))` with zero intermediate buffers. No MIR pass, no NVRTC, no runtime codegen.

**Host integration**: Users write `extern "gpu-kernel"` functions that use `par_iter` internally. Host-side launch uses existing `gpu::run()` / `gpu::run_with_output()`.

**Risk**: LOW. All GPU primitives exist (`spawn_all`, `WARP_RESULT`, `block_scope`). The iterator is a type-level composition layer.

**Deliverable**: `data.par_iter().map(|x| x * 2.0 + 1.0).collect_into(out)` runs correctly inside a kernel, producing correct f32 results verified against CPU.

### Phase 2: Host-Side `gpu::run_data()` — Reduce Boilerplate

**What**: Add `gpu::run_data(kernel_name, &input_data) -> Vec<T>` to `gpu.rs`. This handles upload, launch, download in one call.

**Fusion**: Same as Phase 1 (kernel-side, compile-time).

**User experience**: Two files (kernel + host), but host side is one line.

**Risk**: LOW. Thin wrapper over existing `gpu::run_with_output`.

### Phase 3 (Future): MIR Pass or NVRTC for Host-Side `par_iter`

**What**: Enable `gpu::par_iter(&host_data).map(f).collect()` from the host side. Either:
- (A) MIR pass extracts closures and generates GPU kernel, or
- (B) Procedural macro generates CUDA C source, NVRTC compiles at runtime

**Fusion**: Automatic kernel generation from host-side iterator chains.

**Risk**: HIGH. Cross-compilation-boundary code generation. Novel compiler infrastructure.

**Prerequisite**: Phase 1 and Phase 2 must be complete and battle-tested. The kernel-side trait API must be stable. The host-side launch infrastructure must be reliable.

**Decision point**: Choose MIR pass vs NVRTC based on:
- If the auto-fusion epic (fusion-analysis) has already built NVRTC codegen infrastructure → use NVRTC (Approach B). The infrastructure exists, the marginal cost is low.
- If NVRTC codegen is not available → consider MIR pass (Approach A). But this is months of work.

### Phase 4 (Future): Cross-Boundary Optimization

**What**: MIR pass that fuses iterator chains across function call boundaries, even when LLVM's inliner declines. Also: fused filter+map with warp ballot intrinsics.

**Fusion**: Compiler-guaranteed fusion regardless of optimization level.

**Risk**: MEDIUM. The WarpCooperativeTransform demonstrates that MIR passes for nvptx64 are feasible. But closure composition in MIR is significantly more complex than barrier insertion.

**Prerequisite**: Phase 3 MIR pass infrastructure (if built).

---

## 6. MIR Pass Architecture (Reference, Not To Build Now)

This section documents what the MIR pass WOULD look like, for reference when Phase 3/4 is eventually approached.

### 6.1 Pass Registration

```rust
// In compiler/rustc_mir_transform/src/lib.rs, add:
mod iterator_fusion;

// In the pass pipeline, after monomorphization and inlining:
&iterator_fusion::IteratorFusionTransform,
```

### 6.2 Pass Structure

```rust
// compiler/rustc_mir_transform/src/iterator_fusion.rs

pub(super) struct IteratorFusionTransform;

impl<'tcx> MirPass<'tcx> for IteratorFusionTransform {
    fn is_enabled(&self, sess: &Session) -> bool {
        // Only on nvptx64
        sess.target.arch == Arch::Nvptx64
    }

    fn run_pass(&self, tcx: TyCtxt<'tcx>, body: &mut Body<'tcx>) {
        // Step 1: Find iterator chain construction sites
        let chains = detect_iterator_chains(tcx, body);
        if chains.is_empty() {
            return;
        }

        // Step 2: For each chain, verify fusability
        for chain in &chains {
            if !is_fusable(tcx, body, chain) {
                continue;
            }

            // Step 3: Compose closure bodies
            let fused_body = compose_closures(tcx, body, chain);

            // Step 4: Replace chain with direct spawn_all + fused body
            replace_with_spawn_all(tcx, body, chain, fused_body);
        }
    }
}
```

### 6.3 Chain Detection

The pass would look for sequences of MIR call sites that match `GpuParallelIterator` trait methods:

```rust
struct IteratorChain {
    /// The GpuParIter::new() call that creates the base iterator
    source: BasicBlock,
    /// Sequence of adapter calls (map, enumerate, zip, ...)
    adapters: Vec<AdapterCall>,
    /// The terminal call (collect_into, for_each, fold, sum, ...)
    terminal: TerminalCall,
}

struct AdapterCall {
    bb: BasicBlock,
    kind: AdapterKind, // Map, Enumerate, Zip, Filter
    closure_def_id: DefId, // The closure passed to the adapter
}
```

Detection works by resolving call targets to `DefId`s and checking if they belong to the `GpuParallelIterator` trait impl. This is the same technique used by `WarpCooperativeTransform` to detect `Future::poll` calls.

### 6.4 Closure Composition

The most complex part. Given two closure MIR bodies `f: A -> B` and `g: B -> C`, produce a composed closure `h: A -> C` where `h(x) = g(f(x))`.

In MIR terms:
1. Clone the basic blocks of `f` into the current function
2. Clone the basic blocks of `g` into the current function
3. Wire `f`'s return value as `g`'s input argument
4. The composed body's entry is `f`'s entry, and its exit is `g`'s exit

This is essentially MIR-level inlining, which rustc already does. The pass could reuse `Inliner` infrastructure from `rustc_mir_transform/src/inline.rs`.

### 6.5 Why This Is Not Needed Yet

The library already achieves the same result through monomorphization:

1. Rust instantiates `GpuMap<GpuMap<GpuParIter<f32>, F>, G>`
2. The `get_unchecked` method for the outer `GpuMap` calls `self.f` on the result of `self.inner.get_unchecked(i)`
3. LLVM inlines both closures and the `get_unchecked` call chain
4. The result is a single fused loop body

The MIR pass would produce the exact same result, just earlier in the compilation pipeline. There is no correctness or performance benefit unless LLVM's inliner fails — which it won't for the small closures used in iterator chains.

---

## 7. Key Decisions and Rationale

### D1: No MIR pass for MVP

**Decision**: Phase 1 uses library-only composition. No compiler changes.

**Rationale**: Rust's monomorphization + LLVM inlining already fuses iterator chains. Building a MIR pass for something the compiler already does is wasted effort. The WarpCooperativeTransform MIR pass exists because async/await CANNOT work without it (correctness). Iterator fusion CAN work without a MIR pass (it's an optimization, and one that's already performed).

### D2: `collect_into` instead of `collect`

**Decision**: The kernel-side API uses `collect_into(output: GpuSliceMut<T>)`, not `collect() -> GpuVec<T>`.

**Rationale**: GPU kernels run in `no_std` with no allocator. Output buffers must be pre-allocated by the host before kernel launch. The host-side API (Phase 2) will provide `collect()` semantics by pre-allocating the output buffer.

### D3: Host-side `gpu::run_data()` before `gpu::par_iter()`

**Decision**: Phase 2 adds a launch wrapper, not full host-side par_iter.

**Rationale**: `gpu::run_data(kernel, &input)` is a 50-line function that removes 80% of the boilerplate. `gpu::par_iter(&data).map(f).collect()` requires either a MIR pass or NVRTC codegen — months of work for the remaining 20% of boilerplate reduction.

### D4: Defer MIR pass to Phase 3/4

**Decision**: MIR pass is documented but not built until Phase 1/2 are battle-tested.

**Rationale**: The existing `WarpCooperativeTransform` shows MIR passes are feasible but expensive to maintain (patched compiler, version-locked to nightly, regenerate patches on every rustc update). Adding a second MIR pass doubles the maintenance burden. Do it only when there's a clear need that the library cannot meet.

### D5: NVRTC as preferred codegen for host-side auto-kernels

**Decision**: When Phase 3 arrives, prefer NVRTC codegen (Approach B) over MIR pass (Approach A).

**Rationale**: The auto-fusion epic (fusion-analysis.2) already designs NVRTC codegen infrastructure for fused elementwise kernels. Reusing that infrastructure for `par_iter` auto-kernel-generation is lower effort and lower risk than a new MIR pass. The codegen templates from fusion-analysis.2 Section 4 directly apply to iterator map chains.

---

## 8. Risks and Mitigations

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| LLVM fails to inline deeply nested iterator chains | Low | Medium — function call per element | Add `#[inline(always)]` to all `get_unchecked` impls. Verify with `--emit=llvm-ir`. |
| Closure capture exceeds 256-byte scratch buffer | Very low | High — runtime panic | Document the limit. Add a compile-time `const_assert!(size_of::<closure>() <= 256)` if possible via proc macro. |
| `spawn_all` overhead dominates for tiny data | Medium | Low — use CPU for small data | Document: GPU par_iter benefits start at ~10K elements. Below that, CPU iteration is faster. |
| MIR pass in Phase 3 is more complex than estimated | High | Medium — delays Phase 3 | NVRTC codegen (Approach B) is the fallback. Both achieve the same user-facing API. |
| Host-side `par_iter` creates false expectation of zero-cost GPU | Medium | Low — documentation | Clearly document: each `gpu::par_iter()` call involves data upload + kernel launch + download. This is NOT free. Amortize over large data. |

## Files Changed: none (design document only)
