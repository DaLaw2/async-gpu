# iter-runtime.2 — Chained iterator fusion — zero intermediate buffers in pipeline

## Status: DONE

## Summary

Chained `.map()` calls on `GpuParallelIterator` produce zero intermediate buffers. Rust monomorphization creates nested adapter types (`GpuMap<GpuMap<GpuParIter>>`) that LLVM fully inlines into a single expression per element. PTX inspection confirms all operations execute in registers within one loop iteration. Three fusion patterns verified: dual map, triple map + sum, and map + filter + count.

## 1. Chained map fusion — PTX evidence

### Kernel: `par_iter_chained_map_collect`

Source:
```rust
src.par_iter()
    .map(|x: f32| x * 2.0)   // map #1
    .map(|x: f32| x + 1.0)   // map #2
    .collect_into(dst);
```

Compile-time type: `GpuMap<GpuMap<GpuParIter<f32>, C1>, C2>`

PTX inner loop (spawn_all trampoline):
```ptx
$L__BB304_3:                            // =>This Inner Loop Header
    ld.volatile.b32  %r5, [%rd7];       // read_volatile(input[i])
    add.rn.f32       %r6, %r5, %r5;     // x * 2.0 (compiler uses x+x)
    add.rn.f32       %r7, %r6, 0f3F800000;  // + 1.0
    st.volatile.b32  [%rd8], %r7;       // write_volatile(output[i])
    ; ... stride update + loop back
```

**Zero intermediate buffers.** Both maps inlined into two `add.rn.f32` instructions between one load and one store. The nested `get_unchecked()` chain is fully flattened.

### Kernel: `par_iter_triple_map_sum`

Source:
```rust
src.par_iter()
    .map(|x: f32| x + 1.0)   // map #1
    .map(|x: f32| x * 3.0)   // map #2
    .map(|x: f32| x - 0.5)   // map #3
    .sum();
```

Compile-time type: `GpuMap<GpuMap<GpuMap<GpuParIter<f32>, C1>, C2>, C3>`

PTX inner loop:
```ptx
$L__BB318_3:                            // =>This Inner Loop Header
    ld.volatile.b32  %r5, [%rd12];          // read_volatile(input[i])
    add.rn.f32       %r6, %r5, 0f3F800000;  // + 1.0  (map #1)
    mul.rn.f32       %r7, %r6, 0f40400000;  // * 3.0  (map #2)
    add.rn.f32       %r8, %r7, 0fBF000000;  // - 0.5  (map #3)
    add.rn.f32       %r9, %r9, %r8;         // acc += result (fold)
    ; ... stride update + loop back
```

**Three chained maps fused into three consecutive PTX instructions.** No intermediate buffers, no function calls, pure register chain. The fold accumulation is also inlined.

## 2. Map + filter fusion — PTX evidence

### Kernel: `par_iter_map_filter_count`

Source:
```rust
src.par_iter()
    .map(|x: f32| x * x)                   // map: square
    .filter(move |x: &f32| *x > threshold) // filter: > threshold
    .count();
```

PTX inner loop:
```ptx
$L__BB314_3:                            // =>This Inner Loop Header
    ld.volatile.b32  %r6, [%rd12];      // read_volatile(input[i])
    mul.rn.f32       %r7, %r6, %r6;     // x * x (map)
    setp.gt.f32      %p3, %r7, %r2;     // squared > threshold? (filter)
    selp.b64         %rd8, 1, 0, %p3;   // branchless: pass ? 1 : 0
    add.s64          %rd14, %rd14, %rd8; // count += pass
    ; ... stride update + loop back
```

**Map fused into filter.** The squared value exists only in register `%r7` — never written to memory. The compiler even optimized the conditional count increment to a branchless `selp` (select-on-predicate) instruction, which is more efficient than a branch on GPU.

## 3. Type-level composition — zero heap allocation

The adapter chain is composed entirely at the type level:

| Chain | Monomorphized type | sizeof |
|-------|-------------------|--------|
| `.map(f)` | `GpuMap<GpuParIter<f32>, F>` | 16 bytes (ptr+len+ZST) |
| `.map(f).map(g)` | `GpuMap<GpuMap<GpuParIter, F>, G>` | 16 bytes (ZST closures) |
| `.map(f).map(g).map(h)` | `GpuMap<GpuMap<GpuMap<...>, F>, G>, H>` | 16 bytes (ZST closures) |
| `.map(f).filter(p)` | `GpuFilter<GpuMap<GpuParIter, F>, P>` | 16 bytes (ZST closures) |

All closures with no captures are zero-sized types (ZST). The entire iterator chain is the same size as the base `GpuParIter` (pointer + length = 16 bytes). This fits easily in the 256-byte warp scratch buffer used by `spawn_all`.

## 4. Host-side verification

Created host-side tests in `tests_par_iter.rs` that:
1. Upload input data to GPU
2. Launch the kernel with 1 block x 128 threads (4 warps)
3. Download output and compare against CPU reference

Tests:
- `run_single_map_baseline_test` — baseline: single `.map(|x| x*2.0+1.0)`
- `run_chained_map_test` — two separate `.map()` calls, verifies identical output to baseline
- `run_map_filter_count_test` — `.map(square).filter(>thresh).count()`, verifies count matches CPU
- `run_triple_map_sum_test` — three `.map()` calls + `.sum()`, verifies sum matches CPU

## 5. Key finding: no MIR pass needed

The design document (iter-design.3) predicted that library-level fusion via monomorphization would handle all practical cases. This experiment confirms it:

- **Map chains**: Fully fused regardless of depth (tested 1, 2, 3 maps)
- **Map + filter**: Map output never materializes in memory
- **Map + fold/sum**: Accumulation inlined into the same loop body
- **No function calls**: Every `get_unchecked()` call is inlined away
- **No heap allocation**: All adapter types are `Copy`, fit in registers

A MIR pass for iterator fusion is unnecessary for the current architecture. Rust's monomorphization + LLVM inlining produces optimal code without compiler intervention.

## Files changed

- `crates/kernel/gpu-kernel-std/src/par_iter_demo.rs` — 3 new kernels (chained_map, map_filter_count, triple_map_sum)
- `crates/test/gpu-test-harness/src/tests_par_iter.rs` — NEW: host-side verification tests
- `crates/test/gpu-test-harness/src/main.rs` — registered tests_par_iter module + ONLY_TEST handler
- `crates/core/gpu-host/kernel.ptx` — rebuilt with new kernels
- `crates/core/gpu-host/kernel_std.ptx` — rebuilt with new kernels
