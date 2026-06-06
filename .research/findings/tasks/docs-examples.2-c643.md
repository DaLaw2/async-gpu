# docs-examples.2: Create examples for transparent-data, dyn-dispatch, auto-tuning

## Summary

Created three new standalone examples in `examples/std/`, each demonstrating a
post-cycle-607 feature that previously lacked a standalone example.

## Examples Created

### 1. transparent-data (`examples/std/transparent-data/`)
- Shows `GpuArray<T>` with automatic host-device residency management
- Demonstrates the full residency state machine: HostOnly -> Synced -> DeviceOnly -> Synced
- Uses inline PTX (f(x) = x * 2.0 + 1.0) to avoid kernel build dependency
- Zero explicit cudaMemcpy — all transfers happen via `bind_gpu_array()` and `Deref`
- Depends on: `gpu-host` (default features)

### 2. dyn-dispatch (`examples/std/dyn-dispatch/`)
- Launches pre-compiled GPU kernels (`test_gpu_dyn_trait`, `test_gpu_box_dyn_trait`)
- Shows `&dyn Trait` and `Box<dyn Trait>` with vtable-based dispatch on GPU
- Uses `GpuStdModule` with `__HOSTCALL_BUF` device global injection
- Loads cubin for fast startup, falls back to PTX JIT
- Depends on: `gpu-host`, `cudarc`

### 3. auto-tuning (`examples/std/auto-tuning/`)
- Compiles a vector-add CUDA kernel via NVRTC
- Uses `AutoTuner::tune_block_size()` to find optimal block size
- Demonstrates `TuningCache` for caching results by (kernel, size bucket, device)
- Shows `tune_or_cached()` cache-hit path (zero benchmark calls)
- Prints a formatted comparison report via `AutoTuner::format_report()`
- Depends on: `gpu-host`, `cudarc`

## Build Verification

All three examples pass `cargo check`:
- `transparent-data`: compiles with `gpu-host` only
- `dyn-dispatch`: compiles with `gpu-host` + `cudarc`
- `auto-tuning`: compiles with `gpu-host` + `cudarc`

## Design Decisions

1. **Flat crate structure**: Followed the existing example pattern (single crate with
   `[workspace]` at top, no host/kernel split) rather than the host/kernel split
   mentioned in the task description, because all 14 existing `examples/std/` examples
   use the flat pattern.

2. **Inline PTX for transparent-data**: Used a small inline PTX kernel rather than
   depending on the full kernel build pipeline, matching the pattern in
   `gpu_array.rs` tests. This keeps the example self-contained.

3. **Pre-compiled kernels for dyn-dispatch**: Leveraged the existing
   `test_gpu_dyn_trait` and `test_gpu_box_dyn_trait` kernels from gpu-kernel-test
   rather than writing new kernels, since the kernel code is already well-documented
   and tested.

4. **NVRTC for auto-tuning**: Used `cudarc::nvrtc::compile_ptx` with inline CUDA C
   (matching the monte-carlo example pattern) to keep the example self-contained.
