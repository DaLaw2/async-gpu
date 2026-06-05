# split-design.3: Multi-Cubin Host Loader API Design

**Task**: Design host loader multi-cubin API with backward compatibility
**Date**: 2026-06-05 | **Cycle**: 613

## Summary

The host loader (gpu-host) currently assumes a single monolithic PTX/cubin. The 4-crate kernel split requires the host to load 4 independent PTX/cubin pairs. This design adds per-crate PTX constants (`KERNEL_CORE`, `KERNEL_COMPUTE`, etc.), a kernel catalog for auto-discovery, backward-compatible aliases, and a `.module()` builder method — while keeping every existing call site working with zero changes.

## 1. PTX Module Design (`ptx` module in lib.rs)

### Current State (lib.rs:137-157)

```rust
pub mod ptx {
    pub const KERNEL: &str = include_str!("../kernel.ptx");
    pub const KERNEL_STD: &str = include_str!("../kernel_std.ptx");
    // + 5 test-specific PTX constants (EMBASSY_TEST, etc.)
}
```

`KERNEL` and `KERNEL_STD` are identical copies of the same ~9.9 MB unified PTX. There is one cubin: `kernel_std.cubin` (187 MB).

### Proposed Design

```rust
pub mod ptx {
    // ── Per-crate PTX (new, canonical) ──────────────────────────
    /// Core kernels: basic ops, math helpers, infrastructure.
    pub const KERNEL_CORE: &str = include_str!("../kernel_core.ptx");
    /// ML/compute kernels: GEMM, transformer, CNN, physics, search.
    pub const KERNEL_COMPUTE: &str = include_str!("../kernel_compute.ptx");
    /// I/O kernels: hostcall, pipeline, hybrid warp print.
    pub const KERNEL_IO: &str = include_str!("../kernel_io.ptx");
    /// Test/demo kernels: std demos, warp tests, thread tests, par_iter.
    pub const KERNEL_TEST: &str = include_str!("../kernel_test.ptx");

    // ── Per-crate cubin (new) ───────────────────────────────────
    /// Pre-compiled cubin for kernel_core (fast load).
    pub const CUBIN_CORE: &[u8] = include_bytes!("../kernel_core.cubin");
    pub const CUBIN_COMPUTE: &[u8] = include_bytes!("../kernel_compute.cubin");
    pub const CUBIN_IO: &[u8] = include_bytes!("../kernel_io.cubin");
    pub const CUBIN_TEST: &[u8] = include_bytes!("../kernel_test.cubin");

    // ── Backward-compatible aliases (deprecated) ────────────────
    /// Alias: KERNEL → KERNEL_COMPUTE (the largest, most-used module).
    #[deprecated(note = "use ptx::KERNEL_COMPUTE for ML kernels, \
                         or ptx::ALL for auto-discovery")]
    pub const KERNEL: &str = KERNEL_COMPUTE;

    /// Alias: KERNEL_STD → KERNEL_TEST (test/demo kernels).
    #[deprecated(note = "use ptx::KERNEL_TEST for test kernels")]
    pub const KERNEL_STD: &str = KERNEL_TEST;

    // ── Auto-discovery catalog ──────────────────────────────────
    /// All PTX modules with their cubins, for APIs that search across modules.
    pub const ALL: &[PtxModule] = &[
        PtxModule { name: "core",    ptx: KERNEL_CORE,    cubin: CUBIN_CORE },
        PtxModule { name: "compute", ptx: KERNEL_COMPUTE, cubin: CUBIN_COMPUTE },
        PtxModule { name: "io",      ptx: KERNEL_IO,      cubin: CUBIN_IO },
        PtxModule { name: "test",    ptx: KERNEL_TEST,    cubin: CUBIN_TEST },
    ];

    /// A PTX/cubin pair with a human-readable name.
    pub struct PtxModule {
        pub name: &'static str,
        pub ptx: &'static str,
        pub cubin: &'static [u8],
    }

    // ── Legacy test PTX constants (unchanged) ───────────────────
    pub const EMBASSY_TEST: &str = include_str!("../embassy_test.ptx");
    pub const ASYNC_HOSTCALL_TEST: &str = include_str!("../async_hostcall_test.ptx");
    pub const STD_BUILD_TEST: &str = include_str!("../std_build_test.ptx");
    pub const ASYNC_PIPELINE_TEST: &str = include_str!("../async_pipeline_test.ptx");
    pub const MULTI_WARP_TEST: &str = include_str!("../multi_warp_test.ptx");
}
```

### Key Decisions

1. **KERNEL aliases to KERNEL_COMPUTE** (not a concatenation of all): KernelRegistry loads only ML kernels, and `get_kernel()` in gpu.rs uses KERNEL for the same purpose. The overwhelming majority of KERNEL usage is ML compute. Aliasing to KERNEL_COMPUTE gives backward compat with zero behavior change — the compute module will contain the same ~60 ML kernel functions.

2. **KERNEL_STD aliases to KERNEL_TEST**: The only users of KERNEL_STD are the `#[gpu_test]` macro and gpu-test-harness tests. These launch test/demo kernels (test_gpu_assert_basic, test_gpu_vec_operations, etc.) which all live in the proposed kernel_test crate.

3. **Embedded cubins via `include_bytes!`**: Today, cubins are loaded from disk at runtime (the macro reads `kernel_std.cubin` from a relative path). With split cubins, embedding them as constants eliminates path-resolution complexity. The 187 MB unified cubin will shrink to ~4 smaller cubins (rough estimate: core ~5 MB, compute ~150 MB, io ~20 MB, test ~15 MB). Only the cubins actually referenced get linked into the final binary (LTO/dead code elim for `&[u8]` constants).

4. **`PtxModule` struct + `ALL` array**: Provides a structured catalog for auto-discovery APIs. The `name` field enables meaningful error messages ("kernel 'foo' not found in modules: core, compute, io, test").

### Alternative Considered: Concatenated PTX

An alternative is to concatenate all 4 PTX files into one at compile time and maintain the single-module illusion. **Rejected** because: (a) it defeats the purpose of the split (incremental rebuild), (b) CUDA's JIT compiler scales super-linearly with PTX size, and (c) users who only need compute kernels shouldn't pay for test kernel binary size.

## 2. gpu.rs Changes

### 2a. `get_kernel()` — Internal, No Change Needed

```rust
fn get_kernel(dev, kernel_name) -> Result<CudaFunction> {
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::ptx::KERNEL);  // → KERNEL_COMPUTE alias
    ...
}
```

`KERNEL` aliases to `KERNEL_COMPUTE`, so this keeps working. The functions loaded by `get_kernel()` (used by `gpu::run()`, `gpu::run_with_output()`, `gpu::launch()`) are all compute/basic kernels present in KERNEL_COMPUTE.

**Caveat**: If a user calls `gpu::run("some_io_kernel")`, it will fail with KernelNotFound because `get_kernel()` only searches KERNEL_COMPUTE. This is acceptable — `gpu::run()` was always the "quick convenience" API. Users wanting IO kernels should use `gpu::run_zero_param()` or `gpu::custom()` with an explicit module.

### 2b. `run_zero_param()` family — No Change Needed

These already accept `ptx_src: &str` as a parameter:
```rust
pub fn run_zero_param(ptx_src: &str, kernel_name: &str) -> Result<()>
pub fn run_zero_param_with_cubin(ptx_src: &str, cubin: &[u8], ...) -> Result<()>
```

Callers already pass the specific PTX they want. Post-split, they'll pass the appropriate per-crate constant:
```rust
// Before: gpu::run_zero_param(ptx::KERNEL_STD, "test_hello")
// After:  gpu::run_zero_param(ptx::KERNEL_TEST, "test_hello")
// Compat: ptx::KERNEL_STD still works (deprecated alias)
```

### 2c. `GpuStdModule` — No Change Needed

Same situation as `run_zero_param` — already takes `ptx_src` as parameter.

### 2d. `CustomLaunchBuilder` — Add `.module()` Method

The builder API's `.ptx()` already allows explicit PTX selection. Add a `.module()` convenience that selects both PTX and cubin from a `PtxModule`:

```rust
impl CustomLaunchBuilder {
    /// Use a specific kernel module (PTX + cubin pair).
    ///
    /// Convenience for `.ptx(module.ptx).cubin(module.cubin.to_vec())`.
    pub fn module(self, m: &ptx::PtxModule) -> Self {
        self.ptx(m.ptx).cubin(m.cubin.to_vec())
    }
}
```

Note: `.ptx()` currently takes `&'static str`, which `PtxModule.ptx` satisfies. `.cubin()` takes `Vec<u8>`, so we need `.to_vec()` on the `&[u8]` — this copies the cubin data. An alternative is to add a `.cubin_bytes()` method that takes `&'static [u8]` to avoid the copy, or change `.cubin()` signature to accept `&[u8]` (breaking, but the field is private so only internal callers affected).

**Recommended**: Add `cubin_static(data: &'static [u8])` to avoid the copy:

```rust
pub fn cubin_static(mut self, data: &'static [u8]) -> Self {
    if !data.is_empty() {
        self.cubin_data = Some(CubinData::Static(data));
    }
    self
}
```

This requires changing `cubin_data` from `Option<Vec<u8>>` to `Option<CubinData>`:

```rust
enum CubinData {
    Owned(Vec<u8>),
    Static(&'static [u8]),
}
impl CubinData {
    fn as_slice(&self) -> &[u8] {
        match self { Self::Owned(v) => v, Self::Static(s) => s }
    }
}
```

### 2e. Auto-Discovery Fallback — New `gpu::run_any()` Convenience

For users who don't know which module a kernel lives in:

```rust
/// Launch a kernel, searching all PTX modules for it.
///
/// Tries each module in `ptx::ALL` until the kernel is found.
/// This is slower than specifying the module directly but convenient
/// for interactive exploration.
pub fn run_any(kernel_name: &'static str) -> Result<()> {
    for m in crate::ptx::ALL {
        match run_zero_param_with_cubin(m.ptx, m.cubin, kernel_name, 128, (1,1,1)) {
            Ok(()) => return Ok(()),
            Err(GpuHostError::KernelNotFound(_)) => continue,
            Err(e) => return Err(e),
        }
    }
    Err(GpuHostError::KernelNotFound(kernel_name))
}
```

**Cost**: In the worst case, loads 4 modules before finding the kernel. The cubin fast-path makes each load sub-second, so worst case is ~4 seconds. For production code, users should specify the module directly.

**Alternative**: A compile-time kernel→module mapping (generated by the build script). Deferred — not needed for the initial split.

### 2f. `prepare()` Default PTX Fallback

Currently: `self.ptx_src.unwrap_or(crate::ptx::KERNEL)` (line 772). After the alias, this resolves to KERNEL_COMPUTE. Users building a custom kernel with `gpu::custom("my_kernel").prepare()` without calling `.ptx()` will get KERNEL_COMPUTE by default, which is the correct behavior (most custom launches use compute kernels).

## 3. KernelRegistry Changes

### Current State

```rust
impl KernelRegistry {
    pub fn new(device: Arc<CudaDevice>, ptx_src: &str) -> Result<Self> { ... }
    pub fn init_default() -> Result<(Arc<CudaDevice>, Arc<Self>)> {
        // Uses crate::ptx::KERNEL
    }
}
```

All call sites pass `crate::ptx::KERNEL` explicitly or use `init_default()`.

### Changes Required: None

The `KERNEL` alias resolves to `KERNEL_COMPUTE`. All ~60 ML kernel functions in `ML_KERNELS` (registry.rs:34-103) are from compute files (compute_gemm.rs, compute_transformer.rs, compute_cnn.rs, etc.) which all map to the gpu-kernel-compute crate.

**Verification**: Cross-referencing ML_KERNELS entries with split-design.1 file-to-crate mapping:
- CNN ops → compute_cnn.rs → gpu-kernel-compute
- GEMM ops → compute_gemm.rs → gpu-kernel-compute
- Transformer ops → compute_transformer.rs → gpu-kernel-compute
- Physics ops → compute_physics.rs → gpu-kernel-compute
- Fused ops → compute_fused.rs → gpu-kernel-compute
- Persistent → compute_persistent.rs → gpu-kernel-compute
- Backward kernels → gpu-kernel-compute (autograd kernels live in compute files)

All ML_KERNELS are in gpu-kernel-compute. No changes needed.

**Future enhancement**: If KernelRegistry ever needs kernels from multiple modules, change `new()` to accept `&[&str]` (multiple PTX sources) and load them into separate cudarc modules with a module prefix. Not needed now.

## 4. gpu-test-macro Changes

### Current State

```rust
// In expanded code:
gpu_host::gpu::run_zero_param_with_cubin(
    gpu_host::ptx::KERNEL_STD,
    &cubin,
    #kernel_name,
    ...
)
```

And cubin path: `../../core/gpu-host/kernel_std.cubin`

### Proposed Changes

#### Phase 1 (Alias Period): No Code Changes

`KERNEL_STD` aliases to `KERNEL_TEST`, so the macro expansion works unchanged. The cubin path remains `kernel_std.cubin` during the transition (the build script produces both the old and new names).

#### Phase 2 (After Split Ships): Update Macro

```rust
// Updated expansion:
let cubin = gpu_host::ptx::CUBIN_TEST;  // embedded, no file I/O

gpu_host::gpu::run_zero_param_with_cubin(
    gpu_host::ptx::KERNEL_TEST,
    cubin,
    #kernel_name,
    ...
)
```

Benefits:
- No relative path resolution (today's `../../core/gpu-host/kernel_std.cubin` is fragile)
- Cubin is embedded at compile time via `include_bytes!`
- Test binary is self-contained

#### Optional: `ptx` Attribute for Module Selection

```rust
#[gpu_test(ptx = "io")]  // loads KERNEL_IO instead of KERNEL_TEST
fn test_pipeline_feature() {}
```

This is low priority — most gpu_test kernels are test/demo kernels in gpu-kernel-test.

## 5. Cubin File Layout

### On Disk (build artifacts in gpu-host/)

```
crates/core/gpu-host/
├── kernel_core.ptx          # PTX from gpu-kernel-core
├── kernel_core.cubin         # Pre-compiled cubin
├── kernel_compute.ptx        # PTX from gpu-kernel-compute
├── kernel_compute.cubin      # Pre-compiled cubin (largest, ~150 MB)
├── kernel_io.ptx             # PTX from gpu-kernel-io
├── kernel_io.cubin
├── kernel_test.ptx           # PTX from gpu-kernel-test
├── kernel_test.cubin
├── kernel.ptx                # REMOVED after transition (was alias)
├── kernel_std.ptx            # REMOVED after transition (was alias)
├── kernel_std.cubin           # REMOVED after transition
└── ... (embassy_test.ptx etc. unchanged)
```

### Naming Convention

`kernel_{crate_suffix}.{ptx|cubin}` where `crate_suffix` matches the Cargo crate name:
- `gpu-kernel-core` → `kernel_core.ptx/cubin`
- `gpu-kernel-compute` → `kernel_compute.ptx/cubin`
- `gpu-kernel-io` → `kernel_io.ptx/cubin`
- `gpu-kernel-test` → `kernel_test.ptx/cubin`

### Build Script Changes (build-kernel-std.sh → build-kernels.sh)

```bash
# Build each kernel crate independently
for crate in core compute io test; do
    cargo "+$CHANNEL" build -p "gpu-kernel-$crate" --release
    cp "target/nvptx64-nvidia-cuda/release/gpu_kernel_$crate.ptx" \
       "$HOST_DIR/kernel_$crate.ptx"
done

# Pre-compile cubins (can be parallelized)
for crate in core compute io test; do
    $PTXAS --gpu-name sm_75 \
        -o "$HOST_DIR/kernel_$crate.cubin" \
        "$HOST_DIR/kernel_$crate.ptx" &
done
wait
```

Note: `ptxas` for each crate is independent and can run in parallel. The compute cubin will still take ~10 min (most of the PTX), but core/io/test will be much faster due to smaller PTX size.

## 6. Migration Path

### Phase 0: Preparation (Current)
- Design complete (this document)
- No code changes yet

### Phase 1: Build Infrastructure
- Split gpu-kernel-std into 4 crates
- Update build-kernel-std.sh → build-kernels.sh
- Generate 4 PTX/cubin pairs
- Copy all 4 into gpu-host/

### Phase 2: Host Loader Update (Backward Compatible)
- Add per-crate constants (KERNEL_CORE, etc.) and PtxModule/ALL
- Add deprecated aliases: `KERNEL = KERNEL_COMPUTE`, `KERNEL_STD = KERNEL_TEST`
- Add `.module()` to CustomLaunchBuilder
- Add `CubinData` enum for zero-copy cubin embedding
- **Zero call site changes** — everything compiles with deprecation warnings

### Phase 3: Call Site Migration
- Update KernelRegistry call sites: `ptx::KERNEL` → `ptx::KERNEL_COMPUTE` (suppress warnings)
- Update gpu-test-macro: `ptx::KERNEL_STD` → `ptx::KERNEL_TEST`, use embedded cubin
- Update gpu-test-harness constants
- Update examples

### Phase 4: Cleanup
- Remove deprecated aliases
- Remove kernel.ptx / kernel_std.ptx / kernel_std.cubin (old unified files)
- Remove file-based cubin loading from macro (use embedded)

## 7. Open Questions

1. **Binary size**: Embedding 4 cubins via `include_bytes!` could add ~190 MB to the gpu-host rlib. Today, only the ~10 MB PTX is embedded (cubin loaded from disk at runtime). Options:
   - Keep cubins on disk (current approach) — but fragile paths
   - Embed only the cubin the user needs via Cargo features: `features = ["cubin-compute"]`
   - Embed all and rely on dead-code elimination (linker may not drop unused `&[u8]` statics)
   - **Recommendation**: Keep cubins as disk files for now, embed only PTX. Add `ptx::cubin_path(module_name) -> PathBuf` helper for runtime loading.

2. **Feature-gated PTX**: Should unused PTX modules be behind features? E.g., a user who only needs compute shouldn't compile in 5 MB of test PTX. This could be `default = ["ptx-compute"]` with optional `ptx-io`, `ptx-test`, `ptx-core`.

3. **Module loading overhead**: Loading 4 separate CUDA modules has more overhead than 1 (4x cuModuleLoad calls). For users needing kernels from multiple modules in a hot path, consider a `MultiModule` wrapper that loads all needed modules once and provides unified kernel lookup.
