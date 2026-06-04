# fusion-codegen.1 — Investigation: Code Generation Strategies for Fused Kernels

## Status: done

## Summary

Investigated three code generation strategies for fused kernels: NVRTC template codegen, pre-compiled PTX templates, and dynamic JIT. NVRTC template codegen is the clear winner for the async-gpu project — it aligns with existing infrastructure (9 NVRTC compilation sites already in the codebase), requires no new tooling, and the design from fusion-analysis.2 provides ready-to-use templates.

## 1. How NVRTC Is Currently Used — The Established Pattern

The project has a consistent, well-tested NVRTC compilation pattern used across 9 sites in `crates/core/gpu-host/src/nn/ops/`. Every site follows the same 4-step structure:

### Step 1: Static CUDA C source (compile-time constant)
```rust
static KERNEL_SRC: &str = r#"
extern "C" __global__ void my_kernel(...) { ... }
"#;
// or: include_str!("flash_attn_v3.cu")  for larger kernels
```

### Step 2: OnceLock-guarded compilation (compile once per process)
```rust
use std::sync::OnceLock;
static COMPILED: OnceLock<bool> = OnceLock::new();
COMPILED.get_or_init(|| {
    let ptx = compile_ptx(KERNEL_SRC).expect("NVRTC compile failed");
    dev.load_ptx(ptx, "module_name", &["function_name"])
        .expect("PTX load failed");
    true
});
```

### Step 3: Function lookup from cudarc
```rust
let func = dev.get_func("module_name", "function_name")
    .ok_or(NnError::KernelNotFound { name: "function_name" })?;
```

### Step 4: Configure and launch
```rust
let config = LaunchConfig { grid_dim, block_dim, shared_mem_bytes };
unsafe { func.launch(config, (param1, param2, ...)).map_err(NnError::Cuda)?; }
```

### Existing NVRTC sites in the codebase:

| Module | Kernel | Cache mechanism | Compile options |
|--------|--------|----------------|-----------------|
| `reshape.rs` | `elementwise_add_oop` | `OnceLock<bool>` | Default |
| `norm.rs` | `layer_norm_residual` | `OnceLock<bool>` | Default |
| `norm.rs` | `layer_norm_residual_dual` | `OnceLock<bool>` (named `COMPILED_DUAL`) | Default |
| `gemm.rs` | `gemm_f32_v4` | `OnceLock<bool>` | `sm_75, fmad, fast_math` |
| `gemm.rs` | `gemm_f32_v4_1` | `OnceLock<bool>` (named `COMPILED_V41`) | `sm_75, fast_math` |
| `attention.rs` | `flash_attn_v3` | `OnceLock<bool>` | `sm_75, fast_math` |
| `attention.rs` | `flash_attn_tiled` | `OnceLock<bool>` | Default |
| `conv.rs` | `direct_conv2d` + `_tiled` | `get_func().is_none()` guard | `sm_75, fast_math` |
| `conv.rs` | `winograd_filter_transform` + etc. | `get_func().is_none()` guard | `sm_75, fast_math` |

**Key observation**: The `OnceLock` pattern is used for static kernels (known at compile time), while `get_func().is_none()` is used for the conv.rs kernels. Neither supports **dynamic** kernel sources — they assume a fixed CUDA C string known at build time.

### Compilation overhead:
- Simple elementwise kernels: ~50ms (e.g., `elementwise_add_oop`)
- Complex tiled kernels: ~100-200ms (e.g., `gemm_f32_v4`, `flash_attn_v3`)
- Overhead is one-time per process (cached by `OnceLock`)
- For fusion: expect ~50-100ms per fused elementwise chain (simpler than GEMM)

## 2. NVRTC Template Codegen (Recommended Approach)

### How it works

The fusion-analysis.2 design defines per-op CUDA C code fragments. The codegen composes these fragments by string concatenation into a complete kernel, then compiles via `cudarc::nvrtc::compile_ptx()`.

### Template architecture

```
┌─────────────────────────────────────────────┐
│  Kernel template (header + loop structure)  │
│  ┌─────────────────────────────────────┐    │
│  │  Per-op code fragment (BiasAdd)     │    │
│  ├─────────────────────────────────────┤    │
│  │  Per-op code fragment (Gelu)        │    │
│  ├─────────────────────────────────────┤    │
│  │  Per-op code fragment (...)         │    │
│  └─────────────────────────────────────┘    │
│  Kernel template (tail + scalar fallback)   │
└─────────────────────────────────────────────┘
            ↓ String concatenation
┌─────────────────────────────────────────────┐
│  Complete CUDA C source string              │
│  (unique function name via hash)            │
└─────────────────────────────────────────────┘
            ↓ cudarc::nvrtc::compile_ptx()
┌─────────────────────────────────────────────┐
│  PTX module → dev.load_ptx()                │
└─────────────────────────────────────────────┘
            ↓ dev.get_func()
┌─────────────────────────────────────────────┐
│  CudaFunction → launch()                   │
└─────────────────────────────────────────────┘
```

### Why this fits the project perfectly

1. **Infrastructure exists**: `cudarc::nvrtc::compile_ptx()` and `dev.load_ptx()` are already used 9 times. No new dependencies needed.

2. **Templates are already designed**: fusion-analysis.2 provides complete CUDA C fragments for all 6 elementwise ops (Gelu, Relu, Silu, Sigmoid, BiasAdd, ElemAdd), with both float4-vectorized and scalar-tail variants.

3. **Float4 vectorization is proven**: Every NVRTC kernel in the project uses float4 loads (`reinterpret_cast<const float4*>`). The fusion template follows the same pattern.

4. **Module naming is straightforward**: Dynamic kernels use a hash-based module name (e.g., `fused_elem_a7f3b2c1`) to avoid collisions with static kernels in the `nn_kernels` module.

### What needs to change vs. the OnceLock pattern

The current `OnceLock<bool>` pattern caches a **single fixed kernel** per process. Fused kernels are **dynamic** — different op chains produce different CUDA C source. This requires a new caching layer:

```rust
/// Cache for dynamically-generated fused kernels.
struct FusionCache {
    // key: hash(ops + shape params) → compiled module/function names
    compiled: Mutex<HashMap<u64, CompiledFusedKernel>>,
}

struct CompiledFusedKernel {
    module_name: String,   // e.g., "fused_elem_a7f3b2c1"
    func_name: String,     // e.g., "fused_elementwise_a7f3b2c1"
}
```

This is a natural extension of the existing pattern. The `Mutex<HashMap>` replaces `OnceLock<bool>` for the dynamic case.

### Cache key design

The cache key must capture everything that affects the generated CUDA C source:

```rust
fn cache_key(ops: &[OpKind], shape_params: &[u64]) -> u64 {
    use std::hash::{Hash, Hasher};
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    ops.hash(&mut hasher);         // which ops, in order
    shape_params.hash(&mut hasher); // n_cols for BiasAdd, etc.
    hasher.finish()
}
```

Shape params included in the key:
- **BiasAdd**: `n_cols` (affects `idx % n_cols` in the code)
- **ElemAdd**: nothing extra (pure element-wise)
- **Gelu/Relu/Silu/Sigmoid**: nothing extra (stateless)

**Note**: The total element count `n` is a kernel *argument*, not a codegen parameter. Different batch sizes reuse the same compiled kernel with different `n` values. This is critical for cache hit rate — a GPT-2 model with constant sequence length will compile each chain once.

## 3. Pre-compiled PTX Template Approach (Not Recommended)

### How it would work

Instead of generating CUDA C and invoking NVRTC, we would generate PTX assembly directly:

```ptx
.entry fused_bias_gelu (
    .param .u64 input, .param .u64 output,
    .param .u64 bias, .param .u32 n_cols, .param .u32 n
) {
    .reg .u32 %tid, %idx;
    .reg .f32 %v0, %v1, %v2, %v3;
    // ... manual PTX for bias add + gelu ...
}
```

### Assessment

| Factor | NVRTC Template | PTX Template |
|--------|---------------|-------------|
| Code correctness | CUDA C is human-readable; NVRTC validates | PTX is low-level; bugs are hard to spot |
| Float4 vectorization | `reinterpret_cast<float4*>` — trivial | `ld.global.v4.f32` — requires manual address alignment |
| Math functions | `tanhf()`, `expf()` — built-in | Must use `ex2.approx.f32` + polynomial approximation |
| Shared memory | Straightforward `__shared__` | Manual `.shared` declarations and `st.shared`/`ld.shared` |
| Debug/iterate | Change C string, recompile | Change PTX, hope register alloc is right |
| Compile overhead | ~50-100ms per chain | 0ms (skip NVRTC) |
| Project precedent | 9 existing sites | 0 existing dynamic PTX generation |

**The only advantage of PTX templates is skipping NVRTC compilation overhead (~50-100ms)**. But:

1. This overhead is **one-time per process** — cached after first compilation.
2. For GPT-2, we expect ~5-6 unique fused kernels → total one-time cost: ~300-600ms.
3. This is amortized over hundreds of forward passes during training/inference.
4. The project already tolerates NVRTC overhead (8.6 MB kernel.ptx takes 15+ minutes to *load*; the NVRTC compile is negligible in comparison).

**The complexity cost of PTX templates vastly outweighs the compilation savings.** Not recommended.

### Note on existing PTX

The project has pre-compiled PTX (via `kernel.ptx` from the `gpu-kernel-std` crate compiled with `cargo build --target nvptx64-nvidia-cuda`). This is a **Rust-to-PTX** pipeline for warp-cooperative and std-based kernels, not a PTX template system. It does not support dynamic kernel generation.

## 4. Dynamic JIT Approach (This IS the NVRTC Approach)

"Dynamic JIT" is not a separate strategy — it describes the **execution model**, which is the same regardless of whether we use NVRTC or PTX templates:

```
First call:  generate source → compile → cache → launch
Later calls: cache lookup → launch
```

The question is what source we generate (CUDA C vs PTX), and the answer is CUDA C via NVRTC.

### Cache behavior for GPT-2 inference

With constant shapes (seq_len=128, d_model=768, 3072 FFN hidden):

| Iteration | Action | Overhead |
|-----------|--------|----------|
| 1 | Compile ~5-6 fused kernels via NVRTC | ~300-600ms total |
| 2+ | All cache hits | ~100ns per lookup |

For training with variable batch sizes: the cache still works because element count `n` is a runtime argument, not a codegen parameter. Only `n_cols` (BiasAdd column count) is baked into the kernel. Since model architecture is fixed, the same compiled kernels serve all batch sizes.

### Persistence across runs (optional, future)

The current OnceLock-based caching is process-scoped. For long-running servers, the ~300-600ms first-run overhead is negligible. For short-lived processes (e.g., benchmarking), an optional disk cache could store compiled PTX:

```rust
// Future optimization — NOT needed for Phase 1
let ptx_cache_path = format!("/tmp/async_gpu_cache/{}.ptx", hex_key);
if Path::new(&ptx_cache_path).exists() {
    let ptx = Ptx::from_file(&ptx_cache_path);
    dev.load_ptx(ptx, module_name, &[func_name])?;
} else {
    let ptx = compile_ptx(cuda_src)?;
    std::fs::write(&ptx_cache_path, ptx.as_bytes())?;
    dev.load_ptx(ptx, module_name, &[func_name])?;
}
```

This is a straightforward extension but adds filesystem dependencies. Defer to a later phase.

## 5. Practical Recommendation for Phase 1

### Strategy: NVRTC Template Codegen with HashMap Cache

**Phase 1 scope** (matching fusion-analysis.2 Phase 3):

1. **Implement `FusionCodegen` module** alongside the existing `FusionOptimizer` in `nn/fusion.rs` (or a new `nn/fusion_codegen.rs`):
   - `fn codegen_elementwise_chain(ops: &[TapeEntry]) -> String` — generates CUDA C source
   - `fn compile_fused_kernel(dev: &Arc<CudaDevice>, cuda_src: &str, key: u64) -> Result<CompiledFusedKernel>` — compiles and loads

2. **FusionCache**: `Mutex<HashMap<u64, CompiledFusedKernel>>` with process-scoped lifetime. Replace the per-kernel `OnceLock` pattern with a single shared cache.

3. **Support P5-P10 patterns** (from fusion-analysis.2):
   - P5: BiasAdd → Gelu
   - P6: BiasAdd → Silu
   - P7: BiasAdd → Relu
   - P8: BiasAdd → Sigmoid
   - P9: ElemAdd → Gelu
   - P10: Arbitrary 2-5 elementwise chain

4. **Do NOT change existing hand-fused kernels** (P1: MatmulBiasGelu, P3: ElemAddLayerNorm). These are already compiled from PTX and routed by the FusionOptimizer. The codegen layer is for elementwise chains only.

5. **Do NOT attempt GEMM epilogue codegen** (P4: MatmulBias). GEMM kernels have complex tiling strategies. Epilogue fusion requires modifying the GEMM output write loop in the existing V4/V4.1 CUDA C source — this is a kernel-level change, not a codegen template.

### Implementation outline

```rust
// nn/fusion_codegen.rs

use std::collections::HashMap;
use std::sync::Mutex;

pub struct FusionCodegen {
    cache: Mutex<HashMap<u64, CompiledFusedKernel>>,
}

struct CompiledFusedKernel {
    module_name: String,
    func_name: String,
}

impl FusionCodegen {
    pub fn new() -> Self {
        Self { cache: Mutex::new(HashMap::new()) }
    }

    /// Get or compile a fused elementwise kernel.
    pub fn get_or_compile(
        &self,
        group: &FusionGroup,
        tape: &[TapeEntry],
        dev: &Arc<CudaDevice>,
    ) -> Result<CudaFunction> {
        let key = Self::cache_key(group, tape);
        let mut cache = self.cache.lock().unwrap();

        if let Some(compiled) = cache.get(&key) {
            return dev.get_func(&compiled.module_name, &compiled.func_name)
                .ok_or(NnError::KernelNotFound { name: "fused" });
        }

        // Generate CUDA C source from op chain
        let (cuda_src, func_name) = Self::codegen(group, tape, key);
        let module_name = format!("fused_{key:016x}");

        // Compile via NVRTC
        let opts = cudarc::nvrtc::CompileOptions {
            arch: Some("sm_75"),
            use_fast_math: Some(true),
            ..Default::default()
        };
        let ptx = cudarc::nvrtc::compile_ptx_with_opts(&cuda_src, opts)?;
        dev.load_ptx(ptx, &module_name, &[&func_name])?;

        cache.insert(key, CompiledFusedKernel {
            module_name: module_name.clone(),
            func_name: func_name.clone(),
        });

        dev.get_func(&module_name, &func_name)
            .ok_or(NnError::KernelNotFound { name: "fused" })
    }

    fn codegen(group: &FusionGroup, tape: &[TapeEntry], key: u64) -> (String, String) {
        // Use the template + per-op fragments from fusion-analysis.2 Section 4
        // String concatenation of CUDA C code
        todo!()
    }

    fn cache_key(group: &FusionGroup, tape: &[TapeEntry]) -> u64 {
        // Hash ops + shape-affecting params
        todo!()
    }
}
```

### Expected performance

| Metric | Value |
|--------|-------|
| Fused chains in GPT-2 (P5-P10 only) | ~0-2 per block (most chains are already covered by P1/P3/P4) |
| NVRTC compile per unique chain | ~50-100ms |
| Total first-run overhead | ~100-200ms |
| Cache hit latency | ~100ns (HashMap lookup) |
| Memory per cached kernel | ~200 bytes (module name + function name; PTX lives in cudarc's device state) |

### Why NOT a more sophisticated approach

- **No IR needed**: The fusion chains are simple (2-5 elementwise ops). An intermediate representation adds complexity for no benefit. Direct string concatenation is sufficient.
- **No optimization passes needed**: The generated code is already near-optimal — float4 vectorization, no redundant loads, fused math ops. NVRTC's PTX compiler handles register allocation.
- **No backward codegen needed** (Phase 1): The design explicitly recommends inference-only fusion first. Training backward passes use the existing unfused tape entries.

## Files Referenced

- `crates/core/gpu-host/src/nn/fusion.rs` — existing FusionOptimizer (detection only)
- `crates/core/gpu-host/src/nn/ops/reshape.rs:217-293` — NVRTC pattern: elementwise_add_oop
- `crates/core/gpu-host/src/nn/ops/norm.rs:83-237` — NVRTC pattern: layer_norm_residual
- `crates/core/gpu-host/src/nn/ops/norm.rs:239-406` — NVRTC pattern: layer_norm_residual_dual
- `crates/core/gpu-host/src/nn/ops/gemm.rs:347-530` — NVRTC pattern: gemm_f32_v4
- `crates/core/gpu-host/src/nn/ops/attention.rs:215-282` — NVRTC pattern: flash_attn_v3
- `crates/core/gpu-host/src/nn/ops/conv.rs:919-1035` — NVRTC pattern: direct_conv2d (get_func guard)
- `crates/core/gpu-host/src/nn/registry.rs` — KernelRegistry with static PTX module
- `crates/core/gpu-host/src/nn/autograd/tape.rs` — TapeEntry, OpKind, OpMeta
- `.research/findings/tasks/fusion-analysis.2-c607.md` — fusion design with NVRTC templates
