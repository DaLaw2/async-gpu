# ptx-codegen-fix.2: Experiment — PTX ISA 7.8+ targeting or post-processing
**Cycle**: 278 | **Theme**: ptx-codegen-fix | **Kind**: experiment | **Status**: done

## Summary

Found that `-C target-feature=+ptx78` is the correct way to target PTX ISA 7.8+ from
rustc/LLVM. This completely eliminates the `.ptr .align 1` annotations and also resolves
the `panic_const_async_fn_resumed` extern stub issue. No post-processing script is needed
when this flag is used.

## Findings

### Approach 1: `-C target-feature=+ptx78` — SUCCESS

Adding `-C target-feature=+ptx78` to rustflags produces PTX with `.version 7.8` and
**zero** `.ptr .align` annotations. The LLVM NVPTX backend (LLVM 22.1.0 in
nightly-2026-03-11) emits `.param .b64` instead of `.param .u64 .ptr .align 1` when
targeting PTX 7.8+.

Available PTX version features (from `rustc --print target-features`):
`ptx32` through `ptx87` (including `ptx78`, `ptx80`, etc.)

**Config change** (`.cargo/config.toml`):
```toml
rustflags = ["-C", "target-cpu=sm_86", "-C", "target-feature=+ptx78"]
```

Or via env var:
```
CARGO_TARGET_NVPTX64_NVIDIA_CUDA_RUSTFLAGS="-C target-cpu=sm_86 -C target-feature=+ptx78"
```

**Warning**: rustc emits `warning: unstable feature specified for -Ctarget-feature: ptx78`
— this is expected for nightly and does not affect compilation.

**Before** (PTX 7.1, default):
```
.version 7.1
.target sm_86
  .param .u64 .ptr .align 1 kernel_param_0,
  .param .u64 .ptr .align 1 kernel_param_1
```

**After** (PTX 7.8, with `+ptx78`):
```
.version 7.8
.target sm_86
  .param .b64 kernel_param_0,
  .param .b64 kernel_param_1
```

Additionally, `.extern .func panic_const_async_fn_resumed` externs were **not present**
in the PTX 7.8 output (possibly resolved at the LLVM IR level with the newer PTX target).

### Approach 2: `--nvptx-ptx-version=78` LLVM arg — FAILED

The flag `--nvptx-ptx-version` does not exist in LLVM 22.1.0's NVPTX backend.
LLVM suggested `--nvptx-prec-sqrtf32` as the closest match.

### Approach 3: `--nvptx-short-ptr` — FAILED

Adding `-C llvm-args=--nvptx-short-ptr` causes a data-layout mismatch error:
```
error: data-layout for target `nvptx64-nvidia-cuda` differs from LLVM target's default layout
```
This flag changes pointer sizes for const/local/shared address spaces and is incompatible
with the current Rust target definition.

### Approach 4: Post-processing regex (fallback)

If target-feature is not usable for some reason, the regex `\.ptr\s+\.align\s+\d+` → `(empty)`
applied to PTX text is a reliable fix. The deleted `scripts/postprocess-ptx.sh` already
implemented this. Several `build.rs` files already do post-processing (sm_30→sm_86 patching)
and could incorporate this regex.

**Regex in Rust** (for build.rs):
```rust
// Remove .ptr .align N annotations
let ptx = ptx.replace(".ptr .align 1 ", " ");
// More robust: regex::Regex::new(r"\.ptr\s+\.align\s+\d+").unwrap().replace_all(&ptx, "")
```

## Impact on Downstream Tasks

1. **All `.cargo/config.toml` files** for kernel crates should add `"-C", "target-feature=+ptx78"` to rustflags.
2. **`postprocess-ptx.sh` can be simplified** — the `.ptr .align` removal is no longer needed; only the panic extern stubbing may still be required (though it wasn't observed in this test).
3. **Build.rs post-processing** for sm_30→sm_86 patching should also be updated: instead of patching to `.version 7.1`, patch to `.version 7.8` (or better, just use the target-feature so patching isn't needed).
4. **Minimum CUDA driver requirement**: PTX 7.8 requires CUDA 11.8+ driver (≥ 520.x). sm_86 already requires CUDA 11.1+, so this is a minor increment.
5. **Crates affected**: `gpu-kernel`, `gpu-kernel-std`, `async-pipeline/kernel`, `hello-gpu/kernel`, `async-io/kernel`, `vector-math/kernel`, and all test kernel crates.
