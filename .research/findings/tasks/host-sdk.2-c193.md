# host-sdk.2: Extract reusable API types from gpu-host
**Cycle**: 193 | **Theme**: host-sdk | **Kind**: design | **Status**: done

## Summary
Analyzed gpu-host crate structure and designed its public API surface. The crate already
has a lib.rs with 6 public modules, but main.rs doesn't use it (declares private copies).
Design: make main.rs consume the library, deprecate mapped_mem.rs, add feature-gated
model/tokenizer, and add convenience re-exports.

## Findings

### Q: What types should be public: GpuDevice, KernelBuilder, HostcallRuntime?
A: The existing type names are different but serve the same role:
- **`GpuRuntime`** (runtime.rs) — wraps `Arc<CudaDevice>`, handles PTX loading + launch helpers
- **`MappedBuffer<T>`** (memory.rs) — RAII pinned device-mapped memory with volatile access
- **`HostcallBuffer`** (hostcall.rs) — hostcall buffer allocation + listener loop
- **`GpuHostError`** / **`GpuKernelErrorInfo`** (error.rs) — error hierarchy
- **`StdinSource`** trait (hostcall.rs) — extensible stdin for testing
- **`Gpt2Weights`** / **`Gpt2Tokenizer`** (model.rs, tokenizer.rs) — domain-specific, optional

These names are fine — no renaming needed.
**Confidence**: high

### Q: Can gpu-host be both a lib crate and a bin crate?
A: **Yes**, it already has both lib.rs and main.rs. But they're disconnected:
- lib.rs: `pub mod error; pub mod hostcall; pub mod memory; pub mod model; pub mod runtime; pub mod tokenizer;`
- main.rs: `mod error; mod hostcall; mod mapped_mem;` + 12 test modules (private copies)

The fix: main.rs should `use gpu_host::{error, hostcall, ...}` instead of `mod`.
- `mapped_mem.rs` is legacy (raw pointer helpers) — tests should migrate to `MappedBuffer<T>`
- This migration is non-trivial (21 test files use mapped_mem) but can be gradual
**Confidence**: high

### Q: What is the minimal public API surface?
A: Organized by layer:

```
gpu_host (crate root)
├── GpuRuntime              // re-export from runtime
├── MappedBuffer<T>         // re-export from memory
├── HostcallBuffer          // re-export from hostcall
├── GpuHostError, Result    // re-export from error
│
├── runtime                 // GpuRuntime, launch helpers
├── memory                  // MappedBuffer<T>
├── hostcall                // HostcallBuffer, StdinSource, CannedStdin, RealStdin
├── error                   // GpuHostError, GpuKernelErrorInfo, check_kernel_result()
│
├── model  [feature = "gpt2"]   // Gpt2Weights, load_gpt2_weights(), ModelError
└── tokenizer [feature = "gpt2"]// Gpt2Tokenizer, GPT2_VOCAB_SIZE, TokenizerError
```

**Feature gates:**
- `default = ["gpt2"]` — for backward compatibility
- `gpt2` — enables model + tokenizer modules + safetensors + tiktoken-rs deps

**Re-exports at crate root** (convenience):
```rust
pub use runtime::GpuRuntime;
pub use memory::MappedBuffer;
pub use hostcall::HostcallBuffer;
pub use error::{GpuHostError, Result};
```

## Design: Implementation Plan

### Step 1: Fix main.rs → library consumer (host-sdk.2 scope)
- Change `mod error;` → `use gpu_host::error;` in main.rs
- Change `mod hostcall;` → `use gpu_host::hostcall;` in main.rs
- Keep `mod mapped_mem;` (test-only, will deprecate later)
- Keep `mod tests_*;` (binary-only test modules)
- Add root re-exports in lib.rs

### Step 2: Feature-gate model/tokenizer (host-sdk.2 scope)
- Add `[features]` to Cargo.toml: `gpt2 = ["dep:safetensors", "dep:tiktoken-rs"]`
- Gate `pub mod model` and `pub mod tokenizer` behind `#[cfg(feature = "gpt2")]`

### Step 3: Standalone example (host-sdk.3 — separate task)
- New example binary depends on `gpu-host` as library
- Demonstrates: init device → load PTX → launch kernel → handle hostcall

### Step 4: Migrate mapped_mem → MappedBuffer (host-sdk.3/4 scope)
- Gradually replace `alloc_mapped_u32`/`free_mapped_mem` with `MappedBuffer<u32>`
- Delete mapped_mem.rs when all tests migrated

## ADR: gpu-host as library + binary

**Decision**: gpu-host is both a library (reusable SDK) and a binary (integration test runner).

**Rationale**:
- Library users get `GpuRuntime`, `HostcallBuffer`, `MappedBuffer` as reusable building blocks
- The binary remains the integration test runner with embedded PTX
- Feature-gating model/tokenizer keeps the core SDK lean
- Tests stay in main.rs (not lib.rs) — no test code leaks into library

**Consequences**:
- External crates can `gpu-host = { path = "..." }` to use the SDK
- model/tokenizer are opt-in via `features = ["gpt2"]`
- Tests access library types via `use gpu_host::*`, not private `mod`

## Open Questions
None — design is clear, implementation is straightforward.

## Impact on Downstream Tasks
- **host-sdk.3**: Can now create standalone example using `gpu_host::{GpuRuntime, HostcallBuffer}`
- **host-sdk.4**: Examples will demonstrate the clean public API
- **host-sdk.5**: Build automation can target the library crate
