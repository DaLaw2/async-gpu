# gpu-tls.1 — `#[thread_local]` on nvptx64-nvidia-cuda

**Cycle**: 207
**Status**: completed
**Result**: NOT SUPPORTED

## Question

Does LLVM's nvptx backend support `#[thread_local]`? If yes, what PTX address space does it map to?

## Experiment

Created a minimal test module in `crates/gpu-libc/src/tls_test.rs`:

```rust
#[thread_local]
static TLS_COUNTER: core::cell::Cell<u32> = core::cell::Cell::new(0);

#[inline(never)]
pub fn tls_increment() -> u32 {
    let val = TLS_COUNTER.get();
    TLS_COUNTER.set(val + 1);
    val
}
```

Compiled with `#![feature(thread_local)]` on nightly-2026-03-11.

## Results

### Phase 1: `cargo build` (rlib only) — PASSES

`cargo build --release --target nvptx64-nvidia-cuda -Zbuild-std=core` succeeds.
This is misleading — `rlib` only produces LLVM bitcode/metadata, not actual machine code.
The `#[thread_local]` attribute is accepted at the Rust/LLVM IR level but never lowered to PTX.

### Phase 2: `--emit=asm` (PTX codegen) — FAILS

```
rustc-LLVM ERROR: Cannot select: 0x...: i64 = GlobalTLSAddress<ptr addrspace(1) @...TLS_COUNTER> 0
In function: ...tls_increment
```

LLVM's nvptx backend has no instruction selection pattern for `GlobalTLSAddress`.
This is a hard LLVM backend limitation — no workaround at the Rust level.

## Analysis

- **`#[thread_local]` is not supported on nvptx64.** LLVM's nvptx backend does not implement TLS.
- This makes sense: CUDA's execution model has no OS-level thread-local storage.
  GPU "threads" are hardware lanes, not OS threads with TLS segments.
- The `cargo build` success is a trap — it only checks types and produces bitcode,
  the fatal error occurs during PTX codegen (instruction selection).

### PTX Address Space Mapping (for reference)

| PTX space   | LLVM addrspace | Purpose              |
|-------------|----------------|----------------------|
| `.global`   | 1              | Device global memory |
| `.shared`   | 3              | Block shared memory  |
| `.local`    | 5              | Per-thread local mem |
| `.const`    | 4              | Constant memory      |

PTX `.local` (addrspace 5) is per-thread stack memory, but it is not TLS —
it's automatic storage for function-scoped variables. There is no PTX equivalent
of `__thread` / `thread_local!`.

## Alternatives for Per-Thread State on GPU

1. **Inline PTX with `%tid` register**: Compute thread ID and index into a global array.
   ```rust
   fn thread_id() -> u32 { /* inline asm: mov.u32 %r, %tid.x */ }
   static mut THREAD_DATA: [u32; MAX_THREADS] = [0; MAX_THREADS];
   ```

2. **`.local` memory via stack allocation**: Use local variables (automatically placed in
   per-thread `.local` space by the compiler). Works for function-scoped state only.

3. **Shared memory with lane indexing**: Use `addrspace(3)` shared memory indexed by
   `threadIdx` within a block.

4. **Parameter-passing**: Thread state passed explicitly as function arguments —
   most idiomatic for Rust.

## Conclusion

`#[thread_local]` cannot be used on nvptx64. The LLVM nvptx backend fundamentally
does not support TLS. Per-thread state must use manual thread-ID indexing into
global/shared arrays, stack-local variables, or explicit parameter passing.
