# panic-std-intercept.1: Trace full panic path through patched std on GPU

## Summary

The std panic path on GPU is **fully functional end-to-end**: `panic!()` fires the `#[panic_handler]` in `panicking.rs`, which calls `default_hook` via `panic_with_hook`, formats the message through `Stderr` (which routes to `gpu_stdout_write` via hostcall), then calls `__rust_start_panic` which calls `process::abort()` → `core::intrinsics::abort()`. The thread name, OS ID, and panic_count TLS all work correctly via the ADR-17 gpu_tid()-indexed storage. The **key finding** is that the std path already produces a CPU-like panic message with thread name and warp ID, but then aborts the entire kernel via `trap;` (from `intrinsics::abort`) without sending the panic message through the SERVICE_PANIC hostcall — meaning the no_std path's rich metadata (blockIdx, threadIdx) is **not included**.

## Findings

### Q1: Exact code path when `panic!("msg")` fires in a std-enabled GPU kernel

**Confidence: 95%**

1. `panic!("msg")` expands to `core::panic!` which invokes the `#[panic_handler]` lang item
2. In patched std, this is `panicking::panic_handler()` (line 611 of panicking.rs)
3. `panic_handler` creates a `FormatStringPayload` or `StaticStrPayload`, extracts the `Location`
4. Calls `__rust_end_short_backtrace(|| panic_with_hook(payload, loc, can_unwind, force_no_backtrace))`
5. `panic_with_hook` (line 776):
   a. Calls `panic_count::increase(true)` — uses thread_local (works via gpu_tid()-indexed array)
   b. Checks for recursive panics (MustAbort cases)
   c. Reads `HOOK` static (RwLock<Hook>, using no_threads RwLock = Cell-based)
   d. Calls `default_hook(info)` (or custom hook if set)
6. `default_hook` (line 240):
   a. Calls `panic_output()` which returns `Some(Stderr::new())` (cuda PAL, line 77 of stdio/cuda.rs)
   b. Calls `thread::with_current_name(|name| ...)` — gets thread name from TLS
   c. Calls `thread::current_os_id()` — returns warp ID from `gpu_thread_current_id()`
   d. Formats: `"\nthread '{name}' ({tid}) panicked at {location}:\n{msg}"`
   e. Writes to `Stderr` which calls `gpu_stdout_write()` → hostcall SERVICE_PRINT
7. Calls `panic_count::finished_panic_hook()`
8. Since `panic = "abort"` is configured: calls `rust_panic(payload)` → `__rust_start_panic()`
9. `__rust_start_panic` (panic_abort crate) calls `__rust_abort()` → `process::abort()` → `abort_internal()` → `core::intrinsics::abort()`
10. On nvptx64, `core::intrinsics::abort()` emits `trap;` instruction, killing the warp

**Key insight**: The panic message **does** get printed via hostcall PRINT before the abort. The abort (trap) then kills the warp. But there is no SERVICE_PANIC hostcall in this path — the no_std panic_handler!() macro is what uses SERVICE_PANIC.

### Q2: What do `thread::with_current_name` and `thread::current_os_id` return on nvptx64?

**Confidence: 90%**

- `current_os_id()`: Calls `imp::current_os_id()` which is `sys::thread::cuda::current_os_id()` (line 101-103 of sys/thread/cuda.rs). This returns `Some(gpu_thread_current_id() as u64)`, which is the **warp index** (threadIdx.x / 32). Falls back to Rust ThreadId if unavailable.

- `with_current_name()`: Calls `try_with_current(|thread| thread.name())` via TLS. For the main warp (warp 0), if `std::thread::current()` was never explicitly set, it checks if this is the "main" thread via `main_thread::get()`. For spawned threads (via `std::thread::spawn`), the thread name is set by the ThreadInit trampoline. **For the main warp, the name will likely be `None` (displayed as `<unnamed>`)** unless std runtime initialization sets it.

### Q3: Does `panic_count` (thread_local) work correctly on GPU with ADR-17 TLS?

**Confidence: 95%**

**Yes, it works correctly.** The `panic_count` module uses `thread_local! { static LOCAL_PANIC_COUNT: Cell<(usize, bool)> }` which on CUDA (`target_os = "cuda"`) uses the `gpu_threads` TLS module. This module indexes into a `[MaybeUninit<T>; 1024]` array using `gpu_tid()` (flat thread index from PTX registers). Each warp lane has its own slot. The `GLOBAL_PANIC_COUNT` uses `AtomicUsize` which works correctly on nvptx64.

**Potential issue**: The `no_threads` RwLock (used by `HOOK` static) uses a `Cell<isize>` which is NOT thread-safe. On GPU, if two warps panic simultaneously, they could race on the RwLock's Cell. However, since all warp execution in `gpu_main_poll` is serialized through warp 0 (main) + worker warps that run one closure at a time, and panic typically kills the warp, this is unlikely to be a practical issue. Still, a concurrent panic from two warps would be UB.

### Q4: What does the final panic output look like for a std-enabled kernel?

**Confidence: 90%**

The output IS produced and looks like:

```
thread '<unnamed>' (0) panicked at src/my_kernel.rs:42:5:
my panic message
```

Where:
- `<unnamed>` or thread name if set (warp 0 main thread is likely `<unnamed>`)
- `(0)` is the warp ID (from `gpu_thread_current_id()`)
- Location comes from `#[track_caller]` — standard Rust source location
- Message is the panic payload

This is written via `Stderr::write()` → `gpu_stdout_write()` → hostcall SERVICE_PRINT → host stdout. After printing, `process::abort()` → `core::intrinsics::abort()` → PTX `trap;`.

**Important**: The `trap;` instruction will kill the entire block (all warps), not just the panicking warp. The no_std path's `set_warp_trapped()` (which sets STATUS_TRAPPED for BlockScope detection) is NOT called in the std path.

### Q5: How does `default_hook` format the panic message? Where would warp/thread ID need to be injected?

**Confidence: 95%**

The format happens at lines 262-273 of panicking.rs:

```rust
thread::with_current_name(|name| {
    let name = name.unwrap_or("<unnamed>");
    let tid = thread::current_os_id();
    writeln!(dst, "\nthread '{name}' ({tid}) panicked at {location}:\n{msg}")
});
```

The `tid` is already the warp ID (from Q2). To add GPU-specific metadata:

1. **Inject point**: Inside `default_hook`, after getting `tid`, also read `blockIdx.x` and `threadIdx.x` via gpu_runtime or inline PTX
2. **Format change**: `"\nthread '{name}' (warp {warp_id}, block {block_id}, lane {lane_id}) panicked at {location}:\n{msg}"`
3. **Alternative**: Override `current_os_id()` in `sys::thread::cuda` to encode warp+block+lane into the u64, then format it specially in default_hook

The simplest injection point is modifying the `default_hook` in patched-std to detect `target_os = "cuda"` and read GPU registers directly.

### Q6: Are there any blockers to making the patched std panic handler produce CPU-identical output with GPU thread metadata?

**Confidence: 85%**

**No hard blockers, but several issues to address:**

1. **Missing SERVICE_PANIC hostcall**: The std path uses SERVICE_PRINT (generic stdout), not SERVICE_PANIC (structured panic with metadata). The host-side `handle_panic` formats `[GPU PANIC] block=X thread=Y: msg` in red. The std path just prints plain text. **Decision needed**: Should std panics use SERVICE_PANIC for structured output, or is the SERVICE_PRINT output sufficient?

2. **Missing `set_warp_trapped()`**: The std path's `process::abort()` calls `core::intrinsics::abort()` → `trap;` without first calling `set_warp_trapped()`. This means `BlockScope::join_all()` will spin forever waiting for the panicking warp. **This is a real bug for std-enabled kernels using thread::spawn.**

3. **Missing `write_panic_to_result()`**: The std path doesn't write to the `GpuKernelResult` buffer, so the host can't inspect kernel error status programmatically. The no_std path does this.

4. **`trap;` kills the whole block**: `core::intrinsics::abort()` emits `trap;` which on CUDA kills all warps in the block. The no_std path also uses `trap;`, but it first marks the warp as trapped so other warps can detect it. In the std path, there's no warning.

5. **RwLock race (theoretical)**: The `no_threads` RwLock on HOOK is not truly thread-safe for multi-warp access. If two warps panic concurrently, they'd race on Cell<isize>. Low priority since concurrent panics are rare and the first trap kills everyone.

6. **Thread name**: Main warp likely shows `<unnamed>` rather than a meaningful name. Could be improved by setting the thread name at kernel entry.

## Unexpected Discoveries

1. **Stderr routes through stdout**: The CUDA PAL's `Stderr::write()` delegates to `gpu_stdout_write()` (same as stdout). This means panic output goes through the same SERVICE_PRINT hostcall as regular println! — no special formatting on the host side.

2. **The no_threads RwLock uses Cell**: The HOOK static uses a non-thread-safe RwLock on CUDA because it falls through to the `no_threads` implementation. This is technically unsound with multiple warps, though practically benign.

3. **`process::abort()` on CUDA = `core::intrinsics::abort()` = PTX `trap;`**: This is the "unsupported" platform fallback. It works, but it's a blunt instrument that kills the entire block.

4. **The std panic path IS functional**: Despite using the "unsupported" PAL for CUDA, the full panic formatting pipeline works because: (a) gpu_threads TLS provides per-thread state, (b) Stderr routes through gpu_stdout_write hostcall, (c) current_os_id returns warp ID, (d) panic_abort calls abort which maps to trap.

## Open Questions

1. Should the std panic path be modified to call SERVICE_PANIC instead of SERVICE_PRINT, or should it print via stderr and then call SERVICE_PANIC for structured metadata?
2. Where should `set_warp_trapped()` be injected in the std path? Options: (a) custom `process::abort()` for CUDA, (b) wrap `__rust_start_panic`, (c) custom panic hook
3. Should the thread name be set to something like `"gpu-main"` or `"warp-0"` at kernel entry?
4. How should `blockIdx` be surfaced? The current `current_os_id()` only returns warp index, not block index.

## Impact on Downstream Tasks

- **panic-std-intercept.2** (modify default_hook): The injection point is clear — lines 262-273 of panicking.rs. Need to add `cfg(target_os = "cuda")` block that reads block/thread/lane via PTX inline asm.
- **panic-std-intercept.3** (set_warp_trapped): Must hook into the abort path. Options: (a) modify `abort_internal()` in unsupported PAL for CUDA, or (b) use a custom panic hook that calls set_warp_trapped before returning.
- **panic-std-intercept.4** (write_panic_to_result): Same injection point as set_warp_trapped — needs to happen before trap.
- The RwLock soundness issue is low priority but should be tracked for the sync story.
