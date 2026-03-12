# gpu-panic.1: GPU Panic Handler Design
**Cycle**: 61 | **Theme**: gpu-panic | **Kind**: design | **Status**: done

## Summary

Design a GPU panic handler that routes panic messages through the existing hostcall protocol instead of the current `loop {}` infinite hang. The panic handler formats the message, encodes thread/block metadata, sends it via a new `SERVICE_PANIC` opcode, and terminates the thread with the PTX `trap` instruction.

## Design

### Service Opcode

Add `SERVICE_PANIC = 10` to gpu-protocol.

### Panic Payload Layout (lane 0)

```
Slot 0: metadata (u64)
  - Bits 15..0:  threadIdx.x (u16)
  - Bits 31..16: blockIdx.x (u16)
  - Bits 47..32: message length (u16)
  - Bits 63..48: reserved (zero)
Slot 1-7: panic message bytes (up to 56 bytes, truncated)
```

This packs all metadata into slot 0, leaving 56 bytes for the message — same capacity as SERVICE_PRINT. Thread and block IDs allow the host to identify which GPU thread panicked.

### GPU-side: `#[panic_handler]` Implementation

The panic handler needs to:

1. **Access the hostcall buffer pointer** — requires a global static `*mut u8` set by each kernel before any code that might panic. This is the same pattern used by gpu-libc for hostcall routing.

2. **Extract the message** — `PanicInfo::message()` returns `Option<&fmt::Arguments>`. On `no_std` with no allocator, we can't format into a `String`. Instead:
   - For static messages (`panic!("msg")`): extract the `&str` payload directly
   - For formatted messages (`panic!("{}", val)`): we can use a small fixed-size buffer with `core::fmt::Write`
   - Fallback: send a fixed message like `"panic occurred"` if extraction fails

3. **Send via hostcall** — Use the existing `gpu_hostcall_request` or a simplified inline version. **Critical**: the panic handler must not itself panic (no double-panic). Use a "best effort" approach:
   - If packet pool is exhausted: skip hostcall, go directly to `trap`
   - If hostcall times out: go directly to `trap`
   - No retry logic

4. **Terminate the thread** — After sending (or failing to send), execute `trap; exit;` via inline PTX assembly. The `trap` instruction signals an exception to the CUDA runtime. On SM70+, `trap` terminates the thread (not the entire kernel).

### GPU-side: Buffer Pointer Access

Two approaches:

**Option A: Global static (recommended)**
```rust
// In gpu-runtime or gpu-kernel
static mut HOSTCALL_BUF: *mut u8 = core::ptr::null_mut();

pub unsafe fn gpu_panic_init(buf: *mut u8) {
    HOSTCALL_BUF = buf;
}
```
Each kernel calls `gpu_panic_init(buf)` at the start. The panic handler reads `HOSTCALL_BUF`. If null, skip hostcall and just trap.

**Option B: Kernel parameter (not feasible)**
The `#[panic_handler]` function signature is fixed (`fn(&PanicInfo) -> !`) — there's no way to pass the buffer pointer as an argument. A global is the only option.

**Decision: Option A.** This is the same pattern already used by gpu-libc for stdin routing.

### GPU-side: Fixed-Buffer Formatting

For formatted panic messages, we need a small `no_std`-compatible formatter:

```rust
struct PanicBuf {
    buf: [u8; 56],
    pos: usize,
}

impl core::fmt::Write for PanicBuf {
    fn write_str(&mut self, s: &str) -> core::fmt::Result {
        let bytes = s.as_bytes();
        let remaining = 56 - self.pos;
        let copy_len = bytes.len().min(remaining);
        // byte-by-byte copy (no memcpy on GPU)
        for i in 0..copy_len {
            self.buf[self.pos + i] = bytes[i];
        }
        self.pos += copy_len;
        Ok(()) // Always succeed — silently truncate
    }
}
```

### Host-side: SERVICE_PANIC Handler

On the host side, when `SERVICE_PANIC` is received:

1. **Decode metadata** from slot 0: threadIdx.x, blockIdx.x, message length
2. **Read message bytes** from slots 1-7
3. **Print to stderr** with formatted output:
   ```
   [GPU PANIC] block=0 thread=5: index out of bounds: len is 10 but index is 42
   ```
4. **Set a "panic detected" flag** — the host listener should record that a panic occurred. After the kernel finishes (or is aborted), the host can report the panic to the caller.
5. **Response**: Set `CONTROL_READY` (no error bit needed — the GPU thread will trap regardless). This unblocks the GPU thread's spin-wait so it can proceed to `trap` promptly.

### Behavior with `trap` Instruction

Per NVIDIA PTX ISA documentation:
- **`trap`**: Abort execution and generate an error. On SM70+, the thread is terminated. The CUDA runtime reports `cudaErrorLaunchFailure` or `CUDA_ERROR_ILLEGAL_INSTRUCTION` depending on the error mode.
- **Impact on other threads**: `trap` terminates the executing thread's execution. Other threads in the same warp/block/grid may continue or may also be terminated depending on the GPU's error handling mode.
- **Host detection**: The host will see the kernel return with an error code. `cuStreamSynchronize` or `cuCtxSynchronize` will return a CUDA error.

**Important**: After `trap`, the GPU thread is dead — it won't release the hostcall packet. The host listener must handle orphaned packets (detect timeout and reclaim). However, since we send the hostcall BEFORE trapping and wait for the response, the packet should be properly released before trap in the normal case.

### Panic Handler Flow

```
GPU thread hits panic!("msg")
  │
  ├─ #[panic_handler] called
  │   ├─ Read HOSTCALL_BUF global
  │   ├─ If null → skip to trap
  │   ├─ Format message into PanicBuf (56 bytes max)
  │   ├─ Pop free packet (if pool exhausted → skip to trap)
  │   ├─ Fill: SERVICE_PANIC, metadata + message
  │   ├─ Push to ready stack, ring doorbell
  │   ├─ Spin-wait for CONTROL_READY (with timeout)
  │   ├─ Release packet to free stack
  │   └─ Execute `trap; exit;`
  │
Host listener
  │
  ├─ Receive SERVICE_PANIC packet
  │   ├─ Decode threadIdx.x, blockIdx.x, message
  │   ├─ Print to stderr: "[GPU PANIC] block=X thread=Y: msg"
  │   ├─ Set panic_detected flag
  │   └─ Send CONTROL_READY response
```

### Where to Implement

**GPU-side**:
- `gpu-protocol/src/lib.rs`: Add `SERVICE_PANIC = 10` and panic payload constants
- `gpu-runtime/src/lib.rs`: Add `HOSTCALL_BUF` global static, `gpu_panic_init()`, panic handler module with `register_panic_handler()` macro or `#[panic_handler]` function
- Note: `#[panic_handler]` can only be defined once per binary. Currently each kernel crate defines its own `loop {}` handler. The gpu-runtime crate should provide a proper handler, and kernel crates should remove their `loop {}` versions.

**Host-side**:
- `gpu-host/src/hostcall.rs`: Add `SERVICE_PANIC` arm to dispatch match, `handle_panic()` method

### Caveats and Limitations

1. **56-byte message limit**: Panic messages longer than 56 bytes are truncated. This is sufficient for most error messages but formatted backtraces won't fit.

2. **Double-panic**: If the panic handler itself encounters an error (pool exhaustion, timeout), it must NOT panic — it skips to `trap` silently.

3. **Multi-thread panic**: Multiple threads can panic simultaneously. Each sends its own SERVICE_PANIC packet. The host prints all of them.

4. **Async context**: If a panic occurs inside an Embassy future, the executor won't run further. The panic handler sends the hostcall synchronously (spin-wait), so it works regardless of the execution context.

5. **std crates**: For kernel crates using `-Zbuild-std=std`, Rust's std has its own panic machinery that ultimately calls the platform's `abort()`. The gpu-libc `abort()` already calls `trap`. We need to ensure the std panic path also routes through our hostcall-based handler — this may require patching the vendored std's panic hook.

## Findings

### Q: How to encode panic message in hostcall packet (SERVICE_PANIC opcode)?
A: New `SERVICE_PANIC = 10` opcode. Metadata (threadIdx, blockIdx, msg_len) packed into slot 0, message bytes in slots 1-7. Same 56-byte capacity as PRINT.
**Confidence**: high

### Q: Can trap instruction terminate just the panicking thread?
A: Yes, on SM70+. `trap` terminates the executing thread. Other threads may continue depending on error mode. The CUDA runtime reports an error when the host synchronizes.
**Confidence**: high

### Q: How does host detect and report panic (vs normal hostcall)?
A: Host receives SERVICE_PANIC packet, decodes metadata, prints to stderr with `[GPU PANIC]` prefix. Sets a `panic_detected` flag. After kernel completes, CUDA runtime also returns an error code due to `trap`.
**Confidence**: high

### Q: What metadata to include (thread ID, block ID, warp ID)?
A: threadIdx.x and blockIdx.x packed into slot 0. Warp ID can be derived from threadIdx.x (thread / 32) so it's not separately needed. Message length also in slot 0.
**Confidence**: high

## Unexpected Discoveries

- The `#[panic_handler]` can only be defined once per binary. Currently 5+ kernel crates each define their own `loop {}` handler. The gpu-runtime crate providing a proper handler means all kernel crates must remove their duplicate definitions — this is a positive cleanup.
- The panic handler in std-build-test crates goes through std's own panic machinery, which calls our gpu-libc `abort()` → `trap`. For full message reporting in std crates, we'd need to hook into std's panic hook. This is a stretch goal, not required for the initial implementation.

## Open Questions

- Should the host listener attempt to kill/abort the kernel after receiving a panic? Or just report and let `trap` handle it?
- For std-crates: can we set a custom `std::panic::set_hook()` that routes through hostcall before the default handler calls abort?

## Impact on Downstream Tasks

- **gpu-panic.2**: Direct implementation guide — all design decisions made.
- **All kernel crates**: Will need to remove `loop {}` panic handlers and call `gpu_panic_init(buf)` at kernel entry.
- **gpu-runtime**: Gets a new `panic` module with the handler implementation.
- **host-scaling**: Panic reporting will work correctly with multi-threaded listener (each SERVICE_PANIC is an independent packet).
