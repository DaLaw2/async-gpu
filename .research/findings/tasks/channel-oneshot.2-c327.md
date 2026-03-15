# channel-oneshot.2: Implement Oneshot<T> sender + receiver on GPU
**Cycle**: 327 | **Theme**: channel-oneshot | **Kind**: experiment | **Status**: done

## Summary
Implemented `OneshotSlot<T>`, `OneshotSender<T>`, `OneshotReceiver<T>`, and `oneshot()` factory
function in `gpu_runtime::channel` module. Compiles to valid PTX on nvptx64 target.

## Implementation

### Module: `gpu_runtime::channel` (lib.rs line ~4342)

**OneshotSlot<T: Copy>** — `#[repr(C)]` pre-allocated slot:
- `state: UnsafeCell<u32>` — atomic state (EMPTY=0, SENT=1, CLOSED=2)
- `_pad: u32` — alignment padding for 8-byte aligned value
- `value: UnsafeCell<MaybeUninit<T>>` — inline storage

**OneshotSender<T: Copy>** — sending half:
- `send(self, value: T)` — writes value via `write_volatile`, then `sys_store_release_u32(SENT)`
- `forget(self)` after send to prevent Drop from setting CLOSED
- Drop impl: sets state to CLOSED (sender dropped without sending)

**OneshotReceiver<T: Copy>** — receiving half, implements `Future`:
- `poll()` reads state via `sys_load_acquire_u32`
- SENT → read value via `read_volatile`, return `Poll::Ready(Ok(value))`
- CLOSED → return `Poll::Ready(Err(OneshotClosed))`
- EMPTY → return `Poll::Pending`

**oneshot<T: Copy>(slot)** — factory function:
- Resets slot state to EMPTY
- Returns `(OneshotSender, OneshotReceiver)` pair

### Key Design Decisions
1. **T: Copy bound** — no Drop support on GPU, avoids destructor complexity
2. **Pre-allocated slot** — no heap, caller provides memory
3. **No CAS needed** — SPSC pattern, plain acquire/release sufficient
4. **Sender Drop → CLOSED** — receiver detects abandoned channel
5. **`core::mem::forget` after send** — prevents double state write

## Verification
- Compiled successfully with `cargo +nightly-2026-03-11 build --release` (nvptx64 target)
- No warnings or errors in gpu-runtime crate

## Impact on Downstream Tasks
- **channel-oneshot.3**: Add integration with executor — spawn tasks that use oneshot channels
- **channel-waker.2**: Extend receiver to store waker for wake-on-send
