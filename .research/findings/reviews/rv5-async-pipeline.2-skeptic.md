# rv5 — Skeptic Review: warp-macro rewrite + warp_future module migration

**Task**: async-pipeline.2 (warp-macro rewrite + runtime refactor)
**Reviewer**: skeptic
**Date**: 2026-03-12

---

## 1. The Variable Capture Model

### 1.1 Basic capture: `let fd = warp_open!(...); warp_close!(buf, fd);`

The macro generates `let fd = self.fd;` inside the `fill_payload` closure for
`warp_close`. This is **correct** — `fd` is a `u64` field on the struct, written
by the WAIT handler of the preceding `warp_open` call, and the closure captures
a local copy.

### 1.2 Expression substitution: `warp_close!(buf, fd + 1)`

The macro treats each argument as a `syn::Expr`. The generated code for CLOSE is:

```rust
let fd = self.fd; // capture
let fd_val = fd + 1 as u64; // <-- precedence bug!
```

**BUG**: `fd + 1 as u64` parses as `fd + (1 as u64)` due to Rust operator
precedence — the `as` binds tighter than `+`. The result is still correct
*in this case* because `fd` is already `u64`, so `fd + 1u64` works. But for
more complex expressions like `fd * 2 + 1`, the `as u64` cast only applies to
the final token. This is fragile but not currently broken for u64 inputs.

**Verdict**: Low risk — expressions involving `u64` values work by accident, but
the pattern is brittle. A parenthesized cast `(#expr) as u64` would be safer.

### 1.3 Undefined variable: `warp_close!(buf, my_fd)` where `my_fd` was never defined

The macro does NOT check that referenced variables were previously defined via
`let`. The generated code will emit `let my_fd = self.my_fd;` but the struct has
no `my_fd` field — **compile error**. This is safe (caught at compile time) and
is the expected behavior.

### 1.4 Unnecessary captures

The macro captures ALL known variables before EVERY closure, even if the closure
doesn't use them. For example, if `fd` was captured from a prior call:

```rust
// PRINT closure for warp_print!(buf, b"hello")
let fd = self.fd; // captured but unused
let msg: &[u8] = b"hello";
...
```

**Verdict**: Harmless — Rust compiler will optimize away unused locals. Could
generate an `#[allow(unused)]` but not a correctness issue.

### 1.5 Variable shadowing: `let fd = warp_open!(...); let fd = warp_open!(...);`

Both calls push `fd` to `user_vars`, creating TWO `fd: u64` fields in the struct:

```rust
struct MyFuture {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
    fd: u64, // <-- duplicate field!
}
```

**BUG (BLOCKING)**: Rust does not allow duplicate field names. This will produce
a **compile error**. The macro should either reject shadowing or rename fields
(e.g., `fd`, `fd_1`). Users writing natural sequential code will hit this.

---

## 2. Warp Convergence Guarantees

### 2.1 WAIT state: who writes `self.var`?

Looking at the generated WAIT arm:

```rust
if let Some(val) = warp_hostcall_wait_u64(...) {
    if wcx.is_leader() { self.fd = val; }
    return WarpPoll::Pending;
}
WarpPoll::Pending
```

The `self.fd = val` write is correctly guarded by `is_leader()` — only lane 0
writes the struct field. **Correct.**

However, `val` is already broadcast to all lanes inside `warp_hostcall_wait_u64`
(via two `shfl_sync_idx_u32` calls for lo/hi halves). So every lane has the
correct value, but only lane 0 writes to the struct. This is fine because on the
next poll, the struct field is read by lane 0 and broadcast via the capture
mechanism.

### 2.2 Sync points: are they sufficient?

`warp_hostcall_submit` has two `syncwarp()` calls:
1. After `fill_payload` (ensures payload writes visible before header write)
2. After pushing to ready stack + doorbell (ensures state transition visible)

`warp_hostcall_wait_u64` has one `syncwarp()` after the broadcast.

The WarpExecutor adds a `syncwarp()` after every `Pending` return before the
next poll.

**Verdict**: Sufficient. The old hand-written code had the same pattern.

### 2.3 Stale `pkt_idx` on backpressure

In `warp_hostcall_submit`, if `hc_pop_free` returns `NULL_INDEX`:
- `idx` is broadcast as `NULL_INDEX`
- The function returns `WarpPoll::Pending` immediately
- `*pkt_idx_cell = idx` is NOT executed (it's after the NULL check)

Wait — looking more carefully at the code:

```rust
let idx = broadcast_u32(wcx.active_mask, idx_raw) as u16;
if idx == NULL_INDEX {
    return WarpPoll::Pending;  // early return, pkt_idx_cell NOT written
}
*pkt_idx_cell = idx;  // only reached on success
```

**Correct.** `pkt_idx_cell` is only written on successful allocation. On retry,
the INIT state runs again and gets a fresh packet. No stale index issue.

BUT: there's a subtlety. The `state_cell` is only written by lane 0 inside the
`if wcx.is_leader()` block. On backpressure, the state stays at INIT, and the
next poll re-enters the same INIT arm. **Correct behavior.**

---

## 3. Payload Layout Bugs

### 3.1 SERVICE_OPEN: macro vs protocol vs host

**Protocol comment** (gpu-protocol/src/lib.rs:265-268):
```
Slot 0: flags (u64)
Slot 1: path length (u64)
Slots 2-7: path bytes (up to 48 bytes)
```

**Host parser** (gpu-host/src/hostcall.rs:610-613):
```rust
let slot0 = read_volatile(payload as *const u64);
let path_len = (slot0 & 0xFFFF_FFFF) as usize;  // low 32 bits
let flags = (slot0 >> 32) as u32;                 // high 32 bits
```

**Macro** (warp-macro gen_payload_fill for Open):
```rust
let slot0 = (path.len() as u64) | (flags << 32);  // path_len low, flags high
write_volatile(payload as *mut u64, slot0);
// path bytes at payload+8
```

**Hand-written** (FileTransformFuture, gpu-kernel line 2343):
```rust
let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
```

The macro matches the host parser and hand-written code: slot0 = path_len(low32)
| flags(high32), path bytes at offset 8. **Correct.**

Note: the protocol COMMENT is wrong — it says slot 0 = flags, slot 1 = path_len,
but the actual implementation packs both into slot 0. The comment is stale
documentation. This is not a macro bug.

### 3.2 SERVICE_WRITE: macro vs host

**Host** (gpu-host/src/hostcall.rs:671-678):
```rust
let fd = read_volatile(payload as *const u64);           // offset 0
let data_len = read_volatile(payload.add(8) as *const u64); // offset 8
let data_ptr = payload.add(16);                           // offset 16
```

**Macro**:
```rust
write_volatile(payload as *mut u64, fd_val);        // offset 0
write_volatile(payload.add(8) as *mut u64, dlen);   // offset 8
let dst = payload.add(16);                           // offset 16
```

**Correct.** Layout matches.

### 3.3 SERVICE_BULK_READ/BULK_WRITE: macro vs host

**Macro**:
```rust
write_volatile(payload as *mut u64, fd_val);              // offset 0
write_volatile(payload.add(8) as *mut u64, sb_off_val);   // offset 8
write_volatile(payload.add(16) as *mut u64, len_val);     // offset 16
```

This matches the hand-written FileTransformFuture and the sideband module's
`gpu_bulk_write`/`gpu_bulk_read` implementations. **Correct.**

### 3.4 PRINT: lane-0 only writes

The old `gpu_hostcall_print` function (hostcall module line 251-319) writes all
bytes from a single thread — it was never a cooperative 32-lane write. The
WarpPrintFuture PoC (gpu-kernel line 1179+) was the one with cooperative writes,
but that was a hand-written experiment, not the standard path.

The macro generates lane-0-only writes (via `fill_payload` closure which is only
called on lane 0 in `warp_hostcall_submit`). This matches the standard hostcall
path. **Correct.**

---

## 4. Missing Features That Will Bite Users

### 4.1 No error handling for FILE_ERROR_SENTINEL

When `warp_open` returns `FILE_ERROR_SENTINEL` (u64::MAX), the macro stores it
as a normal `fd` value. Subsequent `warp_read!(buf, fd, 56)` will send an
invalid fd to the host, which returns an error response, but the macro stores
that error as a "bytes_read" value (also FILE_ERROR_SENTINEL = u64::MAX).

**Risk**: Silent failure propagation. The user has no way to check for errors
within the macro-generated state machine. The `ready_value` is hardcoded to
`true`, so the kernel always reports success.

**Severity**: Medium. This is a design limitation, not a bug. The macro is
currently designed for "happy path" demos. Real error handling requires either:
- A `warp_check!` macro that reads the result and branches
- Conditional state transitions based on return values
- Or: document that error handling is the user's responsibility post-completion

### 4.2 No sideband allocation within macro

`warp_bulk_read!` and `warp_bulk_write!` take `sb_offset` as an argument, but
users must call `sideband_alloc()` and `sideband_reset()` before the macro.
However, the macro rejects ANY statement that isn't a `warp_*!()` call or a
`let var = warp_*!()` binding.

**BUG (BLOCKING)**: Users CANNOT use `warp_bulk_read!` or `warp_bulk_write!`
in practice because they cannot call `sideband_alloc()` within the macro body.
They would need to pass a pre-allocated offset as a function parameter, which
works but is severely limiting and undocumented.

### 4.3 `ready_value` hardcoded to `true`

The generated `WarpPoll::Ready(true)` is always returned. If the return type
is `bool`, users cannot return `false` on error. If the return type is anything
other than `bool`, the code won't compile because `true` isn't the right type.

Wait — the return type is parsed from the function signature:
```rust
let return_type = match &input_fn.sig.output {
    ReturnType::Default => quote! { () },
    ReturnType::Type(_, ty) => quote! { #ty },
};
let ready_value = quote! { true };
```

**BUG (BLOCKING)**: If the user writes `-> u32`, the macro generates
`WarpPoll::Ready(true)` which fails because `true` is not `u32`.
Only `-> bool` (returning `true`) and `-> ()` (wait, `true` isn't `()` either)
work. Actually, even `-> ()` is broken because `true` is not `()`.

Correction: looking at the existing usage in gpu-kernel:
```rust
#[warp_macro::warp_async]
unsafe fn warp_macro_print_test(buf: *mut u8) -> bool {
```

Only `-> bool` works. Any other return type is a **compile error**. The macro
should either:
- Infer the ready value from the return type (e.g., `()` → `()`, `bool` → `true`)
- Let the user specify it
- Or document that only `-> bool` is supported

### 4.4 No `warp_stdin!` or `warp_time!`

These services exist in the protocol (`SERVICE_STDIN`, `SERVICE_TIME`) but are
not supported by the macro. Not a bug, but should be documented.

---

## 5. Breaking Changes

### 5.1 PTX output differences

The old hand-written code called `hc_pop_free`, `write_volatile`, `hc_push`
directly inline. The new macro calls `warp_hostcall_submit` which calls the
same functions. Since everything is `#[inline(always)]`, the PTX output should
be identical after inlining.

**Verdict**: No performance regression expected.

### 5.2 PRINT cooperative writes

The old `WarpPrintFuture` PoC used 32-lane cooperative byte writes where each
lane wrote `payload[8 + lane_id]`. The new macro uses lane-0-only writes.

For a 32-byte message, the old code did 32 parallel writes; the new code does
32 sequential writes on lane 0. For 56-byte messages, old did 32 parallel + 24
sequential; new does 56 sequential.

**Impact**: Negligible. The bottleneck is the host-side processing and the
spin-wait, not the payload fill. Each `write_volatile` is a single byte store
instruction — 56 sequential stores take ~56 cycles vs ~2 cycles for 32 parallel
stores. On a GPU clock of 1.5GHz, this is 37ns vs 1.3ns difference. The
hostcall round-trip is measured in microseconds.

### 5.3 Kernel entry point signature

The generated kernel includes ALL function parameters plus `result: *mut u32`:
```rust
pub unsafe extern "ptx-kernel" fn warp_macro_print_test(buf: *mut u8, result: *mut u32)
```

The existing `warp_macro_print_test` example already uses this signature.
New kernels with additional parameters (e.g., `sideband: *mut u8`) will need
the host launcher updated to pass the extra argument.

**Verdict**: Compatible with existing usage. New parameter additions are
expected to require host-side updates.

---

## 6. Verdict

### Blocking Issues (must fix before production)

1. **Variable shadowing causes compile error** (Section 1.5): Two `let fd =
   warp_open!(...)` statements produce duplicate struct fields. Fix: detect
   duplicates and either error or auto-rename.

2. **Return type locked to `bool`** (Section 4.3): `ready_value` is hardcoded
   to `true`. Any return type other than `bool` fails to compile. Fix: infer
   default from type or require explicit specification.

3. **Sideband operations impossible inside macro** (Section 4.2): Users cannot
   call `sideband_alloc()` within the `#[warp_async]` body, making
   `warp_bulk_read!`/`warp_bulk_write!` unusable without pre-allocation via
   function parameters.

### Non-blocking Issues (should fix)

4. **No error propagation** (Section 4.1): FILE_ERROR_SENTINEL silently becomes
   a normal value. Need at minimum documentation; ideally a check mechanism.

5. **Stale protocol documentation** (Section 3.1): The comment in gpu-protocol
   says slot0=flags, slot1=path_len, but the actual layout packs both into
   slot0. Should update the comment.

6. **Expression cast fragility** (Section 1.2): `#expr as u64` should be
   `(#expr) as u64` to handle complex expressions correctly.

### Assessment

The core architecture — `warp_hostcall_submit`/`warp_hostcall_wait_u64` in
gpu-runtime, called by macro-generated state machines — is **sound**. Warp
convergence is maintained correctly, payload layouts match the host parser, and
the `#[inline(always)]` chain ensures no PTX linking issues.

The macro itself works for the current happy-path demo (`warp_print!` only,
`-> bool` return type, no variable shadowing). For production use with file I/O
pipelines, the three blocking issues above must be addressed.

**Recommendation**: Fix blocking issues 1-3 before declaring the macro ready
for general use. The runtime migration (warp_future module) is clean and can
ship as-is.
