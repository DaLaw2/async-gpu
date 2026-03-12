# rv5 — Deep Review: warp-macro rewrite + gpu-runtime warp_future module
**Task**: async-pipeline.2 | **Reviewer**: proposer | **Date**: 2026-03-12

## Files Reviewed
- `crates/warp-macro/src/lib.rs` (709 lines) — proc macro rewrite
- `crates/gpu-runtime/src/lib.rs` — warp_future module (lines 706-933)
- `crates/gpu-kernel/src/lib.rs` — usage site (line 1653-1657)

---

## 1. Correctness

### Memory Safety & Pointer Validity

**PASS with notes.** All pointer arithmetic uses `buf.add(offset)` with offsets
derived from `gpu_protocol` constants (`PKT_OFF_PAYLOAD`, etc.). The payload
pointer passed to `fill_payload` closures is `pkt.add(PKT_OFF_PAYLOAD)`, which
is correct — the payload starts at `PACKET_HEADER_SIZE` (32 bytes) into the
packet.

**Alignment concern (minor):** `core::ptr::write_volatile(payload as *mut u64, ...)`
assumes 8-byte alignment of the payload region. Since packets are allocated at
`PACKET_SIZE` (2112) boundaries from the buffer base, and the header is 32 bytes,
the payload is always 32-byte aligned. This is safe.

**Address space:** All pointers are in GPU global memory (CUDA mapped allocations).
The `fill_payload` closure receives a raw `*mut u8` — there is no address space
annotation, but NVPTX treats all pointers as generic by default, resolving to
global at runtime. Correct for this use case.

### Warp Convergence

**PASS.** The generated `poll_warp` broadcasts state from lane 0:
```rust
let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };
```
All lanes enter the same match arm. Within each arm:
- INIT states call `warp_hostcall_submit` which has internal `syncwarp` barriers
- WAIT states call `warp_hostcall_wait_u64` which broadcasts the result via `shfl`

All lanes return the same `WarpPoll` variant (either all `Pending` or all `Ready`),
maintaining convergence.

**Subtle correctness in WAIT arms:** The generated code is:
```rust
WAIT_STATE => unsafe {
    if let Some(val) = warp_hostcall_wait_u64(...) {
        // on_ready code
    }
    WarpPoll::Pending  // fallthrough
}
```
When the host has NOT responded, `warp_hostcall_wait_u64` returns `None`, all
lanes skip the `if let` body, and all return `Pending`. When the host HAS
responded, all lanes get `Some(val)` (broadcast), all enter the body, and the
body returns `Ready` or `Pending` (with state advanced). The fallthrough
`WarpPoll::Pending` is unreachable in the ready case because the body always
returns. **Correct.**

### State Machine Transitions

**PASS.** For N calls, states are numbered:
- Call i: INIT = 2*i, WAIT = 2*i+1
- DONE = 2*N

Transitions:
- INIT(2i) → calls `warp_hostcall_submit(..., next_state=2i+1)` → sets state to WAIT(2i+1)
- WAIT(2i+1) → calls `warp_hostcall_wait_u64(..., next_state=2(i+1))` → sets state to INIT(2(i+1)) or DONE(2N)

No off-by-one. The `done_state` arm returns `Ready(true)`, and the wildcard `_`
arm returns `Pending` as a safety net. Correct.

### Variable Capture

**PASS.** The `gen_payload_fill` function generates:
```rust
let var = self.var;  // for each known variable
```
This copies the `u64` value out of `self` before the closure body executes. Since
the closure is `FnOnce(*mut u8)` and only called by lane 0 inside
`warp_hostcall_submit`, this avoids borrow conflicts with `&mut self.state` and
`&mut self.pkt_idx`. The `known_vars_so_far` vector correctly accumulates only
variables from *prior* calls, so a variable is never read before it is written.

### Payload Layout Correctness

Comparing generated payload fills with the hand-written versions in `gpu-kernel/src/lib.rs`:

| Service | Macro gen | Hand-written | Match? |
|---------|-----------|-------------|--------|
| PRINT | slot0=msg_len, bytes@8, meta@64 | slot0=msg_len, bytes@8, meta@64 | YES |
| OPEN | slot0=(path_len \| flags<<32), path@8 | slot0=(path_len \| flags<<32), path@8 | YES |
| CLOSE | slot0=fd | slot0=fd | YES |
| READ | slot0=fd, slot1@8=max_bytes | slot0=fd, slot1@8=max_len | YES |
| WRITE | slot0=fd, slot1@8=data_len, data@16 | slot0=fd, slot1@8=data_len, data@16 | YES |
| BULK_READ | slot0=fd, slot1@8=sb_offset, slot2@16=length | slot0=fd, slot1@8=offset, slot2@16=len | YES |
| BULK_WRITE | (same as BULK_READ) | (same) | YES |

All payload layouts match the protocol spec and the existing hand-written code.

---

## 2. Soundness of Code Generation

### Crate References

**PASS.** Generated code references:
- `gpu_runtime::warp_future::*` — correct, the module is `pub mod warp_future`
- `gpu_runtime::panic::gpu_panic_init` — correct
- `gpu_protocol::SERVICE_*` — correct, re-exported via `pub use gpu_protocol`
- `gpu_protocol::NULL_INDEX` — correct
- `gpu_atomics::lane_id` — correct, re-exported via `pub use gpu_atomics`

All paths resolve because `gpu-kernel` depends on `gpu-runtime` which re-exports
`gpu-protocol` and `gpu-atomics`.

### Match Arm Exhaustiveness

**PASS.** Arms cover states 0..2N-1 (every INIT and WAIT), plus the explicit
`done_state` arm and wildcard `_`. No state is missed. The wildcard handles
impossible states defensively.

### `#[inline(always)]` Contract

**PASS.** The generated `poll_warp` and `new` methods are both `#[inline(always)]`.
The kernel entry point is `extern "ptx-kernel"` (not inlined — it's the entry).
`warp_hostcall_submit` and `warp_hostcall_wait_u64` in `gpu-runtime` are both
`#[inline(always)]`. The full call chain will be inlined into the kernel, which
is required because there is no PTX linker.

### Kernel Entry Point

**PASS.** The generated kernel mirrors the original function's parameters plus
an appended `result: *mut u32`:
```rust
pub unsafe extern "ptx-kernel" fn fn_name(buf: *mut u8, ..., result: *mut u32)
```
This is correct — the host launcher must pass the extra result pointer.

### NULL_INDEX Handling

**PASS.** `pkt_idx` is initialized to `gpu_protocol::NULL_INDEX` in the
constructor. Inside `warp_hostcall_submit`, if `hc_pop_free` returns
`NULL_INDEX`, the function returns `Pending` without modifying `pkt_idx`.
On the next poll, the same INIT state is retried. This is backpressure handling
and matches the hand-written `WarpPrintFuture` behavior.

### Missing `CONTROL_FILLED` Before Push

**ISSUE (in warp_hostcall_submit, not macro):** In `warp_hostcall_submit`
(gpu-runtime lines 861-878), the control word is set to `CONTROL_FILLED` and
the packet is pushed to the ready stack within the same `if wcx.is_leader()`
block. However, the control word is NOT first reset to 0 before being set to
`CONTROL_FILLED`. Looking more carefully:

```rust
sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
```

There is no prior `control = 0` write. Compare with `gpu_hostcall_request`
(line 213) which does:
```rust
sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, 0);  // reset
// ... fill payload ...
sys_store_release_u32(pkt.add(PKT_OFF_CONTROL) as *mut u32, CONTROL_FILLED);
```

The warp version skips the reset. If a recycled packet still has `CONTROL_READY`
set from a previous round-trip, and the host reads it between push and the
single `CONTROL_FILLED` store... actually, the host checks `CONTROL_FILLED`
only after popping from the ready stack, and the store is release-ordered before
the push. The packet was already released (control cleared to 0 by host) before
being returned to the free pool. So the control word should already be in a
clean state when popped. **Not a bug, but the defensive reset-to-0 pattern used
in the synchronous path is safer.** Severity: low.

---

## 3. API Ergonomics

### Macro API

**GOOD.** The syntax is intuitive:
```rust
#[warp_async]
unsafe fn my_pipeline(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"file.txt", FILE_OPEN_READ);
    warp_close!(buf, fd);
    warp_print!(buf, b"Done");
}
```

Each macro call maps 1:1 to a hostcall. The `let var = warp_xxx!(...)` pattern
naturally captures return values. The `buf` parameter threading is explicit,
which is appropriate for unsafe GPU code.

### Error Messages

**GOOD.** The macro produces clear compile-time errors:
- Wrong macro name: "unsupported macro `foo!`. Supported: warp_print, ..."
- Wrong arg count: "warp_open! expects 2 argument(s) after `buf`, found 1"
- Non-macro statement: "function body must contain only warp_*!() calls..."
- Destructuring: "only simple `let var = ...` bindings are supported"
- Missing init: "`let` bindings must have an initializer"

These are actionable and specific. Well done.

### Footguns

1. **`buf` name is mandatory first arg.** If the user names it `buffer`, all
   macro calls must use `buffer`. The error message explains this, but it could
   surprise users. Acceptable for now.

2. **Return type is ignored.** The macro hardcodes `ready_value = true` regardless
   of the declared return type. If a user writes `-> u32`, the generated code
   still returns `true` (which is `bool`). This would cause a type error at
   compile time, so it is caught, but the error message would be confusing.
   **Recommend: validate return type is `bool` or remove the return type
   parameter entirely.**

3. **No intermediate computation.** Users cannot write `let x = fd + 1;` or
   any non-macro statement between calls. The error is clear, but this limits
   expressiveness. Acceptable for v1.

---

## 4. Edge Cases

### Empty Function Body (0 calls)

**HANDLED.** Lines 535-542 check `calls.is_empty()` and emit a compile error:
"#[warp_async] requires at least one warp_*!() call". Good.

### Single Call

**CORRECT.** With 1 call: INIT=0, WAIT=1, DONE=2. The state machine has 3
states. The WAIT arm transitions directly to DONE and returns `Ready`. Verified
by mental execution.

### Multiple Parameters (buf + sideband)

**CORRECT.** The kernel entry point correctly mirrors all function parameters:
```rust
unsafe fn pipeline(buf: *mut u8, sideband: *mut u8) -> bool { ... }
```
Generates:
```rust
pub unsafe extern "ptx-kernel" fn pipeline(buf: *mut u8, sideband: *mut u8, result: *mut u32) { ... }
```
The struct stores both `buf` and `sideband` as fields. Payload closures can
reference `self.sideband` if the user passes it to macro args. Works correctly.

### Variable Shadowing

**ISSUE (medium).** If the user writes:
```rust
let fd = warp_open!(buf, b"a.txt", FILE_OPEN_READ);
let fd = warp_open!(buf, b"b.txt", FILE_OPEN_READ);
```
Both `fd` entries are pushed to `user_vars`, generating two `fd: u64` fields
in the struct. This would cause a **compile error** (duplicate field names).
The macro does not detect or report this — the user gets a confusing rustc
error about duplicate fields instead of a clear macro diagnostic.

**Recommendation:** Check for duplicate variable names in `user_vars` and emit
a clear error: "duplicate variable name `fd` — use distinct names for each
warp_*!() result".

### Buffer Overflow: warp_print with >56 bytes

**ISSUE (high).** The PRINT payload fill does NOT clamp `msg.len()`:
```rust
let msg: &[u8] = #msg;
let msg_len = msg.len() as u64;
core::ptr::write_volatile(payload as *mut u64, msg_len);  // writes actual length
let dst = payload.add(8);
let mut __i = 0usize;
while __i < msg.len() {  // copies ALL bytes, no clamp
    core::ptr::write_volatile(dst.add(__i), msg[__i]);
    __i += 1;
}
```

If `msg` is longer than `PRINT_MAX_MSG_LEN` (56 bytes), this writes past the
end of the lane 0 payload region into the metadata area (payload+64) and
potentially into lane 1's slots. Compare with `gpu_hostcall_print` (line 276-280)
which clamps:
```rust
let copy_len = if msg_len > PRINT_MAX_MSG_LEN as u32 { PRINT_MAX_MSG_LEN as u32 } else { msg_len };
```

**Similarly for OPEN:** path bytes are not clamped to `FILE_MAX_PATH_LEN` (56).
**And for WRITE:** data bytes are clamped to 48 via `while __i < data.len() && __i < 48`,
which is correct but hardcodes 48 instead of using `FILE_MAX_WRITE_LEN`.

**Recommendation:** Add length clamping for PRINT and OPEN, and use protocol
constants instead of magic numbers for WRITE.

### Unused Result Variable

**SAFE.** If the user writes `let _ = warp_open!(...)`, the `extract_local_ident`
function parses `_` as a valid `syn::Ident`. It becomes a struct field `_: u64`
and the assignment `self._ = val` is generated. In Rust, `_` as a field name is
actually valid syntax (though unusual). The value is written but never read.
No correctness issue, but slightly wasteful.

---

## 5. Performance

### Register Pressure

The generated struct contains:
- Function params (1-2 pointers = 2-4 registers)
- `state: u32` (1 register)
- `pkt_idx: u16` (1 register, likely widened to u32)
- User variables (1 register each, u64 = 2 regs on 32-bit NVPTX)

For a typical 3-call pipeline with 2 result vars: ~8-10 registers for struct
fields. NVPTX SM 8.6 has 255 registers per thread. This is negligible.

### Closure Capture Overhead

The `fill_payload` closure captures variables by value (u64 copies). Since
`fill_payload` is `FnOnce` and `#[inline(always)]`, the closure is completely
eliminated by inlining — no heap allocation, no vtable, just the inlined body.
**Zero overhead.**

### Comparison with Hand-Inlined Code

The hand-written `WarpPrintFuture` (gpu-kernel lines ~1170-1320) does the same
work: broadcast state, match on state, call submit/wait helpers. The macro
generates structurally identical code. After inlining, the PTX output should
be equivalent. The macro version adds no extra instructions.

The only difference: the hand-written version uses the older pattern of inlining
the entire hostcall protocol (pop, fill, push, doorbell, spin-wait) instead of
calling `warp_hostcall_submit`/`warp_hostcall_wait_u64`. The new helpers split
submit and wait into separate calls, which is actually **better** because it
yields back to the executor between submit and response, allowing the warp
scheduler to overlap with other work.

---

## 6. Verdict

**PASS with required fixes.**

### Must Fix (before merge)

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 1 | **High** | PRINT payload: no length clamp — buffer overflow if msg > 56 bytes | `gen_payload_fill` Print arm, warp-macro line 348-355 |
| 2 | **High** | OPEN payload: no path length clamp — same overflow risk | `gen_payload_fill` Open arm, warp-macro line 378-383 |
| 3 | **Medium** | WRITE payload: hardcoded `48` instead of `FILE_MAX_WRITE_LEN` constant | `gen_payload_fill` Write arm, warp-macro line 424 |

### Should Fix (recommended)

| # | Severity | Issue | Location |
|---|----------|-------|----------|
| 4 | Medium | Duplicate variable names produce confusing rustc error | `extract_warp_calls` / `user_vars` accumulation |
| 5 | Low | Return type hardcoded to `true` — misleading if user declares non-bool | `warp_async` main, line 527 |
| 6 | Low | Defensive `control = 0` reset missing in `warp_hostcall_submit` | gpu-runtime warp_future line 870 |

### Design Quality

The macro rewrite is well-structured. The separation into `ServiceKind`,
`WarpCall`, `MacroArgs`, and `gen_payload_fill` is clean and extensible.
Adding new services requires only: (1) add a `ServiceKind` variant, (2) add
`expected_args`, (3) add a `gen_payload_fill` match arm. The error messages
are specific and helpful.

The `warp_hostcall_submit` / `warp_hostcall_wait_u64` split in gpu-runtime is
a solid design that enables true cooperative scheduling — the warp yields between
submit and wait, unlike the old synchronous spin-in-place approach.

Overall: well-executed, production-ready after fixing the buffer overflow issues.
