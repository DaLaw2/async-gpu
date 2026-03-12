# bs21 — Proposer Analysis: GPU-Native Async Pipeline

**Date**: 2026-03-12
**Scope**: Deep brainstorm for the GPU-Native Async Pipeline epic — expanding `#[warp_async]` from print-only to general I/O, multi-step pipelines, and GPU-autonomous compute.

---

## 1. Systems Analysis (Memory Models, ABI, Unsafe Boundaries)

### 1.1 Warp-Level Hostcall Wrapper Design

The current hostcall infrastructure (`gpu_hostcall_request`, `gpu_hostcall_print`, etc.) is **per-thread**: a single thread pops a packet, fills all payload slots, submits, and spin-waits. For WarpFuture, we need **warp-cooperative wrappers** where:

1. **Lane 0 only** pops the packet, fills header fields (service ID, active_mask, control).
2. **All lanes** can contribute per-lane data to their respective payload slots (lanes 0-31 each own 8 × u64 slots).
3. **Lane 0 only** pushes to ready stack and rings doorbell.
4. **All lanes** spin-wait on the same control word (convergent wait — already proven in `warp_print!` codegen).
5. **All lanes** read their per-lane response from their respective payload slots.
6. **Lane 0 only** releases the packet back to the free stack.

**Key insight**: The packet payload layout already supports this perfectly. Each packet has 32 × 8 = 256 u64 slots. Currently only lane 0's slots are used (7 usable slots after the length field). But for warp-level I/O, each lane can read/write its own region at `payload_slot_offset(lane_id, slot)`.

**Warp wrapper pattern** (pseudocode for `warp_open`):
```
fn warp_open(buf, path, path_len, flags) -> (u64_fd, bool_success):
    // Lane 0: pop packet, write service=OPEN
    // Lane 0: write path + flags to lane-0 payload slots
    // syncwarp()
    // Lane 0: mark FILLED, push ready, ring doorbell
    // All lanes: spin-wait on CONTROL_READY
    // All lanes: read fd from lane-0 payload slot 0 (broadcast via shfl)
    // Lane 0: release packet
    // Return (fd, success)
```

For operations where the **result is shared** (OPEN returns one fd for the whole warp), we broadcast from lane 0. For operations where **results differ per lane** (hypothetical per-lane read), each lane reads its own slot.

### 1.2 Memory Layout: Per-Lane vs Shared Results

| Operation | Request Data | Response Data | Pattern |
|-----------|-------------|---------------|---------|
| OPEN | Shared (path, flags) | Shared (fd) | Lane 0 fills, broadcast response |
| CLOSE | Shared (fd) | Shared (status) | Lane 0 fills, broadcast response |
| WRITE | Shared or per-lane | Shared (bytes_written) | Lane 0 fills, broadcast response |
| READ | Shared (fd, max_len) | Shared (data in lane-0 slots) | Lane 0 fills, all copy from lane-0 area |
| BULK_WRITE | Shared (fd, sideband_offset, len) | Shared (bytes_written) | Lane 0 fills, broadcast response |
| BULK_READ | Shared (fd, sideband_offset, len) | Shared (bytes_read) | Lane 0 fills, broadcast response |
| PRINT | Shared (msg) | None (ack only) | Already implemented in warp_print |

**Conclusion**: All current services are "shared request, shared response" — the warp collectively does one I/O operation. Per-lane I/O (each lane reads a different file) is **not** a WarpFuture use case; that's per-thread executor territory. This is architecturally sound — WarpFuture is for uniform I/O, per-thread for divergent compute.

### 1.3 Unsafe Boundaries

The current `warp_print!` codegen in the proc macro directly emits raw pointer arithmetic, `write_volatile`, `sys_store_release_u32`, etc. This is maximally unsafe — every generated line is `unsafe`. The ergonomic improvement is that users write `warp_print!(buf, msg)` and the macro handles all the protocol details.

For new warp macros, the same pattern applies. The unsafe boundary is **the entire generated WarpFuture impl**. The trait itself is `unsafe trait WarpFuture` precisely because correctness depends on maintaining warp convergence — a property that cannot be checked at compile time.

**Recommendation**: Keep the proc macro approach. The unsafe surface area is inherent to GPU programming; the macro's value is in **preventing protocol bugs**, not in providing safety guarantees.

### 1.4 State Machine Memory Overhead

Each state in the WarpFuture state machine adds fields to the struct. Current `warp_print` uses:
- `buf: *mut u8` (8 bytes)
- `state: u32` (4 bytes)
- `pkt_idx: u16` (2 bytes)

For a multi-operation pipeline (open → read → process → write → close), we need:
- `buf: *mut u8` (8 bytes)
- `sideband: *mut u8` (8 bytes) — if bulk operations used
- `state: u32` (4 bytes)
- `pkt_idx: u16` (2 bytes)
- `fd: u64` (8 bytes) — opened file descriptor, carried across states
- Per-step result storage (variable)

Total: ~30-50 bytes per WarpFuture instance. With one instance per warp (not per lane), this is negligible. The struct lives in **registers** (LLVM will SROA it) since it's stack-allocated — no heap, no global memory access.

---

## 2. Compiler Analysis (Proc Macro Capabilities & Limitations)

### 2.1 What the Proc Macro Can Do Today

The current `#[warp_async]` proc macro:
1. Parses function body as a list of `warp_print!()` macro invocations.
2. Generates a struct + `WarpFuture` impl with 2N states (INIT + WAIT per call).
3. Generates a `ptx-kernel` entry point that creates the struct and runs `WarpExecutor::run()`.

It operates on **Rust syntax tokens** (via `syn`) — it sees macro names, identifiers, and expressions. It does NOT see types, resolved paths, or semantic information.

### 2.2 Extending to New Warp I/O Macros

**Can we add `warp_open!`, `warp_read!`, `warp_write!`, `warp_close!`, `warp_bulk_read!`, `warp_bulk_write!`?**

Yes, and the extension is mechanical. Each new macro maps to:
1. A new extraction function (like `try_extract_from_macro` for `warp_print`).
2. A new code generation template for the INIT and WAIT states.
3. Service-specific payload filling logic.
4. Service-specific response reading logic.

**Concrete design for `warp_open!(buf, path_bytes, flags) -> fd`**:

INIT state:
```rust
// Lane 0: pop packet
// Lane 0: write service=SERVICE_OPEN, path, path_len, flags to payload
// syncwarp
// Lane 0: mark FILLED, push ready, ring doorbell, advance state
// syncwarp
// return Pending
```

WAIT state:
```rust
// All lanes: broadcast pkt_idx from lane 0
// All lanes: spin-load control word
// If READY:
//   All lanes: broadcast fd from lane 0's payload slot 0
//   Lane 0: release packet, advance state
//   Store fd in self.fd for use by later states
//   syncwarp
//   return Pending (or Ready if last)
// return Pending
```

**Key difference from warp_print**: `warp_open!` has a **return value** (`fd`) that must be stored in the WarpFuture struct and accessible to later states. This requires:
- The proc macro to track `let` bindings: `let fd = warp_open!(buf, path, FLAGS_READ);`
- Each bound variable becomes a field on the generated struct.
- Later macro invocations can reference these bound variables.

### 2.3 Supporting `let` Bindings

This is the critical extension. The current proc macro rejects everything except `warp_print!()`. To support pipelines:

```rust
#[warp_async]
unsafe fn file_pipeline(buf: *mut u8, sideband: *mut u8) -> bool {
    let fd = warp_open!(buf, b"/tmp/data.txt", 0);
    let n = warp_read!(buf, fd, &mut data_buf, 256);
    // per_thread! { compute on data_buf }
    warp_write!(buf, fd, &result_buf, n);
    warp_close!(buf, fd);
}
```

The proc macro needs to:
1. Parse `Stmt::Local` (let bindings) where the RHS is a warp macro.
2. Add the bound variable as a field on the generated struct (with appropriate type).
3. In the WAIT state for that operation, store the response value into `self.field_name`.
4. In subsequent states, replace references to `field_name` with `self.field_name`.

**Complexity**: Medium. `syn` can parse `let` bindings trivially. The challenge is **type inference** — the proc macro doesn't know the type of `fd`. Solutions:
- **Option A**: Require explicit type annotations: `let fd: u64 = warp_open!(...)`. Simple, explicit.
- **Option B**: Each warp macro has a known return type (warp_open returns u64, warp_read returns usize). Hard-code in the macro.
- **Recommendation**: Option B — the return type of each warp macro is fixed and known.

### 2.4 Supporting `if` Expressions

```rust
#[warp_async]
unsafe fn conditional_pipeline(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"/tmp/input.txt", 0);
    let n = warp_read!(buf, fd, &mut data_buf, 256);
    if n > 128 {
        warp_print!(buf, b"Large data, writing output");
        warp_write!(buf, fd_out, &processed, n);
    }
    warp_close!(buf, fd);
}
```

**This is where WarpFuture semantics constrain us**. In a WarpFuture, ALL lanes must follow the same control path. An `if` expression would cause warp divergence if lanes disagree on the condition.

However, for **data-independent conditions** (e.g., `if n > 128` where `n` was broadcast from lane 0), all lanes see the same value and take the same branch. The proc macro can support this:

1. Parse `Stmt::Expr(Expr::If(...))` where the condition references only warp-broadcast values.
2. Generate a BRANCH state that evaluates the condition (broadcast from lane 0) and sets the next state accordingly.
3. Both branches must contain only warp macro calls (or be empty).

**Enforcement**: The proc macro **cannot** verify at compile time that the condition is warp-uniform. This must be a documented invariant: "Conditions in `#[warp_async]` must evaluate identically on all lanes."

**Feasibility**: Doable but adds significant complexity. Recommend deferring `if` support until the basic pipeline (linear sequence of warp macros) is proven.

### 2.5 Supporting `per_thread!` Blocks

```rust
#[warp_async]
unsafe fn hybrid_pipeline(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"/tmp/data.bin", 0);
    let n = warp_read!(buf, fd, &mut shared_buf, 1024);
    per_thread! {
        // Each lane processes its slice of shared_buf
        let my_slice = &shared_buf[lane_id * 32..(lane_id + 1) * 32];
        let result = process(my_slice);
        result_buf[lane_id] = result;
    }
    warp_bulk_write!(buf, sideband, fd_out, &result_buf, 32 * 32);
    warp_close!(buf, fd);
}
```

The proc macro can handle this by:
1. Parsing `per_thread!` as a macro invocation containing arbitrary code.
2. Generating a COMPUTE state that:
   - All lanes enter together (state broadcast from lane 0).
   - Executes the block body (arbitrary per-lane divergent code).
   - `syncwarp(active_mask)` reconverges all lanes.
   - Lane 0 advances state.
   - `syncwarp(active_mask)` ensures all lanes see new state.
3. The key constraint: **per_thread! blocks must not contain warp macros** (no yielding).

**Enforcement**: The proc macro scans the `per_thread!` body for warp macro invocations and emits a compile error if found. This IS enforceable at compile time.

### 2.6 Fundamental Limitations of Proc Macros vs Compiler Support

| Capability | Proc Macro | Compiler Support |
|-----------|-----------|-----------------|
| Linear warp macro sequences | Yes | Yes |
| Let bindings with warp macros | Yes (with type table) | Yes (with type inference) |
| Static condition branches | Partial (no uniformity check) | Could verify uniformity |
| Loops (`while`, `for`) | Very difficult | Possible (desugar into states) |
| Nested async (warp_await) | No (can't desugar futures) | Yes (like normal async/await) |
| Error propagation (`?`) | No | Yes |
| Generic over hostcall types | No | Yes |
| Correct lifetime tracking | No | Yes |

**The proc macro approach is sufficient for Phase 1** (demonstrating the vision). Moving to compiler support (Phase 2) only makes sense if the design is validated and there's demand for more complex control flow.

---

## 3. GPU Architecture Analysis

### 3.1 Conditional I/O Paths in WarpFuture

All lanes in a WarpFuture must follow the same state transitions. This means:
- **Data-independent conditions**: OK. `if file_size > THRESHOLD` where `file_size` was read from a hostcall (broadcast to all lanes) — all lanes agree.
- **Per-lane conditions**: NOT OK for I/O. `if my_data[lane_id] > threshold` — different lanes might want different I/O operations. This must be done in a per-thread block.

**The "WarpFuture for I/O, per-thread for compute" model handles this naturally**:
1. Warp reads data (all lanes get the same data, or each lane gets its slice).
2. Per-thread block: each lane independently decides what to do with its data.
3. Per-thread block results are collected.
4. Warp writes results (uniform I/O operation).

For truly heterogeneous I/O (lane 3 needs file A, lane 7 needs file B), the answer is: **don't use WarpFuture for that**. Use per-thread futures with the Embassy executor. WarpFuture is specifically for the common case where the warp collectively does uniform I/O.

### 3.2 Warp-to-Warp Divergence (Different Warps, Different Paths)

Different warps within a block can independently run different WarpFutures. This is the natural scaling model:
- Warp 0: runs an image processing pipeline (read → filter → write)
- Warp 1: runs a data validation pipeline (read → validate → report)
- Warp 2-31: run pure compute kernels

Each warp's WarpFuture is independent. No inter-warp synchronization needed (except through shared memory if desired). This is already supported by the existing architecture — the WarpExecutor is warp-local.

### 3.3 Mapping to Real Workloads

**Image processing pipeline** (1 warp):
```
OPEN(input.png) → READ(header) → [per_thread: decode pixels] →
READ(chunk) → [per_thread: apply filter] → WRITE(output_chunk) →
... repeat for all chunks ... → CLOSE
```

**ETL pipeline** (1 warp per record batch):
```
OPEN(input.csv) → BULK_READ(batch) → [per_thread: parse + validate] →
[per_thread: transform] → BULK_WRITE(output.parquet, transformed) →
PRINT(stats) → CLOSE
```

**Data validation** (1 warp per file):
```
OPEN(data.json) → BULK_READ(all) → [per_thread: validate schema] →
if errors > 0 { BULK_WRITE(errors.log, error_list) }
PRINT(summary) → CLOSE
```

These all follow the pattern: **uniform I/O (WarpFuture) interleaved with divergent compute (per_thread)**.

### 3.4 Register Pressure Analysis

Current warp_print WarpFuture struct: 14 bytes → ~4 registers.
Multi-step pipeline struct: ~50 bytes → ~13 registers.
Per-lane data buffers (e.g., 256-byte read buffer): stack-spilled to local memory.

SM86 (RTX 3060) has 65536 registers per SM and 1536 max threads per SM. At 32 regs/thread, that's full occupancy. A complex WarpFuture might use 20-25 registers for state, leaving 40+ for compute in per_thread blocks. **Register pressure is not a concern** for realistic pipeline depths (5-10 states).

For deep pipelines (20+ states), LLVM will start spilling to local memory. Local memory is L1-cached on SM86 so the performance impact is moderate (~5-10 cycles per spill vs 1 cycle for register access).

### 3.5 Occupancy Impact

Each WarpFuture instance exists once per warp (32 threads share it via lane 0). The per-warp overhead is negligible compared to per-thread overhead. A kernel with 48 warps per SM (1536 threads) would have 48 WarpFuture instances — well within register/local memory budgets.

The bigger occupancy concern is the **spin-wait loop** in `WarpExecutor::run()`. While a warp is waiting for a hostcall response (~20µs), it occupies an SM warp slot doing nothing useful. On SM86 with 48 warp slots, losing 1-2 warps to spin-wait is acceptable. Losing 40+ would crater throughput. This is why **pipelining multiple warps** is important — while warp 0 waits for I/O, warps 1-47 can compute.

---

## 4. Vision Feasibility Assessment

### 4.1 What "GPU Self-Coordinating Multi-Step Pipeline" Looks Like

**Current model** (CPU-directed):
```
CPU: launch kernel_read() → wait → launch kernel_process() → wait → launch kernel_write() → wait
```
Each step is a separate kernel launch. CPU orchestrates. Kernel launch overhead: ~5-10µs each.

**Async pipeline model** (GPU-autonomous):
```
GPU kernel (single launch):
  async {
    let fd = warp_open!(buf, "input.bin", READ);
    loop {
      let n = warp_bulk_read!(buf, sideband, fd, &mut chunk, CHUNK_SIZE);
      if n == 0 { break; }
      per_thread! { process(&mut chunk[lane_id * stride..(lane_id+1) * stride]); }
      warp_bulk_write!(buf, sideband, fd_out, &chunk, n);
    }
    warp_close!(buf, fd);
    warp_close!(buf, fd_out);
  }
```

**One kernel launch**. GPU coordinates everything. Data flows through without returning to CPU. The hostcall protocol provides the I/O, the WarpFuture state machine provides the control flow.

### 4.2 Concrete Demo: File Transform Pipeline

**Minimum viable demo** to prove the vision:

1. GPU kernel opens an input file (`/tmp/input.txt`).
2. Reads content via bulk_read into sideband buffer.
3. Per-thread block: each of 32 lanes processes a portion (e.g., to_uppercase on ASCII).
4. Writes transformed content to output file via bulk_write.
5. Prints summary message.
6. Closes both files.

This demonstrates:
- Multi-step I/O sequencing (open → read → write → close).
- Data-dependent flow (read returns N bytes, write uses N bytes).
- WarpFuture ↔ per_thread hybrid execution.
- GPU autonomy (no CPU intervention between steps).

### 4.3 Minimum Feature Set for End-to-End Demo

| Feature | Status | Needed For Demo |
|---------|--------|----------------|
| `warp_print!` in proc macro | Done | Summary output |
| `warp_open!` in proc macro | **New** | File access |
| `warp_close!` in proc macro | **New** | File cleanup |
| `warp_bulk_read!` in proc macro | **New** | Data input |
| `warp_bulk_write!` in proc macro | **New** | Data output |
| `let` bindings in proc macro | **New** | Carrying fd, byte counts |
| `per_thread!` blocks in proc macro | **New** | Compute phase |
| `sideband` parameter support | **New** | Bulk I/O needs sideband ptr |
| `if` expressions | Deferred | Not needed for linear demo |
| Loops (`while`, `for`) | Deferred | Not needed for single-pass demo |
| Persistent kernel | Deferred | Single-launch kernel suffices |

### 4.4 What This Does NOT Require

- No new hostcall services (all 13 exist and work).
- No protocol changes (packet format unchanged).
- No new crates (extend warp-macro + gpu-runtime).
- No compiler changes (proc macro only).
- No changes to host listener.

**The entire epic is a proc macro extension + a demo kernel**. The infrastructure is complete.

---

## 5. Concrete Recommendations

### 5.1 New Theme: `async-pipeline`

**Goal**: Extend `#[warp_async]` to support multi-step I/O pipelines and demonstrate GPU-autonomous compute with a concrete file-transform demo.

**Success criteria**:
1. `#[warp_async]` supports `warp_open!`, `warp_close!`, `warp_read!`, `warp_write!`, `warp_bulk_read!`, `warp_bulk_write!`.
2. `let` bindings carry results between steps.
3. `per_thread!` blocks work for divergent compute.
4. End-to-end demo: GPU opens file → reads → transforms → writes → closes, all in one kernel.

### 5.2 Tasks

#### `async-pipeline.1` — Warp-level hostcall wrappers in gpu-runtime
- **Kind**: experiment
- **Complexity**: Medium (2-3 hours)
- **Dependencies**: None
- **Description**: Implement warp-cooperative wrapper functions in `gpu_runtime::hostcall` for OPEN, CLOSE, READ, WRITE, BULK_READ, BULK_WRITE. These are the building blocks the proc macro will call. Each wrapper follows the pattern: lane 0 pops/fills/submits, all lanes spin-wait, broadcast result, lane 0 releases. Hand-written first, verify with a test kernel.
- **Deliverables**: `gpu_runtime::hostcall::warp_hostcall_open()`, `warp_hostcall_close()`, `warp_hostcall_read()`, `warp_hostcall_write()`, `warp_hostcall_bulk_read()`, `warp_hostcall_bulk_write()`.
- **Rationale**: Establish the runtime support before extending the macro. These functions can be tested independently with hand-written WarpFuture impls.

#### `async-pipeline.2` — Extend `#[warp_async]` with `let` bindings and new warp macros
- **Kind**: experiment
- **Complexity**: High (3-5 hours)
- **Dependencies**: `async-pipeline.1`
- **Description**: Extend the proc macro to:
  1. Accept `let var: type = warp_xxx!(...)` statements alongside bare `warp_print!()`.
  2. Add extraction + codegen for `warp_open!`, `warp_close!`, `warp_read!`, `warp_write!`, `warp_bulk_read!`, `warp_bulk_write!`.
  3. Generate struct fields for let-bound variables.
  4. In generated WAIT states, store response values into struct fields.
  5. In generated INIT states, reference struct fields for arguments (e.g., `self.fd` in `warp_write`).
  6. Accept `sideband: *mut u8` as optional second parameter.
- **Deliverables**: Updated `warp-macro` crate, compile-time tests.

#### `async-pipeline.3` — Add `per_thread!` block support to proc macro
- **Kind**: experiment
- **Complexity**: Medium (2-3 hours)
- **Dependencies**: `async-pipeline.2`
- **Description**: Extend `#[warp_async]` to accept `per_thread! { ... }` blocks. Generate COMPUTE states following the hybrid executor pattern (enter together → execute divergently → syncwarp → advance state → syncwarp). Verify that warp macros inside `per_thread!` are rejected at compile time.
- **Deliverables**: `per_thread!` support in warp-macro, error on nested warp macros.

#### `async-pipeline.4` — End-to-end file transform demo
- **Kind**: experiment
- **Complexity**: Medium (2-3 hours)
- **Dependencies**: `async-pipeline.3`
- **Description**: Write a complete demo kernel using `#[warp_async]`:
  ```rust
  #[warp_async]
  unsafe fn file_transform(buf: *mut u8, sideband: *mut u8) -> bool {
      let fd_in = warp_open!(buf, b"/tmp/input.txt", FILE_OPEN_READ);
      let n = warp_bulk_read!(buf, sideband, fd_in, &mut data, 1024);
      per_thread! {
          // Each lane transforms its portion (e.g., XOR, uppercase, etc.)
          let start = lane_id as usize * 32;
          let end = start + 32;
          for i in start..end {
              if i < n as usize { data[i] ^= 0x20; }
          }
      }
      let fd_out = warp_open!(buf, b"/tmp/output.txt", FILE_OPEN_WRITE_CREATE);
      let _written = warp_bulk_write!(buf, sideband, fd_out, &data, n);
      warp_print!(buf, b"Transform complete!");
      warp_close!(buf, fd_in);
      warp_close!(buf, fd_out);
  }
  ```
  Run with host listener, verify output file matches expected transformation.
- **Deliverables**: Working demo, host-side test harness, documented output.
- **Rationale**: This is the "proof of vision" — one kernel launch, GPU self-coordinates 7 I/O operations + compute.

#### `async-pipeline.5` — Conditional paths (`if` support)
- **Kind**: investigation
- **Complexity**: Medium-High (3-4 hours)
- **Dependencies**: `async-pipeline.4`
- **Description**: Investigate adding `if` expression support to `#[warp_async]`. Design the state machine transformation for branches (BRANCH state → true-path states → false-path states → join state). Document the warp-uniformity constraint and explore compile-time enforcement options. Implement if feasible.
- **Rationale**: Deferred until linear pipelines work. Conditional paths are valuable but add significant macro complexity.

#### `async-pipeline.6` — Loop support investigation
- **Kind**: investigation
- **Complexity**: High (4-5 hours)
- **Dependencies**: `async-pipeline.4`
- **Description**: Investigate `while` and `for` loop support in `#[warp_async]`. Loops are fundamentally harder than branches because the number of states is dynamic. Options: (a) unroll at compile time if bound is const, (b) encode as a loop within a single state, (c) use runtime state counter. Document tradeoffs.
- **Rationale**: Loops enable the "process all chunks" pattern needed for real workloads. But the proc macro approach may hit fundamental limits here — this investigation determines whether compiler support is needed.

#### `async-pipeline.7` — Performance characterization
- **Kind**: experiment
- **Complexity**: Medium (2-3 hours)
- **Dependencies**: `async-pipeline.4`
- **Description**: Measure end-to-end latency of the file transform demo. Break down: kernel launch time, per-hostcall round-trip time in warp-cooperative mode (vs per-thread mode), compute phase duration, total pipeline time. Compare with equivalent CPU-directed multi-kernel approach. Document the overhead of GPU autonomy.
- **Deliverables**: Latency breakdown table, comparison data, analysis of when GPU-autonomous pipelines are beneficial vs CPU-directed.

### 5.3 Priority Ordering

| Priority | Task | Rationale |
|----------|------|-----------|
| 1 | `async-pipeline.1` | Runtime support — everything depends on this |
| 2 | `async-pipeline.2` | Macro extension — enables the demo |
| 3 | `async-pipeline.3` | per_thread! — completes the hybrid model |
| 4 | `async-pipeline.4` | End-to-end demo — proves the vision |
| 5 | `async-pipeline.7` | Performance data — validates the approach |
| 6 | `async-pipeline.5` | Conditional paths — nice-to-have |
| 7 | `async-pipeline.6` | Loops — future work, may need compiler |

### 5.4 What Can Be Done NOW vs Later

**NOW** (no new research needed):
- `async-pipeline.1`: All hostcall services exist. Warp wrappers are mechanical.
- `async-pipeline.2`: Proc macro extension is well-understood. No unknowns.
- `async-pipeline.3`: Hybrid executor pattern proven in `hybrid-executor.1` and `.2`.
- `async-pipeline.4`: All pieces exist — assembly required.

**LATER** (requires more research):
- `async-pipeline.5`: Branch state machine design needs careful thought.
- `async-pipeline.6`: May hit proc macro limits; compiler support investigation.
- Persistent kernels (parked theme `persistent-kernel`): Needs CUDA cooperative launch.
- GPU task scheduling: Out of scope for this epic.

### 5.5 Themes to Park or Defer

- **`persistent-kernel`** (already parked): Keep parked. Not needed for the demo.
- **`gpu-task-sched`** (already parked): Keep parked. GPU-side task scheduling is a Phase 3 topic.
- **`inter-warp-comm`** (already parked): Keep parked. Not needed until multi-warp pipelines.

### 5.6 Risk Assessment

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|-----------|
| Warp wrapper bugs (deadlock) | Medium | High | Test each wrapper independently before macro integration |
| Proc macro complexity explosion | Low | Medium | Keep linear-only first; defer branches/loops |
| Register pressure in complex state machines | Low | Low | SM86 has 255 regs/thread; pipeline uses <30 |
| Sideband bump allocator fragmentation | Low | Medium | Reset at pipeline start; single-use pattern |
| LLVM optimizer breaks WarpFuture states | Low | High | Test with opt-level 2 and 3; verify PTX output |

---

## 6. Summary

The GPU-Native Async Pipeline epic is **highly feasible** with the current infrastructure. All 13 hostcall services work, the WarpFuture trait is proven, and the hybrid executor model (WarpFuture for I/O, per-thread for compute) has been validated.

The work is primarily a **proc macro extension** (`async-pipeline.1` through `.3`) followed by an **integration demo** (`.4`). No new crates, no protocol changes, no compiler modifications needed.

The minimum viable demo (file transform: open → bulk_read → per-thread process → bulk_write → close) can be completed in **4-6 tasks over 2-3 sessions**. This directly demonstrates the user's vision: "GPU self-coordinating its entire compute flow — data loading, conditional processing, result output — all expressed in async/await on GPU side, no CPU intervention per step."

Conditional paths and loops are valuable but should be deferred until the linear pipeline is proven. Loops may require moving beyond proc macros to compiler support, which would be a Phase 2 decision gate.
