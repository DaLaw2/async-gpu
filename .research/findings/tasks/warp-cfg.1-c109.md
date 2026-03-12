# warp-cfg.1: Design CFG-based state machine generation
**Cycle**: 109 | **Theme**: warp-cfg | **Kind**: design | **Status**: done

## Summary

This document designs the CFG (Control Flow Graph) extension for the `#[warp_async]` proc macro, enabling `if`/`else`, `loop`/`break`, and `match` constructs that contain `warp_*!()` calls. The core constraint is warp uniformity: lane 0 evaluates all branch conditions and broadcasts the decision to all 32 lanes via `shfl.sync.idx.b32`, so every lane takes the same path. The design introduces a DAG-based state numbering scheme, cross-yield variable promotion to struct fields, and explicit broadcast gates at every branch point.

## Design

### 1. CFG Representation

The macro internally builds a **Control Flow Graph** (CFG) before assigning state numbers. Each node in the CFG represents either a *compute block* (pure Rust with no warp calls) or a *yield point* (a `warp_*!()` call that becomes an INIT+WAIT state pair).

```
struct CfgGraph {
    nodes: Vec<CfgNode>,
    edges: Vec<CfgEdge>,
    entry: NodeId,
    exit: NodeId,         // synthetic "Done" node
}

enum CfgNode {
    /// A warp_*!() call — becomes an INIT + WAIT state pair.
    YieldPoint {
        id: NodeId,
        call: WarpCall,
    },
    /// Pure compute (assignments, expressions) — no state needed;
    /// inlined into the preceding or following state's match arm.
    Compute {
        id: NodeId,
        stmts: Vec<Stmt>,
    },
    /// Branch decision point — lane 0 evaluates condition, broadcasts choice.
    Branch {
        id: NodeId,
        kind: BranchKind,
    },
    /// Merge point after branch arms reconverge.
    Join {
        id: NodeId,
    },
    /// Loop header — target of back-edges.
    LoopHeader {
        id: NodeId,
    },
    /// Synthetic entry/exit.
    Entry,
    Exit,
}

enum BranchKind {
    IfElse {
        /// The condition expression (evaluated by lane 0, broadcast as u32 0/1).
        cond: Expr,
        then_entry: NodeId,
        else_entry: Option<NodeId>,
    },
    Match {
        /// The scrutinee expression.
        scrutinee: Expr,
        /// (pattern, arm_entry_node) pairs. Pattern index = broadcast value.
        arms: Vec<(Pat, NodeId)>,
    },
    LoopContinueOrBreak {
        /// Condition for break (None = unconditional break).
        break_cond: Option<Expr>,
        continue_target: NodeId,  // back-edge to LoopHeader
        break_target: NodeId,     // forward-edge past loop
    },
}

struct CfgEdge {
    from: NodeId,
    to: NodeId,
    kind: EdgeKind,
}

enum EdgeKind {
    Fallthrough,
    BranchTrue,
    BranchFalse,
    BackEdge,   // loop continue
    BreakEdge,  // loop break
}
```

**Key invariant**: Every `Branch` node must have a corresponding `Join` node where control flow reconverges. The macro enforces that all branch arms that contain `warp_*!()` calls are structurally balanced — both arms exist and both converge to the same join point.

### 2. Parsing Strategy

The macro parses the function body using `syn`'s AST, extending the current `extract_warp_calls` to a recursive `build_cfg` function that walks `Vec<Stmt>` and builds the CFG.

#### Statement classification

Each statement in the function body is classified into one of:

| Statement type | CFG treatment |
|---|---|
| `warp_*!(...)` or `let x = warp_*!(...)` | YieldPoint node |
| `let x = <non-warp-expr>;` | Compute node (accumulated) |
| `x = <expr>;` (assignment) | Compute node (accumulated) |
| `if cond { ... } else { ... }` | Branch + recursive descent into arms |
| `loop { ... }` | LoopHeader + recursive descent into body |
| `match expr { ... }` | Branch + recursive descent into arms |
| `break` / `break <expr>` | BreakEdge to post-loop Join |
| `continue` | BackEdge to LoopHeader |
| Other expressions | Compute node (if no warp calls inside) |

#### Recursive descent algorithm

```
fn build_cfg(stmts: &[Stmt], graph: &mut CfgGraph, current: NodeId) -> NodeId:
    let acc_compute = vec![]  // accumulate compute stmts

    for stmt in stmts:
        match classify(stmt):
            WarpCall(call) =>
                // Flush accumulated compute as a Compute node
                if !acc_compute.is_empty():
                    let cn = graph.add(Compute { stmts: acc_compute.drain() })
                    graph.edge(current, cn)
                    current = cn

                let yp = graph.add(YieldPoint { call })
                graph.edge(current, yp)
                current = yp

            ComputeStmt(s) =>
                acc_compute.push(s)

            IfElse(cond, then_stmts, else_stmts) =>
                // Check: does either arm contain warp_*!() calls?
                let then_has_yield = contains_warp_call(&then_stmts)
                let else_has_yield = else_stmts.map(|s| contains_warp_call(s))

                if !then_has_yield && !else_has_yield.unwrap_or(false):
                    // No yields — treat entire if/else as compute
                    acc_compute.push(stmt)
                    continue

                // Flush compute
                flush_compute(...)

                // Create Branch + Join nodes
                let branch = graph.add(Branch { IfElse { cond, ... } })
                let join = graph.add(Join {})
                graph.edge(current, branch)

                let then_exit = build_cfg(then_stmts, graph, branch)
                graph.edge(then_exit, join)

                if let Some(else_stmts) = else_stmts:
                    let else_exit = build_cfg(else_stmts, graph, branch)
                    graph.edge(else_exit, join)
                else:
                    // ERROR: if with yields in then-arm MUST have else arm
                    // (otherwise some states would be skipped non-uniformly)
                    compile_error!(...)

                current = join

            Loop(body_stmts) =>
                flush_compute(...)
                let header = graph.add(LoopHeader {})
                graph.edge(current, header)

                // build_cfg for loop body, with header as context for break/continue
                let body_exit = build_cfg_loop(body_stmts, graph, header)
                // body_exit naturally has a back-edge to header
                graph.edge(body_exit, header, BackEdge)

                let post_loop = graph.add(Join {})
                // break edges target post_loop (set during build_cfg_loop)
                current = post_loop

            Match(scrutinee, arms) =>
                // Similar to IfElse but with N arms
                ...

    // Flush remaining compute
    if !acc_compute.is_empty():
        let cn = graph.add(Compute { stmts: acc_compute })
        graph.edge(current, cn)
        current = cn

    return current
```

#### `contains_warp_call` helper

A quick recursive check on `TokenStream` / `Vec<Stmt>` that returns `true` if any `warp_*!()` macro invocation appears anywhere inside. This is used to decide whether an `if`/`loop`/`match` needs CFG treatment or can be treated as opaque compute.

### 3. State Assignment

States are assigned by a post-order traversal of the CFG. Each `YieldPoint` node gets two consecutive state numbers (INIT, WAIT), exactly as the current linear scheme. Non-yield nodes (Compute, Branch, Join, LoopHeader) do NOT get their own states — they are inlined into adjacent states' match arms.

#### Algorithm

```
fn assign_states(graph: &CfgGraph) -> HashMap<NodeId, (u32, u32)>:
    let counter = 0
    let states = HashMap::new()

    // Topological order (respecting back-edges for loops)
    for node in graph.topo_order():
        match node:
            YieldPoint { id, .. } =>
                states[id] = (counter, counter + 1)  // (INIT, WAIT)
                counter += 2

            // Branch, Join, LoopHeader, Compute — no state numbers
            // They are folded into the WAIT arm of the preceding yield,
            // or the INIT arm of the following yield.
            _ => ()

    let done_state = counter
    return (states, done_state)
```

**Branch decisions** are inlined into the preceding state's match arm. When the WAIT state of a yield point completes, and the next thing is a Branch node, the branch evaluation + broadcast happens right there inside that WAIT arm before transitioning to the next INIT state.

#### State numbering example for `if/else`:

```
// Source:
let fd = warp_open!(...);        // states 0(INIT), 1(WAIT)
if condition {
    warp_print!(...);            // states 2(INIT), 3(WAIT)
} else {
    warp_write!(...);            // states 4(INIT), 5(WAIT)
}
warp_close!(...);                // states 6(INIT), 7(WAIT)
// Done = 8
```

The branch decision happens inside state 1 (WAIT for `warp_open`). When the host responds, state 1's match arm:
1. Stores the fd result
2. Evaluates `condition` on lane 0
3. Broadcasts the choice (0 or 1) to all lanes
4. Transitions to state 2 or state 4 based on the broadcast value

### 4. Variable Scoping

Variables that are live across yield points become fields of the generated struct. The current implementation already does this for `let x = warp_*!()` bindings. The extension handles:

#### 4a. Variables defined in compute blocks before yields

```rust
let x = some_computation();  // compute
warp_print!(..., x);         // yield — x must survive across the state transition
```

Any `let` binding in a `Compute` node whose value is referenced in a later node (across a yield boundary) is **promoted** to a `u64` struct field, just like warp call results.

#### 4b. Variables defined in one branch arm, used after join

Variables defined inside a branch arm are only valid inside that arm. If a variable defined in `then` is referenced after the join point, it's a compile error — the macro cannot know which arm executed (even though at runtime it's deterministic, the Rust type system sees both paths).

**Rule**: Variables defined inside branch arms are scoped to those arms. To pass data out of an `if`/`match`, define the variable before the branch and assign inside:

```rust
let mut result: u64 = 0;         // promoted to struct field
if condition {
    result = warp_read!(...);     // assigns in then-arm
} else {
    result = warp_read!(...);     // assigns in else-arm
}
warp_print!(..., result);         // uses result after join
```

#### 4c. Loop variables

Variables modified inside a loop body that are referenced after `break` follow the same promotion rule. The variable must be declared before the loop and becomes a struct field.

#### Liveness analysis

The macro performs a simple liveness analysis on the CFG:

1. For each variable defined in a `Compute` or `YieldPoint` node, check if it is referenced in any node that is separated by at least one yield point.
2. If yes, promote it to a struct field (type `u64`).
3. If no, keep it as a local variable within the match arm.

**Implementation shortcut**: Since the current macro already requires all `let` bindings to be `warp_*!()` results (all `u64`), the extension can initially require that all cross-yield variables are explicitly typed as `u64` and declared at function scope. A future enhancement could add type inference.

### 5. Branch Broadcast Protocol

The critical mechanism: lane 0 evaluates the condition, broadcasts the result, and all lanes use the broadcast value to pick the same branch.

#### If/Else Protocol

Generated code inside a WAIT arm (after hostcall completion):

```rust
// Inside state N (WAIT for preceding yield):
// ... hostcall result handling ...

// Branch decision
let mut __branch: u32 = 0;
if wcx.is_leader() {
    __branch = if /* user condition */ { 1 } else { 0 };
}
let __branch = unsafe { broadcast_u32(wcx.active_mask, __branch) };
if wcx.is_leader() {
    self.state = if __branch != 0 { THEN_INIT_STATE } else { ELSE_INIT_STATE };
}
return WarpPoll::Pending;
```

On the next poll, all lanes read `self.state` via `broadcast_u32` (already done at the top of `poll_warp`) and enter the correct match arm.

#### Match Protocol

For match with N arms, lane 0 evaluates the scrutinee and determines which arm index (0..N-1) matches. That index is broadcast:

```rust
let mut __arm: u32 = 0;
if wcx.is_leader() {
    __arm = match /* scrutinee */ {
        pattern_0 => 0,
        pattern_1 => 1,
        pattern_2 => 2,
        _ => N,  // default/wildcard
    };
}
let __arm = unsafe { broadcast_u32(wcx.active_mask, __arm) };
if wcx.is_leader() {
    self.state = match __arm {
        0 => ARM_0_INIT_STATE,
        1 => ARM_1_INIT_STATE,
        2 => ARM_2_INIT_STATE,
        _ => ARM_DEFAULT_INIT_STATE,
    };
}
return WarpPoll::Pending;
```

#### Loop Protocol

For `loop { ... }`, the back-edge is a state transition from the last yield in the loop body back to the first yield's INIT state. For `break`, it transitions to the post-loop state.

Break condition broadcast:

```rust
// At the break decision point inside the loop:
let mut __do_break: u32 = 0;
if wcx.is_leader() {
    __do_break = if /* break condition */ { 1 } else { 0 };
}
let __do_break = unsafe { broadcast_u32(wcx.active_mask, __do_break) };
if wcx.is_leader() {
    self.state = if __do_break != 0 { POST_LOOP_STATE } else { LOOP_BODY_INIT_STATE };
}
return WarpPoll::Pending;
```

### 6. Code Generation Examples

#### 6.1 Simple if/else with warp_*!() in both arms

**Input:**
```rust
#[warp_async]
unsafe fn branch_example(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"data.txt", FILE_OPEN_READ);
    if fd != 0 {
        warp_print!(buf, b"File opened");
    } else {
        warp_print!(buf, b"Open failed");
    }
    warp_close!(buf, fd);
}
```

**Generated state machine:**
- State 0: INIT — submit `warp_open`
- State 1: WAIT — receive fd, evaluate `fd != 0`, broadcast, transition to 2 or 4
- State 2: INIT — submit `warp_print("File opened")`
- State 3: WAIT — receive ack, transition to 6
- State 4: INIT — submit `warp_print("Open failed")`
- State 5: WAIT — receive ack, transition to 6
- State 6: INIT — submit `warp_close`
- State 7: WAIT — receive ack, transition to Done(8)

**Generated code (abbreviated):**
```rust
struct BranchExample {
    buf: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
}

impl BranchExample {
    #[inline(always)]
    fn new(buf: *mut u8) -> Self {
        Self { buf, state: 0, pkt_idx: gpu_protocol::NULL_INDEX, fd: 0 }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for BranchExample {
    type Output = bool;

    #[inline(always)]
    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{WarpPoll, broadcast_u32};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // State 0: INIT warp_open
            0 => unsafe {
                gpu_runtime::warp_future::warp_hostcall_submit(
                    self.buf, wcx, gpu_protocol::SERVICE_OPEN,
                    |payload| {
                        let path: &[u8] = b"data.txt";
                        let flags = (gpu_protocol::FILE_OPEN_READ) as u64;
                        let path_len = if path.len() > gpu_protocol::FILE_MAX_PATH_LEN {
                            gpu_protocol::FILE_MAX_PATH_LEN
                        } else { path.len() };
                        let slot0 = (path_len as u64) | (flags << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut __i = 0usize;
                        while __i < path_len {
                            core::ptr::write_volatile(dst.add(__i), path[__i]);
                            __i += 1;
                        }
                    },
                    1, &mut self.state, &mut self.pkt_idx,
                )
            }

            // State 1: WAIT warp_open + BRANCH decision
            1 => unsafe {
                if let Some(val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    // next_state is a placeholder — we override below
                    1, &mut self.state,
                ) {
                    if wcx.is_leader() { self.fd = val; }
                    // Branch: lane 0 evaluates condition, broadcasts
                    let mut __branch: u32 = 0;
                    if wcx.is_leader() {
                        __branch = if self.fd != 0 { 1 } else { 0 };
                    }
                    let __branch = broadcast_u32(wcx.active_mask, __branch);
                    if wcx.is_leader() {
                        self.state = if __branch != 0 { 2 } else { 4 };
                    }
                    return WarpPoll::Pending;
                }
                WarpPoll::Pending
            }

            // State 2: INIT warp_print("File opened") — then arm
            2 => unsafe {
                gpu_runtime::warp_future::warp_hostcall_submit(
                    self.buf, wcx, gpu_protocol::SERVICE_PRINT,
                    |payload| { /* fill "File opened" */ },
                    3, &mut self.state, &mut self.pkt_idx,
                )
            }

            // State 3: WAIT — then arm complete, go to join state 6
            3 => unsafe {
                if let Some(_val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    6, &mut self.state,
                ) {
                    return WarpPoll::Pending;
                }
                WarpPoll::Pending
            }

            // State 4: INIT warp_print("Open failed") — else arm
            4 => unsafe {
                gpu_runtime::warp_future::warp_hostcall_submit(
                    self.buf, wcx, gpu_protocol::SERVICE_PRINT,
                    |payload| { /* fill "Open failed" */ },
                    5, &mut self.state, &mut self.pkt_idx,
                )
            }

            // State 5: WAIT — else arm complete, go to join state 6
            5 => unsafe {
                if let Some(_val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    6, &mut self.state,
                ) {
                    return WarpPoll::Pending;
                }
                WarpPoll::Pending
            }

            // State 6: INIT warp_close — post-join
            6 => unsafe {
                gpu_runtime::warp_future::warp_hostcall_submit(
                    self.buf, wcx, gpu_protocol::SERVICE_CLOSE,
                    |payload| {
                        let fd = self.fd;
                        let fd_val = (fd) as u64;
                        core::ptr::write_volatile(payload as *mut u64, fd_val);
                    },
                    7, &mut self.state, &mut self.pkt_idx,
                )
            }

            // State 7: WAIT warp_close — done
            7 => unsafe {
                if let Some(_val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
                    self.buf, wcx, self.pkt_idx,
                    8, &mut self.state,
                ) {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            }

            // Done
            8 => WarpPoll::Ready(true),
            _ => WarpPoll::Pending,
        }
    }
}
```

#### 6.2 Loop with warp_*!() and break

**Input:**
```rust
#[warp_async]
unsafe fn read_loop(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"input.txt", FILE_OPEN_READ);
    loop {
        let n = warp_read!(buf, fd, 256);
        if n == 0 {
            break;
        }
        warp_print!(buf, b"Read chunk");
    }
    warp_close!(buf, fd);
}
```

**State assignment:**
- State 0/1: warp_open INIT/WAIT
- State 2/3: warp_read INIT/WAIT (loop body start)
- State 3 includes break-decision: if n == 0, goto 6 (post-loop); else goto 4
- State 4/5: warp_print INIT/WAIT
- State 5 WAIT transitions back to 2 (loop back-edge)
- State 6/7: warp_close INIT/WAIT (post-loop)
- Done = 8

**Key generated arms:**

```rust
// State 3: WAIT warp_read + loop break decision
3 => unsafe {
    if let Some(val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
        self.buf, wcx, self.pkt_idx,
        3, &mut self.state, // placeholder
    ) {
        if wcx.is_leader() { self.n = val; }
        // Break decision
        let mut __do_break: u32 = 0;
        if wcx.is_leader() {
            __do_break = if self.n == 0 { 1 } else { 0 };
        }
        let __do_break = broadcast_u32(wcx.active_mask, __do_break);
        if wcx.is_leader() {
            self.state = if __do_break != 0 { 6 } else { 4 };
        }
        return WarpPoll::Pending;
    }
    WarpPoll::Pending
}

// State 5: WAIT warp_print — back-edge to loop header (state 2)
5 => unsafe {
    if let Some(_val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
        self.buf, wcx, self.pkt_idx,
        2, &mut self.state,  // back-edge: go to loop header
    ) {
        return WarpPoll::Pending;
    }
    WarpPoll::Pending
}
```

#### 6.3 Match with warp_*!() in arms

**Input:**
```rust
#[warp_async]
unsafe fn match_example(buf: *mut u8) -> bool {
    let status = warp_read!(buf, 1, 8);
    match status {
        0 => { warp_print!(buf, b"EOF"); }
        1 => { warp_print!(buf, b"OK"); }
        _ => { warp_print!(buf, b"Error"); }
    }
}
```

**State assignment:**
- State 0/1: warp_read INIT/WAIT
- State 1 includes match dispatch: broadcast arm index (0, 1, or 2)
- State 2/3: warp_print("EOF") — arm 0
- State 4/5: warp_print("OK") — arm 1
- State 6/7: warp_print("Error") — arm 2 (wildcard)
- All three WAIT states (3, 5, 7) transition to Done(8)

**Key generated code:**

```rust
// State 1: WAIT warp_read + match dispatch
1 => unsafe {
    if let Some(val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
        self.buf, wcx, self.pkt_idx,
        1, &mut self.state,
    ) {
        if wcx.is_leader() { self.status = val; }
        let mut __arm: u32 = 0;
        if wcx.is_leader() {
            __arm = match self.status {
                0 => 0u32,
                1 => 1u32,
                _ => 2u32,
            };
        }
        let __arm = broadcast_u32(wcx.active_mask, __arm);
        if wcx.is_leader() {
            self.state = match __arm {
                0 => 2,
                1 => 4,
                _ => 6,
            };
        }
        return WarpPoll::Pending;
    }
    WarpPoll::Pending
}
```

#### 6.4 Nested: if inside loop

**Input:**
```rust
#[warp_async]
unsafe fn nested_example(buf: *mut u8) -> bool {
    let fd = warp_open!(buf, b"log.txt", FILE_OPEN_READ);
    loop {
        let n = warp_read!(buf, fd, 512);
        if n == 0 {
            break;
        }
        if n > 256 {
            warp_print!(buf, b"Large chunk");
        } else {
            warp_print!(buf, b"Small chunk");
        }
    }
    warp_close!(buf, fd);
}
```

**State assignment:**
- 0/1: warp_open INIT/WAIT
- 2/3: warp_read INIT/WAIT (loop header yield)
- 3: break decision (n == 0 → goto 10, else evaluate inner if)
- 3: inner if decision (n > 256 → goto 4, else goto 6)
  - *Note*: Both decisions happen in state 3's match arm sequentially. First the break check, then if not breaking, the branch check. Two broadcasts in the same arm.
- 4/5: warp_print("Large chunk") INIT/WAIT
- 6/7: warp_print("Small chunk") INIT/WAIT
- 5 and 7: both back-edge to state 2 (loop continue)
- 8/9: warp_close INIT/WAIT (post-loop, but renumbered to 8/9 since 10=done... let's recalculate)

Actually, with proper numbering:
- 0/1: warp_open
- 2/3: warp_read (loop body)
- 4/5: warp_print("Large chunk")
- 6/7: warp_print("Small chunk")
- 8/9: warp_close
- Done = 10

State 3 match arm:

```rust
3 => unsafe {
    if let Some(val) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
        self.buf, wcx, self.pkt_idx,
        3, &mut self.state,
    ) {
        if wcx.is_leader() { self.n = val; }

        // First decision: break?
        let mut __do_break: u32 = 0;
        if wcx.is_leader() {
            __do_break = if self.n == 0 { 1 } else { 0 };
        }
        let __do_break = broadcast_u32(wcx.active_mask, __do_break);

        if __do_break != 0 {
            // Break path
            if wcx.is_leader() { self.state = 8; }
            return WarpPoll::Pending;
        }

        // Second decision: large or small?
        let mut __branch: u32 = 0;
        if wcx.is_leader() {
            __branch = if self.n > 256 { 1 } else { 0 };
        }
        let __branch = broadcast_u32(wcx.active_mask, __branch);
        if wcx.is_leader() {
            self.state = if __branch != 0 { 4 } else { 6 };
        }
        return WarpPoll::Pending;
    }
    WarpPoll::Pending
}
```

Both WAIT states 5 and 7 transition back to state 2 (loop continue):

```rust
5 => unsafe {
    if let Some(_) = gpu_runtime::warp_future::warp_hostcall_wait_u64(
        self.buf, wcx, self.pkt_idx, 2, &mut self.state,
    ) { return WarpPoll::Pending; }
    WarpPoll::Pending
}
// State 7 is identical but for the other print
```

### 7. Limitations

#### 7.1 Per-lane branching to warp calls is forbidden

```rust
// REJECTED — compile error:
if lane_id() == 0 {
    warp_print!(...);  // Only lane 0 would execute this — warp divergence!
}
```

The macro cannot statically prove a condition is warp-uniform in all cases. The rule is: **any condition guarding a `warp_*!()` call is assumed to be warp-uniform and will be evaluated only by lane 0**. If the user passes a per-lane condition, the behavior is incorrect but the macro cannot detect this at compile time. Documentation must clearly state this contract.

#### 7.2 if without else (when then-arm has yields)

```rust
// REJECTED — compile error:
if condition {
    warp_print!(...);  // yield in then-arm
}
// No else arm — what states would we skip? Ambiguous control flow.
```

If the `then` arm contains `warp_*!()` calls, the `else` arm MUST also exist. The else arm may be empty of yields (just compute or empty), in which case it's a single-edge to the join point — but it must be syntactically present.

**Exception**: If neither arm contains yields, the entire `if` is treated as compute and no branch broadcast is needed.

#### 7.3 Early return

```rust
// REJECTED — compile error:
if error_condition {
    return false;  // No mechanism for early warp termination
}
```

Early `return` inside branching constructs is not supported. The state machine must run to the `Done` state. A future extension could add a dedicated `Error`/`Abort` terminal state.

#### 7.4 while loops

`while cond { ... }` is syntactic sugar for `loop { if !cond { break; } ... }`. The macro could desugar this, but initially only `loop` + `break` is supported. `while` support can be added as a straightforward transformation.

#### 7.5 Nested loops

Nested `loop` constructs are supported in principle (each loop gets its own LoopHeader and back-edge/break-edge targets), but the complexity of state numbering increases. The initial implementation should support single-level loops and add nesting in a follow-up.

#### 7.6 break with value

`break <expr>` is not supported. Loops in `#[warp_async]` functions do not return values. Use a mutable variable declared before the loop and assign to it before `break`.

#### 7.7 Variable types

All cross-yield variables must be `u64`. The macro does not perform type inference — it promotes all captured variables to `u64` fields. Users must cast to/from `u64` explicitly if they need other types.

#### 7.8 for loops and iterators

Not supported. `for x in iter { ... }` requires iterator trait machinery that is not available in `no_std` GPU context. Use `loop` with manual indexing.

#### 7.9 match arm patterns with bindings

```rust
match status {
    x @ 1..=10 => { warp_print!(...); }  // Pattern binding
    _ => {}
}
```

Pattern bindings in match arms are not supported initially. Only literal/const patterns and wildcards are supported. The scrutinee value is already available via the struct field.

## Open Questions

1. **State numbering strategy for unbalanced branches**: If the then-arm has 3 yields and the else-arm has 1 yield, the state space is sparse (some states only reachable from one branch). Should we use a flat numbering (wastes some discriminant values but is simple) or a compact scheme? **Recommendation**: Flat numbering is simpler and the state count is small (typically <100). Use flat.

2. **warp_hostcall_wait_u64 next_state override**: The current `warp_hostcall_wait_u64` takes a `next_state` parameter and writes it on completion. For branch decisions, we need to override this after the wait completes. Two approaches:
   - Pass a dummy `next_state` to `warp_hostcall_wait_u64` and immediately overwrite `self.state` with the branch target. (Simple, works today.)
   - Refactor `warp_hostcall_wait_u64` to not set state, letting the caller always set it. (Cleaner but changes the existing API.)
   **Recommendation**: Use the dummy-then-overwrite approach for backward compatibility.

3. **Static warp-uniformity checking**: Can we add a `#[warp_uniform]` attribute or lint to help users avoid per-lane conditions? This is a nice-to-have for a future task.

## Impact on Downstream Tasks

- **warp-cfg.2** (implementation): Direct consumer of this design. Should implement the `CfgGraph`, `build_cfg`, `assign_states`, and code generation as described.
- **warp-future runtime**: No changes needed — the generated code uses the existing `warp_hostcall_submit`/`warp_hostcall_wait_u64`/`broadcast_u32` APIs.
- **gpu-kernel**: Existing hand-written kernels are unaffected. New kernels can use the extended `#[warp_async]` once implemented.
- **Testing**: Need a test kernel with if/else, loop, and match constructs to validate the generated state machines. Should use the existing PTX build pipeline.
