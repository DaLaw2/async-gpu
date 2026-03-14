# tla-model.1 — TLA+ Modeling Investigation for Hostcall CAS Protocol

**Status**: Complete
**Theme**: tla-model
**Type**: Investigation
**Date**: 2026-03-15

## Goal

Determine how to model the async_gpu hostcall lock-free two-stack protocol in
TLA+, covering CAS operations, memory ordering, scope, invariants, liveness,
and agent count.

## Findings

### Q1: TLA+ Constructs for CAS Compare-and-Swap

CAS maps naturally to a PlusCal **macro** — an atomic block with no internal
labels. The standard pattern (used in the NEU lock-free stack example and
tla-rust) is:

```
macro CAS(addr, expected, desired) {
  if (addr = expected) {
    addr := desired;
    cas_ok := TRUE;
  } else {
    cas_ok := FALSE;
  }
}
```

Because PlusCal macros execute atomically (no interleaving within a macro),
this correctly models the hardware CAS instruction. The key insight: each
*label* in PlusCal defines an atomic step. CAS must be inside a single label
(or be a macro invoked within one) to preserve atomicity.

For the tagged-pointer CAS used in async_gpu (64-bit tagged value with
ABA epoch + packet index), the TLA+ model would compare the full tagged
value, not just the index. This is critical — the ABA tag is the entire
reason we can use CAS safely.

### Q2: Memory Ordering in TLA+

**TLA+ is inherently sequentially consistent (SC).** All actions
(label-delimited blocks) are totally ordered in the behavior trace. There is
no built-in mechanism for modeling store buffers or reordering.

This is *advantageous* for our use case for two reasons:

1. **SC is strictly stronger than acquire/release.** Any bug found under SC
   is a real bug. Any algorithm correct under SC is correct under SC but
   *not* necessarily under weaker models. However, for our protocol the
   critical question is the reverse: are there bugs that only appear under
   weak ordering?

2. **Our protocol already enforces SC-like guarantees.** The GPU uses
   `st.release.sys` / `ld.acquire.sys` / `atom.cas.sys` — system-scope
   atomics that synchronize across GPU SMs and host CPU. The host uses Rust
   `Ordering::Acquire` / `Ordering::Release` / `Ordering::AcqRel`. These
   form correct release-acquire pairs:
   - GPU release-store of `CONTROL_FILLED` → host acquire-load of control
   - Host release-store of `CONTROL_READY` → GPU acquire-load of control
   - CAS on stack heads: system-scope, full atomic RMW

   Because every shared-memory communication is protected by a
   release-acquire pair, and TLA+ checks the *algorithm logic* under SC,
   modeling under SC is sufficient for finding logic bugs (e.g., packet
   used-after-free, double-push, stack corruption).

**Recommendation**: Model under SC (default TLA+). If we later want to
explicitly check for missing fences, we could add a "store buffer" variable
per agent that delays writes — but this is an advanced extension, not needed
for the initial model.

### Q3: Minimal Model Scope

**Phase 1 — Packet Lifecycle FSM (minimal, high value)**:
- Two Treiber stacks: `free_stack` and `ready_stack` (global, unsharded)
- Packet pool: `N` packets, each with state `{Free, Filling, Ready, Processing, Done}`
- GPU agents: pop free → fill → push ready → spin-wait
- Host agent: detect doorbell → swap-drain ready → process → set CONTROL_READY
- Doorbell: monotonic counter

This is the core protocol and catches the most dangerous bugs: packet
double-use, stack corruption, lost packets, ABA.

**Phase 2 — Sharding (optional extension)**:
- Replace single stacks with `S` shard pairs (free + ready per shard)
- GPU block → shard mapping via `blockIdx % num_shards`
- Host scans all shard ready stacks

Sharding changes contention but not correctness semantics if the per-shard
stacks are independent Treiber stacks. It adds model complexity (state
space) without adding new bug classes — **defer to Phase 2**.

**Phase 3 — Doorbell optimization (low priority)**:
The doorbell is a monotonic counter used purely for host wake-up. It does
not affect correctness — the host will eventually scan ready stacks. The
doorbell only affects liveness (latency). **Defer or omit.**

### Q4: Safety Invariants

1. **No double-ownership**: A packet index appears in at most one location
   at any time — either on the free stack, on the ready stack, owned by a
   GPU thread, or owned by the host. Formally:
   ```
   ∀ i ∈ Packets:
     (i ∈ free_stack_set) + (i ∈ ready_stack_set) +
     (∃ g ∈ GPU: gpu_owns[g] = i) + (host_owns = i) ≤ 1
   ```

2. **No packet loss** (conservation): The total number of packets across
   all locations equals `N`:
   ```
   |free_stack_set| + |ready_stack_set| + |gpu_owned_set| + |host_owned_set| = N
   ```

3. **No ABA corruption**: After a successful CAS pop, the popped packet's
   `next` pointer is not stale. (The ABA tag increment on push prevents
   this — the model should confirm it.)

4. **Packet state consistency**: A packet on the ready stack has
   `CONTROL_FILLED` set. A packet returned to the GPU has `CONTROL_READY`
   set. A packet on the free stack has control cleared.

5. **No use-after-free**: A GPU thread does not read/write a packet after
   pushing it to the ready stack. The host does not read/write a packet
   after setting `CONTROL_READY`.

6. **Type invariant**: Tagged pointers contain valid indices (0..N-1 or
   NULL_INDEX). Tags are non-negative.

### Q5: Liveness Properties

1. **Progress (weak fairness)**: If a GPU thread wants to submit a
   hostcall and free packets exist, it eventually succeeds. Under weak
   fairness for all processes:
   ```
   □(gpu_wants_packet ∧ free_stack ≠ empty) ⇒ ◇(gpu_has_packet)
   ```

2. **Response delivery**: Every packet pushed to the ready stack is
   eventually processed by the host and marked `CONTROL_READY`:
   ```
   □(packet_state[i] = Ready ⇒ ◇(packet_state[i] = Done))
   ```

3. **Recycling**: Every packet eventually returns to the free stack
   (full lifecycle completion):
   ```
   □(packet_state[i] = Done ⇒ ◇(packet_state[i] = Free))
   ```

4. **No starvation**: Under fair scheduling, no GPU thread spins forever
   on CAS. (Treiber stack CAS is lock-free but not wait-free — a thread
   can starve under adversarial scheduling. With weak fairness, TLC will
   check this.)

### Q6: Agent Count for Meaningful Verification

**Minimum**: 2 GPU threads + 1 host thread + 2 packets

This is sufficient to expose:
- CAS contention (two GPUs racing for the same free packet)
- ABA problem (pop-push-pop sequence)
- Concurrent push to ready stack
- Host drain while GPU is pushing

**Recommended**: 3 GPU threads + 1 host thread + 3 packets

Three GPU threads allow modeling the full ABA scenario:
1. Thread A begins pop, reads head and next
2. Thread B pops head, uses it, pushes it back
3. Thread A's CAS succeeds with stale `next` (ABA!)

With tagged pointers, thread A's CAS should fail (tag mismatch). Three
threads + three packets give enough state space to exercise this.

**State space consideration**: TLC explores all interleavings. With `G`
GPU threads, `H` host threads, and `P` packets, state space grows
exponentially. For `G=3, H=1, P=3`, the state space is manageable (likely
< 10M states). Going to `G=4, P=4` may push into minutes/hours of model
checking — acceptable but not needed for initial validation.

### Q7: Existing TLA+ References for Treiber Stacks

1. **NEU CS3650 lock_free_stack_ABA.tla** — PlusCal spec of a lock-free
   stack that deliberately exposes the ABA bug by splitting the pop CAS
   across labels. Shows how to model CAS as a macro and detect ABA via an
   `onStack` ghost variable. Directly applicable to our tagged-pointer
   design.
   URL: https://course.ccs.neu.edu/cs3650/parent/tla+/PlusCal-examples/advanced/lock_free_stack_ABA.tla

2. **tla-rust (spacejam)** — TLA+ models for Rust lock-free structures
   including ring buffers, stacks, and epoch-based GC. Uses short labeled
   blocks for CAS atomicity. Demonstrates invariant checking for
   conservation properties.
   URL: https://github.com/spacejam/tla-rust

3. **tlaplus/Examples** — Official TLA+ example repository with various
   concurrent algorithm specs.
   URL: https://github.com/tlaplus/Examples

## Recommended Model Architecture

```
MODULE HostcallProtocol

CONSTANTS
  GPU_THREADS    \* e.g., {g1, g2, g3}
  NUM_PACKETS    \* e.g., 3
  NULL           \* sentinel for empty stack

VARIABLES
  free_head,          \* tagged pointer (record: {tag, idx})
  ready_head,         \* tagged pointer
  packet_next,        \* function: packet_idx -> tagged pointer
  packet_state,       \* function: packet_idx -> {Free, Filling, Ready, Processing, Done}
  packet_control,     \* function: packet_idx -> subset of {FILLED, READY, ERROR}
  gpu_pc,             \* function: gpu_thread -> program counter label
  gpu_local,          \* function: gpu_thread -> local variables (old_head, next, pkt_idx)
  host_pc,            \* host program counter
  host_local,         \* host local variables (drained list, current packet)
  doorbell            \* monotonic counter
```

Key design decisions:
- **Tagged pointers as records** `[tag |-> Nat, idx |-> 0..N ∪ {NULL}]`
- **CAS as macro** comparing full tagged value
- **GPU thread as PlusCal process** with labels: PopFree → Fill → PushReady → RingDoorbell → SpinWait → Release
- **Host as PlusCal process** with labels: PollDoorbell → SwapDrain → WalkList → Process → SetReady
- **Ghost variable** `packet_owner` tracking exclusive ownership for invariant checking

## Conclusions

1. TLA+ is well-suited for verifying the hostcall CAS protocol. CAS maps
   to atomic macros, and the packet lifecycle FSM is a natural fit for
   state-machine specification.

2. Sequential consistency (TLA+ default) is sufficient because the real
   protocol uses system-scope acquire/release atomics that establish the
   same happens-before relationships.

3. Start with Phase 1 (unsharded, 3 GPU + 1 host + 3 packets). This
   catches the critical bugs (double-ownership, ABA, packet loss) with
   manageable state space.

4. The NEU lock_free_stack_ABA.tla example is a near-perfect starting
   template — adapt its CAS macro and stack representation, extend with
   the two-stack (free/ready) protocol and host-side swap-drain.

5. Key invariants: no double-ownership, packet conservation, ABA
   prevention, state consistency. Key liveness: progress and response
   delivery under weak fairness.
