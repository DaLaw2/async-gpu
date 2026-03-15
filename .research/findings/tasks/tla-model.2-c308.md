# tla-model.2 — TLA+ Specification for Hostcall CAS Protocol

**Status**: Complete
**Theme**: tla-model
**Type**: Implementation
**Date**: 2026-03-15

## Goal

Write the TLA+ / PlusCal specification for the hostcall lock-free two-stack protocol, including safety invariants, liveness properties, and a TLC model configuration.

## Deliverables

### Files Created

1. **`formal/HostcallProtocol.tla`** — Main TLA+ specification
   - PlusCal algorithm with C-syntax (`{...}` blocks)
   - Hand-written TLA+ translation (between BEGIN/END TRANSLATION markers)
   - Can also be re-translated via TLA+ Toolbox (Ctrl+T)

2. **`formal/HostcallProtocol.cfg`** — TLC model configuration
   - Constants: 3 GPU threads, 3 packets, null sentinel
   - Safety invariants enabled by default
   - Liveness properties commented out (run separately due to state constraint interaction)
   - State constraint enabled (MAX_TAG = 6)

3. **`formal/README.md`** — Usage guide
   - Prerequisites, how to run (Toolbox and CLI)
   - Invariant and property explanations
   - Source code mapping table

## Design Decisions

### CAS as PlusCal Macro
The `CAS(addr, expected, desired, ok)` macro executes atomically (no labels inside), correctly modeling the hardware CAS instruction. The full tagged pointer `[tag, idx]` is compared, preventing ABA.

### Tagged Pointers
Modeled as TLA+ records `[tag |-> Nat, idx |-> PACKETS ∪ {NULL}]`. Tags increment monotonically on every push. This directly maps to the upper-32/lower-32 split in the real 64-bit tagged pointers.

### State Constraint
Monotonic counters (tags, doorbell) require bounding for finite model checking. `MAX_TAG = 6` provides enough headroom for 2 full lifecycles per GPU thread with 3 threads. Safety checking is sound under this constraint; liveness requires care (documented in README).

### Ghost Variable: gpu_owns
A `gpu_owns` function tracks which GPU thread owns which packet. This is purely for invariant checking — it has no operational effect. It enables the `NoDoubleOwnership` and `PacketConservation` invariants.

### Host Swap-Drain
The host atomically replaces `ready_head` with `NullTP` (keeping the tag for ABA safety), then walks the drained linked list sequentially. This matches the real protocol's `swap` + iterate pattern.

### Label Granularity
Each label maps to one atomic memory operation in the real PTX/Rust code:
- `PopFree_Read` = load acquire of free_head
- `PopFree_Next` = load acquire of packet_next[head]
- `PopFree_CAS` = CAS on free_head
- `Fill` = write payload + set CONTROL_FILLED (combined since they target owned memory)
- etc.

This granularity is critical: splitting CAS across labels would introduce artificial interleavings, while merging too many operations would hide real races.

## Safety Invariants

| ID | Name | Formula Summary |
|----|------|----------------|
| 1 | NoDoubleOwnership | Each packet in at most one location |
| 2 | PacketConservation | Total packets = |PACKETS| across all locations |
| 3 | StateConsistency | Free stack packets are "Free", ready stack packets are "Ready" |
| 4 | StacksDisjoint | Free ∩ Ready = ∅ |
| 5 | TypeOK | All values in expected domains |

## Liveness Properties

| ID | Name | Formula |
|----|------|---------|
| 1 | ResponseDelivery | Ready ~> Done |
| 2 | PacketRecycling | Done ~> Free |
| 3 | FullLifecycle | Filling ~> Free |

## Known Limitations

1. **Sequential consistency only** — TLA+ models SC; weak memory ordering bugs (missing fences) would not be caught. However, the real protocol uses system-scope acquire/release atomics that establish equivalent ordering.

2. **Unsharded** — models a single free_stack + ready_stack pair. Sharding is an orthogonal extension.

3. **No payload modeling** — packet contents are not modeled; only lifecycle states. Sufficient for protocol correctness.

4. **Bounded tags** — state constraint truncates behaviors at MAX_TAG = 6. Increase for deeper exploration at the cost of longer checking time.
