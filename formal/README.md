# Formal Verification: Hostcall CAS Protocol

## What This Verifies

This TLA+ specification models the **async_gpu hostcall packet lifecycle protocol** — a lock-free two-stack system where GPU threads communicate with a host CPU thread through shared-memory packets.

The protocol uses two **Treiber stacks** (lock-free linked-list stacks) with **tagged compare-and-swap (CAS)** to prevent the ABA problem:

```
FREE (free stack)
  -> GPU pops via tagged CAS -> FILLING (GPU owns)
  -> GPU pushes to ready stack -> READY
  -> Host swap-drains ready stack -> PROCESSING (host owns)
  -> Host writes response -> DONE
  -> GPU reads response + pushes to free -> FREE
```

The model uses **3 GPU threads**, **1 host thread**, and **3 packets** — the minimum configuration needed to expose ABA scenarios and CAS contention bugs.

## Prerequisites

Install one of:

1. **TLA+ Toolbox** (GUI, recommended for exploration):
   Download from https://github.com/tlaplus/tlaplus/releases

2. **Command-line TLC** (for CI or scripting):
   Download `tla2tools.jar` from the same release page.

3. **VSCode Extension** (alternative GUI):
   Install the "TLA+" extension by Markus Kuppe.

## How to Run

### Option A: TLA+ Toolbox

1. Open `HostcallProtocol.tla` in the Toolbox
2. Translate PlusCal: **File -> Translate PlusCal Algorithm** (Ctrl+T)
   - Note: The file already includes a manual TLA+ translation, so this step is optional. If the Toolbox complains about checksum mismatch, re-translate to update.
3. Create a new model: **TLC Model Checker -> New Model**
4. In model settings:
   - Set constants as shown in `HostcallProtocol.cfg`
   - Add invariants: `TypeOK`, `NoDoubleOwnership`, `PacketConservation`, `StateConsistency`, `StacksDisjoint`
   - Add temporal properties: `ResponseDelivery`, `PacketRecycling`, `FullLifecycle`
5. Run the model checker

### Option B: Command Line

```bash
# From the formal/ directory:
java -jar /path/to/tla2tools.jar -config HostcallProtocol.cfg HostcallProtocol.tla
```

Or with explicit memory settings for large state spaces:

```bash
java -Xmx4g -jar /path/to/tla2tools.jar \
    -config HostcallProtocol.cfg \
    -workers auto \
    HostcallProtocol.tla
```

### Running Safety Only (Faster)

To check only safety invariants without liveness (much faster), edit `HostcallProtocol.cfg` and comment out the `PROPERTY` lines:

```
\* PROPERTY ResponseDelivery
\* PROPERTY PacketRecycling
\* PROPERTY FullLifecycle
```

## Invariants Explained

| Invariant | What It Checks |
|-----------|---------------|
| `TypeOK` | All variables have valid types (tagged pointers contain valid indices, states are valid enum values, counters are natural numbers) |
| `NoDoubleOwnership` | A packet appears in **at most one** location at any time — free stack, ready stack, GPU-owned, or host-owned. Prevents data races. |
| `PacketConservation` | The total number of packets across all locations equals 3 (no packets created or destroyed). Detects lost packets. |
| `StateConsistency` | Packets on the free stack are in `Free` state; packets on the ready stack are in `Ready` state. Ensures state transitions are correct. |
| `StacksDisjoint` | Free and ready stacks never share a packet. Catches stack corruption. |

## Liveness Properties Explained

| Property | What It Checks |
|----------|---------------|
| `ResponseDelivery` | Every `Ready` packet **eventually** becomes `Done` (host processes it). Ensures no packet is lost on the ready stack. |
| `PacketRecycling` | Every `Done` packet **eventually** returns to `Free`. Ensures no packet leak after processing. |
| `FullLifecycle` | Every `Filling` packet **eventually** returns to `Free`. Ensures the full lifecycle completes. |

Liveness properties require **weak fairness** (every continuously enabled action eventually executes), which is provided by the `fair process` declarations in the PlusCal spec.

## State Constraint

The model uses a `StateConstraint` to bound monotonically increasing values (tagged pointer tags and doorbell counter) at `MAX_TAG = 6`. Without this bound, TLC would generate infinite states because tags increment on every push.

This bound is safe for safety checking: if there is a bug in the protocol logic, it will manifest within the bounded state space. The tag values themselves do not affect correctness — they only need to be unique enough to prevent ABA within any reachable interleaving.

**Important**: Liveness checking with state constraints can produce false counterexamples because the constraint truncates infinite behaviors. The default configuration has liveness properties commented out. To check liveness:
1. Run safety-only first (default config)
2. Then uncomment `PROPERTY` lines and either increase `MAX_TAG` or remove the constraint

## Expected Results

When the model checker finishes successfully, you should see output like:

```
Model checking completed. No error has been found.
  Finished in XXs at (timestamp)
  NNN states generated, NNN distinct states found, 0 states left on queue.
```

**If any invariant fails**, TLC will produce a counterexample trace showing the exact sequence of steps that led to the violation. This trace maps directly to the protocol's atomic operations.

## Mapping to Source Code

| TLA+ Concept | Rust/PTX Implementation |
|--------------|------------------------|
| `free_head` / `ready_head` | `AtomicU64` stack head pointers in `HostcallSharedState` |
| `pkt_next[p]` | Per-packet `next` field in `HostcallPacket` |
| `pkt_state[p]` | `CONTROL` byte in packet header (`FILLED` / `READY`) |
| CAS macro | `atom.cas.acq_rel.sys` (GPU PTX) / `AtomicU64::compare_exchange` (host) |
| Tagged pointer `[tag, idx]` | Upper 32 bits = epoch tag, lower 32 bits = packet index |
| `doorbell` | Monotonic `AtomicU32` counter, GPU increments, host polls |
| GPU `PopFree_*` labels | `hostcall_submit()` free stack pop loop |
| GPU `PushReady_*` labels | `hostcall_submit()` ready stack push loop |
| Host `SwapDrain_*` labels | `HostcallHandler::poll()` swap-drain loop |
| Host `SetDone` label | `HostcallHandler` writes response + sets `CONTROL_READY` |

## Extending the Model

### More Threads/Packets

Edit `HostcallProtocol.cfg`:
```
CONSTANT
    GPU_THREADS = {"g1", "g2", "g3", "g4"}
    PACKETS     = {"p1", "p2", "p3", "p4"}
    INIT_FREE_SEQ = <<"p1", "p2", "p3", "p4">>
```

**Warning**: State space grows exponentially. 4 GPU threads + 4 packets may take hours.

### Sharding

The current model uses a single pair of stacks (unsharded). To model the sharded variant, duplicate `free_head`/`ready_head` per shard and assign GPU threads to shards via `blockIdx % num_shards`.
