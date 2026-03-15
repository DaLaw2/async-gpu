--------------------------- MODULE HostcallProtocol ---------------------------
(*
 * TLA+ / PlusCal specification of the async_gpu hostcall CAS protocol.
 *
 * This models the packet lifecycle through two lock-free Treiber stacks
 * (free_stack and ready_stack) shared between GPU threads and a host thread.
 *
 * Packet Lifecycle FSM (maps to crates/async-gpu-hostcall):
 *   FREE (on free stack)
 *     -> GPU pops via tagged CAS -> FILLING (GPU owns)
 *     -> GPU pushes to ready stack -> READY
 *     -> Host swap-drains ready stack -> PROCESSING (host owns)
 *     -> Host writes response + sets CONTROL_READY -> DONE
 *     -> GPU reads response + pushes back to free -> FREE
 *
 * Tagged CAS prevents the ABA problem: each push increments a monotonic tag
 * in a 64-bit tagged pointer. CAS compares the full (tag, idx) pair.
 *
 * Model parameters: 3 GPU threads, 1 host thread, 3 packets.
 * With these parameters, TLC should explore the state space in minutes.
 *
 * Mapping to real code:
 *   free_head / ready_head  -> AtomicU64 stack head pointers in shared memory
 *   packet_next             -> per-packet "next" field (linked list node)
 *   packet_state            -> derived from CONTROL byte in packet header
 *   CAS macro               -> atom.cas.acq_rel.sys (GPU) / compare_exchange (host)
 *   doorbell                -> monotonic doorbell counter
 *
 * To use:
 *   1. Open in TLA+ Toolbox
 *   2. Translate PlusCal: File -> Translate PlusCal Algorithm (or Ctrl+T)
 *   3. Create model with HostcallProtocol.cfg settings
 *   4. Run TLC model checker
 *)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    GPU_THREADS,    \* Set of GPU thread IDs, e.g., {"g1", "g2", "g3"}
    PACKETS,        \* Set of packet indices, e.g., {"p1", "p2", "p3"}
    NULL,           \* Sentinel for empty stack
    INIT_FREE_SEQ   \* Sequence of packets for initial free stack ordering
                    \* e.g., <<"p1", "p2", "p3">>

(* Packet states matching the protocol FSM *)
States == {"Free", "Filling", "Ready", "Processing", "Done"}

(* Tagged pointer constructor *)
TP(t, i) == [tag |-> t, idx |-> i]

(* Null tagged pointer *)
NullTP == TP(0, NULL)

(*
 * Build the initial free stack from INIT_FREE_SEQ.
 * Returns <<head, nextFn>> where head is the tagged pointer to the first
 * packet and nextFn maps each packet to its next tagged pointer.
 *
 * For INIT_FREE_SEQ = <<p1, p2, p3>>:
 *   head = TP(0, p1)
 *   next[p1] = TP(0, p2), next[p2] = TP(0, p3), next[p3] = NullTP
 *)
RECURSIVE BuildNextFn(_, _)
BuildNextFn(seq, i) ==
    IF i >= Len(seq) THEN [p \in PACKETS |-> NullTP]
    ELSE LET rest == BuildNextFn(seq, i+1)
         IN  [p \in PACKETS |-> IF p = seq[i] THEN TP(0, seq[i+1]) ELSE rest[p]]

InitHead == IF Len(INIT_FREE_SEQ) > 0 THEN TP(0, INIT_FREE_SEQ[1]) ELSE NullTP
InitNext == BuildNextFn(INIT_FREE_SEQ, 1)

----------------------------------------------------------------------------
(*
 * Collect the set of packet indices reachable from a stack head.
 * Used in invariant checking only (not in the algorithm).
 *)
RECURSIVE StackSet(_, _)
StackSet(head, nextFn) ==
    IF head.idx = NULL THEN {}
    ELSE {head.idx} \union StackSet(nextFn[head.idx], nextFn)

(*
 * Walk a linked list from head, collecting indices into a sequence.
 * Used by host swap-drain to iterate packets in FIFO order.
 *)
RECURSIVE StackToSeq(_, _)
StackToSeq(head, nextFn) ==
    IF head.idx = NULL THEN << >>
    ELSE <<head.idx>> \o StackToSeq(nextFn[head.idx], nextFn)

----------------------------------------------------------------------------

(*--algorithm HostcallProtocol {

variables
    \* --- Shared state ---
    free_head  = InitHead;            \* Head of free Treiber stack
    ready_head = NullTP;              \* Head of ready Treiber stack (initially empty)
    pkt_next   = InitNext;            \* Per-packet next pointers
    pkt_state  = [p \in PACKETS |-> "Free"];  \* Per-packet lifecycle state
    doorbell   = 0;                   \* Monotonic doorbell counter
    host_seen  = 0;                   \* Last doorbell value host observed

    \* Ghost variable for ownership tracking (invariant checking only)
    gpu_owns   = [g \in GPU_THREADS |-> NULL];

    \* Host process variables
    drain_list  = << >>;   \* Packets drained from ready stack
    drain_idx   = 0;       \* Current processing index in drain_list
    h_old_head  = NullTP;  \* Snapshot of ready_head for swap-drain CAS

define {
    \* ==============================================================
    \*  TYPE INVARIANT
    \* ==============================================================
    TPValid(tp) == tp.tag \in Nat /\ tp.idx \in PACKETS \union {NULL}

    TypeOK ==
        /\ TPValid(free_head)
        /\ TPValid(ready_head)
        /\ \A p \in PACKETS: TPValid(pkt_next[p])
        /\ \A p \in PACKETS: pkt_state[p] \in States
        /\ doorbell \in Nat
        /\ host_seen \in Nat

    \* ==============================================================
    \*  DERIVED SETS (for invariants)
    \* ==============================================================
    FreeSet  == StackSet(free_head, pkt_next)
    ReadySet == StackSet(ready_head, pkt_next)

    GpuOwnedSet == {gpu_owns[g] : g \in {g2 \in GPU_THREADS : gpu_owns[g2] /= NULL}}

    \* Host-owned: packets in drain_list from drain_idx onward.
    \* When drain_idx = 0 or > Len(drain_list), host owns nothing.
    HostOwnedSet ==
        IF drain_idx >= 1 /\ Len(drain_list) >= drain_idx
        THEN {drain_list[i] : i \in drain_idx..Len(drain_list)}
        ELSE {}

    \* ==============================================================
    \*  SAFETY INVARIANTS
    \* ==============================================================

    (*
     * INV-1: No double-ownership.
     * Each packet is in at most one location (free stack, ready stack,
     * GPU-owned, or host-owned) at any time. Violation would mean a
     * packet is being written by two agents simultaneously.
     *)
    NoDoubleOwnership ==
        \A p \in PACKETS:
            Cardinality({
                s \in {"free", "ready", "gpu", "host"} :
                    \/ (s = "free"  /\ p \in FreeSet)
                    \/ (s = "ready" /\ p \in ReadySet)
                    \/ (s = "gpu"   /\ p \in GpuOwnedSet)
                    \/ (s = "host"  /\ p \in HostOwnedSet)
            }) <= 1

    (*
     * INV-2: Packet conservation.
     * The total number of packets across all locations equals |PACKETS|.
     * No packet is ever created or destroyed.
     *)
    PacketConservation ==
        Cardinality(FreeSet \union ReadySet \union GpuOwnedSet \union HostOwnedSet)
            = Cardinality(PACKETS)

    (*
     * INV-3: Packet state consistency.
     * Packets on specific stacks must have matching lifecycle state.
     *)
    StateConsistency ==
        /\ \A p \in FreeSet:  pkt_state[p] = "Free"
        /\ \A p \in ReadySet: pkt_state[p] = "Ready"

    (*
     * INV-4: Valid stack structure.
     * Free and ready stacks do not share any packets.
     *)
    StacksDisjoint == FreeSet \intersect ReadySet = {}

    \* Combined safety invariant
    SafetyInvariant ==
        /\ TypeOK
        /\ NoDoubleOwnership
        /\ PacketConservation
        /\ StateConsistency
        /\ StacksDisjoint

    (*
     * STATE CONSTRAINT for finite model checking.
     * Tags and doorbell are monotonically increasing, so without a bound
     * TLC would never terminate. We bound all tags and the doorbell to
     * MAX_TAG. This is safe because the protocol's correctness does not
     * depend on the absolute tag value — only on tags being unique enough
     * to prevent ABA within the state space we explore.
     *
     * With 3 GPU threads doing at most ~2 iterations each, tag values
     * stay well within this bound.
     *)
    MAX_TAG == 6

    StateConstraint ==
        /\ free_head.tag  <= MAX_TAG
        /\ ready_head.tag <= MAX_TAG
        /\ doorbell       <= MAX_TAG
        /\ \A p \in PACKETS: pkt_next[p].tag <= MAX_TAG
}

\* ==================================================================
\*  CAS MACRO
\*
\*  Atomically compares addr to expected; if equal, sets addr to
\*  desired and sets ok to TRUE. Otherwise sets ok to FALSE.
\*
\*  This is a PlusCal macro, so it executes atomically (no labels
\*  allowed inside). This correctly models the hardware CAS instruction.
\*
\*  Maps to: atom.cas.acq_rel.sys on GPU, AtomicU64::compare_exchange on host.
\* ==================================================================
macro CAS(addr, expected, desired, ok) {
    if (addr = expected) {
        addr := desired;
        ok := TRUE;
    } else {
        ok := FALSE;
    }
}

\* ==================================================================
\*  GPU THREAD PROCESS
\*
\*  Each GPU thread loops forever:
\*    1. Pop a packet from the free stack (tagged CAS, retry on failure)
\*    2. Fill the packet with request data
\*    3. Push the packet to the ready stack (tagged CAS, retry on failure)
\*    4. Ring the doorbell (atomic increment)
\*    5. Spin-wait until host marks packet as Done
\*    6. Read response, push packet back to free stack
\*
\*  Labels correspond to atomic steps in the real PTX code:
\*    PopFree_Read    -> ld.acquire.sys(free_head)
\*    PopFree_Next    -> ld.acquire.sys(packet_next[head.idx])
\*    PopFree_CAS     -> atom.cas.acq_rel.sys(free_head, old, new)
\*    Fill            -> memcpy + st.release.sys(control, FILLED)
\*    PushReady_Read  -> ld.acquire.sys(ready_head)
\*    PushReady_Link  -> st(packet_next[pkt], head)
\*    PushReady_CAS   -> atom.cas.acq_rel.sys(ready_head, old, new)
\*    RingDoorbell    -> atom.add.sys(doorbell, 1)
\*    SpinWait        -> ld.acquire.sys(control) loop
\*    ReadResp        -> read response payload
\*    RelFree_Read    -> ld.acquire.sys(free_head)
\*    RelFree_Link    -> st(packet_next[pkt], head)
\*    RelFree_CAS     -> atom.cas.acq_rel.sys(free_head, old, new)
\* ==================================================================
fair process (gpu \in GPU_THREADS)
variables
    ok      = FALSE,
    lh      = NullTP,   \* local snapshot of stack head
    ln      = NullTP,   \* local snapshot of next pointer
    my_pkt  = NULL;     \* currently owned packet index
{
    \* ---- Pop from free stack ----
    PopFree_Read:
        lh := free_head;
        if (lh.idx = NULL) {
            \* No free packets available, spin
            goto PopFree_Read;
        };

    PopFree_Next:
        \* Read next pointer of the head node
        ln := pkt_next[lh.idx];

    PopFree_CAS:
        \* CAS: try to advance free_head from lh to ln
        \* The tag in lh prevents ABA: if someone popped lh.idx and
        \* pushed it back, the tag would be different.
        CAS(free_head, lh, ln, ok);
        if (~ok) {
            goto PopFree_Read;
        };
        \* Success! We own lh.idx now.
        my_pkt := lh.idx;
        gpu_owns[self] := my_pkt;

    \* ---- Fill packet with request ----
    Fill:
        pkt_state[my_pkt] := "Filling";
        \* Clear next pointer before pushing to ready stack
        pkt_next[my_pkt] := NullTP;

    \* ---- Push to ready stack ----
    PushReady_Read:
        lh := ready_head;

    PushReady_Link:
        \* Link our packet's next to current ready head
        pkt_next[my_pkt] := lh;

    PushReady_CAS:
        \* CAS: try to set ready_head to our packet (with incremented tag)
        CAS(ready_head, lh, TP(lh.tag + 1, my_pkt), ok);
        if (~ok) {
            goto PushReady_Read;
        };
        \* Packet is now on the ready stack
        pkt_state[my_pkt] := "Ready";
        gpu_owns[self] := NULL;  \* Release ownership

    \* ---- Ring doorbell ----
    RingDoorbell:
        doorbell := doorbell + 1;

    \* ---- Spin-wait for host response ----
    SpinWait:
        if (pkt_state[my_pkt] /= "Done") {
            goto SpinWait;
        };

    \* ---- Read response ----
    ReadResp:
        \* GPU reads the host's response from the packet.
        \* Re-acquire ownership for the release phase.
        gpu_owns[self] := my_pkt;

    \* ---- Release: push back to free stack ----
    RelFree_Read:
        lh := free_head;

    RelFree_Link:
        pkt_next[my_pkt] := lh;

    RelFree_CAS:
        CAS(free_head, lh, TP(lh.tag + 1, my_pkt), ok);
        if (~ok) {
            goto RelFree_Read;
        };
        pkt_state[my_pkt] := "Free";
        gpu_owns[self] := NULL;
        my_pkt := NULL;

    \* ---- Loop: do another hostcall ----
    GpuLoop:
        goto PopFree_Read;
}

\* ==================================================================
\*  HOST THREAD PROCESS
\*
\*  The host thread runs a polling loop:
\*    1. Poll doorbell counter for new work
\*    2. Swap-drain the ready stack (atomic exchange with NULL)
\*    3. Walk the drained list, processing each packet
\*    4. Mark each packet Done (CONTROL_READY) so GPU can proceed
\*
\*  Labels:
\*    PollDoorbell  -> check doorbell > last_seen
\*    SwapDrain_R   -> ld.acquire(ready_head)
\*    SwapDrain_CAS -> CAS(ready_head, old, NullTP) — drain entire stack
\*    ProcLoop      -> iterate through drained packets
\*    ProcPacket    -> execute hostcall (syscall dispatch)
\*    SetDone       -> st.release(control, CONTROL_READY)
\* ==================================================================
fair process (host = "host")
variables
    hok = FALSE;
{
    PollDoorbell:
        if (doorbell = host_seen) {
            \* No new work, keep polling
            goto PollDoorbell;
        };
        host_seen := doorbell;

    SwapDrain_R:
        h_old_head := ready_head;
        if (h_old_head.idx = NULL) {
            \* Ready stack already empty
            goto PollDoorbell;
        };

    SwapDrain_CAS:
        \* Atomic swap: replace ready_head with NullTP (keep tag for ABA safety)
        CAS(ready_head, h_old_head, TP(h_old_head.tag, NULL), hok);
        if (~hok) {
            goto SwapDrain_R;
        };
        \* Build list of drained packets
        drain_list := StackToSeq(h_old_head, pkt_next);
        drain_idx := 1;

    ProcLoop:
        if (drain_idx > Len(drain_list)) {
            \* All packets in this batch processed
            drain_list := << >>;
            drain_idx := 0;
            goto PollDoorbell;
        };

    ProcPacket:
        \* Host processes the request (execute file I/O, etc.)
        pkt_state[drain_list[drain_idx]] := "Processing";

    SetDone:
        \* Host writes response and marks packet Done
        pkt_state[drain_list[drain_idx]] := "Done";
        \* Clear next pointer (packet is no longer on any stack)
        pkt_next[drain_list[drain_idx]] := NullTP;
        drain_idx := drain_idx + 1;
        goto ProcLoop;
}

}
*)

\* ================================================================
\*  BEGIN TRANSLATION
\*
\*  To generate the TLA+ translation from the PlusCal above:
\*    1. Open this file in the TLA+ Toolbox
\*    2. File -> Translate PlusCal Algorithm (Ctrl+T)
\*    3. The toolbox will insert TLA+ code between these markers
\*
\*  Alternatively, use the command-line translator:
\*    java -cp tla2tools.jar pcal.trans HostcallProtocol.tla
\* ================================================================

\* BEGIN TRANSLATION (chksum(pcal) = "placeholder")
VARIABLES free_head, ready_head, pkt_next, pkt_state, doorbell, host_seen,
          gpu_owns, drain_list, drain_idx, h_old_head, pc, ok, lh, ln,
          my_pkt, hok

vars == << free_head, ready_head, pkt_next, pkt_state, doorbell, host_seen,
           gpu_owns, drain_list, drain_idx, h_old_head, pc, ok, lh, ln,
           my_pkt, hok >>

ProcSet == (GPU_THREADS) \cup {"host"}

Init == (* Global variables *)
        /\ free_head = InitHead
        /\ ready_head = NullTP
        /\ pkt_next = InitNext
        /\ pkt_state = [p \in PACKETS |-> "Free"]
        /\ doorbell = 0
        /\ host_seen = 0
        /\ gpu_owns = [g \in GPU_THREADS |-> NULL]
        /\ drain_list = << >>
        /\ drain_idx = 0
        /\ h_old_head = NullTP
        (* Process gpu *)
        /\ ok = [self \in GPU_THREADS |-> FALSE]
        /\ lh = [self \in GPU_THREADS |-> NullTP]
        /\ ln = [self \in GPU_THREADS |-> NullTP]
        /\ my_pkt = [self \in GPU_THREADS |-> NULL]
        (* Process host *)
        /\ hok = FALSE
        /\ pc = [self \in ProcSet |-> CASE self \in GPU_THREADS -> "PopFree_Read"
                                        [] self = "host" -> "PollDoorbell"]

(* --- GPU thread actions --- *)

PopFree_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PopFree_Read"
    /\ lh' = [lh EXCEPT ![self] = free_head]
    /\ IF lh'[self].idx = NULL
          THEN /\ pc' = [pc EXCEPT ![self] = "PopFree_Read"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "PopFree_Next"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok >>

PopFree_Next(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PopFree_Next"
    /\ ln' = [ln EXCEPT ![self] = pkt_next[lh[self].idx]]
    /\ pc' = [pc EXCEPT ![self] = "PopFree_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, my_pkt, hok >>

PopFree_CAS(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PopFree_CAS"
    /\ IF free_head = lh[self]
          THEN /\ free_head' = ln[self]
               /\ ok' = [ok EXCEPT ![self] = TRUE]
          ELSE /\ ok' = [ok EXCEPT ![self] = FALSE]
               /\ UNCHANGED free_head
    /\ IF ~ok'[self]
          THEN /\ pc' = [pc EXCEPT ![self] = "PopFree_Read"]
               /\ UNCHANGED << my_pkt, gpu_owns >>
          ELSE /\ my_pkt' = [my_pkt EXCEPT ![self] = lh[self].idx]
               /\ gpu_owns' = [gpu_owns EXCEPT ![self] = lh[self].idx]
               /\ pc' = [pc EXCEPT ![self] = "Fill"]
    /\ UNCHANGED << ready_head, pkt_next, pkt_state, doorbell, host_seen,
                    drain_list, drain_idx, h_old_head, lh, ln, hok >>

Fill(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "Fill"
    /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Filling"]
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = NullTP]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_Read"]
    /\ UNCHANGED << free_head, ready_head, doorbell, host_seen, gpu_owns,
                    drain_list, drain_idx, h_old_head, ok, lh, ln, my_pkt, hok >>

PushReady_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_Read"
    /\ lh' = [lh EXCEPT ![self] = ready_head]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_Link"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok >>

PushReady_Link(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_Link"
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = lh[self]]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_state, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

PushReady_CAS(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_CAS"
    /\ IF ready_head = lh[self]
          THEN /\ ready_head' = TP(lh[self].tag + 1, my_pkt[self])
               /\ ok' = [ok EXCEPT ![self] = TRUE]
          ELSE /\ ok' = [ok EXCEPT ![self] = FALSE]
               /\ UNCHANGED ready_head
    /\ IF ~ok'[self]
          THEN /\ pc' = [pc EXCEPT ![self] = "PushReady_Read"]
               /\ UNCHANGED << pkt_state, gpu_owns >>
          ELSE /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Ready"]
               /\ gpu_owns' = [gpu_owns EXCEPT ![self] = NULL]
               /\ pc' = [pc EXCEPT ![self] = "RingDoorbell"]
    /\ UNCHANGED << free_head, pkt_next, doorbell, host_seen,
                    drain_list, drain_idx, h_old_head, lh, ln, my_pkt, hok >>

RingDoorbell(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RingDoorbell"
    /\ doorbell' = doorbell + 1
    /\ pc' = [pc EXCEPT ![self] = "SpinWait"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

SpinWait(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "SpinWait"
    /\ IF pkt_state[my_pkt[self]] /= "Done"
          THEN /\ pc' = [pc EXCEPT ![self] = "SpinWait"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "ReadResp"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

ReadResp(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "ReadResp"
    /\ gpu_owns' = [gpu_owns EXCEPT ![self] = my_pkt[self]]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_Read"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

RelFree_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_Read"
    /\ lh' = [lh EXCEPT ![self] = free_head]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_Link"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok >>

RelFree_Link(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_Link"
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = lh[self]]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_state, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

RelFree_CAS(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_CAS"
    /\ IF free_head = lh[self]
          THEN /\ free_head' = TP(lh[self].tag + 1, my_pkt[self])
               /\ ok' = [ok EXCEPT ![self] = TRUE]
          ELSE /\ ok' = [ok EXCEPT ![self] = FALSE]
               /\ UNCHANGED free_head
    /\ IF ~ok'[self]
          THEN /\ pc' = [pc EXCEPT ![self] = "RelFree_Read"]
               /\ UNCHANGED << pkt_state, gpu_owns, my_pkt >>
          ELSE /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Free"]
               /\ gpu_owns' = [gpu_owns EXCEPT ![self] = NULL]
               /\ my_pkt' = [my_pkt EXCEPT ![self] = NULL]
               /\ pc' = [pc EXCEPT ![self] = "GpuLoop"]
    /\ UNCHANGED << ready_head, pkt_next, doorbell, host_seen,
                    drain_list, drain_idx, h_old_head, lh, ln, hok >>

GpuLoop(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "GpuLoop"
    /\ pc' = [pc EXCEPT ![self] = "PopFree_Read"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

(* --- Host thread actions --- *)

PollDoorbell ==
    /\ pc["host"] = "PollDoorbell"
    /\ IF doorbell = host_seen
          THEN /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
               /\ UNCHANGED host_seen
          ELSE /\ host_seen' = doorbell
               /\ pc' = [pc EXCEPT !["host"] = "SwapDrain_R"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

SwapDrain_R ==
    /\ pc["host"] = "SwapDrain_R"
    /\ h_old_head' = ready_head
    /\ IF ready_head.idx = NULL
          THEN /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
          ELSE /\ pc' = [pc EXCEPT !["host"] = "SwapDrain_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx,
                    ok, lh, ln, my_pkt, hok >>

SwapDrain_CAS ==
    /\ pc["host"] = "SwapDrain_CAS"
    /\ IF ready_head = h_old_head
          THEN /\ ready_head' = TP(h_old_head.tag, NULL)
               /\ hok' = TRUE
          ELSE /\ hok' = FALSE
               /\ UNCHANGED ready_head
    /\ IF ~hok'
          THEN /\ pc' = [pc EXCEPT !["host"] = "SwapDrain_R"]
               /\ UNCHANGED << drain_list, drain_idx >>
          ELSE /\ drain_list' = StackToSeq(h_old_head, pkt_next)
               /\ drain_idx' = 1
               /\ pc' = [pc EXCEPT !["host"] = "ProcLoop"]
    /\ UNCHANGED << free_head, pkt_next, pkt_state, doorbell, host_seen,
                    gpu_owns, h_old_head, ok, lh, ln, my_pkt >>

ProcLoop ==
    /\ pc["host"] = "ProcLoop"
    /\ IF drain_idx > Len(drain_list)
          THEN /\ drain_list' = << >>
               /\ drain_idx' = 0
               /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
          ELSE /\ pc' = [pc EXCEPT !["host"] = "ProcPacket"]
               /\ UNCHANGED << drain_list, drain_idx >>
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, h_old_head, ok, lh, ln, my_pkt, hok >>

ProcPacket ==
    /\ pc["host"] = "ProcPacket"
    /\ pkt_state' = [pkt_state EXCEPT ![drain_list[drain_idx]] = "Processing"]
    /\ pc' = [pc EXCEPT !["host"] = "SetDone"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok >>

SetDone ==
    /\ pc["host"] = "SetDone"
    /\ pkt_state' = [pkt_state EXCEPT ![drain_list[drain_idx]] = "Done"]
    /\ pkt_next' = [pkt_next EXCEPT ![drain_list[drain_idx]] = NullTP]
    /\ drain_idx' = drain_idx + 1
    /\ pc' = [pc EXCEPT !["host"] = "ProcLoop"]
    /\ UNCHANGED << free_head, ready_head, doorbell, host_seen, gpu_owns,
                    drain_list, h_old_head, ok, lh, ln, my_pkt, hok >>

(* --- Next-state relation --- *)

gpu(self) ==
    \/ PopFree_Read(self)
    \/ PopFree_Next(self)
    \/ PopFree_CAS(self)
    \/ Fill(self)
    \/ PushReady_Read(self)
    \/ PushReady_Link(self)
    \/ PushReady_CAS(self)
    \/ RingDoorbell(self)
    \/ SpinWait(self)
    \/ ReadResp(self)
    \/ RelFree_Read(self)
    \/ RelFree_Link(self)
    \/ RelFree_CAS(self)
    \/ GpuLoop(self)

host == \/ PollDoorbell
        \/ SwapDrain_R
        \/ SwapDrain_CAS
        \/ ProcLoop
        \/ ProcPacket
        \/ SetDone

Next == (\E self \in GPU_THREADS: gpu(self)) \/ host

Spec == Init /\ [][Next]_vars
            /\ \A self \in GPU_THREADS: WF_vars(gpu(self))
            /\ WF_vars(host)

\* END TRANSLATION

----------------------------------------------------------------------------
(*
 * ================================================================
 *  LIVENESS (TEMPORAL) PROPERTIES
 *
 *  These use the ~> (leads-to) operator and require fairness
 *  assumptions (provided by the fair process declarations above
 *  via WF_vars in Spec).
 * ================================================================
 *)

(*
 * LIVE-1: Response delivery.
 * Every packet that reaches "Ready" state is eventually processed
 * by the host and marked "Done".
 *)
ResponseDelivery ==
    \A p \in PACKETS:
        (pkt_state[p] = "Ready") ~> (pkt_state[p] = "Done")

(*
 * LIVE-2: Packet recycling.
 * Every packet that reaches "Done" state eventually returns to "Free".
 *)
PacketRecycling ==
    \A p \in PACKETS:
        (pkt_state[p] = "Done") ~> (pkt_state[p] = "Free")

(*
 * LIVE-3: Full lifecycle.
 * Every packet that enters "Filling" eventually returns to "Free".
 *)
FullLifecycle ==
    \A p \in PACKETS:
        (pkt_state[p] = "Filling") ~> (pkt_state[p] = "Free")

=============================================================================
\* Modification History
\* Created 2026-03-15 for async_gpu hostcall protocol verification
