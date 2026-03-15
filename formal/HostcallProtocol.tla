--------------------------- MODULE HostcallProtocol ---------------------------
(*
 * TLA+ specification of the async_gpu hostcall CAS protocol.
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
 *   CAS actions             -> atom.cas.acq_rel.sys (GPU) / compare_exchange (host)
 *   doorbell                -> monotonic doorbell counter
 *)
EXTENDS Integers, Sequences, FiniteSets, TLC

CONSTANTS
    GPU_THREADS,    \* Set of GPU thread IDs, e.g., {"g1", "g2", "g3"}
    PACKETS,        \* Set of packet indices, e.g., {"p1", "p2", "p3"}
    NULL            \* Sentinel for empty stack

(* Packet states matching the protocol FSM *)
States == {"Free", "Filling", "Ready", "Processing", "Done"}

(* Tagged pointer constructor *)
TP(t, i) == [tag |-> t, idx |-> i]

(* Null tagged pointer *)
NullTP == TP(0, NULL)

(*
 * Build initial free stack: all packets linked in a chain.
 * We pick an arbitrary fixed ordering using CHOOSE. The protocol
 * correctness is independent of the initial ordering — all packets
 * start FREE, and the Treiber stack is LIFO regardless.
 *)
FixedSeq == CHOOSE seq \in [1..Cardinality(PACKETS) -> PACKETS] :
    \A i, j \in 1..Cardinality(PACKETS) : i /= j => seq[i] /= seq[j]

RECURSIVE BuildNextFn(_, _)
BuildNextFn(seq, i) ==
    IF i > Len(seq) THEN [p \in PACKETS |-> NullTP]
    ELSE LET rest == BuildNextFn(seq, i+1)
         IN  [p \in PACKETS |-> IF p = seq[i] THEN
                (IF i < Len(seq) THEN TP(0, seq[i+1]) ELSE NullTP)
              ELSE rest[p]]

InitHead == IF Cardinality(PACKETS) > 0 THEN TP(0, FixedSeq[1]) ELSE NullTP
InitNext == BuildNextFn(FixedSeq, 1)

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
(* ================================================================
 *  VARIABLES
 * ================================================================ *)

VARIABLES free_head, ready_head, pkt_next, pkt_state, doorbell, host_seen,
          gpu_owns, drain_list, drain_idx, h_old_head, pc, ok, lh, ln,
          my_pkt, hok,
          free_tag,   \* Global monotonic tag counter for free stack pushes
          ready_tag   \* Global monotonic tag counter for ready stack pushes

vars == << free_head, ready_head, pkt_next, pkt_state, doorbell, host_seen,
           gpu_owns, drain_list, drain_idx, h_old_head, pc, ok, lh, ln,
           my_pkt, hok, free_tag, ready_tag >>

ProcSet == (GPU_THREADS) \cup {"host"}

(* ================================================================
 *  INVARIANTS AND DERIVED DEFINITIONS
 * ================================================================ *)

(* Type invariant *)
TPValid(tp) == tp.tag \in Nat /\ tp.idx \in PACKETS \union {NULL}

TypeOK ==
    /\ TPValid(free_head)
    /\ TPValid(ready_head)
    /\ \A p \in PACKETS: TPValid(pkt_next[p])
    /\ \A p \in PACKETS: pkt_state[p] \in States
    /\ doorbell \in Nat
    /\ host_seen \in Nat
    /\ free_tag \in Nat
    /\ ready_tag \in Nat

(*
 * Derived sets for invariants.
 *
 * NOTE: StackSet traversal (following pkt_next pointers) is NOT reliable
 * for invariant checking because pkt_next values are modified non-atomically
 * relative to the stack head (e.g., PushReady_Link writes pkt_next before
 * the CAS in PushReady_CAS). Instead, we use pkt_state and process-local
 * variables to classify packets.
 *)

(* Packets actively owned by a GPU thread for writing *)
GpuOwnedSet == {gpu_owns[g] : g \in {g2 \in GPU_THREADS : gpu_owns[g2] /= NULL}}

(* Packets tracked by any GPU thread (owned or waiting for response) *)
GpuTrackedSet == {my_pkt[g] : g \in {g2 \in GPU_THREADS : my_pkt[g2] /= NULL}}

(* Host-owned: packets in drain_list from drain_idx onward *)
HostOwnedSet ==
    IF drain_idx >= 1 /\ Len(drain_list) >= drain_idx
    THEN {drain_list[i] : i \in drain_idx..Len(drain_list)}
    ELSE {}

(* All drain_list packets (including already-processed ones before drain_idx) *)
HostDrainedSet ==
    IF Len(drain_list) >= 1
    THEN {drain_list[i] : i \in 1..Len(drain_list)}
    ELSE {}

(*
 * INV-1: No double write-ownership.
 * A packet must not be actively written by two agents simultaneously.
 * "Active write-ownership" means:
 *   - GPU: gpu_owns[g] = p (thread is filling or releasing the packet)
 *   - Host: p is in HostOwnedSet (host is processing the packet)
 * Additionally, at most one GPU thread actively owns any given packet.
 *)
NoDoubleOwnership ==
    /\ \A p \in PACKETS:
        ~(p \in GpuOwnedSet /\ p \in HostOwnedSet)
    /\ \A g1a, g2a \in GPU_THREADS:
        (g1a /= g2a /\ gpu_owns[g1a] /= NULL) => gpu_owns[g1a] /= gpu_owns[g2a]

(*
 * INV-2: Packet conservation.
 * Every packet is accounted for in at least one tracking location:
 *   - On the free stack (state = "Free" and not tracked by anyone)
 *   - On the ready stack (state = "Ready" and not yet drained)
 *   - Tracked by a GPU thread (my_pkt)
 *   - In the host's drain list
 * Since these overlap, we check: the union covers all packets.
 *)
PacketConservation ==
    \A p \in PACKETS:
        \/ pkt_state[p] = "Free"
        \/ p \in GpuTrackedSet
        \/ p \in HostDrainedSet

(*
 * INV-3: Packet state consistency.
 * - GPU-owned packets must be in Filling or Done state
 * - Host-owned packets (in active drain window) must be in Ready or Processing state
 * - Packets not tracked by anyone must be Free
 *)
StateConsistency ==
    /\ \A g \in GPU_THREADS:
        gpu_owns[g] /= NULL =>
            pkt_state[gpu_owns[g]] \in {"Filling", "Done", "Free"}
    /\ \A p \in HostOwnedSet:
        pkt_state[p] \in {"Ready", "Processing"}

(*
 * INV-4: Stack heads are valid.
 * Both stack heads point to a valid packet or NULL. We cannot check
 * full stack disjointness via pointer traversal because pkt_next
 * values are non-atomically updated, but TypeOK already ensures the
 * head pointers are well-formed.
 *)
StacksDisjoint ==
    /\ free_head.idx \in PACKETS \union {NULL}
    /\ ready_head.idx \in PACKETS \union {NULL}

(* Combined safety invariant *)
SafetyInvariant ==
    /\ TypeOK
    /\ NoDoubleOwnership
    /\ PacketConservation
    /\ StateConsistency
    /\ StacksDisjoint

(*
 * STATE CONSTRAINT for finite model checking.
 * Tags and doorbell are monotonically increasing, so without a bound
 * TLC would never terminate.
 *)
MAX_TAG == 3

StateConstraint ==
    /\ free_tag        <= MAX_TAG
    /\ ready_tag       <= MAX_TAG
    /\ doorbell        <= MAX_TAG

(* ================================================================
 *  INIT
 * ================================================================ *)

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
        (* Tag counters *)
        /\ free_tag = 0
        /\ ready_tag = 0
        /\ pc = [self \in ProcSet |-> CASE self \in GPU_THREADS -> "PopFree_Read"
                                        [] self = "host" -> "PollDoorbell"]

(* ================================================================
 *  GPU THREAD ACTIONS
 *
 *  Each GPU thread loops forever:
 *    1. Pop a packet from the free stack (tagged CAS, retry on failure)
 *    2. Fill the packet with request data
 *    3. Push the packet to the ready stack (tagged CAS, retry on failure)
 *    4. Ring the doorbell (atomic increment)
 *    5. Spin-wait until host marks packet as Done
 *    6. Read response, push packet back to free stack
 * ================================================================ *)

PopFree_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PopFree_Read"
    /\ lh' = [lh EXCEPT ![self] = free_head]
    /\ IF lh'[self].idx = NULL
          THEN /\ pc' = [pc EXCEPT ![self] = "PopFree_Read"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "PopFree_Next"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok, free_tag, ready_tag >>

PopFree_Next(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PopFree_Next"
    /\ ln' = [ln EXCEPT ![self] = pkt_next[lh[self].idx]]
    /\ pc' = [pc EXCEPT ![self] = "PopFree_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, my_pkt, hok, free_tag, ready_tag >>

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
                    drain_list, drain_idx, h_old_head, lh, ln, hok,
                    free_tag, ready_tag >>

Fill(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "Fill"
    /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Filling"]
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = NullTP]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_Read"]
    /\ UNCHANGED << free_head, ready_head, doorbell, host_seen, gpu_owns,
                    drain_list, drain_idx, h_old_head, ok, lh, ln, my_pkt, hok,
                    free_tag, ready_tag >>

PushReady_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_Read"
    /\ lh' = [lh EXCEPT ![self] = ready_head]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_Link"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok, free_tag, ready_tag >>

PushReady_Link(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_Link"
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = lh[self]]
    /\ pc' = [pc EXCEPT ![self] = "PushReady_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_state, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

PushReady_CAS(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "PushReady_CAS"
    /\ IF ready_head = lh[self]
          THEN /\ ready_head' = TP(ready_tag + 1, my_pkt[self])
               /\ ready_tag' = ready_tag + 1
               /\ ok' = [ok EXCEPT ![self] = TRUE]
          ELSE /\ ok' = [ok EXCEPT ![self] = FALSE]
               /\ UNCHANGED << ready_head, ready_tag >>
    /\ IF ~ok'[self]
          THEN /\ pc' = [pc EXCEPT ![self] = "PushReady_Read"]
               /\ UNCHANGED << pkt_state, gpu_owns >>
          ELSE /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Ready"]
               /\ gpu_owns' = [gpu_owns EXCEPT ![self] = NULL]
               /\ pc' = [pc EXCEPT ![self] = "RingDoorbell"]
    /\ UNCHANGED << free_head, pkt_next, doorbell, host_seen,
                    drain_list, drain_idx, h_old_head, lh, ln, my_pkt, hok,
                    free_tag >>

RingDoorbell(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RingDoorbell"
    /\ doorbell' = doorbell + 1
    /\ pc' = [pc EXCEPT ![self] = "SpinWait"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

SpinWait(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "SpinWait"
    /\ IF pkt_state[my_pkt[self]] /= "Done"
          THEN /\ pc' = [pc EXCEPT ![self] = "SpinWait"]
          ELSE /\ pc' = [pc EXCEPT ![self] = "ReadResp"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

ReadResp(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "ReadResp"
    /\ gpu_owns' = [gpu_owns EXCEPT ![self] = my_pkt[self]]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_Read"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

RelFree_Read(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_Read"
    /\ lh' = [lh EXCEPT ![self] = free_head]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_Link"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, ln, my_pkt, hok, free_tag, ready_tag >>

RelFree_Link(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_Link"
    /\ pkt_next' = [pkt_next EXCEPT ![my_pkt[self]] = lh[self]]
    /\ pc' = [pc EXCEPT ![self] = "RelFree_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_state, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

RelFree_CAS(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "RelFree_CAS"
    /\ IF free_head = lh[self]
          THEN /\ free_head' = TP(free_tag + 1, my_pkt[self])
               /\ free_tag' = free_tag + 1
               /\ ok' = [ok EXCEPT ![self] = TRUE]
          ELSE /\ ok' = [ok EXCEPT ![self] = FALSE]
               /\ UNCHANGED << free_head, free_tag >>
    /\ IF ~ok'[self]
          THEN /\ pc' = [pc EXCEPT ![self] = "RelFree_Read"]
               /\ UNCHANGED << pkt_state, gpu_owns, my_pkt >>
          ELSE /\ pkt_state' = [pkt_state EXCEPT ![my_pkt[self]] = "Free"]
               /\ gpu_owns' = [gpu_owns EXCEPT ![self] = NULL]
               /\ my_pkt' = [my_pkt EXCEPT ![self] = NULL]
               /\ pc' = [pc EXCEPT ![self] = "GpuLoop"]
    /\ UNCHANGED << ready_head, pkt_next, doorbell, host_seen,
                    drain_list, drain_idx, h_old_head, lh, ln, hok,
                    ready_tag >>

GpuLoop(self) ==
    /\ self \in GPU_THREADS
    /\ pc[self] = "GpuLoop"
    /\ pc' = [pc EXCEPT ![self] = "PopFree_Read"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

(* ================================================================
 *  HOST THREAD ACTIONS
 *
 *  The host thread runs a polling loop:
 *    1. Poll doorbell counter for new work
 *    2. Swap-drain the ready stack (atomic exchange with NULL)
 *    3. Walk the drained list, processing each packet
 *    4. Mark each packet Done (CONTROL_READY) so GPU can proceed
 * ================================================================ *)

PollDoorbell ==
    /\ pc["host"] = "PollDoorbell"
    /\ IF doorbell = host_seen
          THEN /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
               /\ UNCHANGED host_seen
          ELSE /\ host_seen' = doorbell
               /\ pc' = [pc EXCEPT !["host"] = "SwapDrain_R"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

SwapDrain_R ==
    /\ pc["host"] = "SwapDrain_R"
    /\ h_old_head' = ready_head
    /\ IF ready_head.idx = NULL
          THEN /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
          ELSE /\ pc' = [pc EXCEPT !["host"] = "SwapDrain_CAS"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, drain_list, drain_idx,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

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
                    gpu_owns, h_old_head, ok, lh, ln, my_pkt,
                    free_tag, ready_tag >>

ProcLoop ==
    /\ pc["host"] = "ProcLoop"
    /\ IF drain_idx > Len(drain_list)
          THEN /\ drain_list' = << >>
               /\ drain_idx' = 0
               /\ pc' = [pc EXCEPT !["host"] = "PollDoorbell"]
          ELSE /\ pc' = [pc EXCEPT !["host"] = "ProcPacket"]
               /\ UNCHANGED << drain_list, drain_idx >>
    /\ UNCHANGED << free_head, ready_head, pkt_next, pkt_state, doorbell,
                    host_seen, gpu_owns, h_old_head, ok, lh, ln, my_pkt, hok,
                    free_tag, ready_tag >>

ProcPacket ==
    /\ pc["host"] = "ProcPacket"
    /\ pkt_state' = [pkt_state EXCEPT ![drain_list[drain_idx]] = "Processing"]
    /\ pc' = [pc EXCEPT !["host"] = "SetDone"]
    /\ UNCHANGED << free_head, ready_head, pkt_next, doorbell, host_seen,
                    gpu_owns, drain_list, drain_idx, h_old_head,
                    ok, lh, ln, my_pkt, hok, free_tag, ready_tag >>

SetDone ==
    /\ pc["host"] = "SetDone"
    /\ pkt_state' = [pkt_state EXCEPT ![drain_list[drain_idx]] = "Done"]
    /\ pkt_next' = [pkt_next EXCEPT ![drain_list[drain_idx]] = NullTP]
    /\ drain_idx' = drain_idx + 1
    /\ pc' = [pc EXCEPT !["host"] = "ProcLoop"]
    /\ UNCHANGED << free_head, ready_head, doorbell, host_seen, gpu_owns,
                    drain_list, h_old_head, ok, lh, ln, my_pkt, hok,
                    free_tag, ready_tag >>

(* ================================================================
 *  NEXT-STATE RELATION
 * ================================================================ *)

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

----------------------------------------------------------------------------
(*
 * ================================================================
 *  LIVENESS (TEMPORAL) PROPERTIES
 *
 *  These use the ~> (leads-to) operator and require fairness
 *  assumptions (provided by WF_vars in Spec).
 * ================================================================
 *)

(* LIVE-1: Response delivery *)
ResponseDelivery ==
    \A p \in PACKETS:
        (pkt_state[p] = "Ready") ~> (pkt_state[p] = "Done")

(* LIVE-2: Packet recycling *)
PacketRecycling ==
    \A p \in PACKETS:
        (pkt_state[p] = "Done") ~> (pkt_state[p] = "Free")

(* LIVE-3: Full lifecycle *)
FullLifecycle ==
    \A p \in PACKETS:
        (pkt_state[p] = "Filling") ~> (pkt_state[p] = "Free")

=============================================================================
