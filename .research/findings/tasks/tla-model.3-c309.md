# tla-model.3: Multi-agent contention model, verify safety + liveness
**Cycle**: 309 | **Theme**: tla-model | **Kind**: experiment | **Status**: done

## Summary
Both safety and liveness properties of the CAS hostcall protocol have been formally verified using TLC model checker.

## Safety Verification (3 GPU + 1 host + 3 packets)
- **367,246,473 states generated, 91,811,618 distinct states**
- **Depth 88**, runtime ~12 minutes
- All 5 invariants pass: TypeOK, NoDoubleOwnership, PacketConservation, StateConsistency, StacksDisjoint

## Liveness Verification (2 GPU + 1 host + 2 packets)
- **336,856 states generated, 112,285 distinct states**
- **Depth 80**, runtime 25 seconds
- All 3 temporal properties pass:
  - **ResponseDelivery**: Ready packets eventually become Done (host always processes)
  - **PacketRecycling**: Done packets eventually return to Free (GPU always releases)
  - **FullLifecycle**: Filling packets eventually complete full lifecycle back to Free

## Key Finding: ABA Prevention Confirmed
During development, the model initially used local snapshot tags for push operations, which TLC correctly identified as an ABA vulnerability. Fix: global monotonic tag counters (`free_tag`, `ready_tag`) ensure tags never repeat. The real protocol uses 48-bit epoch counters in a 64-bit tagged pointer, providing the same guarantee.

## Files
- `formal/HostcallProtocol.tla` — 750-line TLA+ spec (pure TLA+, no PlusCal)
- `formal/HostcallProtocol.cfg` — Safety config (3+3)
- `formal/HostcallProtocol-liveness.cfg` — Liveness config (2+2, ACTION_CONSTRAINT)

## Impact on Downstream Tasks
- formal-verification epic: 4/5 criteria met (model, safety, liveness, multi-agent all verified; findings documented here)
- Remaining criterion 5 (edge cases/improvements) — no bugs found, protocol design is sound
