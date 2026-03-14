# Brainstorm 84 — Skeptic: Post-v0.1.0 Strategic Directions
**Cycle**: 294 | **Date**: 2026-03-15 | **Level**: Deep

## Challenges to Proposer's Analysis

### On Multi-Warp Concurrent Async (P1)

The proposer claims this is "must do" but overlooks a critical constraint: **the research agent cannot test GPU code**. Writing a multi-warp kernel that cannot be tested is of questionable value — untested GPU code is likely broken GPU code. The host infrastructure "supports" multi-warp in theory, but whether it actually works under concurrent load (race conditions, ABA bugs, packet pool exhaustion) is unknown until tested on hardware.

**Counter-recommendation**: Design the architecture and write the code, but tag it as UNTESTED and leave testing to the user. Alternatively, focus on work that can be verified without GPU hardware.

### On Performance Benchmarks (P2)

Agreed that benchmarks are important, but benchmarks without a GPU to run them on are just benchmark code. The agent should write the infrastructure (kernel + host), but the actual numbers are zero until executed.

### On Network I/O (P4)

The proposer dismisses this as a "party trick" but it is actually the most implementable direction: the host-side service handlers are pure Rust (std::net), compile on stable, and CI can verify they build. The GPU-side Futures follow the established pattern exactly. This is HIGH feasibility work that the agent can complete and verify (compilation) without GPU hardware.

**Counter-recommendation**: Elevate network I/O to P2 for the agent's work queue. It is the most productive direction given the constraint of no GPU hardware.

### On Safety Improvements (P4)

The proposer estimates 10+ tasks but this is inflatable. The 822 unsafe blocks are concentrated in a few patterns:
- CUDA API calls through cudarc (not our unsafety — it is the FFI boundary)
- Raw pointer arithmetic for packet/buffer access (could benefit from safe wrappers)
- Inline PTX assembly (inherently unsafe, but SAFETY comments explain invariants)

A focused audit of the hostcall protocol's unsafe code (the CAS operations, pointer arithmetic) would be higher value than auditing every CUDA wrapper call. Estimate: 3-4 tasks for the critical paths only.

### On Real-World Workloads (P3)

The proposer correctly notes the library gap (no no_std JSON/CSV parser for nvptx64). But the existing parallel-search is already a real workload (byte pattern search). A more ambitious version (multi-file search, regex-like patterns) builds incrementally on proven code rather than requiring new parser libraries.

## Consensus Points

1. The project needs to transition from "it compiles" to "it is useful"
2. Multi-warp is the central thesis proof
3. Documentation and polish are diminishing returns — the project is well-documented
4. Upstreaming, ecosystem, and multi-GPU are premature

## Skeptic's Priority for the Agent

Given the constraint that the agent cannot test GPU code:

1. **Network I/O** — most implementable, fully verifiable via CI compilation
2. **Multi-warp design** — architecture doc only, code marked UNTESTED
3. **Safety audit** — focused on critical hostcall protocol code paths
4. **State.toml archival** — 311 completed tasks and 30+ completed themes are cluttering state
