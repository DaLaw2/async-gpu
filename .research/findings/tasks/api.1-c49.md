# api.1: API surface design
**Cycle**: 49 | **Theme**: api | **Kind**: design | **Status**: done

## Summary
Analyzed the current crate dependency graph and identified API friction points.
Designed a two-tier public API: GPU-side gpu-runtime facade and host-side
GpuContext builder. Key insight: hostcall helpers (hc_pop_free, hc_push,
gpu_hostcall_request) are duplicated across 5+ test crates and should be
consolidated into a shared gpu-hostcall crate.

## Findings

### Q: What is public vs internal?
A: **Public API (GPU-side):**
- gpu-protocol: packet constants, layout helpers, error encoding (already clean)
- gpu-atomics: system-scope atomics, fences, spin-loads (already clean)
- gpu-critical-section: no-op critical-section for Embassy (already clean)
- NEW gpu-hostcall: consolidated hostcall helpers (hc_pop_free, hc_push,
  gpu_hostcall_request, gpu_hostcall_release, typed wrappers for each service)

**Public API (host-side):**
- HostcallBuffer: pinned memory allocation, dev_ptr, listen loop
- alloc_mapped_mem / free_mapped_mem: typed GPU-CPU shared memory
- Service handlers: PRINT, OPEN, WRITE, READ, CLOSE, STDIN, TIME, ABORT

**Internal (not exported):**
- Low-level CAS implementations (internals of gpu-atomics)
- PTX inline asm blocks
- Host listener polling mechanics

**Confidence**: high

### Q: How many crates should a user depend on?
A: Ideally 2: `gpu-runtime` (GPU-side, re-exports everything) and `gpu-host`
(host-side). Currently a kernel author needs gpu-protocol + gpu-atomics +
gpu-critical-section + embassy-executor + manual hostcall code. A facade crate
reduces this to `gpu-runtime` + `embassy-executor`.

**Confidence**: high

### Q: What is the minimal code for hello-world?
A: GPU kernel: ~30 lines (import gpu-runtime, define ptx-kernel fn, call
gpu_hostcall_print()). Host: ~40 lines (CudaDevice::new, load PTX, create
HostcallBuffer, launch kernel, listen, shutdown). Total: ~70 lines for a
working GPU-to-host print.

**Confidence**: medium (depends on how much boilerplate we can hide)

### Q: How to handle PTX compilation in build process?
A: Options:
1. build.rs that invokes `cargo +nightly build --target nvptx64-nvidia-cuda`
   for the kernel crate and copies the PTX output — complex but automated
2. Manual two-step: build kernel, copy PTX, build host — current approach
3. Proc macro `#[gpu_kernel]` — too ambitious for now

Recommend option 1 for api.3, with fallback to option 2 documented.

**Confidence**: medium

## Design Decision

### ADR-5: GPU-side API consolidation

**Decision**: Create `gpu-runtime` facade crate that re-exports:
- `gpu_protocol::*` (constants, layout)
- `gpu_atomics::*` (atomics, fences)
- `gpu_critical_section` (Embassy support)
- New `gpu_hostcall` module with consolidated helpers

**Rationale**: Reduces kernel author imports from 4 crates to 1. Hostcall
helpers are currently copy-pasted across 5 test crates with subtle variations.
Consolidation prevents bugs from diverging implementations.

**Trade-off**: One more crate in the dependency graph, but users see fewer
imports. Fat LTO still merges everything anyway.

### Crate dependency graph (proposed)

```
gpu-runtime (facade)
  ├─ gpu-protocol (re-exported)
  ├─ gpu-atomics (re-exported)
  ├─ gpu-critical-section (re-exported)
  └─ gpu-hostcall (new, re-exported)
       ├─ gpu-protocol
       └─ gpu-atomics
```

## Impact on Downstream Tasks
- api.2: implement gpu-runtime + gpu-hostcall crates
- api.3: create example using gpu-runtime
- Record ADR-5 in decisions.md during api.2
