# arch-doc.1: Design ARCHITECTURE.md structure
**Cycle**: 288 | **Theme**: arch-doc | **Kind**: design | **Status**: done

## Summary
Researched all major subsystems. ARCHITECTURE.md will cover 6 sections: Overview, Hostcall Protocol, Warp-Cooperative MIR Pass, PAL Layer, Crate Map, Build Pipeline. Key constants and design decisions documented.

## Structure

1. **Overview** — What this project does, one-paragraph summary
2. **Hostcall Protocol** — Two-stack lock-free design, packet layout, service dispatch, sideband bulk I/O
3. **Warp-Cooperative Async** — MIR pass mechanics, bar.warp.sync + shfl.sync insertion, lane-0 leadership
4. **Platform Adaptation Layer** — gpu-libc shim, patched std, memory allocator, errno mapping
5. **Crate Map** — Dependency graph, GPU vs host split, what each crate does
6. **Build Pipeline** — Toolchain setup, kernel compilation, PTX post-processing, host build

**Confidence**: high
