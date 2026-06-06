# docs-refresh.2: Rewrite ARCHITECTURE.md

## Summary

Complete rewrite of `docs/ARCHITECTURE.md` from 348 lines of outdated content
to 396 lines reflecting the current 15-crate architecture. All sections written
from source code inspection, not from the old document.

## Findings

1. **Crate map**: Documented all 15 crates across 4 layers (Facade, Core,
   Kernel, Test) with accurate descriptions derived from Cargo.toml and
   lib.rs files.

2. **Build system**: Two-workspace model documented (host x86_64 vs GPU
   nvptx64). Build-kernels.sh flow captured: sequential PTX builds, parallel
   ptxas in prod mode, PTX auto-discovery catalog.

3. **Compilation pipeline**: Full flow from Rust source through MIR passes
   (StateTransform, WarpCooperative), LLVM nvptx64, PTX, optional cubin,
   to runtime loading via cuModuleLoadData.

4. **Key subsystems added**: AutoScheduler (3 variants), AutoTuner (warmup
   search + TuningCache), FlightRecorder, GPU panic handler, CUDA streams,
   par_iter, 3-tier channels, GPU executor, structured concurrency
   (BlockScope/GridScope), tiered memory (SharedRef/GlobalRef), GpuArray<T>
   (4-state residency).

5. **Neural network stack**: nn module tree (GpuTensor, ops, layers, autograd,
   fusion, models), ONNX runtime (proto + executor + fusion).

6. **Hostcall protocol**: Retained accurate protocol description, condensed
   services listing to avoid duplicating the protocol spec.

7. **Data flow diagram**: Compact side-by-side compile-time/run-time ASCII
   diagram showing the full path.

## Open Questions

1. The `warp-macro` crate documented in the old ARCHITECTURE.md does not
   exist in the repo. The old doc's `#[warp_async]` proc macro section was
   intentionally omitted since there is no corresponding crate.

2. `docs/` is still gitignored. The rewrite improves local documentation
   quality but remains invisible to collaborators until un-gitignored.
