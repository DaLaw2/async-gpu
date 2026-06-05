# split-loader theme synthesis

**Theme**: split-loader — Host-side multi-PTX loading for kernel split
**Epic**: kernel-split (T0)

## Status

Tasks split-loader.1 and split-loader.2 complete. Per-crate PTX
constants, backward aliases, `PtxModule` catalog, and `.module()`
builder are all in place. Zero call-site breakage.

## What shipped

- 4 canonical constants: `KERNEL_{CORE,COMPUTE,IO,TEST}`
- 2 aliases: `KERNEL -> KERNEL_COMPUTE`, `KERNEL_STD -> KERNEL_TEST`
- `PtxModule` struct + `ALL` catalog for auto-discovery
- `.module(&PtxModule)` on `CustomLaunchBuilder`
- KernelRegistry, GpuStdModule, gpu-test-macro all verified unchanged
- build.rs builds all 4 kernel crates and copies PTX

## What remains

- Per-crate cubin builds (ptxas for each crate independently)
- Cubin embedding via `include_bytes!` + `PtxModule.cubin` field
- Deprecation warnings on aliases + call-site migration (Phase 3)
- `gpu::run_any()` cross-module kernel search (Phase 3)

## PTX size breakdown

core: 1.0 MB | compute: 1.4 MB | io: 1.8 MB | test: 7.2 MB
