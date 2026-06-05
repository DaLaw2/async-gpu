# split-loader theme synthesis

**Theme**: split-loader — Host-side multi-PTX loading for kernel split
**Epic**: kernel-split (T0)

## Status

Task split-loader.1 (per-crate PTX constants) complete. The host now
embeds PTX from all 4 kernel crates with backward-compatible aliases.

## What shipped

- 4 canonical constants: `KERNEL_{CORE,COMPUTE,IO,TEST}`
- 2 aliases: `KERNEL → KERNEL_COMPUTE`, `KERNEL_STD → KERNEL_TEST`
- build.rs builds all 4 kernel crates and copies PTX
- Zero call-site breakage; all existing code compiles unchanged

## What remains

- Cubin embedding (`include_bytes!` for pre-compiled cubins) — deferred
- `PtxModule` struct + `ALL` catalog for auto-discovery — deferred
- `.module()` builder on `CustomLaunchBuilder` — deferred
- Deprecation warnings on aliases + call-site migration — Phase 3
- `gpu::run_any()` cross-module kernel search — Phase 3

## PTX size breakdown

core: 1.0 MB | compute: 1.4 MB | io: 1.8 MB | test: 7.2 MB
Total: ~11.4 MB (vs 9.9 MB monolithic — slight overhead from duplicated
std shims across crates, expected to shrink with shared-core dedup).
