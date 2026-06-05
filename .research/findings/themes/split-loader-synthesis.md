# split-loader theme synthesis

**Theme**: split-loader — Host-side multi-PTX loading for kernel split
**Epic**: kernel-split (T0)

## Status

Tasks split-loader.1 through split-loader.4 complete. Multi-PTX loading,
per-crate cubin support, and dev-mode build profiles all in place.

## What shipped

- 4 canonical constants: `KERNEL_{CORE,COMPUTE,IO,TEST}` + aliases
- `PtxModule` struct + `ALL` catalog for auto-discovery
- `.module(&PtxModule)` on `CustomLaunchBuilder`
- Per-crate cubin builds (ptxas for each crate independently)
- Cubin embedding via `include_bytes!` + `PtxModule.cubin` field
- Dev-mode profiles: opt-level 1 default, `--prod` for full opt
- build-kernels.sh and build-kernel-test.sh support `--prod` flag

## What remains

- Deprecation warnings on aliases + call-site migration (Phase 3)
- `gpu::run_any()` cross-module kernel search (Phase 3)

## PTX size breakdown (dev / prod)

core: 1.3/1.0 MB | compute: 1.9/1.4 MB | io: 2.3/1.8 MB | test: 5.9/6.9 MB
