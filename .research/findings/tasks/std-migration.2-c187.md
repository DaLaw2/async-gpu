# std-migration.2: Factor shared hostcall protocol into std-compatible library
**Cycle**: 187 | **Theme**: std-migration | **Kind**: design | **Status**: done

## Summary
No code changes needed. gpu-runtime (no_std) already works as a dependency of std-based
crates. gpu-kernel-std successfully uses `gpu_runtime::hostcall::*` functions. The existing
architecture eliminates code duplication without any refactoring.

## Findings

### Q: Can gpu-runtime be made std-compatible (remove no_std requirement)?
A: Not necessary. Rust's `#![no_std]` crates can be used as dependencies by `std` crates
without any modification. gpu-kernel-std already depends on gpu-runtime and uses its hostcall
functions. The `no_std` attribute only means the crate itself doesn't pull in std — it doesn't
prevent std-based code from depending on it.

Dependency chain (working):
```
gpu-kernel-std (std, -Zbuild-std=std)
  └── gpu-runtime (no_std)
        ├── gpu-atomics (no_std)
        ├── gpu-protocol (no_std)
        └── gpu-critical-section (no_std)
```

This is the ideal architecture: protocol and runtime code are `no_std` (works on GPU),
while kernel crates choose whether to use `std` or `no_std`.

**Confidence**: high (verified by successful gpu-kernel-std build)

### Q: What is the minimal shared interface between std and no_std kernel crates?
A: The existing `gpu_runtime::hostcall` module IS the shared interface:
- `gpu_hostcall_print()` — used by both gpu-kernel and gpu-kernel-std
- `gpu_hostcall_request()` — used for file I/O, stdin, time, etc.
- `gpu_hostcall_release()` — packet lifecycle
- `gpu_panic_init()` / `gpu_result_init()` — error reporting
- All `gpu_protocol::*` constants

No code was duplicated between std-build-test (430+ lines of inline PTX) and the new
gpu-kernel-std (0 lines duplicated — delegates to gpu-runtime).

**Confidence**: high

## Impact on Downstream Tasks
- std-migration.3 unblocked — can proceed with porting async pipeline kernels
- No ADR needed — existing architecture is correct
