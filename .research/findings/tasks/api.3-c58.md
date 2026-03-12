# api.3: End-to-end example with build script
**Cycle**: 58 | **Theme**: api | **Kind**: experiment | **Status**: done

## Summary
Created `examples/hello-gpu/` — a complete, standalone example demonstrating how to
build and run a GPU kernel using `gpu-runtime`. Includes a kernel crate, host binary,
and build.rs that auto-compiles the kernel to PTX. Both `vector_add` (pure compute) and
`hello_gpu` (hostcall PRINT) work end-to-end.

## Findings

### Q: Can a build.rs automate PTX compilation and embedding?
A: **Yes, with caveats.** The build.rs invokes `cargo +nightly-2025-08-25 build --release`
on the kernel crate and copies the resulting PTX to `OUT_DIR` for `include_str!`.

**Critical caveat**: llvm-bitcode-linker on the latest nightly emits `.target sm_30` and
`.version 6.0` in the PTX header even when compiled with `-C target-cpu=sm_86`. The
actual instructions ARE sm_86 (e.g., `nanosleep`, `.sys`-scope atomics), so the CUDA JIT
rejects the PTX as invalid. The build.rs must either:
1. Patch the header: `.target sm_30` → `.target sm_86`, `.version 6.0` → `.version 7.1`
2. Or use the pinned `nightly-2025-08-25` (which emits correct headers) with env cleanup

We implemented option 1 (patch if needed) for robustness across toolchain versions.

**Additional caveat**: When build.rs spawns cargo as a subprocess, it inherits the parent
cargo's environment variables (`CARGO`, `RUSTC`, `CARGO_TARGET_DIR`, etc.). These must be
removed via `env_remove()` to prevent interference with the kernel build.

**Confidence**: high (verified on RTX 3060, SM_86)

### Q: What does a minimal but complete GPU kernel project look like?
A: Two crates in the same directory:

```
examples/hello-gpu/
├── kernel/                    # GPU kernel crate
│   ├── Cargo.toml             # depends on gpu-runtime
│   ├── .cargo/config.toml     # nvptx64 target, build-std, sm_86
│   └── src/lib.rs             # #![no_std] kernel functions
└── host/                      # Host binary
    ├── Cargo.toml             # depends on cudarc, gpu-protocol
    ├── build.rs               # auto-compiles kernel to PTX
    └── src/main.rs            # CUDA init, PTX load, kernel launch
```

The kernel crate depends only on `gpu-runtime` (one dependency). The host crate needs
`cudarc` for CUDA driver API and `gpu-protocol` for shared constants. The hostcall
listener is included inline in `main.rs` (a simplified version of `gpu-host`'s listener).

**Confidence**: high

### Q: Can a newcomer follow the example without reading internals?
A: **Mostly.** The example is self-contained and includes:
- `vector_add`: simple compute kernel (no hostcall needed)
- `hello_gpu`: hostcall PRINT via `gpu_runtime::prelude::*`

A newcomer would need to understand:
1. `#![no_std]` + `extern "ptx-kernel"` for GPU entry points
2. The hostcall buffer setup (pinned mapped memory)
3. The host listener pattern (poll doorbell → grab ready stack → dispatch)

The `gpu-runtime` prelude makes kernel code ergonomic — just `use gpu_runtime::prelude::*`
and call `gpu_hostcall_print()`. No need to understand CAS, tagged pointers, or packet layout.

**Confidence**: medium (no user testing performed, but code is straightforward)

## Files Created
- `examples/hello-gpu/kernel/Cargo.toml`
- `examples/hello-gpu/kernel/.cargo/config.toml`
- `examples/hello-gpu/kernel/src/lib.rs`
- `examples/hello-gpu/host/Cargo.toml`
- `examples/hello-gpu/host/build.rs`
- `examples/hello-gpu/host/src/main.rs`

## Unexpected Discoveries
- llvm-bitcode-linker PTX target/version header mismatch is a significant usability
  issue — newcomers would hit CUDA_ERROR_INVALID_PTX with no obvious cause
- The parent cargo environment leaks into build.rs subprocess — `env_remove()` is essential
- The pinned nightly-2025-08-25 produces correct PTX headers; latest nightly does not

## Open Questions
- Should we file a bug against llvm-bitcode-linker for the target header issue?
- Would a `gpu-host-lib` crate (library version of gpu-host) simplify the host side?

## Impact on Downstream Tasks
- The PTX target header bug should be documented in README
- The build.rs pattern can be reused for other examples or user projects
- api theme is now effectively complete (api.1 design + api.2 facade + api.3 example)
