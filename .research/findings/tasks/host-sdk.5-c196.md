# host-sdk.5: Automate build system — single cargo command
**Cycle**: 196 | **Theme**: host-sdk | **Kind**: experiment | **Status**: done

## Summary
Build automation is already implemented across all three examples via build.rs
scripts. Each example's build.rs: (1) invokes `cargo +nightly` on the kernel
crate, (2) copies the resulting PTX to OUT_DIR, (3) the host `include_str!`s it.
A single `cargo build` compiles both host and kernel.

## Findings
### Q: Can a build.rs compile the kernel PTX automatically?
A: Yes, already implemented in all 3 examples. The pattern is:
1. `Command::new("cargo").args(["+nightly-2026-03-11", "build", "--release"])`
   with `.current_dir(&kernel_dir)` and env var cleanup to isolate from parent cargo
2. Copy PTX from kernel's target dir to `OUT_DIR`
3. Patch `.target sm_30` → `.target sm_86` if needed (LLVM codegen quirk)
4. Host code: `const KERNEL_PTX: &str = include_str!(concat!(env!("OUT_DIR"), "/kernel.ptx"))`
**Confidence**: high

### Q: How to handle the nightly + nvptx64 toolchain requirement?
A: Build.rs specifies the exact nightly version (`+nightly-2026-03-11`).
When the toolchain is unavailable (e.g., during `cargo clippy`), build.rs
falls back to cached PTX if available, or panics with a clear message.
The kernel crate's `.cargo/config.toml` provides target and build-std settings.
**Confidence**: high

## Unexpected Discoveries
None — the automation was implemented as part of host-sdk.3 and applied to all
examples in host-sdk.4.

## Open Questions
None.

## Impact on Downstream Tasks
- public-api criterion 3 ("Build system automated") is met
- host-sdk theme is now complete: all 5 tasks done
