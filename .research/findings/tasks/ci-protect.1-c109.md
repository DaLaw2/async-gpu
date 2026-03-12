# ci-protect.1: Audit and harden CI coverage
**Cycle**: 109 | **Theme**: ci-protect | **Kind**: investigation | **Status**: done

## Summary

Current CI covers only 3 of 14 crates (gpu-host, gpu-protocol, warp-macro) plus the hello-gpu example. The 11 GPU-side (`no_std`/nvptx64) crates are partially covered through gpu-kernel (which depends on most of them), but 6 leaf crates are not individually checked. Host-side unit tests can be added for gpu-protocol (pure logic, already has doc-tests) and warp-macro (proc-macro, runs on host). The hello-gpu build is already verified in CI and covers the kernel+host end-to-end compile path.

## Findings

### Q: What crates/examples are NOT covered by current CI?

A: The CI has 3 jobs covering these crates:

| CI Job | Crates Covered | Operations |
|--------|---------------|------------|
| lint | gpu-host, gpu-protocol, warp-macro | fmt, clippy, doc-tests (protocol only) |
| build-ptx | gpu-kernel (+ transitive deps), hello-gpu/kernel | PTX compile, PTX file check |
| build-host | gpu-host | cargo check |

**Crates NOT directly covered by any CI job:**

1. **gpu-atomics** — nvptx64-only, inline PTX asm. Transitively built via gpu-kernel.
2. **gpu-critical-section** — nvptx64-only. Transitively built via gpu-kernel.
3. **gpu-libc** — nvptx64-only. Transitively built via gpu-kernel.
4. **gpu-runtime** — nvptx64-only. Transitively built via gpu-kernel (and hello-gpu/kernel).
5. **async-hostcall-test** — nvptx64 test kernel. NOT built in CI at all.
6. **async-pipeline-test** — nvptx64 test kernel. NOT built in CI at all.
7. **embassy-test** — nvptx64 test kernel. NOT built in CI at all.
8. **gpu-std-test** — nvptx64 test kernel. NOT built in CI at all.
9. **multi-warp-test** — nvptx64 test kernel. NOT built in CI at all.
10. **std-build-test** — nvptx64 test kernel (builds std, not just core). NOT built in CI at all.

The first 4 (gpu-atomics, gpu-critical-section, gpu-libc, gpu-runtime) get implicit compile coverage through gpu-kernel's dependency tree. But the 6 test kernels (*-test) have no CI coverage at all — a breakage there would go undetected.

**Lint gaps:** gpu-host, gpu-protocol, and warp-macro have fmt+clippy. The 11 GPU crates have no fmt or clippy checks (though clippy may not work with nvptx64 target).

**Confidence**: high

### Q: Can we add host-side unit tests that don't require a GPU?

A: Yes, there are clear opportunities:

1. **gpu-protocol** (best candidate): Pure `no_std` logic crate with zero hardware dependencies. Already has 6 doc-tests that run in CI (`cargo +stable test --doc`). Could add a `#[cfg(test)]` module with unit tests for:
   - `make_tagged` / `tagged_index` / `tagged_tag` round-trip
   - `encode_error` / `error_category` / `error_raw_errno` round-trip
   - `encode_panic_metadata` / `panic_thread_idx` / `panic_block_idx` / `panic_msg_len` round-trip
   - `packet_offset`, `packet_offset_sharded`, `buffer_size`, `buffer_size_sharded` arithmetic
   - `payload_slot_offset` boundary calculations
   - `shard_entry_offset` calculations
   - `null_tagged()` produces NULL_INDEX
   - Edge cases: max index (0xFFFE), zero shards, large shard counts

2. **warp-macro** (good candidate): Proc-macro crate, runs on host. 736 lines of code transforming `#[warp_async]` functions into state machines. Currently has ZERO tests. Could add:
   - Compile-pass tests via `trybuild` crate
   - Output verification tests (expand macro, compare expected tokens)
   - Error case tests (invalid input produces good error messages)

3. **gpu-host** (limited without GPU): Depends on `cudarc` which needs CUDA runtime. The `io_error_to_category()` function is pure logic and testable. The `CannedStdin` implementation is testable. But `HostcallBuffer` requires CUDA to allocate, so most of this crate cannot be unit-tested without a GPU or a mock CUDA layer.

**Confidence**: high

### Q: Should we add a CI job that verifies hello-gpu builds end-to-end?

A: This is **already done** in the `build-ptx` job. It:
1. Builds `hello-gpu/kernel` to PTX (`cargo build --release` in kernel dir)
2. Checks that `hello-gpu/host` compiles (`cargo +stable check`)

What is NOT covered is actually **running** hello-gpu (which would require a GPU in CI). For compile-only verification, the current coverage is adequate.

However, there are gaps worth addressing:
- The 6 test kernel crates (async-hostcall-test, async-pipeline-test, embassy-test, gpu-std-test, multi-warp-test, std-build-test) are not compiled in CI. Adding a `build-test-kernels` job would catch breakages in these important validation crates.
- std-build-test is special: it builds with `build-std = ["std", "core", "panic_abort"]` which is a more aggressive compilation target than gpu-kernel. A CI failure there would be a critical signal.

**Confidence**: high

## Unexpected Discoveries

1. **warp-macro falsely detected as `#![no_std]`**: The grep for `#![no_std]` returned a match in the warp-macro directory, but manual inspection shows warp-macro is a standard proc-macro crate (no `#![no_std]` in lib.rs). The false positive likely came from a file in the target/ directory or a dependency.

2. **No unit tests anywhere**: None of the 14 crates have `#[test]` functions. The only tests in the entire project are doc-tests in gpu-protocol. This is a significant testing gap.

3. **gpu-host has a main.rs**: The gpu-host crate has both `lib.rs` and `main.rs`, suggesting it can be run as a binary. The CI only checks it compiles (`cargo check`), not that it builds as a binary.

## Open Questions

1. Can `cargo clippy` run against nvptx64 crates? If so, adding clippy checks for GPU crates would catch code quality issues. If not, at minimum `cargo fmt --check` should be added (formatting is target-independent).
2. Should CI compile the 6 test kernels in the same `build-ptx` job or a separate job? Same job saves runner time; separate job gives clearer failure signals.
3. Is there value in adding a GPU-equipped CI runner (self-hosted) for integration tests? This would enable actual kernel execution verification.

## Impact on Downstream Tasks

- **ci-protect.2** (if created): Should add PTX compilation for the 6 test kernel crates and `cargo fmt --check` for all crates.
- **ci-protect.3** (if created): Should add unit tests for gpu-protocol (pure logic) and warp-macro (proc-macro compile tests).
- **api-cleanup**: warp-macro's lack of tests is a risk for any API changes to the `#[warp_async]` attribute.
