# toolchain.3: VectorWare gpu-kernel ABI Analysis — DECISION GATE
**Date**: 2026-03-11
**Cycle**: 1
**Theme**: toolchain
**Kind**: investigation
**Status**: done

## Summary

The `extern "gpu-kernel"` ABI has been upstreamed to rust-lang/rust as of January 2025 (PR #135047, merged 2025-01-17), making it available on nightly Rust under `#![feature(abi_gpu_kernel)]`. VectorWare does not maintain a public rustc fork; their blog articles use the older `extern "ptx-kernel"` ABI (already in nightly under `#![feature(abi_ptx)]`), meaning no custom compiler patch is required to reproduce their async/await GPU demos. The decision is to proceed with `ptx-kernel` for immediate work while planning a migration path to `gpu-kernel` as it stabilizes.

---

## Detailed Findings

### Q1: Upstream Status of `gpu-kernel` ABI

The `gpu-kernel` ABI is **present in upstream nightly Rust** and requires no external patches.

Key milestones:
- **PR #135047** ("Add gpu-kernel calling convention" / "Revive amdgpu-kernel calling convention") — merged **2025-01-17**, authored by Flakebi (Sebastian Neubauer), reviewed by workingjubilee.
- **PR #149991** ("Add checks for gpu-kernel calling conv") — also merged; enforces restrictions on gpu-kernel functions.
- **Tracking issue**: rust-lang/rust #135467 — open for stabilization tracking.
- **Feature gate**: `#![feature(abi_gpu_kernel)]`

The feature is **experimental/unstable** on nightly. It is not stabilized, and the tracking issue lists unresolved design questions (see Q4). There is no stable-channel support.

Known active bug: rust-lang/rust #144381 — the amdgcn target fails to build `compiler_builtins`, meaning AMDGPU support is partially broken as of mid-2025. NVPTX support is functional.

### Q2: Differences from `ptx-kernel` ABI

Both ABIs produce the same LLVM calling convention for NVPTX targets (`ptx_kernel`), but they differ in scope and design intent:

| Property | `ptx-kernel` | `gpu-kernel` |
|---|---|---|
| Feature gate | `abi_ptx` | `abi_gpu_kernel` |
| Introduced | ~2017 (issue #38788) | January 2025 (PR #135047) |
| NVPTX translation | `ptx_kernel` (LLVM) | `ptx_kernel` (LLVM) — identical |
| AMDGPU translation | N/A (NVPTX only) | `amdgpu_kernel` (LLVM) |
| SPIR-V plan | N/A | Future (unresolved) |
| Enforced restrictions | Minimal | Enforced by PR #149991: no async, no safe calls, must return `()` or `!`, unmangled export name required |
| Deprecation status | Not deprecated | Intended long-term replacement for `ptx-kernel` |
| Stabilization status | Not stabilized (design concerns) | Not stabilized (design questions open) |

The functional difference on NVPTX is negligible: both produce the same PTX global function entry point. The key differences are: `gpu-kernel` is cross-vendor (covers AMD too), has stricter compile-time validation, and is the direction Rust's compiler team is investing in. The `ptx-kernel` ABI is NVPTX-specific and considered legacy.

Critically: `gpu-kernel` **disallows** `async` kernel functions at the ABI level (enforced in PR #149991). This means async/await logic runs *inside* the kernel body, not at the kernel entry point signature — consistent with how VectorWare's demo works.

### Q3: Public Forks and VectorWare's GitHub Presence

VectorWare's GitHub organization is at **https://github.com/vectorware-inc**.

As of the research date, VectorWare has **one public repository**: `rustlantis`, a UB-free deterministic rustc fuzzer (forked from cbeuw/rustlantis, Apache-2.0 licensed). This is a fuzzing/testing tool, not a compiler fork or GPU-specific patch.

**No public rustc fork exists.** VectorWare's blog explicitly states: "we are cleaning up our changes and preparing to open source them" and "we are keen to work upstream as much as possible." This indicates:
1. They have internal changes not yet public.
2. They intend to upstream rather than maintain a fork.
3. The `extern "gpu-kernel"` ABI that is now in nightly was likely driven by VectorWare team members who are also Rust compiler team members (Flakebi/Sebastian Neubauer).

**Important finding**: VectorWare's published async/await demo (blog post "Async/Await on the GPU") uses `extern "ptx-kernel"`, not `extern "gpu-kernel"`. This means their publicly demonstrated work requires only nightly Rust with `#![feature(abi_ptx)]` — no custom patch needed.

### Q4: RFCs and PRs Tracking `gpu-kernel` Progress

No formal Rust RFC has been filed for `gpu-kernel`. The feature was introduced directly via a PR with tracking issue.

Relevant upstream tracking:
- **Tracking issue #135467**: "Tracking Issue for the `gpu-kernel` ABI" — lists open design questions:
  - Should the ABI be renamed `"device-kernel"` instead of `"gpu-kernel"`?
  - How to handle SPIR-V's `OpEntryPoint` (incompatible entry point model)?
  - What constitutes a valid function signature?
  - Safety semantics: should these always require `unsafe`?
  - How to handle breaking ABI changes across GPU targets?
- **Tracking issue #135024**: "Tracking Issue for amdgcn target" — AMDGPU target itself still experimental.
- **Tracking issue #131513**: "Tracking Issue for GPU-offload" — broader GPU offload initiative (`#![feature(gpu_offload)]`), separate from kernel ABI work.
- **Tracking issue #38788**: `ptx-kernel` ABI — open since 2017, still unstable with design concerns.

No stabilization timeline is set for `gpu-kernel`. The feature is under active development but carries open design questions that prevent stabilization.

### Q5: `ptx-kernel` as Sufficient Fallback

**Yes, `ptx-kernel` is a sufficient fallback for the immediate research goals.**

Evidence:
- VectorWare's own published async/await demo uses `extern "ptx-kernel"`, not `extern "gpu-kernel"`. Their world-first GPU async/await demonstration was accomplished entirely with the existing `ptx-kernel` ABI.
- `ptx-kernel` has been in nightly Rust since ~2017 and is functional for NVPTX targets.
- On NVPTX, `gpu-kernel` and `ptx-kernel` produce identical LLVM IR (both lower to `ptx_kernel` calling convention).
- The practical difference for this project's Phase 1 goals (reproduce VectorWare's async/await on GPU) is zero.

Limitations of `ptx-kernel`:
- NVPTX-only (not portable to AMD).
- No enforced compile-time restrictions (could silently produce incorrect code).
- Not the long-term direction of the Rust project.
- Stabilization remains blocked by design concerns for over 7 years.

### Q6: Estimated Scope of a `gpu-kernel` Rustc Patch

Since `gpu-kernel` is **already in upstream nightly Rust**, writing a custom rustc patch is unnecessary. The feature just needs the nightly compiler + `#![feature(abi_gpu_kernel)]`.

However, if a patch were needed for a hypothetical extension (e.g., adding SPIR-V support, or adding gpu-kernel support to a stable compiler), the estimated scope based on PR #135047 analysis:
- **Core ABI addition**: ~200-400 lines of Rust across `compiler/rustc_abi`, `compiler/rustc_codegen_llvm`, and `compiler/rustc_middle`. PR #135047 was a relatively contained change.
- **Validation/checks**: PR #149991 adds compile-time checks — similar scale.
- **Total estimated scope**: Small-to-medium (~500-800 lines across ~6-10 files), well within a contributor's reach, given the pattern is already established by Flakebi's work.

The actual remaining work for this project: none at the ABI layer. Both `ptx-kernel` and `gpu-kernel` are available in nightly.

### Q7: DECISION

**Decision: Proceed with `ptx-kernel` now; plan incremental migration to `gpu-kernel`.**

Rationale:
1. **No compiler patch needed.** Both ABIs are in upstream nightly Rust. The `extern "ptx-kernel"` ABI (`#![feature(abi_ptx)]`) is sufficient and is what VectorWare's own published demos use.
2. **Not blocked.** The project can begin immediately with `nightly` Rust targeting `nvptx64-nvidia-cuda`.
3. **`gpu-kernel` is available but immature.** It is present in nightly (since January 2025) but has unresolved design questions, a known AMDGPU bug (#144381), and no stabilization timeline. Using it today adds instability risk without benefit.
4. **Migration path is clear.** Once `gpu-kernel` progresses toward stabilization (or once AMDGPU portability becomes a project goal), migration is straightforward — the NVPTX behavior is identical.
5. **No VectorWare fork to track.** VectorWare has no public rustc fork. Their internal changes are being prepared for open-sourcing and upstream contribution. We do not need to wait for or reverse-engineer a private fork.

**Action items from this decision:**
- Use `nightly` Rust + `#![feature(abi_ptx)]` + `extern "ptx-kernel"` for initial kernel work.
- Monitor rust-lang/rust #135467 for `gpu-kernel` stabilization progress.
- When AMD portability is needed, switch to `gpu-kernel` (same NVPTX behavior, adds AMD coverage).
- Do not write a custom rustc patch — it would be redundant with upstream work.

---

## Unexpected Discoveries

1. **VectorWare's async demo uses `ptx-kernel`, not `gpu-kernel`.** Despite VectorWare being credited with the `gpu-kernel` motivation (their team members drove PR #135047), their published technical demonstration uses the older `ptx-kernel`. This suggests the async/await work was done before or in parallel with the `gpu-kernel` upstreaming effort, and the two tracks are somewhat independent.

2. **`gpu-kernel` explicitly blocks `async` at the ABI boundary.** PR #149991 enforces that `extern "gpu-kernel"` functions cannot be `async`. This is not a limitation — it is correct design: async/await runs *inside* the kernel, not at the kernel signature. The kernel entry point is a synchronous `extern` function that calls `block_on` or a similar executor internally.

3. **`gpu-kernel` was upstreamed by Flakebi (Sebastian Neubauer)**, who is affiliated with AMD/GPU work in the Rust community and appears to be a VectorWare team member or close collaborator. The feature effectively landed in upstream Rust as part of VectorWare's infrastructure groundwork.

4. **`ptx-kernel` has been unstable for over 7 years** (tracking issue #38788 opened 2016). It will likely remain unstable indefinitely, but it is functional and widely used by the community (rust-cuda, etc.).

5. **VectorWare's only public GitHub repo is a rustc fuzzer (`rustlantis`).** Their GPU/std work remains private as of the research date, consistent with their stated intent to "clean up and open source" before publishing.

---

## Key Conclusions

- `gpu-kernel` ABI is upstreamed (nightly, not stable). Feature gate: `#![feature(abi_gpu_kernel)]`. No patch required.
- `ptx-kernel` ABI is the practical choice today: proven, functional on NVPTX, used by VectorWare's own published demos.
- No VectorWare rustc fork exists publicly. No reverse engineering needed.
- The project is **not blocked** by ABI availability.
- Both ABIs require `nightly` Rust — there is no stable-channel path for GPU kernel development.
- On NVPTX, `ptx-kernel` and `gpu-kernel` are functionally equivalent.

---

## Open Questions

1. What nightly version range is compatible with `gpu-kernel` and also works with rust-cuda's toolchain? (rust-cuda pins to specific nightlies, e.g., `nightly-2025-06-23`).
2. Will VectorWare open-source their `std` port and executor implementation? Timeline unknown.
3. Does the `async` restriction in `gpu-kernel` (PR #149991) affect any upstream plans to have async kernel entry points? Or is the intended model always synchronous entry + async interior?
4. SPIR-V / Vulkan Compute path: will `gpu-kernel` ever cover it, or will that require a separate ABI (`spirv-kernel`)?
5. With `ptx-kernel` blocked on stabilization for 7+ years, is there risk that nightly breakage could affect project continuity?

---

## Impact on Downstream Tasks

- **toolchain.1 / toolchain.2** (compiler setup, target configuration): Use `nightly` Rust, `nvptx64-nvidia-cuda` target, `#![feature(abi_ptx)]`. Straightforward — no custom toolchain needed.
- **hostcall tasks**: Hostcall mechanism is orthogonal to kernel ABI choice. Both `ptx-kernel` and `gpu-kernel` can coexist with hostcall implementations.
- **gpu-std tasks**: VectorWare's std port is not yet public. Must implement independently. The ABI choice does not block std porting work.
- **async-runtime tasks**: The async executor runs *inside* the kernel (called from the synchronous entry point). This architecture is confirmed by both VectorWare's demo and the `gpu-kernel` restriction in PR #149991. Embassy (embedded executor) is the confirmed approach, requires minimal modification for GPU no_std.
- **integration tasks**: Can begin with `ptx-kernel`. Migration to `gpu-kernel` later when AMDGPU portability matters.

---

## Theme Progress

The `toolchain` theme's critical blocker question — "do we need a custom rustc patch or fork to use `gpu-kernel`?" — is resolved: **No**. Both relevant ABIs are in upstream nightly. The toolchain theme can proceed to build configuration, target setup, and linking infrastructure tasks without waiting for any compiler-level work.

---

## Sources

- rust-lang/rust tracking issue #135467: https://github.com/rust-lang/rust/issues/135467
- rust-lang/rust PR #135047: https://github.com/rust-lang/rust/pull/135047
- rust-lang/rust tracking issue #38788 (ptx-kernel): https://github.com/rust-lang/rust/issues/38788
- VectorWare blog — Rust std on GPU: https://www.vectorware.com/blog/rust-std-on-gpu/
- VectorWare blog — Async/Await on the GPU: https://www.vectorware.com/blog/async-await-on-gpu/
- VectorWare GitHub organization: https://github.com/vectorware-inc
- Rust-GPU/Rust-CUDA kernel ABI guide: https://github.com/Rust-GPU/Rust-CUDA/blob/main/guide/src/guide/kernel_abi.md
- Rust unstable book — abi_ptx: https://doc.rust-lang.org/stable/unstable-book/language-features/abi-ptx.html
- Rust compiler dev guide — GPU offload internals: https://rustc-dev-guide.rust-lang.org/offload/internals.html
- Rust CUDA August 2025 update: https://rust-gpu.github.io/blog/2025/08/11/rust-cuda-update/
