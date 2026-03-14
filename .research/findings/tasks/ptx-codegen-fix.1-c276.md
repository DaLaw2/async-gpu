# ptx-codegen-fix.1: Investigation — root cause of .ptr .align N
**Cycle**: 276 | **Theme**: ptx-codegen-fix | **Kind**: investigation | **Status**: done

## Summary
Investigated the root cause of `.ptr .align N` annotations in kernel entry parameters. Confirmed the source is **LLVM NVPTXAsmPrinter**, not rustc_codegen_llvm. The issue only affects `extern "ptx-kernel"` function parameters that are raw pointers. Three viable mitigation paths identified, with PTX ISA 7.8+ targeting being the least invasive.

## Findings

### Q: Is .ptr .align emitted by LLVM NVPTXAsmPrinter or by rustc metadata?
A: **LLVM NVPTXAsmPrinter.** The `.ptr .align N` annotation is emitted by LLVM's NVPTX backend when lowering function parameters with pointer type to PTX assembly. Rustc's codegen layer passes LLVM IR with pointer parameters; the NVPTX backend then decides how to print them in PTX. The `.ptr .align 1` means "generic pointer with 1-byte alignment" — LLVM's default when no alignment metadata is provided.

Evidence:
- All 30 occurrences in std-build-test PTX are on `.visible .entry` (kernel) parameters
- The pattern is always `.param .u64 .ptr .align 1 kernel_param_N`
- No `.ptr .align` on internal `.func` declarations — only kernel entry points
- This matches LLVM NVPTXAsmPrinter behavior: kernel params get address space annotations

**Confidence**: high

### Q: Can targeting PTX ISA 7.8+ avoid the issue?
A: **Likely yes, but needs testing.** PTX ISA 7.8 (CUDA 11.8+) changed the semantics of `.ptr .align` annotations — they became optional rather than rejected by older ptxas versions. However, the current build targets PTX 7.1 (`.version 7.1` in output).

To target 7.8+, options:
1. Add `-C llvm-args=-nvptx-ptx-version=78` to rustflags (untested — flag may not exist)
2. Use a newer LLVM that defaults to PTX 7.8+ for sm_86
3. Post-process: simple regex `s/\.ptr\s+\.align\s+\d+//g` (current workaround)

The sm_86 target should support PTX 7.5+ (Ampere introduced in CUDA 11.1, PTX 7.1). PTX 7.8 requires CUDA 11.8+ which sm_86 supports.

**Confidence**: medium (needs experimental verification)

### Q: Can we fix this in rustc_codegen_llvm without patching LLVM source?
A: **Partially.** Two approaches at the rustc level:
1. **Set alignment metadata on kernel params**: In `rustc_codegen_llvm`, add alignment attributes to kernel parameter types. This would change `.align 1` to `.align 8` (proper alignment for u64 pointers) but wouldn't remove `.ptr` entirely.
2. **Post-link transform**: Add a PTX post-processing step in the linker wrapper (llvm-bitcode-linker) that strips `.ptr .align`.

Neither approach fixes the root cause in LLVM. The proper fix is in `llvm/lib/Target/NVPTX/NVPTXAsmPrinter.cpp` — suppress `.ptr .align` for kernel parameters when targeting PTX < 7.8, or omit it entirely since it's not required by the PTX ISA for kernel params.

**Confidence**: medium

### Current Impact Assessment

With std-build-test producing 30 `.ptr .align` instances across 15 kernels:
- All are on kernel entry params (raw pointers: `*mut u8`, `*const u32`, `*mut u32`)
- CUDA PTX JIT **rejects** the PTX if `.ptr .align 1` is present (on some driver versions)
- Simple regex post-processing fixes all cases reliably
- The fix is: `content.replace(/.ptr\s+.align\s+\d+/g, '')`

### panic_const Status
**No longer an issue.** With `panic = "abort"` profile and `build-std = ["panic_abort"]`, zero `panic_const` references appear in the PTX output. This was previously a significant concern but is fully resolved by the panic strategy choice.

## Three Mitigation Paths

| Path | Effort | Risk | Durability |
|------|--------|------|------------|
| A: PTX post-processing (regex strip) | Low | Low | Fragile (depends on PTX format) |
| B: Target PTX ISA 7.8+ | Low-Medium | Medium | Good (proper ISA targeting) |
| C: Patch LLVM NVPTXAsmPrinter | High | Medium | Best (root cause fix) |

**Recommendation**: Try Path B first (PTX 7.8+ targeting). If it doesn't work, use Path A (post-processing) as the production workaround while pursuing Path C as a long-term fix.

## Open Questions
- Does LLVM's NVPTX backend support explicit PTX version targeting via command-line flags?
- What's the minimum CUDA driver version required for PTX 7.8? (Need to check user's driver)
- Would upstream LLVM accept a patch to suppress `.ptr .align` for kernel params?

## Impact on Downstream Tasks
- **ptx-codegen-fix.2**: Unblocked — can experiment with PTX version targeting and post-processing
- **std-sysroot-build.3/4**: Need post-processing or PTX version fix before end-to-end testing
