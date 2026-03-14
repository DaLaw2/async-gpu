# toolchain-auto.2: Create scripts/postprocess-ptx.sh
**Cycle**: 274 | **Theme**: toolchain-auto | **Kind**: experiment | **Status**: done

## Summary
Created `scripts/postprocess-ptx.sh` — a PTX post-processing script that fixes two known issues with rustc-generated PTX for nvptx64. Tested successfully on a synthetic PTX file containing both issues.

## Findings

### Q: What PTX post-processing is needed?
A: Two fixes, both identified in warp-verify.2:

1. **`.ptr .align N` removal**: LLVM NVPTX backend emits `.param .u64 .ptr .align 1 name` for raw pointer params in `extern "ptx-kernel"` functions. CUDA PTX JIT rejects this. Fix: regex removal of `.ptr .align \d+`.

2. **Panic extern stubbing**: Async fn coroutines emit `.extern .func panic_const_async_fn_resumed(...)` declarations. Since we compile with `-Zbuild-std=core` there's no `core::panicking` implementation. Fix: replace `.extern .func` with a `.visible .func` body containing `trap; ret;`.

**Confidence**: high — both fixes verified in warp-verify.2 and now automated+tested

### Implementation Notes
- Uses Python for multi-line regex (extern func declarations can span multiple lines with params)
- Python detection handles Windows quirk: `python3` is a Microsoft Store stub, `python` is the real binary
- Only stubs functions with `panic` or `abort` in the name (safety — don't break legit externs)
- `--all <dir>` mode processes all .ptx files recursively

## Impact on Downstream Tasks
- toolchain-auto theme criteria 2/2 met
- async-yield.3 can use this script after compiling PTX with patched rustc
