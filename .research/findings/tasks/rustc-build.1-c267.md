# rustc-build.1: Build patched rustc from rustc-src/ with applied patches
**Cycle**: 267 | **Theme**: rustc-build | **Kind**: experiment | **Status**: done

## Summary
Successfully built patched rustc compiler with WarpCooperativeTransform MIR pass.
Required fixing 5 API errors (nightly 1.96.0 API changes) and MSVC ABI mismatch
with CI-downloaded LLVM (built LLVM from source instead).

## Findings

### Q: Can the patched rustc compile successfully?
A: Yes, after fixing 5 compilation errors caused by API differences between the
original code (written against rustc internals docs) and actual nightly 1.96.0 APIs.
**Confidence**: high

### Errors fixed:
1. `rustc_target::asm::nvptx` module is private → use re-exported `NvptxInlineAsmRegClass`
2. Missing `is_required()` method in `MirPass` trait → added `fn is_required(&self) -> bool { false }`
3. `sess.target.arch == "nvptx64"` type mismatch → use `Arch::Nvptx64` enum variant
4. `tcx.get_attrs_unchecked()` removed → use `tcx.get_all_attrs()` (deprecated but functional)
5. `tcx.trait_of_item()` removed → use `tcx.trait_of_assoc()`
6. `Symbol::intern()` on string literals forbidden by lint → added `warp_cooperative` to `symbol.rs`, use `sym::poll`

### MSVC ABI mismatch:
CI-downloaded LLVM was compiled with newer MSVC that has `__std_find_first_of_trivial_pos_1`,
but local MSVC 14.43.34808 doesn't. Fix: `bootstrap.toml` with `download-ci-llvm = false`,
builds LLVM from source with local MSVC (3997 C++ files, ~30 min).

### New patch file:
- `rustc-patches/rustc_span_src_symbol.patch` — adds `warp_cooperative` to the symbol registry

### Optimized gen-rustc-patches.sh:
Replaced `find + diff -q` (per-file, takes minutes) with `diff -rq` (directory-level, seconds).

## Build output
```
stage1/bin/rustc.exe — the patched compiler with WarpCooperativeTransform
```

## Open Questions
- Will `#[warp_cooperative]` attribute be recognized by the compiler or need registration?
- Does the MIR pass actually trigger on coroutine bodies?

## Impact on Downstream Tasks
- rustc-build.2 unblocked: can now test compiling `#[warp_cooperative] async fn`
