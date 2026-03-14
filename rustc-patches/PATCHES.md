# Rustc Compiler Patches — WarpCooperativeTransform MIR Pass

Patches for the `rustc_mir_transform` crate that add warp-cooperative
async/await support for nvptx64 targets.

Baseline: `rustc 1.96.0-nightly (3b1b0ef4d 2026-03-11)`

## Setup

```bash
# Apply patches to patched-rustc/ (already done if you cloned from rustc-src/)
./scripts/apply-rustc-patches.sh patched-rustc

# Build the patched compiler
cd patched-rustc && python x.py build compiler
```

## Regenerate patches after editing patched-rustc/

```bash
./scripts/gen-rustc-patches.sh    # diffs patched-rustc/ vs rustc-src/
```

## Modified files (patches)

- `compiler/rustc_mir_transform/src/lib.rs` → `rustc-patches/rustc_mir_transform_src_lib.patch`
- `compiler/rustc_span/src/symbol.rs` → `rustc-patches/rustc_span_src_symbol.patch`

## New files

- `compiler/rustc_mir_transform/src/warp_cooperative.rs` → `rustc-patches/warp_cooperative.rs`
