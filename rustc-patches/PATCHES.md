# Rustc Compiler Patches — WarpCooperativeTransform MIR Pass

Patches for the `rustc_mir_transform` crate that add warp-cooperative
async/await support for nvptx64 targets.

Baseline: `rustc 1.96.0-nightly (3b1b0ef4d 2026-03-11)`

## Setup

```bash
# Clone rustc source (matching nightly)
git clone --depth 1 https://github.com/rust-lang/rust.git rustc-src

# Apply patches
./scripts/apply-rustc-patches.sh rustc-src

# Build the patched compiler
cd rustc-src && ./x.py build compiler
```

## Regenerate patches after editing patched-rustc/

```bash
./scripts/gen-rustc-patches.sh
```

## Modified files (patches)

- `compiler/rustc_mir_transform/src/lib.rs` → `rustc-patches/rustc_mir_transform_src_lib.patch`

## New files

- `compiler/rustc_mir_transform/src/warp_cooperative.rs` → `rustc-patches/warp_cooperative.rs`
