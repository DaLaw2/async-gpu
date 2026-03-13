# /maintain ci — Sync CI with actual crate and PTX state

Ensure `.github/workflows/build.yml` and `scripts/ci-lint.sh` match the actual crates and PTX files.

## Language
- Conversation: 繁體中文 | Files: English

## Steps

1. **Scan crates**: List all `crates/*/Cargo.toml` — extract crate names. Classify:
   - **Stable-compatible** (no `#![feature(...)]` in lib.rs): eligible for fmt + clippy
   - **Nightly-only** (has `#![feature(...)]`): fmt only (no stable clippy)
   - **GPU kernel** (has `.cargo/config.toml` with nvptx64 target): PTX build list

2. **Scan PTX stubs**: Grep `crates/gpu-host/src/` for `include_str!("../*.ptx")` — extract filenames.

3. **Compare** against:
   - `scripts/ci-lint.sh`: `CRATES_FMT`, `CRATES_CLIPPY`, `PTX_KERNELS`, PTX stub loop
   - `.github/workflows/build.yml`: fmt/clippy steps, PTX build steps, PTX stub loops

4. **Fix mismatches**: Update the files so they match the scan results.

5. **Report**: Print what was added/removed, or `[OK] CI in sync`.

## Rules
- Do NOT add crates that don't exist
- Do NOT remove crates that exist but were intentionally excluded (check for comments)
- PTX stub list must include every `.ptx` filename from `include_str!()` calls
