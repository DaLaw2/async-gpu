# /maintain nightly — Sync nightly toolchain version

Ensure the nightly version in `rust-toolchain.toml` is the single source of truth.

## Language
- Conversation: 繁體中文 | Files: English

## Steps

1. **Read** `rust-toolchain.toml` → extract channel (e.g., `nightly-2026-03-11`)
2. **Check** these files for hardcoded nightly versions:
   - `scripts/ci-lint.sh` → `NIGHTLY=` variable
   - `.github/workflows/build.yml` → `toolchain:` field
3. **Fix** any mismatches → update hardcoded values to match `rust-toolchain.toml`
4. **Report**: `[FIX] Updated {files}` or `[OK] Nightly version consistent`
