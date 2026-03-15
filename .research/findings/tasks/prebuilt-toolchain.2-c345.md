# prebuilt-toolchain.2 — CI Workflow + Download Script

**Status:** done
**Theme:** prebuilt-toolchain
**Cycle:** 345

## What was created

### 1. `.github/workflows/build-toolchain.yml`

GitHub Actions workflow that builds the patched Rust toolchain and publishes it as a GitHub Release.

**Triggers:**
- `workflow_dispatch` (manual, with optional `--from-scratch` flag)
- Push to `main` when `std-patches/**`, `rustc-patches/**`, or build scripts change

**Key design decisions:**
- **Disk cleanup:** Removes ~25-30GB of pre-installed software (.NET, Android SDK, Haskell/GHC, Boost, Swift, Docker images, etc.) to make room for the ~30GB build
- **Release tagging:** Uses `toolchain-{nightly-channel}` format (e.g., `toolchain-nightly-2026-03-11`)
- **Artifact naming:** `async-gpu-toolchain-{triple}-{channel}.tar.gz`
- **Uses `softprops/action-gh-release@v2`** for creating/updating releases
- **Also uploads as workflow artifact** (30-day retention) as a backup
- **Timeout:** 180 minutes (toolchain builds can take 1-3 hours)
- **CARGO_BUILD_JOBS=2** to reduce memory pressure on the 7GB RAM runner

**Risk:** ubuntu-latest may still run out of disk or RAM. A comment in the workflow notes that larger runners may be needed.

### 2. `scripts/install-toolchain.sh`

Download and install script for end users.

**Features:**
- Auto-reads nightly version from `rust-toolchain.toml` (or accepts explicit argument)
- Downloads from GitHub Releases
- Installs to `~/.rustup/toolchains/async-gpu/`
- Works with `cargo +async-gpu build ...` after install
- Detects existing installation and prompts for reinstall
- Verifies rustc binary and nvptx64 target libs
- Supports both curl and wget
- Can be run as one-liner: `bash <(curl -sL ...)`
- Currently Linux x86_64 only (errors with instructions for other platforms)

### Archive format

The sysroot is packaged directly — the archive contains a `stage2/` (or `stage1/`) directory with `bin/rustc`, `lib/rustlib/`, etc. The install script extracts this into the rustup toolchains directory.

## Future work

- Windows and macOS pre-built toolchains (need separate CI jobs)
- Checksum verification (SHA256) for downloaded archives
- GPG signing of releases
- Automatic nightly rebuild schedule (cron trigger)
