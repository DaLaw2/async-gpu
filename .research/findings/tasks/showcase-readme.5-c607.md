# showcase-readme.5 — Update getting-started guide with SC, channels, executor sections

## Status: done

## What Changed
- Added 3 new run commands to Quick Start: structured-concurrency, gpu-channels, warp-cooperative
- Added structured-concurrency and gpu-channels rows to "All examples" table
- Updated warp-cooperative description in table to reflect its refactored state
- Updated hostcall example count in Crate Map (8 → 10)

## Verified
- All manifest paths confirmed: each example has a Cargo.toml at its root
- All use `cargo run --release` (verified via run.sh scripts)
- 10 hostcall example directories confirmed via filesystem listing

## Files Changed
- README.md
