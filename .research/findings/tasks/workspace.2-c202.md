# workspace.2: Deduplicate .cargo/config.toml across kernel crates
**Cycle**: 202 | **Theme**: workspace-consolidation | **Kind**: experiment | **Status**: done

## Summary
Investigation found that per-crate `.cargo/config.toml` files CANNOT be deduplicated.
Each GPU kernel crate requires its own config because Cargo only reads the nearest
`.cargo/config.toml` when building from that directory (isolated workspace pattern).
A root-level config would apply to ALL crates including host crates.

## Findings
### Q: Can kernel crates inherit target config from workspace root?
A: No. Per-crate `.cargo/config.toml` is required because: (1) GPU crates must be
built from their own directory (`cd crate && cargo build`), (2) root config would
force nvptx64 target on host crates, (3) `build-std` values differ per crate
(core vs core+alloc vs std+core+panic_abort).
**Confidence**: high

### Q: Does per-crate override still work if needed?
A: Yes — this is exactly what we're using. Each crate's `.cargo/config.toml`
provides target, linker, rustflags, and build-std settings for isolated builds.
**Confidence**: high
