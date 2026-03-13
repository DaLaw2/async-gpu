# workspace.1: Consolidate all crates into single Cargo workspace
**Cycle**: 201 | **Theme**: workspace-consolidation | **Kind**: experiment | **Status**: done

## Summary
Investigated workspace consolidation. Full unification is infeasible because GPU
crates require `--target nvptx64-nvidia-cuda -Zbuild-std=core` (nightly only),
which cannot coexist with host x86 builds in a single `cargo build` invocation.
Added gpu-protocol to root workspace (now 3 members: gpu-host, gpu-protocol,
warp-macro). GPU kernel crates remain as isolated workspaces — this is a Cargo
limitation, not a project choice.

## Findings
### Q: Can all 14 crates coexist in one workspace despite mixed stable/nightly targets?
A: **Partially.** Host-compatible crates (gpu-host, gpu-protocol, warp-macro) can
share a workspace. GPU kernel crates CANNOT because:
1. They need `.cargo/config.toml` with `target = "nvptx64-nvidia-cuda"` — workspace
   root config overrides per-crate config when building from root
2. `-Zbuild-std=core` is an unstable flag requiring nightly
3. `crate-type = ["cdylib"]` for nvptx64 is incompatible with host builds
**Confidence**: high

### Q: How to handle GPU kernel crates (nvptx64) vs host crates (x86) in one workspace?
A: They require separate build invocations regardless. GPU: `cd crates/gpu-kernel &&
cargo +nightly build --release`. Host: `cargo build -p gpu-host` from root.
The per-crate `.cargo/config.toml` + `[workspace]` marker pattern is the correct
approach for isolated GPU builds.
**Confidence**: high

### Q: What resolver settings are needed?
A: `resolver = "2"` (already set). No additional settings needed for the host
workspace.
**Confidence**: high

## Unexpected Discoveries
- The epic criterion "Single workspace with all crates — cargo build -p <any-crate>
  works from root" is fundamentally infeasible for mixed x86/nvptx64 projects.
  Cargo does not support multi-target builds with per-crate target overrides.
- Recommendation: update the criterion to reflect what's achievable.

## Impact on Downstream Tasks
- workspace.2 (deduplicate .cargo/config.toml) — assessed below
- workspace.3 (single-source nightly version) — achievable independently
