# Feature Synthesis: audit-build

## Status
Complete. All 43 crates compile successfully.

## Key Findings
- **43 of 43 crates compile successfully** on nightly-2026-06-03 (rustc 1.98.0)
- Fixed: `std-build-test` linker symbol collision — removed ~300 lines of
  duplicated `gpu_stdout_write`/`gpu_stdin_read` code, delegated to gpu-runtime
- All 14 std examples, 10 hostcall host examples, 7 hostcall kernel subcrates,
  4 kernel crates, and 7/7 non-workspace test crates build clean
- ~10 compiler warnings remain (dead fields, unused imports/variables in other crates)
- The `ptx78` unstable feature warning is universal across nvptx64 builds (cosmetic)

## Completed Tasks
| Task | Result |
|------|--------|
| audit-build.1 | Mapped build landscape: 42/43 pass, 1 failure identified |
| audit-build.2 | Fixed std-build-test: removed duplicate symbols, 43/43 pass |

## Remaining Work
- Warning cleanup pass across other crates (separate feature scope)
- Runtime verification of examples (next feature task)

## Architecture Notes
- Workspace contains 5 host-side crates; all others are standalone
- Kernel crates use `.cargo/config.toml` with `target = "nvptx64-nvidia-cuda"`
- Test crates are kernel crates (not host), must be built from their own directory
- `build-kernels.sh` orchestrates the 4 main kernel crate builds
