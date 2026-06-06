# Feature Synthesis: audit-build

## Status
First task (audit-build.1) complete. Build landscape fully mapped.

## Key Findings
- **42 of 43 crates compile successfully** on nightly-2026-06-03 (rustc 1.98.0)
- Single failure: `std-build-test` — linker symbol collision (`gpu_stdin_read` defined
  in both the test crate and its dependency `gpu-runtime`)
- All 14 std examples, 10 hostcall host examples, 7 hostcall kernel subcrates,
  4 kernel crates, and 6/7 non-workspace test crates build clean
- ~12 compiler warnings across the codebase (dead fields, unused imports/variables)
- The `ptx78` unstable feature warning is universal across all nvptx64 builds (cosmetic)

## Failure Inventory
| Crate | Category | Error |
|-------|----------|-------|
| std-build-test | linker | `gpu_stdin_read` multiply defined (test crate vs gpu-runtime) |

## Remaining Work
- Fix `std-build-test` symbol collision (separate task)
- Warning cleanup pass (separate task)
- Runtime verification of examples (next feature task)

## Architecture Notes
- Workspace contains 5 host-side crates; all others are standalone
- Kernel crates use `.cargo/config.toml` with `target = "nvptx64-nvidia-cuda"`
- Test crates are kernel crates (not host), must be built from their own directory
- `build-kernels.sh` orchestrates the 4 main kernel crate builds
