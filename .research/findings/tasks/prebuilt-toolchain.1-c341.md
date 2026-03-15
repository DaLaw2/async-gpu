# prebuilt-toolchain.1: CI workflow for building + publishing patched rustc
**Cycle**: 341 | **Theme**: prebuilt-toolchain | **Kind**: investigation | **Status**: done

## Summary
Investigated the feasibility of building and distributing pre-built patched rustc toolchains via CI. The build process is already automated (`build-toolchain.sh`), but requires ~30GB disk space and significant CPU/RAM. GitHub Actions standard runners are insufficient; need `ubuntu-latest-xl` or self-hosted runners. Distribution via GitHub Releases artifacts is the most practical approach.

## Findings

### Q: What CI resources are needed to build rustc?
A: Based on `build-toolchain.sh`:
- **Disk**: ~30GB for build artifacts (rustc-src + LLVM + build outputs)
- **RAM**: ~8-16GB (LLVM build is the bottleneck)
- **CPU**: 4+ cores recommended (parallel make)
- **Time**: ~30-60 min on 8-core, ~2-3 hours on 2-core
- **Prerequisites**: Python 3, git, cmake, ninja, clang/gcc
- **Standard GitHub Actions runner** (ubuntu-latest): 7GB RAM, 14GB SSD — **NOT ENOUGH**
- **Larger runner** (ubuntu-latest-xl): 16GB RAM, 150GB SSD — sufficient but costs $$$
**Confidence**: high

### Q: How to package the sysroot for distribution?
A: The build produces a sysroot at `patched-rustc/build/{host}/stage2/lib/rustlib/`. Options:
1. **tar.gz of entire sysroot** — ~200-500MB compressed, includes rustc binary + target libs
2. **rustup-compatible toolchain** — more complex, requires custom dist server
3. **Docker image with pre-built toolchain** — alternative approach, ~2-3GB image
- **Recommendation**: tar.gz via GitHub Releases is simplest. Download script extracts to `~/.rustup/toolchains/async-gpu/`.
**Confidence**: high

### Q: How to handle nightly version pinning?
A: `rust-toolchain.toml` specifies the exact nightly version (`channel = "nightly-YYYY-MM-DD"`). The CI workflow should:
1. Read nightly version from `rust-toolchain.toml`
2. Build patched rustc from that nightly's rustc-src
3. Tag release with nightly version (e.g., `toolchain-nightly-2025-08-25`)
4. Users download matching version via script
**Confidence**: high

### Q: GitHub Actions workflow design?
A:
```yaml
name: Build Patched Toolchain
on:
  workflow_dispatch:           # Manual trigger
  push:
    paths: ['std-patches/**', 'rustc-patches/**']  # Only when patches change

jobs:
  build-toolchain:
    runs-on: ubuntu-latest     # May need -xl for disk space
    steps:
      - checkout repo
      - install prerequisites (cmake, ninja, python3)
      - run scripts/build-toolchain.sh
      - tar -czf toolchain.tar.gz patched-rustc/build/*/stage2/
      - upload as GitHub Release artifact
```
- Trigger: manual (`workflow_dispatch`) + on patch changes
- Cost: ~$0.50 per build on standard runners (if they fit), ~$2-5 on XL runners
**Confidence**: medium (disk space constraint may require workarounds)

## Unexpected Discoveries
- `build-toolchain.sh` already handles incremental builds (skip if LLVM already built)
- The `--targets` flag allows building only needed targets (x86_64 + nvptx64)
- Windows build script (`build-toolchain.bat`) also exists

## Open Questions
- Can we use GitHub's free runners with disk cleanup tricks? (delete .cargo, .rustup, other SDKs)
- Should we support Windows pre-built toolchains? (Much harder — MSVC dependency)
- Should we publish to a custom `rustup` component server?

## Impact on Downstream Tasks
- Need experiment task to create the actual CI workflow and test it
- Distribution script also needed (`scripts/install-toolchain.sh`)
