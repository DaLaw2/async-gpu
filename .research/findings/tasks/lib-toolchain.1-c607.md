# lib-toolchain.1 — Minimum User Setup Requirements

## Investigation Summary

Analyzed all setup scripts, env-check, patched-std/patched-rustc structure, and
cargo configs across the entire repo to determine minimum requirements per user
scenario and design the setup.sh flow.

---

## 1. Script Analysis

### `scripts/build-toolchain.sh` — Patched Compiler Builder
- **What**: Clones upstream rustc, applies compiler patches (warp_cooperative MIR
  pass via `rustc-patches/`), applies std patches, builds stage-1 compiler.
- **Prerequisites**: Python 3, git, cmake, ninja/make, clang/gcc, ~30GB disk.
- **Duration**: Hours (full LLVM + rustc build, `download-ci-llvm = false`).
- **Output**: `patched-rustc/build/x86_64-unknown-linux-gnu/stage1/bin/rustc`
  (currently 13GB build dir on this machine).
- **Who needs it**: Only contributors working on the `#[warp_cooperative]` MIR pass.

### `scripts/apply-std-patches.sh` — Patched Std Generator
- **What**: Copies `rustc-src/library/std/` to `patched-std/`, applies 10 patches
  + 6 new files for `target_os = "cuda"` support (alloc, fs, stdio, thread,
  thread_local, io/error, net, random).
- **Prerequisites**: `rustc-src/library/std/` must exist (requires cloning rustc
  upstream: `git clone --depth 1 https://github.com/rust-lang/rust.git rustc-src`).
- **Duration**: Seconds (just copy + patch).
- **Output**: `patched-std/` directory.
- **Note**: patched-std/ already exists on disk; this is a regeneration step.

### `scripts/build-kernel-std.sh` — Kernel Std PTX/Cubin Builder
- **What**: Builds `gpu-kernel-std` crate for nvptx64 with `build-std = ["std",
  "core", "panic_abort"]`, then compiles PTX to cubin with ptxas.
- **Prerequisites**:
  - Patched std in nightly sysroot (copies from `patched-std/` into sysroot).
  - CUDA toolkit (ptxas).
  - Nightly toolchain from `rust-toolchain.toml`.
- **Duration**: Minutes (cargo build-std + ~10min ptxas compilation).
- **Output**: `kernel_std.ptx` + `kernel_std.cubin` in `crates/core/gpu-host/`.
- **Warning**: Mutates the nightly sysroot (backs up stock std, overwrites with
  patched). This is invasive but reversible.

### `scripts/env-check.sh` — Environment Verifier
- **What**: Read-only check of prerequisites. Reports status, never modifies.
- **Checks performed**:
  - rustup installed
  - rustc installed
  - Nightly toolchain from rust-toolchain.toml installed
  - nvptx64-nvidia-cuda target installed
  - rust-src component installed
  - llvm-bitcode-linker component installed
  - NVIDIA GPU present + driver version + compute capability >= 7.0
  - CUDA toolkit (nvcc) — warns but doesn't fail
  - cargo, git
- **Missing checks**:
  - ptxas specifically (checks nvcc but not ptxas)
  - Python 3 (needed for build-toolchain.sh)
  - cmake, ninja/make (needed for build-toolchain.sh)
  - Disk space
  - No check for patched-std or patched-rustc status

---

## 2. User Scenarios & Minimum Requirements

### Scenario A: Stock Nightly (core-only kernels, no patched std)
**Use case**: Write GPU kernels using `#![no_std]` + `core` only. Hostcall I/O via
gpu-protocol. Examples: hello-gpu, async-io, vector-math, parallel-search, tcp-echo.

**Requirements**:
1. Rust via rustup
2. Nightly toolchain `nightly-2026-06-03` (from rust-toolchain.toml)
3. Components: `rust-src`, `llvm-tools`, `llvm-bitcode-linker`, `rustfmt`
4. Target: `nvptx64-nvidia-cuda`
5. NVIDIA GPU with driver (compute capability >= 7.0)
6. CUDA runtime (comes with driver; no toolkit needed for loading)

**Setup commands**:
```bash
rustup toolchain install nightly-2026-06-03 \
  --component rust-src llvm-tools llvm-bitcode-linker rustfmt \
  --target nvptx64-nvidia-cuda
```

**Time**: ~2 minutes (download nightly + components).

**What works**: All `build-std = ["core"]` kernels, all hostcall examples except
warp-cooperative and async-pipeline, all host crates, all tests except std-build-test.

### Scenario B: Patched Std (std::thread, File I/O, println! on GPU)
**Use case**: Write GPU kernels using full Rust std. Examples: thread-demo,
benchmark, mnist-cnn, gpt2-inference, etc.

**Requirements**: Everything from Scenario A, plus:
1. `rustc-src/` clone: `git clone --depth 1 https://github.com/rust-lang/rust.git rustc-src`
2. Run `scripts/apply-std-patches.sh` to generate `patched-std/`
3. Run `scripts/build-kernel-std.sh` (or manually copy patched std into sysroot)
4. CUDA toolkit with ptxas (for PTX-to-cubin compilation)

**Setup commands**:
```bash
git clone --depth 1 https://github.com/rust-lang/rust.git rustc-src
bash scripts/apply-std-patches.sh
bash scripts/build-kernel-std.sh
```

**Time**: ~15 minutes (clone: 2min, patch: seconds, build+ptxas: 10-15min).

**Sysroot mutation warning**: `build-kernel-std.sh` copies patched std files into
the nightly sysroot (`$SYSROOT/lib/rustlib/src/rust/library/std/`). This modifies
the shared toolchain. A backup is made to `std.bak` but it's still invasive.

**What works**: Everything from A + all `build-std = ["std"]` kernels + gpu-kernel-std.

### Scenario C: Contributor (Patched Compiler, MIR Pass)
**Use case**: Develop/test the `#[warp_cooperative]` attribute and the
`WarpCooperativeTransform` MIR pass. Examples: warp-cooperative, async-pipeline.

**Requirements**: Everything from Scenario B, plus:
1. Python 3, cmake, ninja (or make), clang or gcc
2. ~30GB disk space
3. Run `scripts/build-toolchain.sh`

**Setup commands**:
```bash
bash scripts/build-toolchain.sh
# Then use: export RUSTC="patched-rustc/build/x86_64-unknown-linux-gnu/stage1/bin/rustc"
```

**Time**: 2-4 hours (full LLVM + rustc compilation from source).

**What works**: Everything including `#[warp_cooperative]` MIR pass.

---

## 3. Prerequisite Details

| Prerequisite | Scenario A | Scenario B | Scenario C |
|---|---|---|---|
| rustup | Required | Required | Required |
| Nightly `nightly-2026-06-03` | Required | Required | Required |
| rust-src component | Required | Required | Required |
| llvm-bitcode-linker | Required | Required | Required |
| nvptx64 target | Required | Required | Required |
| NVIDIA GPU + driver | Required | Required | Required |
| Compute cap >= 7.0 | Required | Required | Required |
| CUDA toolkit (ptxas) | Not needed | Required | Required |
| rustc-src clone | Not needed | Required | Required |
| Python 3 | Not needed | Not needed | Required |
| cmake + ninja/make | Not needed | Not needed | Required |
| clang or gcc | Not needed | Not needed | Required |
| 30GB disk | Not needed | Not needed | Required |

---

## 4. env-check.sh Gap Analysis

**Already covers**: rustup, rustc, nightly toolchain, nvptx64 target, rust-src,
llvm-bitcode-linker, NVIDIA GPU/driver/compute cap, nvcc, cargo, git.

**Missing (should add)**:
- ptxas check (separate from nvcc — both in CUDA toolkit but ptxas is the one
  actually needed by build-kernel-std.sh)
- Scenario-awareness (report what the user CAN do vs what they CAN'T)
- Patched std status (is sysroot patched? is patched-std/ present?)
- Patched rustc status (is build present?)

---

## 5. setup.sh Design

### Interface
```bash
./setup.sh              # Auto-detect best scenario, interactive prompt
./setup.sh --quick      # Scenario A only (fastest)
./setup.sh --std        # Scenario A + B (std patches)
./setup.sh --full       # Scenario A + B + C (patched compiler)
./setup.sh --check      # Run env-check.sh (alias)
```

### Flow for each mode

#### `--quick` (Scenario A)
1. Check rustup exists (fail with install URL if not)
2. Install nightly toolchain + components + target (idempotent)
3. Verify NVIDIA GPU accessible
4. Run a smoke test: `cargo build --release` on gpu-kernel (PTX builds)
5. Run hello-gpu example if GPU present
6. Print summary: what's working, what needs --std/--full for more

#### `--std` (Scenario A + B)
1. Everything from --quick
2. Check ptxas available (fail with CUDA toolkit install instructions if not)
3. Clone rustc-src if not present (depth 1)
4. Run apply-std-patches.sh
5. Run build-kernel-std.sh
6. Run a smoke test: gpu-std-test or std-build-test
7. Print summary

#### `--full` (Scenario A + B + C)
1. Everything from --std
2. Check Python 3, cmake, ninja/make, clang/gcc
3. Run build-toolchain.sh
4. Print summary with RUSTC path

#### Default (no flag)
1. Run env-check.sh
2. Detect what's already set up
3. Suggest the minimal next step
4. Prompt user to choose scenario

### Key Design Principles
1. **Idempotent**: Re-running is safe, skips already-done steps.
2. **Non-destructive**: Never silently modify sysroot without warning.
3. **Fast-path first**: Most users want Scenario A. Make it 2 minutes.
4. **Fail early, fail clear**: Check prereqs before doing heavy work.
5. **No system modification**: Respect CLAUDE.md rule — only modify repo contents.
   Toolchain install via rustup is user-scoped, which is fine.

---

## 6. Blockers & Issues

### Blocker: Sysroot Mutation in Scenario B
`build-kernel-std.sh` copies patched std into the shared nightly sysroot. This is
problematic because:
- Other projects using the same nightly get patched std unexpectedly.
- A nightly update will wipe the patches.
- The backup (`std.bak`) is fragile.

**Mitigation options** (for setup.sh, not this task):
- Use `RUSTUP_TOOLCHAIN` with a per-project copy of the toolchain.
- Use `--sysroot` override pointing to a local sysroot.
- Document the risk clearly and offer an "undo" command.

### Blocker: rustc-src Clone Size
Even `--depth 1`, the rustc repo is ~500MB+. For Scenario B users who only need
the std source, we could instead:
- Ship just `library/std/` as a subtree/archive.
- Or provide a pre-patched tarball.
But this is a Scenario B optimization, not a blocker.

### Not Automatable
- NVIDIA driver installation (system-level, requires root).
- CUDA toolkit installation (system-level, requires root).
- These must remain "user action required" with clear instructions.

### ptxas vs CUDA Toolkit
`build-kernel-std.sh` needs ptxas but env-check.sh only checks nvcc. The script
searches `/usr/local/cuda*/bin` and `/opt/cuda/bin`. On this machine it's at
`/usr/local/cuda-13.3/bin/ptxas`. Setup.sh should check both and add to PATH hint.

---

## 7. Crate Classification by Scenario

### Scenario A (core-only, stock nightly)
- `crates/core/gpu-host` — host-side runtime (stable Rust)
- `crates/core/gpu-protocol` — wire protocol (stable Rust)
- `crates/core/gpu-atomics` — GPU atomics (stable Rust)
- `crates/core/gpu-runtime` — GPU-side runtime (nightly, core-only)
- `crates/core/gpu-libc` — GPU libc bindings (nightly, core-only)
- `crates/kernel/gpu-kernel` — reference kernel (nightly, `build-std = ["core"]`)
- `crates/macro/warp-macro` — proc macros (stable Rust)
- `crates/test/async-hostcall-test` — core-only test kernel
- `crates/test/async-pipeline-test` — core-only test kernel
- `crates/test/embassy-test` — core-only test kernel
- `crates/test/multi-warp-test` — core-only test kernel
- `crates/test/gpu-std-test` — uses `build-std = ["core", "alloc"]`
- All `examples/hostcall/*` kernels EXCEPT warp-cooperative, async-pipeline

### Scenario B (patched std)
- `crates/kernel/gpu-kernel-std` — `build-std = ["std", "core", "panic_abort"]`
- `crates/test/std-build-test` — `build-std = ["std", "core", "panic_abort"]`
- All `examples/std/*` examples

### Scenario C (patched compiler)
- `examples/hostcall/warp-cooperative/` — uses `#[warp_cooperative]` attribute
- `examples/hostcall/async-pipeline/` — build.rs compiles kernel with patched rustc
