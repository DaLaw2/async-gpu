# Applying the WarpCooperativeTransform MIR Pass to rustc

## Target Compiler Version

- **Toolchain**: `nightly-2026-03-11`
- **rustc commit**: whichever commit `nightly-2026-03-11` resolves to
  (check with `rustc +nightly-2026-03-11 -vV`)

## Prerequisites

- A working Rust development environment (capable of building `x.py`)
- Python 3
- At least 30 GB of free disk space for the rustc build

## Step 1: Clone the rustc source

```bash
git clone https://github.com/rust-lang/rust.git rustc-src
cd rustc-src

# Check out the commit matching nightly-2026-03-11.
# Find the exact SHA from the release channel:
#   curl -s https://static.rust-lang.org/dist/2026-03-11/channel-rust-nightly.toml | grep 'commit ='
git checkout <COMMIT_SHA>
```

## Step 2: Configure the build

```bash
cp config.example.toml config.toml
```

Edit `config.toml`:

```toml
[build]
# Build only the compiler, not the full distribution
build-stage = 1

[rust]
# Enable debug assertions for easier development
debug-assertions = true
```

## Step 3: Place the pass source file

```bash
cp /path/to/warp_cooperative.rs compiler/rustc_mir_transform/src/warp_cooperative.rs
```

The file goes into `compiler/rustc_mir_transform/src/` alongside the existing
passes (`coroutine.rs`, `inline.rs`, etc.).

Note: The pass uses `Symbol::intern("warp_cooperative")` for attribute detection,
so no changes to `rustc_span/src/symbol.rs` are required.

## Step 4: Register the pass in the pipeline

Apply the provided patch:

```bash
git apply /path/to/lib_rs.patch
```

This patch makes two changes to `compiler/rustc_mir_transform/src/lib.rs`:

1. Adds `mod warp_cooperative;` to the module declarations
2. Inserts `&warp_cooperative::WarpCooperativeTransform` into the
   `mir_drops_elaborated_and_const_checked` pass pipeline, immediately after
   `&coroutine::StateTransform`

### Manual application (if the patch does not apply cleanly)

**a)** Find the `declare_passes!` macro invocation in `lib.rs` and add after
`mod coroutine : StateTransform;`:

```rust
    mod warp_cooperative : WarpCooperativeTransform;
```

**b)** Find the `run_runtime_lowering_passes` function.  Inside it
locate the pass list that includes `&coroutine::StateTransform`.  Add the new
pass right after it:

```rust
    // Existing:
    &coroutine::StateTransform,
    // NEW: warp-cooperative transform for nvptx64 async
    &warp_cooperative::WarpCooperativeTransform,
```

## Step 5: Build the patched compiler

```bash
./x.py build compiler
```

This produces a stage-1 compiler in `build/<host-triple>/stage1/bin/rustc`.

## Step 6: Test with a simple async fn

Create a test file `test_warp.rs`:

```rust
#![feature(warp_cooperative)]
#![no_std]

use core::future::Future;
use core::pin::Pin;
use core::task::{Context, Poll};

struct DummyFuture;

impl Future for DummyFuture {
    type Output = u32;
    fn poll(self: Pin<&mut Self>, _cx: &mut Context<'_>) -> Poll<u32> {
        Poll::Ready(42)
    }
}

#[warp_cooperative]
async fn simple_await() -> u32 {
    let val = DummyFuture.await;
    val + 1
}
```

Compile targeting nvptx64:

```bash
build/<host-triple>/stage1/bin/rustc \
    test_warp.rs \
    --edition 2021 \
    --target nvptx64-nvidia-cuda \
    -Z build-std=core \
    --emit=asm \
    -C linker=echo \
    -C target-cpu=sm_86
```

Expected output: diagnostic notes from the pass, e.g.:

```
note: warp_cooperative: `simple_await` —
      0 yield(s), 1 poll(s), 1 suspension(s), 1 return(s)
```

The generated PTX should contain `shfl.sync.idx.b32` (discriminant broadcast)
and `bar.warp.sync` (barrier before return) instructions. The pass now performs
Phase 2 (discriminant broadcast) and Rule 4 (warp barrier before return).

## Directory Layout

```
rustc-patches/
├── PATCHES.md                  ← this file
├── warp_cooperative.rs         ← MIR pass source (→ compiler/rustc_mir_transform/src/)
└── lib_rs.patch                ← unified diff for compiler/rustc_mir_transform/src/lib.rs
```
