# GPU Test

Write GPU tests with `#[gpu_test]` — just like `#[test]`, but runs on the GPU.

## What It Demonstrates

- The `#[gpu_test]` proc macro for zero-boilerplate GPU testing
- Custom thread count and grid dimensions via macro attributes
- GPU-side `assert!()` / `panic!()` propagating as host test failures
- Coexistence of `#[gpu_test]` and regular `#[test]` in the same file

## How It Works

The `#[gpu_test]` macro transforms a stub function:

```rust
use gpu_test_macro::gpu_test;

#[gpu_test]
fn test_arithmetic() {}
```

Into a full GPU test:

```rust
#[test]
fn test_arithmetic() {
    gpu_host::gpu::run_zero_param_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "test_arithmetic",
        128,
        (1, 1, 1),
    ).expect("GPU test 'test_arithmetic' failed");
}
```

The kernel-side function is defined in the kernel crate:

```rust
#[no_mangle]
pub unsafe extern "gpu-kernel" fn test_arithmetic() {
    thread::gpu_main(|| {
        assert_eq!(2 + 3, 5);
        assert!(10 * 4 == 40);
    });
}
```

## Macro Attributes

```rust
// Default: 128 threads, (1,1,1) grid
#[gpu_test]
fn test_basic() {}

// Custom thread count
#[gpu_test(threads = 256)]
fn test_wide() {}

// Custom grid dimensions
#[gpu_test(threads = 128, grid = (2, 1, 1))]
fn test_multiblock() {}
```

## Running

```bash
# Show usage info
cargo run -p gpu-test-example

# Run actual GPU tests (requires kernel cubin/PTX)
cargo test -p gpu-test-harness --test gpu_tests
```

## Key Source Files

- Proc macro: `crates/test/gpu-test-macro/src/lib.rs`
- Test harness: `crates/test/gpu-test-harness/tests/gpu_tests.rs`
- Kernel tests: `crates/kernel/gpu-kernel-test/src/`
