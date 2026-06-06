//! GPU Test Example — demonstrates how to use the `#[gpu_test]` macro.
//!
//! This binary shows the concept of GPU testing. For actual GPU tests,
//! see the `tests/` directory which uses `#[gpu_test]` with `cargo test`.
//!
//! The `#[gpu_test]` macro transforms a stub function into a `#[test]` that:
//! 1. Loads the unified kernel PTX (or cubin for fast loading)
//! 2. Launches the eponymous zero-param GPU kernel
//! 3. Any `assert!()` / `panic!()` in the kernel becomes a host test failure
//!
//! # Usage in your test file:
//!
//! ```rust,ignore
//! use gpu_test_macro::gpu_test;
//!
//! // Basic: 128 threads, 1 block (default)
//! #[gpu_test]
//! fn test_arithmetic() {}
//!
//! // Custom thread count
//! #[gpu_test(threads = 256)]
//! fn test_multithread() {}
//!
//! // Custom grid dimensions
//! #[gpu_test(threads = 128, grid = (2, 1, 1))]
//! fn test_multiblock() {}
//! ```
//!
//! The kernel-side function (in the kernel crate) must have the signature:
//!
//! ```rust,ignore
//! #[no_mangle]
//! pub unsafe extern "gpu-kernel" fn test_arithmetic() {
//!     thread::gpu_main(|| {
//!         assert_eq!(2 + 3, 5);
//!         assert!(10 * 4 == 40);
//!     });
//! }
//! ```

fn main() {
    println!("=== GPU Test Example ===\n");
    println!("This example demonstrates the #[gpu_test] proc macro.\n");
    println!("The #[gpu_test] macro transforms stub functions into GPU tests:");
    println!();
    println!("  use gpu_test_macro::gpu_test;");
    println!();
    println!("  // Launches 'test_arithmetic' kernel on GPU with 128 threads");
    println!("  #[gpu_test]");
    println!("  fn test_arithmetic() {{}}");
    println!();
    println!("  // Custom configuration");
    println!("  #[gpu_test(threads = 256)]");
    println!("  fn test_wide() {{}}");
    println!();
    println!("  // Custom grid for multi-block tests");
    println!("  #[gpu_test(threads = 128, grid = (2, 1, 1))]");
    println!("  fn test_multiblock() {{}}");
    println!();
    println!("How it works:");
    println!("  1. The macro expands to a #[test] function");
    println!("  2. It loads the kernel PTX (or cubin for fast loading)");
    println!("  3. It launches the kernel whose name matches the function");
    println!("  4. assert!/panic! on GPU propagate as host test failures");
    println!();
    println!("To run actual GPU tests:");
    println!("  cargo test -p gpu-test-example --test gpu_tests");
    println!();
    println!("Key source files:");
    println!("  Macro:   crates/test/gpu-test-macro/src/lib.rs");
    println!("  Harness: crates/test/gpu-test-harness/tests/gpu_tests.rs");
    println!("  Kernels: crates/kernel/gpu-kernel-test/src/");
    println!();
    println!("=== Example complete ===");
}
