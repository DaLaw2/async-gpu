//! Integration tests using the `#[gpu_test]` proc macro.
//!
//! Each `#[gpu_test]` function expands to a `#[test]` that loads the unified
//! kernel PTX and launches the eponymous zero-param kernel on the GPU.
//! Standard assert!/assert_eq! inside the kernel cause a trap on failure,
//! which propagates as a CUDA error to the host test.
//!
//! Regular `#[test]` functions coexist with `#[gpu_test]` in the same file,
//! demonstrating that `cargo test` discovers and runs both side by side.

use gpu_test_macro::gpu_test;

// ============================================================================
// GPU tests — each launches a kernel on the GPU
// ============================================================================

/// Test that basic arithmetic assertions work on GPU.
///
/// The kernel `test_gpu_assert_basic` in gpu-kernel-test asserts:
///   2 + 3 == 5, 10 * 4 == 40, 5 < 40, 5 != 40
#[gpu_test]
fn test_gpu_assert_basic() {}

/// Test that Vec operations with assertions work on GPU.
///
/// The kernel `test_gpu_vec_operations` in gpu-kernel-test:
///   - Creates a Vec, pushes i*i for i in 0..10
///   - Asserts length, individual elements, and sum
#[gpu_test]
fn test_gpu_vec_operations() {}

/// Test that thread spawn + join with assertions works on GPU.
///
/// The kernel `test_gpu_thread_spawn` in gpu-kernel-test:
///   - Spawns two threads returning 42 and 99
///   - Joins and asserts the results
#[gpu_test]
fn test_gpu_thread_spawn() {}

// ============================================================================
// GPU tests — standard library on GPU
// ============================================================================

/// Test that Box allocation works on GPU.
///
/// The kernel `test_gpu_box_alloc` in gpu-kernel-test:
///   - Allocates Box<u32> and Box<[u32; 4]>
///   - Asserts dereference values and computed results
#[gpu_test]
fn test_gpu_box_alloc() {}

/// Test that String operations work on GPU.
///
/// The kernel `test_gpu_string_ops` in gpu-kernel-test:
///   - Creates Strings via from(), push_str(), format!()
///   - Asserts length, equality, and contains()
#[gpu_test]
fn test_gpu_string_ops() {}

/// Test that HashMap works on GPU.
///
/// The kernel `test_gpu_hashmap` in gpu-kernel-test:
///   - Inserts 5 entries, verifies get/contains_key/values().sum()
///   - Removes an entry and verifies
#[gpu_test]
fn test_gpu_hashmap() {}

// ============================================================================
// GPU tests — thread spawn with data passing
// ============================================================================

/// Test that threads can compute and return data.
///
/// The kernel `test_gpu_thread_data_passing` in gpu-kernel-test:
///   - Spawns thread computing sum(1..=100) = 5050
///   - Spawns thread computing 10! = 3628800
///   - Joins and asserts results
#[gpu_test]
fn test_gpu_thread_data_passing() {}

/// Test that warps are reused when more tasks than warps are spawned.
///
/// The kernel `test_gpu_thread_reuse` in gpu-kernel-test:
///   - Spawns 6 sequential tasks on 3 available warps
///   - Joins each and verifies result, sums total
#[gpu_test]
fn test_gpu_thread_reuse() {}

// ============================================================================
// GPU tests — cooperative compute
// ============================================================================

/// Test cooperative execution — all warps participate.
///
/// The kernel `test_gpu_cooperative` in gpu-kernel-test:
///   - Uses cooperative() to run all 4 warps in parallel
///   - Each warp writes its ID to a global array
///   - Verifies all warps participated
#[gpu_test]
fn test_gpu_cooperative() {}

/// Test cooperative_map — all warps transform an array.
///
/// The kernel `test_gpu_cooperative_map` in gpu-kernel-test:
///   - Initializes 64-element array with values 0..64
///   - All warps cooperatively double each element
///   - Verifies output[i] = i * 2
#[gpu_test]
fn test_gpu_cooperative_map() {}

/// Test cooperative_reduce — all warps sum partitions.
///
/// The kernel `test_gpu_cooperative_reduce` in gpu-kernel-test:
///   - Initializes 64-element array with values 0..64
///   - All warps cooperatively compute the sum
///   - Verifies total = 2016
#[gpu_test]
fn test_gpu_cooperative_reduce() {}

// ============================================================================
// GPU tests — math, atomics, iterators
// ============================================================================

/// Test GPU math intrinsics (sin, cos, sqrt, exp, log, abs, fma, tanh, sigmoid).
///
/// The kernel `test_gpu_math_intrinsics` in gpu-kernel-test:
///   - Tests each math function with known input/output
///   - Verifies results within tolerance
#[gpu_test]
fn test_gpu_math_intrinsics() {}

/// Test atomic operations across GPU threads.
///
/// The kernel `test_gpu_atomics` in gpu-kernel-test:
///   - Tests store/load, fetch_add, fetch_sub, CAS on AtomicU32
///   - Spawns 3 threads each doing 100 atomic increments
///   - Verifies final count = 300
#[gpu_test]
fn test_gpu_atomics() {}

/// Test iterator chain operations (map, filter, fold, zip, enumerate, chain).
///
/// The kernel `test_gpu_iterator_chain` in gpu-kernel-test:
///   - Tests map+collect, filter+collect, fold, zip+map+sum
///   - Tests enumerate+filter, chain
///   - All on GPU-allocated Vec
#[gpu_test]
fn test_gpu_iterator_chain() {}

// ============================================================================
// CPU tests — regular #[test] functions that coexist with #[gpu_test]
// ============================================================================

/// Verify that regular CPU tests run alongside GPU tests in the same file.
/// This confirms `cargo test` discovers both `#[test]` and `#[gpu_test]` tests.
#[test]
fn test_cpu_sanity_check() {
    assert_eq!(2 + 2, 4, "basic arithmetic should work on CPU");
    let v: Vec<i32> = (1..=5).collect();
    assert_eq!(v.len(), 5, "Vec should have 5 elements");
    assert_eq!(v.iter().sum::<i32>(), 15, "1+2+3+4+5 = 15");
}

/// Verify that the gpu-test-macro crate is accessible and the proc macro
/// attribute compiles correctly (compilation itself is the test here).
#[test]
fn test_gpu_test_macro_is_available() {
    // The fact that the #[gpu_test] functions above compile proves the macro works.
    // This test just confirms we can reference the crate and basic host types.
    let _ptx_snippet: &str = gpu_host::ptx::KERNEL_STD;
    assert!(
        !_ptx_snippet.is_empty(),
        "KERNEL_STD PTX should be non-empty"
    );
}

// ============================================================================
// Failure propagation test — behind a feature flag so it doesn't break CI
// ============================================================================

/// Test that a GPU kernel failure propagates as a Rust test failure.
///
/// This test calls `run_zero_param_with_cubin` with a nonexistent kernel name,
/// which should return an error. This verifies the error propagation path
/// without requiring a kernel that actually panics (which would need a new
/// kernel build cycle).
///
/// Enable with: `cargo test -p gpu-test-harness --test gpu_tests --features test-failure-propagation`
#[cfg(feature = "test-failure-propagation")]
#[test]
fn test_gpu_failure_propagation() {
    let cubin = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let cubin_path =
            std::path::Path::new(manifest).join("../../core/gpu-host/kernel_std.cubin");
        std::fs::read(&cubin_path).unwrap_or_default()
    };

    // Launch a kernel that doesn't exist — should fail with KernelNotFound
    let result = gpu_host::gpu::run_zero_param_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "nonexistent_kernel_that_does_not_exist",
        128,
        (1, 1, 1),
    );

    assert!(
        result.is_err(),
        "launching a nonexistent kernel should return an error"
    );
    let err_msg = format!("{}", result.unwrap_err());
    assert!(
        err_msg.contains("not found") || err_msg.contains("NotFound"),
        "error should indicate kernel was not found, got: {err_msg}"
    );
}
