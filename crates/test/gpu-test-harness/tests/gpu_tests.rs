//! Integration tests using the `#[gpu_test]` proc macro.
//!
//! Each `#[gpu_test]` function expands to a `#[test]` that loads the unified
//! kernel PTX and launches the eponymous zero-param kernel on the GPU.
//! Standard assert!/assert_eq! inside the kernel cause a trap on failure,
//! which propagates as a CUDA error to the host test.

use gpu_test_macro::gpu_test;

/// Test that basic arithmetic assertions work on GPU.
///
/// The kernel `test_gpu_assert_basic` in gpu-kernel-std asserts:
///   2 + 3 == 5, 10 * 4 == 40, 5 < 40, 5 != 40
#[gpu_test]
fn test_gpu_assert_basic() {}

/// Test that Vec operations with assertions work on GPU.
///
/// The kernel `test_gpu_vec_operations` in gpu-kernel-std:
///   - Creates a Vec, pushes i*i for i in 0..10
///   - Asserts length, individual elements, and sum
#[gpu_test]
fn test_gpu_vec_operations() {}

/// Test that thread spawn + join with assertions works on GPU.
///
/// The kernel `test_gpu_thread_spawn` in gpu-kernel-std:
///   - Spawns two threads returning 42 and 99
///   - Joins and asserts the results
#[gpu_test]
fn test_gpu_thread_spawn() {}
