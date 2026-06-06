//! Integration tests using the `#[gpu_test]` proc macro.
//!
//! Each `#[gpu_test]` function expands to a `#[test]` that loads the unified
//! kernel PTX and launches the eponymous zero-param kernel on the GPU.
//! Standard assert!/assert_eq! inside the kernel cause a trap on failure,
//! which propagates as a CUDA error to the host test.
//!
//! Regular `#[test]` functions coexist with `#[gpu_test]` in the same file,
//! demonstrating that `cargo test` discovers and runs both side by side.

use cudarc::driver::DevicePtr;
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

/// Test type-safe cooperative execution with DisjointSlice + WarpIndex.
///
/// The kernel `test_gpu_type_safe_cooperative` in gpu-kernel-test:
///   - alloc_disjoint() + spawn_all_indexed(): safe parallel fill + verify
///   - alloc_disjoint() + cooperative_indexed(): safe cooperative() variant
///   - DisjointSlice::get() immutable reads + bounds checking
#[gpu_test]
fn test_gpu_type_safe_cooperative() {}

/// Test safe cooperative map — zero-unsafe rewrite of test_gpu_cooperative_map.
///
/// The kernel `test_gpu_cooperative_map_safe` in gpu-kernel-test:
///   - Uses cooperative_indexed() + DisjointSlice (zero unsafe in body)
///   - Each warp doubles its partition of a 64-element array
///   - WarpHandle::reduce_sum_u32 for safe warp-level reduction
///   - DisjointSlice::get() for bounds-checked verification
#[gpu_test]
fn test_gpu_cooperative_map_safe() {}

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
// GPU tests — generic monomorphization (gen-mono.2)
// ============================================================================

/// Test generic map with f32 monomorphization.
///
/// The kernel `test_gpu_generic_map_f32` in gpu-kernel-test:
///   - Allocates Vec<f32>, fills with 0.0..16.0
///   - Applies generic_map_inplace::<f32>(scale=2.0, bias=1.0)
///   - Asserts data[i] == i * 2.0 + 1.0
#[gpu_test]
fn test_gpu_generic_map_f32() {}

/// Test generic map with i32 monomorphization.
///
/// The kernel `test_gpu_generic_map_i32` in gpu-kernel-test:
///   - Allocates Vec<i32>, fills with 0..16
///   - Applies generic_map_inplace::<i32>(scale=3, bias=10)
///   - Asserts data[i] == i * 3 + 10
#[gpu_test]
fn test_gpu_generic_map_i32() {}

/// Test generic reduce with f32 monomorphization.
///
/// The kernel `test_gpu_generic_reduce_f32` in gpu-kernel-test:
///   - Creates Vec<f32> with 1.0..=16.0
///   - Calls generic_reduce_sum::<f32>()
///   - Asserts total == 136.0
#[gpu_test]
fn test_gpu_generic_reduce_f32() {}

/// Test generic reduce with i32 monomorphization.
///
/// The kernel `test_gpu_generic_reduce_i32` in gpu-kernel-test:
///   - Creates Vec<i32> with 1..=100
///   - Calls generic_reduce_sum::<i32>()
///   - Asserts total == 5050
#[gpu_test]
fn test_gpu_generic_reduce_i32() {}

/// Test that the same generic body works for both f32 and i32 in one kernel.
///
/// The kernel `test_gpu_generic_dual_type` in gpu-kernel-test:
///   - Calls generic_map_inplace with f32 and i32 data
///   - Calls generic_reduce_sum with f32 and i32 data
///   - Verifies all four operations produce correct type-specific results
#[gpu_test]
fn test_gpu_generic_dual_type() {}

// ============================================================================
// GPU tests — user-defined traits + where bounds (gen-traits.1)
// ============================================================================

/// Test trait-based reduce with f32 monomorphization (GpuReducible).
///
/// The kernel `test_gpu_trait_reduce_f32` in gpu-kernel-test:
///   - Creates Vec<f32> with 1.0..=20.0
///   - Calls trait_reduce::<f32>() using GpuReducible::combine
///   - Asserts total == 210.0
///   - Verifies identity() and combine() trait methods
#[gpu_test]
fn test_gpu_trait_reduce_f32() {}

/// Test trait-based reduce with i32 monomorphization (GpuReducible).
///
/// The kernel `test_gpu_trait_reduce_i32` in gpu-kernel-test:
///   - Creates Vec<i32> with 1..=50
///   - Calls trait_reduce::<i32>() using GpuReducible::combine
///   - Asserts total == 1275
///   - Verifies identity() and combine() trait methods
#[gpu_test]
fn test_gpu_trait_reduce_i32() {}

/// Test where-clause transform (GpuTransformable with explicit where bounds).
///
/// The kernel `test_gpu_where_transform` in gpu-kernel-test:
///   - f32: apply_transform(scale=3.0, offset=10.0) — verifies where T: GpuTransformable
///   - i32: apply_transform(scale=5, offset=-2) — verifies i32 monomorphization
///   - Proves `where T: Trait` syntax works identically to `<T: Trait>` on nvptx64
#[gpu_test]
fn test_gpu_where_transform() {}

/// Test combined GpuReducible + GpuTransformable bounds on same type parameter.
///
/// The kernel `test_gpu_trait_combined` in gpu-kernel-test:
///   - f32: transform(x*2+10) then reduce → 80.0
///   - i32: transform(x*3+1) then reduce → 34
///   - Proves multiple trait bounds monomorphize correctly
#[gpu_test]
fn test_gpu_trait_combined() {}

/// Test custom Vec2f struct implementing GpuReducible + GpuTransformable.
///
/// The kernel `test_gpu_trait_custom_vec2f` in gpu-kernel-test:
///   - Reduces Vec<Vec2f> — per-field f32 additions
///   - Tests identity(), combine(), transform_then_reduce on Vec2f
///   - Proves user-defined #[repr(C)] types work with generic trait code on GPU
#[gpu_test]
fn test_gpu_trait_custom_vec2f() {}

/// Test all types (f32, i32, u32, Vec2f) through same generic function in one kernel.
///
/// The kernel `test_gpu_trait_multi_type` in gpu-kernel-test:
///   - Calls trait_reduce with f32, i32, u32, and Vec2f
///   - Calls apply_transform with f32
///   - Definitive proof that user-defined traits monomorphize for all types on nvptx64
#[gpu_test]
fn test_gpu_trait_multi_type() {}

// ============================================================================
// GPU tests — generic parallel_reduce showcase (gen-demo.1)
// ============================================================================

/// Test generic parallel_reduce at scale: f32, i32, Vec2f with 1024 elements.
///
/// The kernel `test_gpu_generic_reduce_showcase` in gpu-kernel-test:
///   - parallel_reduce::<f32>(1..=1024) = 524800.0
///   - parallel_reduce::<i32>(1..=1024) = 524800
///   - parallel_reduce::<Vec2f> with per-field summation
///   - SAME generic function, three different types
///   - This is the gpu-generics epic litmus test
#[gpu_test]
fn test_gpu_generic_reduce_showcase() {}

/// Test zero-overhead proof: generic reduce vs handwritten produce identical results.
///
/// The kernel `test_gpu_generic_zero_overhead` in gpu-kernel-test:
///   - Runs parallel_reduce::<f32> and handwritten_reduce_f32 on same 2048-element data
///   - Runs parallel_reduce::<i32> and handwritten_reduce_i32 on same 2048-element data
///   - Asserts results match — proving the trait abstraction has zero cost
#[gpu_test]
fn test_gpu_generic_zero_overhead() {}

/// Test generic map-then-reduce composition: transform + reduce fused in one loop.
///
/// The kernel `test_gpu_generic_map_reduce` in gpu-kernel-test:
///   - f32: map(x*2+1) then reduce over 1024 elements
///   - i32: map(x*3-1) then reduce over 100 elements
///   - Vec2f: map(scale+offset) then reduce over 50 elements
///   - Proves GpuReducible + GpuTransformable compose correctly on GPU
#[gpu_test]
fn test_gpu_generic_map_reduce() {}

// ============================================================================
// GPU tests — dynamic dispatch (dyn-probe.2)
// ============================================================================

/// Test &dyn Trait dynamic dispatch on GPU hardware.
///
/// The kernel `test_gpu_dyn_trait` in gpu-kernel-test:
///   - Creates GreeterA (returns 42) and GreeterB (returns 99)
///   - Calls both through &dyn Greeter (vtable lookup + indirect call)
///   - Runtime-selects a trait object and calls it
///   - Writes [42, 99, 42] to output buffer
///
/// This test cannot use `#[gpu_test]` because the kernel takes a `result: *mut u32`
/// parameter for output verification. Uses `GpuStdModule` to inject hostcall via
/// `__HOSTCALL_BUF` device global and pass the result buffer via `launch_raw`.
#[test]
fn test_gpu_dyn_trait() {
    // Load cubin for fast startup (sub-second) or fall back to PTX JIT (~25 min).
    let cubin = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        // kernel_test.cubin is in gpu-host dir (built by ptxas from kernel_test.ptx)
        let cubin_path =
            std::path::Path::new(manifest).join("../../core/gpu-host/kernel_test.cubin");
        std::fs::read(&cubin_path).unwrap_or_default()
    };

    let module = gpu_host::gpu::GpuStdModule::load_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "test_gpu_dyn_trait",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            eprintln!("  [GPU] {}", s.trim());
        })),
    )
    .expect("failed to load test_gpu_dyn_trait kernel");

    // Allocate output buffer: 3 x u32 (val_a, val_b, val_dynamic)
    let result_dev: cudarc::driver::CudaSlice<u32> = module
        .device()
        .alloc_zeros::<u32>(3)
        .expect("failed to allocate output buffer");
    let mut result_ptr: u64 = *result_dev.device_ptr();

    unsafe {
        module
            .launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])
            .expect("test_gpu_dyn_trait kernel launch failed");
    }

    // Brief sleep so hostcall listener can flush remaining println! messages
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module
        .device()
        .dtoh_sync_copy(&result_dev)
        .expect("failed to copy results from device");
    module.finish();

    // Verify dynamic dispatch produced correct results
    assert_eq!(
        result[0], 42,
        "GreeterA via &dyn Greeter should return 42, got {}",
        result[0]
    );
    assert_eq!(
        result[1], 99,
        "GreeterB via &dyn Greeter should return 99, got {}",
        result[1]
    );
    assert_eq!(
        result[2], 42,
        "Runtime-selected &dyn Greeter should return 42, got {}",
        result[2]
    );
}

// ============================================================================
// GPU tests — Box<dyn Trait> dynamic dispatch (dyn-box.1)
// ============================================================================

/// Test Box<dyn Trait> heap-allocated dynamic dispatch on GPU hardware.
///
/// The kernel `test_gpu_box_dyn_trait` in gpu-kernel-test:
///   - Creates Box<dyn Animal> for Cat (returns 1), Dog (returns 2), Parrot (returns 100+n)
///   - Calls speak() through Box<dyn Animal> (heap allocation + vtable lookup + indirect call)
///   - Builds Vec<Box<dyn Animal>> with 5 elements and iterates
///   - Runtime-selects a Box<dyn Animal> via conditional
///   - Writes [1, 2, 142, 234, 2] to output buffer
///
/// Uses `GpuStdModule` to inject hostcall via `__HOSTCALL_BUF` device global.
#[test]
fn test_gpu_box_dyn_trait() {
    // Load cubin for fast startup (sub-second) or fall back to PTX JIT (~25 min).
    let cubin = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let cubin_path =
            std::path::Path::new(manifest).join("../../core/gpu-host/kernel_test.cubin");
        std::fs::read(&cubin_path).unwrap_or_default()
    };

    let module = gpu_host::gpu::GpuStdModule::load_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "test_gpu_box_dyn_trait",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            eprintln!("  [GPU] {}", s.trim());
        })),
    )
    .expect("failed to load test_gpu_box_dyn_trait kernel");

    // Allocate output buffer: 5 x u32 (cat, dog, parrot, vec_sum, runtime_chosen)
    let result_dev: cudarc::driver::CudaSlice<u32> = module
        .device()
        .alloc_zeros::<u32>(5)
        .expect("failed to allocate output buffer");
    let mut result_ptr: u64 = *result_dev.device_ptr();

    unsafe {
        module
            .launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])
            .expect("test_gpu_box_dyn_trait kernel launch failed");
    }

    // Brief sleep so hostcall listener can flush remaining println! messages
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module
        .device()
        .dtoh_sync_copy(&result_dev)
        .expect("failed to copy results from device");
    module.finish();

    // Verify Box<dyn Trait> dynamic dispatch produced correct results
    assert_eq!(
        result[0], 1,
        "Cat via Box<dyn Animal> should return 1, got {}",
        result[0]
    );
    assert_eq!(
        result[1], 2,
        "Dog via Box<dyn Animal> should return 2, got {}",
        result[1]
    );
    assert_eq!(
        result[2], 142,
        "Parrot(42) via Box<dyn Animal> should return 142, got {}",
        result[2]
    );
    assert_eq!(
        result[3], 234,
        "Vec<Box<dyn Animal>> sum should be 234, got {}",
        result[3]
    );
    assert_eq!(
        result[4], 2,
        "Runtime-chosen Box<dyn Animal> should be Dog (returns 2), got {}",
        result[4]
    );
}

// ============================================================================
// GPU tests — &dyn Fn() closures + Drop for Box<dyn Trait> (dyn-box.2)
// ============================================================================

/// Test &dyn Fn() closures and Drop for Box<dyn Trait> on GPU hardware.
///
/// The kernel `test_gpu_dyn_fn_and_drop` in gpu-kernel-test:
///   - Creates &dyn Fn() closures (simple and capturing)
///   - Calls closures through &dyn Fn and via call_fn_dyn helper
///   - Creates Box<dyn Fn()> (heap-allocated closure with capture)
///   - Creates Box<dyn Droppable> and verifies Drop::drop is called via vtable
///   - Creates Vec<Box<dyn Droppable>> with HasDrop + HasDrop2 and verifies
///     all destructors are called with correct dispatch
///   - Writes [11, 20, 15, 21, 105, 42, 1, 201] to output buffer
///
/// Uses `GpuStdModule` to inject hostcall via `__HOSTCALL_BUF` device global.
#[test]
fn test_gpu_dyn_fn_and_drop() {
    // Load cubin for fast startup (sub-second) or fall back to PTX JIT (~25 min).
    let cubin = {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let cubin_path =
            std::path::Path::new(manifest).join("../../core/gpu-host/kernel_test.cubin");
        std::fs::read(&cubin_path).unwrap_or_default()
    };

    let module = gpu_host::gpu::GpuStdModule::load_with_cubin(
        gpu_host::ptx::KERNEL_STD,
        &cubin,
        "test_gpu_dyn_fn_and_drop",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            eprintln!("  [GPU] {}", s.trim());
        })),
    )
    .expect("failed to load test_gpu_dyn_fn_and_drop kernel");

    // Allocate output buffer: 8 x u32
    let result_dev: cudarc::driver::CudaSlice<u32> = module
        .device()
        .alloc_zeros::<u32>(8)
        .expect("failed to allocate output buffer");
    let mut result_ptr: u64 = *result_dev.device_ptr();

    unsafe {
        module
            .launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])
            .expect("test_gpu_dyn_fn_and_drop kernel launch failed");
    }

    // Brief sleep so hostcall listener can flush remaining println! messages
    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module
        .device()
        .dtoh_sync_copy(&result_dev)
        .expect("failed to copy results from device");
    module.finish();

    // Verify &dyn Fn() closures
    assert_eq!(
        result[0], 11,
        "&dyn Fn add_one(10) should return 11, got {}",
        result[0]
    );
    assert_eq!(
        result[1], 20,
        "&dyn Fn double(10) should return 20, got {}",
        result[1]
    );
    assert_eq!(
        result[2], 15,
        "captured closure add_captured(5) should return 15, got {}",
        result[2]
    );
    assert_eq!(
        result[3], 21,
        "captured closure mul_captured(3) should return 21, got {}",
        result[3]
    );
    assert_eq!(
        result[4], 105,
        "Box<dyn Fn>(100) should return 105, got {}",
        result[4]
    );

    // Verify Drop for Box<dyn Trait>
    assert_eq!(
        result[5], 42,
        "HasDrop.value() via Box<dyn Droppable> should return 42, got {}",
        result[5]
    );
    assert_eq!(
        result[6], 1,
        "drop_counter after single Box<dyn> drop should be 1, got {}",
        result[6]
    );
    assert_eq!(
        result[7], 201,
        "drop_counter after Vec<Box<dyn>> drop should be 201, got {}",
        result[7]
    );
}

// ============================================================================
// GPU tests — third-party no_std crate with dyn Trait (dyn-compat.1)
// ============================================================================

/// Test hashbrown (third-party no_std crate) on GPU with internal dyn dispatch.
///
/// The kernel `test_gpu_dyn_compat_hashbrown` in gpu-kernel-test:
///   - Creates hashbrown::HashMap with a custom hasher, inserts/gets entries
///     (internally triggers `&dyn FnMut(usize) -> bool` in find_inner)
///   - Creates hashbrown::HashSet, inserts + contains checks
///   - Inserts 50 entries to force resize (triggers `&dyn Fn(...)` in resize_inner)
///   - Verifies remove, iteration, and value summation
///   - hashbrown is used UNMODIFIED — just `hashbrown = { default-features = false }`
#[gpu_test]
fn test_gpu_dyn_compat_hashbrown() {}

// ============================================================================
// GPU tests — tiered memory (SharedRef / GlobalRef) valid patterns
// ============================================================================

/// Test valid SharedRef patterns within block_scope on GPU.
///
/// The kernel `test_gpu_shared_ref_valid_patterns` in gpu-kernel-test:
///   - alloc_shared → SharedRef, write + read 16 u32 elements
///   - sub_ref for tiling (two non-overlapping 32-element tiles)
///   - Pass SharedRef to helper functions within scope
///   - Cooperative spawn_all with SharedRef (each warp fills its partition)
///   - f32 type with SharedRef
#[gpu_test]
fn test_gpu_shared_ref_valid_patterns() {}

/// Test valid GlobalRef patterns within grid_scope on GPU.
///
/// The kernel `test_gpu_global_ref_valid_patterns` in gpu-kernel-test:
///   - alloc_global → GlobalRef, write + read 16 u32 elements
///   - sub_ref for partitioning
///   - Cross-warp communication (GlobalRef is Send+Sync via spawn_all)
///   - u64 type with GlobalRef
#[gpu_test]
fn test_gpu_global_ref_valid_patterns() {}

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
