//! Dynamic Dispatch on GPU — &dyn Trait and Box<dyn Trait>.
//!
//! This example launches pre-compiled GPU kernels that demonstrate:
//!
//! 1. **&dyn Trait** — stack-allocated data dispatched through vtable
//! 2. **Box<dyn Trait>** — heap-allocated trait objects on GPU
//! 3. **Vec<Box<dyn Trait>>** — heterogeneous collections
//! 4. **Drop for Box<dyn Trait>** — destructor dispatch via vtable
//!
//! The kernel code lives in `crates/kernel/gpu-kernel-test/src/lib.rs`.
//! The host side just launches the kernels and verifies results.
//!
//! ## Kernel-side code (what runs on GPU)
//!
//! ```rust,ignore
//! trait Greeter {
//!     fn greet(&self) -> u32;
//! }
//! struct GreeterA;
//! impl Greeter for GreeterA { fn greet(&self) -> u32 { 42 } }
//! struct GreeterB;
//! impl Greeter for GreeterB { fn greet(&self) -> u32 { 99 } }
//!
//! // Dynamic dispatch through vtable:
//! let a = GreeterA;
//! let val = (&a as &dyn Greeter).greet(); // vtable lookup -> 42
//! ```

use cudarc::driver::DevicePtr;
use gpu_host::gpu::GpuStdModule;
use gpu_host::ptx;

fn main() {
    println!("=== Dynamic Dispatch on GPU ===\n");

    // Try cubin first (sub-second), fall back to PTX JIT (~25 min)
    let cubin = load_cubin();

    // -----------------------------------------------------------------------
    // Test 1: &dyn Trait — stack-allocated dynamic dispatch
    // -----------------------------------------------------------------------
    println!("--- Test 1: &dyn Trait ---\n");
    println!("GPU kernel defines trait Greeter with two impls (GreeterA, GreeterB).");
    println!("Calls greet() through &dyn Greeter — vtable lookup + indirect call.\n");

    let module = GpuStdModule::load_with_cubin(
        ptx::KERNEL_STD,
        &cubin,
        "test_gpu_dyn_trait",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            eprintln!("  [GPU] {}", s.trim());
        })),
    )
    .expect("Failed to load test_gpu_dyn_trait kernel");

    // Allocate output buffer: 3 x u32 (val_a, val_b, val_dynamic)
    let result_dev: cudarc::driver::CudaSlice<u32> = module
        .device()
        .alloc_zeros::<u32>(3)
        .expect("Failed to allocate output buffer");
    let mut result_ptr: u64 = *result_dev.device_ptr();

    unsafe {
        module
            .launch_raw(&[&mut result_ptr as *mut u64 as *mut std::ffi::c_void])
            .expect("Kernel launch failed");
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let result: Vec<u32> = module
        .device()
        .dtoh_sync_copy(&result_dev)
        .expect("Failed to copy results");
    module.finish();

    println!("Results:");
    println!("  GreeterA.greet() via &dyn: {} (expected 42)", result[0]);
    println!("  GreeterB.greet() via &dyn: {} (expected 99)", result[1]);
    println!("  Runtime-selected &dyn:     {} (expected 42)", result[2]);

    assert_eq!(result[0], 42, "GreeterA via &dyn should return 42");
    assert_eq!(result[1], 99, "GreeterB via &dyn should return 99");
    assert_eq!(result[2], 42, "Runtime-selected &dyn should return 42");
    println!("  PASS\n");

    // -----------------------------------------------------------------------
    // Test 2: Box<dyn Trait> — heap-allocated dynamic dispatch
    // -----------------------------------------------------------------------
    println!("--- Test 2: Box<dyn Trait> ---\n");
    println!("GPU kernel creates Box<dyn Animal> for Cat, Dog, Parrot.");
    println!("Heap allocation via GPU hostcall allocator + vtable dispatch.\n");

    let cubin2 = load_cubin();
    let module2 = GpuStdModule::load_with_cubin(
        ptx::KERNEL_STD,
        &cubin2,
        "test_gpu_box_dyn_trait",
        128,
        (1, 1, 1),
        Some(Box::new(|msg| {
            let s = String::from_utf8_lossy(msg);
            eprintln!("  [GPU] {}", s.trim());
        })),
    )
    .expect("Failed to load test_gpu_box_dyn_trait kernel");

    // Allocate output buffer: 5 x u32
    let result_dev2: cudarc::driver::CudaSlice<u32> = module2
        .device()
        .alloc_zeros::<u32>(5)
        .expect("Failed to allocate output buffer");
    let mut result_ptr2: u64 = *result_dev2.device_ptr();

    unsafe {
        module2
            .launch_raw(&[&mut result_ptr2 as *mut u64 as *mut std::ffi::c_void])
            .expect("Kernel launch failed");
    }

    std::thread::sleep(std::time::Duration::from_millis(100));

    let result2: Vec<u32> = module2
        .device()
        .dtoh_sync_copy(&result_dev2)
        .expect("Failed to copy results");
    module2.finish();

    println!("Results:");
    println!("  Cat.speak()   via Box<dyn>:  {} (expected 1)", result2[0]);
    println!("  Dog.speak()   via Box<dyn>:  {} (expected 2)", result2[1]);
    println!("  Parrot(42)    via Box<dyn>:  {} (expected 142)", result2[2]);
    println!("  Vec<Box<dyn>> sum:           {} (expected 234)", result2[3]);
    println!("  Runtime-chosen Box<dyn>:     {} (expected 2)", result2[4]);

    assert_eq!(result2[0], 1, "Cat via Box<dyn> should return 1");
    assert_eq!(result2[1], 2, "Dog via Box<dyn> should return 2");
    assert_eq!(result2[2], 142, "Parrot(42) via Box<dyn> should return 142");
    assert_eq!(result2[3], 234, "Vec<Box<dyn>> sum should be 234");
    assert_eq!(result2[4], 2, "Runtime-chosen Box<dyn> should be Dog (2)");
    println!("  PASS\n");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!("Dynamic dispatch on GPU works identically to CPU Rust:");
    println!("  - &dyn Trait: vtable lookup + indirect call on stack data");
    println!("  - Box<dyn Trait>: heap-allocated data + vtable dispatch");
    println!("  - Vec<Box<dyn Trait>>: heterogeneous collections");
    println!("  - Runtime-selected trait objects via conditionals");
    println!("\n=== All assertions passed ===");
}

/// Try to load pre-compiled cubin for fast kernel loading.
///
/// Falls back to empty (triggers PTX JIT compilation) if cubin not found.
fn load_cubin() -> Vec<u8> {
    let manifest = env!("CARGO_MANIFEST_DIR");
    let cubin_path =
        std::path::Path::new(manifest).join("../../../crates/core/gpu-host/kernel_test.cubin");
    std::fs::read(&cubin_path).unwrap_or_default()
}
