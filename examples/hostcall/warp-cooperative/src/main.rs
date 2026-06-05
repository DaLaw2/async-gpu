//! Cooperative Compute on GPU — showcase example.
//!
//! Demonstrates warp-parallel compute patterns using the `gpu_runtime::thread`
//! cooperative APIs. Each GPU warp (32 SIMT lanes) acts as a logical "thread".
//! Warp 0 runs the main function; the cooperative APIs wake all warps to
//! process data in parallel, then return control to warp 0.
//!
//! # What This Shows
//!
//! 1. **cooperative()** — all warps execute the same closure in parallel.
//!    Data is partitioned by warp ID. The simplest cooperative primitive.
//!
//! 2. **cooperative_map()** — data-parallel transform with explicit
//!    `(src, dst, len)` arguments. No global atomics, no closure captures.
//!    Each warp handles elements where `index % n_warps == warp_id`.
//!
//! 3. **cooperative_reduce()** — multi-warp reduction to a single value.
//!    Each warp computes a partial sum; warp 0 combines them.
//!
//! 4. **cooperative_map_with_params()** — parameterized data-parallel compute.
//!    Extra `u64` parameters carry matrix dimensions, scalars, etc.
//!    Demonstrated with a cooperative matrix multiply (C = A x B).
//!
//! # How It Works
//!
//! The kernel side (in `crates/kernel/gpu-kernel-test/src/thread_test.rs`) uses:
//! - `gpu_runtime::thread::gpu_main` — sets up the warp thread pool
//! - `thread::cooperative(&|| { ... })` — runs closure on all warps
//! - `thread::cooperative_map(src, dst, len, fn)` — data-parallel transform
//! - `thread::cooperative_reduce(src, len, fn)` — multi-warp reduction
//! - `thread::cooperative_map_with_params(src, dst, len, params, fn)` — parameterized map
//!
//! The host side (this file) just launches the kernels and verifies results.
//! No manual PTX loading — just `gpu::custom()` + verify.

use async_gpu::gpu;

/// Path to the pre-compiled cubin for fast kernel loading.
///
/// When available, kernel loading takes <1 second instead of 10+ minutes
/// (PTX JIT compilation). Build with `scripts/build-kernel-test.sh`.
const CUBIN_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../../crates/core/gpu-host/kernel_std.cubin"
);

fn main() {
    println!("=== Cooperative Compute on GPU ===\n");

    let mut all_passed = true;

    // ----------------------------------------------------------------
    // Demo 1: cooperative() — basic all-warps parallel execution
    // ----------------------------------------------------------------
    //
    // All 4 warps execute the same closure. Each warp uses its ID
    // to compute output[i] = i * 2 + 1 for its partition of indices.
    // Warp 0 handles i = 0, 4, 8, ...; warp 1 handles i = 1, 5, 9, ...
    //
    // This demonstrates the fundamental cooperative pattern:
    //   Sequential code → cooperative(&|| { ... }) → Sequential code
    //
    // Expected: output[i] = i * 2 + 1 for i in 0..256
    println!("--- Demo 1: cooperative() — basic parallel execution ---");
    match run_cooperative_basic() {
        Ok(pass) => {
            println!(
                "  Verification: {}\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 2: cooperative_map() — data-parallel transform
    // ----------------------------------------------------------------
    //
    // All warps cooperatively double each element of an array.
    // Unlike cooperative(), data flows through explicit (src, dst, len)
    // parameters — no global atomics needed, no unsafe required.
    //
    // The key API pattern:
    //   thread::cooperative_map(src_ptr, dst_ptr, len, |args| {
    //       // args.warp_id, args.n_warps, args.src, args.dst, args.len
    //       // Each warp processes elements at stride n_warps
    //   });
    //
    // Expected: output[i] = i * 2 for i in 0..256
    println!("--- Demo 2: cooperative_map() — data-parallel transform ---");
    match run_cooperative_map() {
        Ok(pass) => {
            println!(
                "  Verification: {}\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 3: cooperative_reduce() — multi-warp reduction
    // ----------------------------------------------------------------
    //
    // All warps compute partial sums of a 256-element array, then
    // warp 0 combines them into a single total.
    //
    // Each warp sums its partition: warp k sums elements k, k+4, k+8, ...
    // Warp 0 collects all partial results and returns the total.
    //
    // Expected: sum of 0..256 = 32640
    println!("--- Demo 3: cooperative_reduce() — multi-warp reduction ---");
    match run_cooperative_reduce() {
        Ok(pass) => {
            println!(
                "  Verification: {}\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 4: cooperative_map_with_params() — parameterized compute
    // ----------------------------------------------------------------
    //
    // Elementwise multiply by a scalar passed via params[0].
    // The params array carries up to 4 u64 values — useful for
    // matrix dimensions, scalar multipliers, stride values, etc.
    //
    // Expected: output[i] = i * 7 for i in 0..256
    println!("--- Demo 4: cooperative_map_with_params() — scalar multiply ---");
    match run_cooperative_map_ext() {
        Ok(pass) => {
            println!(
                "  Verification: {}\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Demo 5: cooperative matmul — C[8x6] = A[8x4] x B[4x6]
    // ----------------------------------------------------------------
    //
    // Full cooperative matrix multiply using cooperative_map_with_params.
    // Each warp computes its partition of output rows. The params array
    // carries matrix dimensions (M, K, N) and the B pointer.
    //
    // This shows how cooperative compute scales to real workloads:
    // row partitioning across warps, triple-loop matmul per warp.
    //
    // Expected: C = A x B with known input matrices
    println!("--- Demo 5: cooperative matmul — C = A x B ---");
    match run_cooperative_matmul() {
        Ok(pass) => {
            println!(
                "  Verification: {}\n",
                if pass { "PASSED" } else { "FAILED" }
            );
            if !pass {
                all_passed = false;
            }
        }
        Err(e) => {
            println!("  SKIP: {e}\n");
            all_passed = false;
        }
    }

    // ----------------------------------------------------------------
    // Summary
    // ----------------------------------------------------------------
    if all_passed {
        println!("=== All demos passed! ===");
    } else {
        println!("=== Some demos failed or were skipped ===");
    }
}

// ====================================================================
// Demo implementations
// ====================================================================

/// Demo 1: cooperative() — all warps execute the same closure.
///
/// Kernel: `cooperative_compute_test` from gpu-kernel-test/thread_test.rs.
/// Each warp fills its slice of output[i] = i * 2 + 1.
///
/// Launch: 4 warps (128 threads), 256 output elements.
fn run_cooperative_basic() -> Result<bool, Box<dyn std::error::Error>> {
    let ctx = gpu::custom("cooperative_compute_test")
        .threads(128) // 4 warps
        .cubin_file(CUBIN_PATH)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u32>(256)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    let mut pass = true;
    for (i, &val) in values.iter().enumerate().take(256) {
        let expected = (i as u32) * 2 + 1;
        if val != expected {
            println!("  FAIL: output[{i}] = {val}, expected {expected}");
            pass = false;
        }
    }
    if pass {
        println!("  4 warps cooperatively filled 256 elements");
        println!("  output[i] = i * 2 + 1 (verified all 256)");
    }
    Ok(pass)
}

/// Demo 2: cooperative_map() — data-parallel transform, zero global atomics.
///
/// Kernel: `cooperative_map_test` from gpu-kernel-test/thread_test.rs.
/// All warps double each element: output[i] = input[i] * 2.
///
/// Launch: 4 warps (128 threads), 256 elements.
fn run_cooperative_map() -> Result<bool, Box<dyn std::error::Error>> {
    let ctx = gpu::custom("cooperative_map_test")
        .threads(128) // 4 warps
        .cubin_file(CUBIN_PATH)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u32>(256)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    let mut pass = true;
    for (i, &val) in values.iter().enumerate().take(256) {
        let expected = (i as u32) * 2;
        if val != expected {
            println!("  FAIL: output[{i}] = {val}, expected {expected}");
            pass = false;
        }
    }
    if pass {
        println!("  4 warps cooperatively doubled 256 elements");
        println!("  cooperative_map: no global atomics, explicit (src, dst, len)");
    }
    Ok(pass)
}

/// Demo 3: cooperative_reduce() — multi-warp sum reduction.
///
/// Kernel: `cooperative_reduce_test` from gpu-kernel-test/thread_test.rs.
/// All warps compute partial sums; warp 0 combines into total.
///
/// Launch: 4 warps (128 threads), reducing 256 elements.
fn run_cooperative_reduce() -> Result<bool, Box<dyn std::error::Error>> {
    let ctx = gpu::custom("cooperative_reduce_test")
        .threads(128) // 4 warps
        .cubin_file(CUBIN_PATH)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u64>(1)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    // sum of 0..256 = 255 * 256 / 2 = 32640
    let expected = 32640u64;
    let total = values[0];
    let pass = total == expected;

    if pass {
        println!("  4 warps reduced 256 elements to sum = {total}");
        println!("  Each warp computed partial sum of its partition");
    } else {
        println!("  FAIL: total = {total}, expected {expected}");
    }
    Ok(pass)
}

/// Demo 4: cooperative_map_with_params() — scalar multiply via params.
///
/// Kernel: `cooperative_map_ext_test` from gpu-kernel-test/thread_test.rs.
/// Each element multiplied by scalar=7 (passed via params[0]).
///
/// Launch: 4 warps (128 threads), 256 elements.
fn run_cooperative_map_ext() -> Result<bool, Box<dyn std::error::Error>> {
    let ctx = gpu::custom("cooperative_map_ext_test")
        .threads(128) // 4 warps
        .cubin_file(CUBIN_PATH)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u32>(256)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    let mut pass = true;
    for (i, &val) in values.iter().enumerate().take(256) {
        let expected = (i as u32) * 7;
        if val != expected {
            println!("  FAIL: output[{i}] = {val}, expected {expected}");
            pass = false;
        }
    }
    if pass {
        println!("  4 warps multiplied 256 elements by scalar 7");
        println!("  Scalar passed via params[0] — no closure captures needed");
    }
    Ok(pass)
}

/// Demo 5: cooperative matmul — C[8x6] = A[8x4] x B[4x6].
///
/// Kernel: `cooperative_matmul_test` from gpu-kernel-test/thread_test.rs.
/// Row-parallel matmul: each warp computes rows where i % n_warps == warp_id.
///
/// Launch: 4 warps (128 threads), 8x6 = 48 output floats.
fn run_cooperative_matmul() -> Result<bool, Box<dyn std::error::Error>> {
    const M: usize = 8;
    const K: usize = 4;
    const N: usize = 6;

    let ctx = gpu::custom("cooperative_matmul_test")
        .threads(128) // 4 warps
        .cubin_file(CUBIN_PATH)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<f32>(M * N)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;

    // Compute expected C = A x B on the host.
    // A[i][j] = (i * K + j + 1) as f32
    // B[i][j] = ((i * N + j + 1) * 2) as f32
    let mut expected = [0.0f32; M * N];
    for i in 0..M {
        for j in 0..N {
            let mut sum = 0.0f32;
            for p in 0..K {
                let a = (i * K + p + 1) as f32;
                let b = ((p * N + j + 1) * 2) as f32;
                sum += a * b;
            }
            expected[i * N + j] = sum;
        }
    }

    let mut pass = true;
    let mut max_err = 0.0f32;
    for i in 0..M * N {
        let err = (values[i] - expected[i]).abs();
        if err > max_err {
            max_err = err;
        }
        if err > 0.01 {
            println!(
                "  FAIL: C[{}][{}] = {}, expected {}",
                i / N,
                i % N,
                values[i],
                expected[i]
            );
            pass = false;
        }
    }
    if pass {
        println!("  C[{M}x{N}] = A[{M}x{K}] x B[{K}x{N}] — 4 warps, row-parallel");
        println!("  Max error: {max_err:.6} (48 elements verified)");
    }
    Ok(pass)
}
