//! Structured Concurrency on GPU — showcase example.
//!
//! Demonstrates GPU structured concurrency using `BlockScope`, `BlockOneshotSlot`,
//! `spawn_all`, and nested scopes. Each demo launches a pre-built kernel from
//! `gpu-kernel-std` that exercises these primitives:
//!
//! 1. **Producer-Consumer Pipeline** — two warps communicate via a shared-memory
//!    oneshot channel inside a `BlockScope`. The producer fills a buffer, signals
//!    completion; the consumer waits, sums the data, and returns the result.
//!
//! 2. **Cooperative Data-Parallel** — all warps process data in parallel via
//!    `scope.spawn_all()`. Each warp handles a stride of elements. The scope
//!    ensures all warps finish before the manager reads results.
//!
//! 3. **Nested Scopes** — an outer scope allocates a persistent buffer, an inner
//!    scope allocates scratch space for a worker. When the inner scope exits, its
//!    memory is reclaimed (watermark pop) while the outer buffer survives.
//!
//! 4. **Combined** — producer-consumer pipeline followed by cooperative reduction
//!    in a single scope, showing `spawn` + `join_all` + `spawn_all` composability.
//!
//! 5. **Multi-Block GridScope** — grid-level structured concurrency with global
//!    memory allocation, atomic completion tracking, and parallel reduce across
//!    virtual blocks (warps acting as independent blocks within one CTA).
//!
//! # How It Works
//!
//! The kernel side (in `crates/kernel/gpu-kernel-std/src/sc_demo.rs`) uses:
//! - `gpu_runtime::scope::block_scope` — creates a lifetime-bounded scope
//! - `scope.alloc::<T>(n)` — bump-allocates from shared memory
//! - `scope.spawn(closure)` — dispatches a closure to an idle warp
//! - `scope.spawn_all(|wid, n_warps| { ... })` — data-parallel across all warps
//! - `block_oneshot(slot)` — oneshot channel in shared memory (~2 cycle latency)
//! - `gpu_runtime::scope::grid_scope` — grid-level scope with global memory pool
//!
//! The host side (this file) just launches the kernels and verifies results.
//! No manual PTX loading, no buffer management — just `gpu::custom()` + verify.

use async_gpu::gpu;

fn main() {
    println!("=== Structured Concurrency on GPU ===\n");

    let mut all_passed = true;

    // ----------------------------------------------------------------
    // Demo 1: BlockScope Producer-Consumer Pipeline
    // ----------------------------------------------------------------
    //
    // Two warps communicate through a BlockOneshotSlot in shared memory:
    //   - Warp 0 (manager) enters block_scope, allocates buffer + oneshot slot
    //   - Producer warp fills buffer with data[i] = i, signals via oneshot
    //   - Consumer warp waits for signal, sums buffer, returns result
    //   - Manager joins both warps, writes verified sum to output
    //
    // Expected: sum of 0..64 = 2016
    println!("--- Demo 1: Producer-Consumer Pipeline ---");
    match run_sc_kernel("sc_producer_consumer", 2, 2048) {
        Ok(result) => {
            let sum = result[0];
            let success = result[1];
            let expected_sum = (0u32..64).sum::<u32>(); // 2016
            let pass = sum == expected_sum && success == 1;
            println!("  Producer filled 64-element shared buffer");
            println!("  Consumer received signal, computed sum = {sum}");
            println!(
                "  Verification: {} (expected {expected_sum})\n",
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
    // Demo 2: Cooperative Data-Parallel with spawn_all()
    // ----------------------------------------------------------------
    //
    // All warps process data in parallel — each warp handles elements
    // where index % n_warps == warp_id. The scope guarantees all warps
    // complete before warp 0 reads the output.
    //
    // Expected: sum of (i * 2) for i in 0..128 = 16256
    println!("--- Demo 2: Cooperative spawn_all ---");
    match run_sc_kernel("sc_cooperative_parallel", 2, 4096) {
        Ok(result) => {
            let sum = result[0];
            let success = result[1];
            let expected_sum = (0u32..128).map(|i| i * 2).sum::<u32>(); // 16256
            let pass = sum == expected_sum && success == 1;
            println!("  All warps cooperatively doubled 128 elements");
            println!("  Warp 0 summed output = {sum}");
            println!(
                "  Verification: {} (expected {expected_sum})\n",
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
    // Demo 3: Nested Scopes — inner scope scratch, outer buffer persists
    // ----------------------------------------------------------------
    //
    // The shared memory allocator uses a watermark stack:
    //   - Outer scope allocates 64-element buffer (fills with i+10)
    //   - Inner scope allocates 16-element scratch, spawns worker
    //   - Worker computes partial sum of outer_buf[0..16] into scratch
    //   - Inner scope exits: scratch freed (watermark popped)
    //   - Outer buffer still valid — warp 0 reads outer_buf[0] = 10
    //
    // Expected: inner_sum = sum(10..26) = 280, outer_first = 10
    println!("--- Demo 3: Nested Scopes ---");
    match run_sc_kernel("sc_nested_scopes", 4, 2048) {
        Ok(result) => {
            let inner_sum = result[0];
            let outer_first = result[1];
            let mem_reclaimed = result[2];
            let success = result[3];

            // sum(10..26) = 16*10 + sum(0..16) = 160 + 120 = 280
            let expected_inner = (10u32..26).sum::<u32>(); // 280
            let pass =
                inner_sum == expected_inner && outer_first == 10 && mem_reclaimed == 1 && success == 1;

            println!("  Outer scope: 64-element buffer (data[i] = i + 10)");
            println!("  Inner scope: worker summed outer_buf[0..16] = {inner_sum}");
            println!("  After inner scope exit: outer_buf[0] = {outer_first} (still valid)");
            println!(
                "  Memory reclaimed: {}",
                if mem_reclaimed == 1 { "yes" } else { "no" }
            );
            println!(
                "  Verification: {} (expected inner_sum={expected_inner}, outer_first=10)\n",
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
    // Demo 4: Combined — spawn + join_all + spawn_all in one scope
    // ----------------------------------------------------------------
    //
    // Shows composability of structured concurrency primitives:
    //   Phase 1: spawn producer (fills data[i] = i) + consumer (waits, triples data)
    //   Phase 2: join_all — ensures both warps complete
    //   Phase 3: spawn_all — all warps cooperatively compute partial sums
    //   Phase 4: warp 0 reduces partial sums
    //
    // Expected: sum of (i * 3) for i in 0..32 = 1488
    println!("--- Demo 4: Combined spawn + spawn_all ---");
    match run_sc_kernel("sc_combined_demo", 2, 4096) {
        Ok(result) => {
            let final_sum = result[0];
            let success = result[1];
            let expected = (0u32..32).map(|i| i * 3).sum::<u32>(); // 1488
            let pass = final_sum == expected && success == 1;
            println!("  Phase 1: producer fills, consumer triples via oneshot signal");
            println!("  Phase 2: join_all ensures pipeline complete");
            println!("  Phase 3: spawn_all computes partial sums across warps");
            println!("  Final reduced sum = {final_sum}");
            println!(
                "  Verification: {} (expected {expected})\n",
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
    // Demo 5: GridScope — multi-block parallel reduce
    // ----------------------------------------------------------------
    //
    // Grid-level structured concurrency with global memory pool:
    //   - Warp 0 enters grid_scope with a pre-allocated device memory pool
    //   - GridScope allocates input data + partial sums from the pool
    //   - Worker warps act as "virtual blocks", each computing a partial sum
    //   - Workers signal completion via atomic counter in the pool header
    //   - Coordinator waits for all completions, reduces partial sums
    //
    // Expected: sum of 1..=128 = 8256
    println!("--- Demo 5: GridScope Multi-Block Reduce ---");
    match run_grid_reduce() {
        Ok((final_sum, completions, success)) => {
            let expected = 128u32 * 129 / 2; // 8256
            let pass = final_sum == expected && success == 1;
            println!("  GridScope allocated input (128 elements) from global pool");
            println!("  {completions} virtual blocks completed their partial sums");
            println!("  Coordinator reduced to final sum = {final_sum}");
            println!(
                "  Verification: {} (expected {expected})\n",
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

/// Launch a structured concurrency kernel that takes `(result: *mut u32)`.
///
/// Uses `gpu::custom()` with shared memory allocation. 4 warps (128 threads):
/// warp 0 = scope manager, warps 1-3 = workers.
fn run_sc_kernel(
    kernel_name: &'static str,
    n_output: usize,
    shared_mem_bytes: u32,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let ctx = gpu::custom(kernel_name)
        .threads(128)
        .shared_mem(shared_mem_bytes)
        .prepare()?;

    let mut output = ctx.alloc_zeros::<u32>(n_output)?;
    let result = unsafe { ctx.launch((&mut output,))? };
    let values = result.download(&output)?;
    Ok(values)
}

/// Launch the GridScope reduce kernel which needs a global memory pool.
///
/// Kernel signature: `sc_grid_reduce(pool: *mut u8, pool_size: u32, result: *mut u32)`
fn run_grid_reduce() -> Result<(u32, u32, u32), Box<dyn std::error::Error>> {
    let pool_size: u32 = 4096;

    let ctx = gpu::custom("sc_grid_reduce")
        .threads(128)
        .shared_mem(2048)
        .prepare()?;

    // Allocate global memory pool for GridScope (passed as first kernel arg)
    let mut pool = ctx.alloc_zeros::<u8>(pool_size as usize)?;
    let mut output = ctx.alloc_zeros::<u32>(3)?;

    let result = unsafe { ctx.launch((&mut pool, pool_size, &mut output))? };
    let values = result.download(&output)?;

    Ok((values[0], values[1], values[2]))
}
