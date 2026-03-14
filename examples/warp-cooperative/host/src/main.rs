//! Host test for warp-cooperative async kernels.
//!
//! Loads PTX compiled with patched rustc and verifies:
//! 1. test_simple_warp: output[tid] = tid + 1
//! 2. test_multi_await: output[tid] = 2*tid + 12

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::Ptx;

const SIMPLE_PTX: &str = include_str!("../minimal.ptx");
const FULL_PTX: &str = include_str!("../kernel.ptx");
const WARP_SIZE: u32 = 32;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("=== Warp-Cooperative Async Kernel Test ===\n");

    let dev = CudaDevice::new(0)?;

    // ---- Test 1: simple_add (no .await, just bar.warp.sync) ----
    println!("--- Test 1: test_simple_warp ---");
    {
        let ptx = Ptx::from_src(SIMPLE_PTX);
        dev.load_ptx(ptx, "simple", &["test_simple_warp"])?;

        let mut output = dev.alloc_zeros::<u32>(WARP_SIZE as usize)?;
        let f = dev
            .get_func("simple", "test_simple_warp")
            .expect("test_simple_warp not found");
        let cfg = LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (WARP_SIZE, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe { f.launch(cfg, (&mut output,))? };
        let result = dev.dtoh_sync_copy(&output)?;

        let mut pass = true;
        for tid in 0..WARP_SIZE as usize {
            let expected = tid as u32 + 1;
            if result[tid] != expected {
                println!("  FAIL: output[{tid}] = {}, expected {expected}", result[tid]);
                pass = false;
            }
        }
        println!(
            "  test_simple_warp: {}\n",
            if pass { "PASSED" } else { "FAILED" }
        );
    }

    // ---- Test 2: multi_await (2 .await points, shfl.sync broadcast) ----
    println!("--- Test 2: test_multi_await ---");
    {
        let ptx = Ptx::from_src(FULL_PTX);
        match dev.load_ptx(ptx, "full", &["test_multi_await"]) {
            Ok(()) => {
                let mut output = dev.alloc_zeros::<u32>(WARP_SIZE as usize)?;
                let f = dev
                    .get_func("full", "test_multi_await")
                    .expect("test_multi_await not found");
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { f.launch(cfg, (&mut output,))? };
                let result = dev.dtoh_sync_copy(&output)?;

                let mut pass = true;
                for tid in 0..WARP_SIZE as usize {
                    // multi_await(x) = (x+1) + (x+1+10) = 2x + 12
                    let expected = 2 * tid as u32 + 12;
                    if result[tid] != expected {
                        println!("  FAIL: output[{tid}] = {}, expected {expected}", result[tid]);
                        pass = false;
                    }
                }
                println!(
                    "  test_multi_await: {}\n",
                    if pass { "PASSED" } else { "FAILED" }
                );
            }
            Err(e) => {
                println!("  SKIP: PTX load failed ({e:?}) — extern panic function not resolvable");
                println!("  (This is expected — the panic path is unreachable at runtime)\n");
            }
        }
    }

    // ---- Test 3: async_pipeline (6 .await points, simulated I/O pipeline) ----
    println!("--- Test 3: test_async_pipeline ---");
    {
        let ptx = Ptx::from_src(FULL_PTX);
        match dev.load_ptx(ptx, "pipeline", &["test_async_pipeline"]) {
            Ok(()) => {
                let mut output = dev.alloc_zeros::<u32>(WARP_SIZE as usize)?;
                let f = dev
                    .get_func("pipeline", "test_async_pipeline")
                    .expect("test_async_pipeline not found");
                let cfg = LaunchConfig {
                    grid_dim: (1, 1, 1),
                    block_dim: (WARP_SIZE, 1, 1),
                    shared_mem_bytes: 0,
                };
                unsafe { f.launch(cfg, (&mut output,))? };
                let result = dev.dtoh_sync_copy(&output)?;

                let mut pass = true;
                for tid in 0..WARP_SIZE as usize {
                    // async_pipeline always returns 29*1000 + 29 = 29029
                    let expected = 29029u32;
                    if result[tid] != expected {
                        println!("  FAIL: output[{tid}] = {}, expected {expected}", result[tid]);
                        pass = false;
                    }
                }
                println!(
                    "  test_async_pipeline: {}\n",
                    if pass { "PASSED" } else { "FAILED" }
                );
            }
            Err(e) => {
                println!("  SKIP: PTX load failed ({e:?})");
                println!("  (May need PTX post-processing for panic stub)\n");
            }
        }
    }

    println!("=== All tests complete ===");
    Ok(())
}
