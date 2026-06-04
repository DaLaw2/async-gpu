#![allow(clippy::needless_range_loop)]
mod bench_harness;
mod error;
mod hostcall;
mod mapped_mem;
mod tests_basic;
mod tests_benchmark;
mod tests_cnn;
mod tests_gemm;
mod tests_hostcall;
mod tests_inference;
mod tests_pipeline;
mod tests_scaling;
mod tests_search;
mod tests_std;
mod tests_tokenizer;
mod tests_transformer;
mod tests_warp;

use cudarc::driver::CudaDevice;
use error::{GpuHostError, Result};
use std::sync::Arc;

// PTX constants re-exported from the library crate.
const KERNEL_PTX: &str = gpu_host::ptx::KERNEL;
const EMBASSY_PTX: &str = gpu_host::ptx::EMBASSY_TEST;
const ASYNC_HOSTCALL_PTX: &str = gpu_host::ptx::ASYNC_HOSTCALL_TEST;
const STD_BUILD_TEST_PTX: &str = gpu_host::ptx::STD_BUILD_TEST;
const ASYNC_PIPELINE_PTX: &str = gpu_host::ptx::ASYNC_PIPELINE_TEST;
const MULTI_WARP_PTX: &str = gpu_host::ptx::MULTI_WARP_TEST;
const KERNEL_STD_PTX: &str = gpu_host::ptx::KERNEL_STD;

fn main() -> Result<()> {
    println!("=== GPU Kernel Execution Test ===\n");

    let dev = CudaDevice::new(0).map_err(GpuHostError::CudaInit)?;
    println!("CUDA device initialized successfully");

    // Quick filter: ONLY_TEST=generation to skip to generation test
    if let Ok(only) = std::env::var("ONLY_TEST") {
        match only.as_str() {
            "generation" => {
                tests_inference::run_generation_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "forward" => {
                tests_inference::run_f32_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "cpu_ref" => {
                tests_inference::run_cpu_f64_reference_test()?;
                return Ok(());
            }
            "mma_diag" => {
                tests_gemm::run_mma_diag(Arc::clone(&dev))?;
                return Ok(());
            }
            "splitk" => {
                tests_gemm::run_splitk_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mma_fwd" => {
                tests_inference::run_mma_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "kv_cache" => {
                tests_transformer::run_kv_cache_attention_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "kv_gen" => {
                tests_inference::run_kv_cached_generation_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bf16" => {
                tests_gemm::run_bf16_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bf16_fwd" => {
                tests_inference::run_bf16_forward_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "tf32" => {
                tests_gemm::run_tf32_gemm_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_fs" => {
                tests_std::run_std_file_io_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_sysroot_file" => {
                tests_std::run_std_sysroot_file_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_pipeline" => {
                tests_std::run_std_pipeline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_stdin" => {
                tests_std::run_std_stdin_readline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mt_malloc" => {
                tests_std::run_multithread_malloc_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "mt_std" => {
                tests_std::run_multithread_println_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "trace" => {
                tests_hostcall::run_trace_multithread_test(Arc::clone(&dev))?;
                tests_hostcall::run_trace_assert_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "session" => {
                tests_hostcall::run_session_multi_launch_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "cmd" => {
                tests_hostcall::run_multi_cmd_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "pipeline" => {
                tests_hostcall::run_cross_pipeline_test(Arc::clone(&dev))?;
                tests_hostcall::run_pipeline_api_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "converge" => {
                tests_hostcall::run_convergence_test(Arc::clone(&dev))?;
                tests_hostcall::run_autonomous_pipeline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "flight" => {
                tests_hostcall::run_flight_recorder_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_try" => {
                tests_warp::run_warp_try_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_await" => {
                tests_warp::run_warp_await_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "warp_e2e" => {
                tests_warp::run_warp_e2e_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "rustc_async" => {
                tests_warp::run_rustc_async_baseline_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "std_future" => {
                tests_hostcall::run_std_future_print_test(Arc::clone(&dev))?;
                tests_hostcall::run_std_future_two_prints_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_cooperative_future_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_cooperative_two_futures_test(Arc::clone(&dev))?;
                tests_hostcall::run_warp_result_future_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "throughput" => {
                tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "scalability" => {
                tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "file_io_bench" => {
                tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            "executor" => {
                tests_scaling::run_executor_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "channel" | "channel_oneshot" => {
                tests_scaling::run_channel_oneshot_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "channel_mpsc" | "mpsc" => {
                tests_scaling::run_channel_mpsc_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "compute" | "compute_pipeline" => {
                tests_scaling::run_compute_pipeline_demo_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "compute_bench" => {
                tests_scaling::run_compute_benchmark_test(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "async")]
            "tokio_bridge" | "tokio" => {
                let tokio_rt = tokio::runtime::Runtime::new().expect("tokio runtime init");
                tokio_rt
                    .block_on(tests_scaling::run_tokio_bridge_demo_test())
                    .map_err(|e| GpuHostError::Verification {
                        test: "tokio_bridge",
                        detail: format!("{e}"),
                    })?;
                return Ok(());
            }
            "gemm_bench" => {
                tests_gemm::run_gemm_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            #[cfg(feature = "cublas")]
            "attn_bench" => {
                tests_transformer::run_flash_attention_v3_bench(Arc::clone(&dev))?;
                return Ok(());
            }
            "cnn" => {
                tests_cnn::run_batchnorm_silu_test(Arc::clone(&dev))?;
                tests_cnn::run_cnn_ops_test(Arc::clone(&dev))?;
                tests_cnn::run_conv2d_test(Arc::clone(&dev))?;
                tests_cnn::run_yolo_io_test()?;
                tests_cnn::run_yolo_backbone_test(Arc::clone(&dev))?;
                tests_cnn::run_detect_head_test(Arc::clone(&dev))?;
                tests_cnn::run_yolo_end_to_end_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "yolo" => {
                tests_cnn::run_yolo_end_to_end_test(Arc::clone(&dev))?;
                return Ok(());
            }
            "bench" => {
                tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
                tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;
                tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;
                return Ok(());
            }
            _ => println!("Unknown ONLY_TEST={only}, running all tests"),
        }
    }

    tests_basic::run_write_thread_idx(Arc::clone(&dev))?;
    tests_basic::run_vector_add(Arc::clone(&dev))?;
    tests_basic::run_asm_smoke_tests(Arc::clone(&dev))?;
    tests_basic::run_integration_sys_store(Arc::clone(&dev))?;
    tests_basic::run_u64_atomics_tests(Arc::clone(&dev))?;
    tests_basic::run_warp_intrinsics_tests(Arc::clone(&dev))?;

    // Hostcall tests (hostcall.4)
    tests_hostcall::run_hostcall_print_hello(Arc::clone(&dev))?;
    tests_hostcall::run_hostcall_print_multi(Arc::clone(&dev), 4)?;

    // Embassy async/await tests (async-runtime.3)
    tests_hostcall::run_embassy_immediate(Arc::clone(&dev))?;
    tests_hostcall::run_embassy_countdown(Arc::clone(&dev))?;
    tests_hostcall::run_embassy_two_task(Arc::clone(&dev))?;
    tests_hostcall::run_sync_countdown(Arc::clone(&dev))?;

    // File I/O test (gpu-std.3)
    tests_hostcall::run_hostcall_file_test(Arc::clone(&dev))?;

    // Async hostcall tests (integration.1)
    tests_hostcall::run_async_hostcall_single(Arc::clone(&dev))?;
    tests_hostcall::run_async_hostcall_two(Arc::clone(&dev))?;

    // futures_util::future::join on GPU (integration.2)
    tests_hostcall::run_futures_join(Arc::clone(&dev))?;

    // GPU Instant + Time test (gpu-std.4)
    tests_hostcall::run_hostcall_time_test(Arc::clone(&dev))?;

    // Std-build-test (integration.3): -Zbuild-std=std on GPU
    tests_std::run_std_build_test(Arc::clone(&dev))?;

    // Dynamic allocation stress test (product.1)
    tests_std::run_dynamic_alloc_test(Arc::clone(&dev))?;

    // Hostcall latency benchmark (benchmark.2) — run before PAL tests
    tests_benchmark::run_hostcall_latency_benchmark(Arc::clone(&dev))?;

    // Warp divergence measurement (warp-future.2)
    tests_benchmark::run_warp_divergence_measurement(Arc::clone(&dev))?;

    // Bulk data transfer test (large-payload.3)
    tests_pipeline::run_bulk_io_test(Arc::clone(&dev))?;

    // Per-block sharding test (per-block-sharding.2)
    tests_pipeline::run_sharded_hostcall_test(Arc::clone(&dev))?;

    // Sharding benchmark (per-block-sharding.3)
    tests_benchmark::run_sharding_benchmark(Arc::clone(&dev))?;

    // Throughput + scalability benchmarks (bench-suite.2)
    tests_benchmark::run_throughput_benchmark(Arc::clone(&dev))?;
    tests_benchmark::run_scalability_benchmark(Arc::clone(&dev))?;

    // File I/O latency benchmark (bench-suite.3)
    tests_benchmark::run_file_io_benchmark(Arc::clone(&dev))?;

    // Executor demo (executor-impl.4)
    tests_scaling::run_executor_demo_test(Arc::clone(&dev))?;

    // Channel oneshot demo (channel-oneshot.3)
    tests_scaling::run_channel_oneshot_demo_test(Arc::clone(&dev))?;

    // Compute pipeline demo (demo-pipeline.2)
    tests_scaling::run_compute_pipeline_demo_test(Arc::clone(&dev))?;

    // Parallel file grep demo (product.8)
    tests_pipeline::run_parallel_grep_test(Arc::clone(&dev))?;

    // Warp intrinsics test (warp-future.3): syncwarp + shfl.sync.idx
    tests_warp::run_warp_intrinsics_test(Arc::clone(&dev))?;

    // WarpFuture PoC test (warp-future.4)
    tests_warp::run_warp_future_print_test(Arc::clone(&dev))?;

    // WarpFuture multi-hostcall test (warp-future.6)
    tests_warp::run_warp_future_multi_print_test(Arc::clone(&dev))?;

    // WarpFuture proc macro test (warp-future.5)
    tests_warp::run_warp_macro_print_test(Arc::clone(&dev))?;

    // WarpFuture if/else test (warp-cfg.2)
    tests_warp::run_warp_cfg_if_else_test(Arc::clone(&dev))?;

    // WarpFuture loop/break test (warp-cfg.3)
    tests_warp::run_warp_cfg_loop_test(Arc::clone(&dev))?;

    // WarpFuture match test (warp-cfg.4)
    tests_warp::run_warp_cfg_match_test(Arc::clone(&dev))?;

    // WarpFuture nested control flow test (warp-cfg.5)
    tests_warp::run_warp_cfg_nested_test(Arc::clone(&dev))?;

    // Hybrid executor test (hybrid-executor.1)
    tests_warp::run_hybrid_executor_test(Arc::clone(&dev))?;

    // Hybrid stress test (hybrid-executor.2)
    tests_warp::run_hybrid_stress_test(Arc::clone(&dev))?;

    // f32 math validation (ml-workload.1)
    tests_search::run_f32_math_test(Arc::clone(&dev))?;

    // Vector similarity search (ml-workload.2)
    tests_search::run_vector_search_test(Arc::clone(&dev))?;

    // Batch vector search (ml-workload.3)
    tests_search::run_batch_search_test(Arc::clone(&dev))?;

    // File transform pipeline (async-pipeline.1+2)
    tests_pipeline::run_file_transform_test(Arc::clone(&dev))?;

    // Branching pipeline (async-pipeline.3)
    tests_pipeline::run_branching_pipeline_test(Arc::clone(&dev))?;

    // Pipelined I/O + compute (async-pipeline.4)
    tests_pipeline::run_pipelined_compute_test(Arc::clone(&dev))?;

    // Warp-scale Embassy test (async-pipeline.5)
    tests_pipeline::run_warp_scale_async_test(Arc::clone(&dev))?;

    // Autonomous pipeline test (gpu-compute.2)
    tests_pipeline::run_autonomous_pipeline_test(Arc::clone(&dev))?;

    // Tensor Core MMA test (gpu-compute.3)
    tests_search::run_mma_test(Arc::clone(&dev))?;

    // Shared memory + bar.sync test (gpu-compute.4)
    tests_search::run_shared_memory_test(Arc::clone(&dev))?;

    // Tiled GEMM test (gpu-compute.5)
    tests_gemm::run_tiled_gemm_test(Arc::clone(&dev))?;

    // MMA fragment mapping test (gpu-pipeline.1)
    tests_search::run_mma_mapped_test(Arc::clone(&dev))?;

    // Softmax test (gpu-compute.6)
    tests_gemm::run_softmax_test(Arc::clone(&dev))?;

    // Multi-tile GEMM test (gpu-pipeline.2)
    tests_gemm::run_multi_tile_gemm_test(Arc::clone(&dev))?;

    // GEMM + softmax pipeline test (gpu-pipeline.3)
    tests_gemm::run_gemm_softmax_pipeline_test(Arc::clone(&dev))?;

    // Multi-warp GEMM test (gemm-scale.1)
    tests_gemm::run_multi_warp_gemm_test(Arc::clone(&dev))?;

    // Multi-block GEMM test (gemm-scale.2)
    tests_gemm::run_multi_block_gemm_test(Arc::clone(&dev))?;

    // Full GEMM 768x768 test (gemm-scale.3)
    tests_gemm::run_full_gemm_test(Arc::clone(&dev))?;

    // Full GEMM f32-input test (precision-fix.2)
    tests_gemm::run_full_gemm_f32in_test(Arc::clone(&dev))?;

    // LayerNorm test (transformer-layer.1)
    tests_transformer::run_layer_norm_test(Arc::clone(&dev))?;

    // GELU test (transformer-layer.2)
    tests_transformer::run_gelu_test(Arc::clone(&dev))?;

    // Attention test (transformer-layer.3)
    tests_transformer::run_attention_test(Arc::clone(&dev))?;

    // FlashAttention test (attention-scale.3)
    tests_transformer::run_flash_attention_test(Arc::clone(&dev))?;

    // FlashAttention scaling test (attention-scale.4)
    tests_transformer::run_flash_attention_scale_test(Arc::clone(&dev))?;

    // Embedding lookup test (full-inference.1)
    tests_transformer::run_embedding_test(Arc::clone(&dev))?;

    // FFN block test (transformer-layer.4)
    tests_transformer::run_ffn_test(Arc::clone(&dev))?;

    // Transformer layer test (transformer-layer.6)
    tests_transformer::run_transformer_layer_test(Arc::clone(&dev))?;

    // 12-layer GPT-2 forward pass (full-inference.2) + LM head (full-inference.3)
    tests_inference::run_full_forward_test(Arc::clone(&dev))?;

    // Greedy autoregressive generation (full-inference.4)
    tests_inference::run_generation_test(Arc::clone(&dev))?;

    // f32 GEMM forward pass (full-inference.6)
    tests_inference::run_f32_forward_test(Arc::clone(&dev))?;

    // CPU f64 reference forward pass (full-inference.8+9+10)
    tests_inference::run_cpu_f64_reference_test()?;

    // GPT-2 model weight loading test (model-loading.3)
    {
        let model_path =
            gpu_host::model_dir(Some(env!("CARGO_MANIFEST_DIR"))).join("model.safetensors");
        if model_path.exists() {
            println!("\n--- GPT-2 weight loading test (model-loading.3) ---");
            let weights = gpu_host::model::load_gpt2_weights(&model_path).map_err(|e| {
                GpuHostError::Verification {
                    test: "model_loading",
                    detail: format!("{e}"),
                }
            })?;
            println!("  Total params: {}", weights.total_params());
            println!("  Memory: {:.1} MB", weights.memory_bytes() as f64 / 1e6);
            println!("  Layers: {}", weights.layers.len());
            println!("  wte shape: [50257, 768] = {} elements", weights.wte.len());
            println!("  wpe shape: [1024, 768] = {} elements", weights.wpe.len());
            assert_eq!(weights.layers.len(), 12);
            assert_eq!(weights.wte.len(), 50257 * 768);
            assert_eq!(weights.wpe.len(), 1024 * 768);
            assert_eq!(weights.layers[0].c_attn_weight.len(), 768 * 2304);
            assert_eq!(weights.layers[0].mlp_fc_weight.len(), 768 * 3072);

            // Upload embedding table to GPU to verify transfer
            let wte_dev = dev.htod_sync_copy(&weights.wte)?;
            let wte_back: Vec<f32> = dev.dtoh_sync_copy(&wte_dev)?;
            assert_eq!(wte_back.len(), weights.wte.len());
            assert_eq!(wte_back[0], weights.wte[0]);
            assert_eq!(wte_back[1000], weights.wte[1000]);
            println!("  GPU round-trip verified for wte");
            println!("  GPT-2 weight loading — PASSED");
        } else {
            println!(
                "\n--- Skipping GPT-2 weight loading (models/model.safetensors not found) ---"
            );
        }
    }

    // GPT-2 tokenizer test (tokenizer.2)
    tests_tokenizer::run_tokenizer_test()?;

    // GPT-2 tokenizer validation (tokenizer.3)
    tests_tokenizer::run_tokenizer_validation()?;

    // GPU panic handler test (gpu-panic.2) — MUST BE LAST
    // since trap instruction calls process::exit(0)
    tests_pipeline::run_panic_test(Arc::clone(&dev))?;

    // PAL stdout routing test (std-pal.1)
    tests_std::run_std_println_test(Arc::clone(&dev))?;

    // PAL stdin routing test (std-pal.2)
    tests_std::run_std_stdin_test(Arc::clone(&dev))?;

    // std::fs File I/O test (std-fs.4)
    tests_std::run_std_file_io_test(Arc::clone(&dev))?;

    // std::fs File I/O via build-std (std-sysroot-build.4)
    tests_std::run_std_sysroot_file_test(Arc::clone(&dev))?;

    // std pipeline test (std-migration.3)
    tests_std::run_std_pipeline_test(Arc::clone(&dev))?;

    // stdin().read_line() e2e test (std-migration.4)
    tests_std::run_std_stdin_readline_test(Arc::clone(&dev))?;

    // Multi-thread malloc test (std-hardening.3)
    tests_std::run_multithread_malloc_test(Arc::clone(&dev))?;

    // Multi-warp scaling test (product.3)
    tests_scaling::run_multi_warp_test(Arc::clone(&dev))?;

    // 4-step async pipeline test (product.2)
    tests_scaling::run_pipeline_test(Arc::clone(&dev))?;

    // Showcase demo kernel (product.4)
    tests_scaling::run_showcase_test(Arc::clone(&dev))?;

    // Multi-block scaling test (multiblock.1)
    tests_scaling::run_multi_block_test(Arc::clone(&dev))?;

    // Multi-block 512-thread scaling test (multiblock.2)
    tests_scaling::run_multi_block_512_test(Arc::clone(&dev))?;

    // Slab allocator deallocation test (allocator.2)
    tests_scaling::run_slab_dealloc_test(Arc::clone(&dev))?;

    // Concurrent slab allocator test (allocator.3)
    tests_scaling::run_slab_concurrent_test(Arc::clone(&dev))?;

    tests_scaling::run_println_direct_test(Arc::clone(&dev))?;

    tests_scaling::run_error_propagation_test(Arc::clone(&dev))?;

    tests_scaling::run_multi_block_async_test(Arc::clone(&dev))?;

    // Standard impl Future on GPU (warp-future-bridge.1)
    tests_hostcall::run_std_future_print_test(Arc::clone(&dev))?;
    tests_hostcall::run_std_future_two_prints_test(Arc::clone(&dev))?;

    // Warp-cooperative Future polling (warp-future-bridge.2)
    tests_hostcall::run_warp_cooperative_future_test(Arc::clone(&dev))?;

    // Warp-cooperative two sequential futures (warp-future-bridge.3)
    tests_hostcall::run_warp_cooperative_two_futures_test(Arc::clone(&dev))?;

    // Warp-cooperative Result<T,E> broadcasting (warp-future-bridge.4)
    tests_hostcall::run_warp_result_future_test(Arc::clone(&dev))?;

    // Buffered print test (printf-batch.3)
    tests_pipeline::run_buffered_print_test(Arc::clone(&dev))?;

    // Std buffered println! test (println-buffer.2)
    tests_std::run_std_buffered_println_test(Arc::clone(&dev))?;

    // Data-dependent iteration test (data-iter.1)
    tests_pipeline::run_newton_sqrt_test(Arc::clone(&dev))?;

    println!("\nAll tests PASSED.");
    Ok(())
}
