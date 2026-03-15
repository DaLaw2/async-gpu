# demo-pipeline.1: Compute demo scenario and architecture design
**Cycle**: 331 | **Theme**: demo-pipeline | **Kind**: design | **Status**: done

## Summary
Designed a multi-stage GPU compute pipeline demo that showcases async GPU's advantage
over raw CUDA: elimination of kernel launch overhead and GPU-autonomous decision making.

## Demo Choice: Iterative Softmax + LayerNorm Pipeline

### Why This Demo?
1. **Multi-stage**: normalize → activate → reduce → check convergence (4+ stages)
2. **Uses our compute utils**: math, warp, nn modules are all exercised
3. **Shows GPU autonomy**: convergence loop runs entirely on GPU, no host roundtrip
4. **Realistic**: resembles transformer inference pipeline stages
5. **Benchmarkable**: can compare N kernel launches vs single async launch

### Architecture

```
Single Kernel Launch (32 threads = 1 warp)
├── Stage 1: Generate test data (math::sin_f32 + math::cos_f32)
├── Stage 2: Warp softmax (nn::warp_softmax_f32)
├── Stage 3: Element-wise GELU activation (nn::gelu_f32)
├── Stage 4: Warp reduction sum (warp::reduce_sum_f32)
├── Stage 5: Check convergence — if |sum - target| > epsilon, adjust and repeat 2-4
├── Stage 6: Write final results to output buffer
└── Stage 7: Write iteration count + timing to status buffer
```

### CUDA Equivalent (for comparison)
```
for each iteration:
    launch kernel softmax(data, N)         // ~5-20μs launch overhead
    cudaDeviceSynchronize()                // host-GPU sync
    launch kernel gelu(data, N)            // ~5-20μs launch overhead
    cudaDeviceSynchronize()                // host-GPU sync
    launch kernel reduce_sum(data, N, &sum) // ~5-20μs launch overhead
    cudaDeviceSynchronize()                // host-GPU sync
    cudaMemcpy(&host_sum, &sum, ...)       // D2H copy
    if |host_sum - target| < epsilon: break // host-side decision
// Total: 3 launches × K iterations × ~10μs = 30K μs overhead
```

### Async GPU Version
```rust
#[no_mangle]
pub unsafe extern "ptx-kernel" fn compute_pipeline_demo(
    data: *mut f32,       // 32 floats (one per lane)
    output: *mut f32,     // 32 result floats
    status: *mut u32,     // [iteration_count, elapsed_nanos_lo, elapsed_nanos_hi, done_flag]
) {
    let tid = index::thread_idx_x();
    let start = index::clock_nanos();

    // Stage 1: Generate test data
    let x = math::sin_f32(tid as f32 * 0.1) + math::cos_f32(tid as f32 * 0.2);

    // Iterative pipeline (GPU-autonomous)
    let mut val = x;
    let mut iterations = 0u32;
    let target_sum = 16.0f32;  // arbitrary convergence target
    let epsilon = 0.01f32;

    loop {
        // Stage 2: Softmax normalization
        val = nn::warp_softmax_f32(val);

        // Stage 3: GELU activation
        val = nn::gelu_f32(val);

        // Stage 4: Warp sum
        let sum = warp::reduce_sum_f32(val);

        iterations += 1;

        // Stage 5: Convergence check (ON GPU — no host roundtrip!)
        let diff = math::abs_f32(sum - target_sum);
        if diff < epsilon || iterations >= 100 {
            break;
        }

        // Adjust for next iteration (scale toward target)
        val = val * (target_sum / sum);
    }

    // Stage 6: Write results
    *output.add(tid as usize) = val;

    // Stage 7: Timing + stats (thread 0 only)
    if tid == 0 {
        let elapsed = index::clock_nanos() - start;
        core::ptr::write_volatile(status, iterations);
        core::ptr::write_volatile(status.add(1), elapsed as u32);
        core::ptr::write_volatile(status.add(2), (elapsed >> 32) as u32);
        core::ptr::write_volatile(status.add(3), 1); // done flag
    }
}
```

### Host Test
```rust
fn run_compute_pipeline_demo() -> Result<()> {
    let rt = GpuRuntime::new()?;
    let data = rt.alloc_mapped::<f32>(32)?;
    let output = rt.alloc_mapped::<f32>(32)?;
    let status = rt.alloc_mapped::<u32>(4)?;

    let cfg = GpuRuntime::launch_config((1,1,1), (32,1,1), 0);
    rt.launch("compute_pipeline_demo", &cfg, &[data, output, status])?;

    // Poll for done flag
    while status[3] != 1 { std::thread::sleep(Duration::from_millis(1)); }

    let iterations = status[0];
    let nanos = status[1] as u64 | ((status[2] as u64) << 32);
    println!("Converged in {} iterations, {} ns", iterations, nanos);

    // Verify: output should sum to ~target_sum
    let sum: f32 = (0..32).map(|i| output[i]).sum();
    assert!((sum - 16.0).abs() < 1.0, "sum={} not near target", sum);
}
```

### What This Proves
1. **Zero launch overhead**: One kernel launch, multiple compute stages
2. **GPU-autonomous iteration**: Convergence check on GPU, no host roundtrip
3. **Composable utils**: math/warp/nn modules compose naturally
4. **Timing instrumentation**: GPU-side profiling via clock_nanos
5. **Clean API**: Much simpler than equivalent CUDA code

### Files to Create/Modify
- `examples/compute-pipeline/kernel/src/lib.rs` — GPU kernel
- `examples/compute-pipeline/kernel/Cargo.toml` — kernel crate config
- `examples/compute-pipeline/host/src/main.rs` — host driver + benchmark
- `examples/compute-pipeline/host/Cargo.toml` — host crate config
- `crates/kernel/gpu-kernel/src/compute_demo.rs` — alt: kernel in gpu-kernel crate
- `crates/core/gpu-host/src/tests_scaling.rs` — host test

## Decision: Implementation Approach
Put the demo kernel in gpu-kernel (existing compute infrastructure) rather than a new
example crate, to avoid build system complexity. The host test goes in tests_scaling.rs.

A standalone example can be created later once the pattern is proven.

## Impact on Downstream Tasks
- **demo-pipeline.2**: Implement based on this design
- **demo-pipeline.3**: Benchmark timing data will come from status buffer
