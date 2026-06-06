// Auto-Fusion — tape-level kernel fusion detection and fused kernel execution.
//
// Demonstrates async-gpu's auto-fusion optimizer:
// 1. Build a simulated autograd tape with a chain of operations
// 2. Run the FusionOptimizer to detect fusable patterns
// 3. Use FusionCodegen to JIT-compile a fused CUDA kernel via NVRTC
// 4. Execute fused vs unfused kernels and compare correctness + performance
//
// Key APIs:
// - FusionOptimizer::analyze() — detects fusable op sequences in a tape
// - FusionPlan — describes fusion groups (start, end, fused_op)
// - FusionCodegen::get_or_compile() — JIT compiles fused elementwise kernels

use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};

use gpu_host::nn::autograd::{OpKind, OpMeta, TapeEntry, TensorId};
use gpu_host::nn::fusion::{FusionCodegen, FusionOptimizer, FusionPlan};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Create a tape entry with the given op kind and tensor IDs.
fn entry(op: OpKind, inputs: &[u32], output: u32) -> TapeEntry {
    TapeEntry {
        op,
        inputs: inputs.iter().map(|&id| TensorId(id)).collect(),
        output: TensorId(output),
        saved: vec![],
        meta: OpMeta::None,
    }
}

/// Create a matmul tape entry with proper metadata.
fn matmul_entry(a: u32, b: u32, output: u32) -> TapeEntry {
    TapeEntry {
        op: OpKind::Matmul,
        inputs: vec![TensorId(a), TensorId(b)],
        output: TensorId(output),
        saved: vec![TensorId(a), TensorId(b)],
        meta: OpMeta::Matmul {
            m: 128,
            k: 768,
            n: 3072,
        },
    }
}

/// CPU reference: GELU approximation (tanh-based).
fn cpu_gelu(x: f32) -> f32 {
    let sqrt_2_over_pi = 0.7978845608_f32;
    let coeff = 0.044715_f32;
    let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
    0.5 * x * (1.0 + inner.tanh())
}

/// Pretty-print a fusion plan.
fn print_plan(plan: &FusionPlan) {
    println!("  Fusion groups detected: {}", plan.len());
    println!("  Kernel launches saved:  {}", plan.launches_saved());
    for (i, g) in plan.groups.iter().enumerate() {
        println!(
            "    Group {}: tape[{}..{}] -> {} (inputs: {:?}, output: {:?})",
            i, g.start, g.end, g.fused_op, g.inputs, g.output
        );
    }
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Auto-Fusion Example ===\n");

    // -----------------------------------------------------------------------
    // Demo 1: Detect fusion patterns in a GPT-2 transformer block
    // -----------------------------------------------------------------------
    println!("--- Demo 1: Fusion Detection (GPT-2 Block) ---\n");
    println!("Simulating a GPT-2 transformer block tape:");
    println!("  LayerNorm -> Matmul -> BiasAdd -> Attention -> Matmul -> BiasAdd");
    println!("  -> ElemAdd -> LayerNorm -> Matmul -> BiasAdd -> Gelu");
    println!("  -> Matmul -> BiasAdd -> ElemAdd\n");

    let tape = vec![
        // 0: LayerNorm (standalone)
        entry(OpKind::LayerNorm, &[100], 101),
        // 1-2: Matmul -> BiasAdd (QKV projection)
        matmul_entry(101, 102, 103),
        entry(OpKind::BiasAdd, &[103], 104),
        // 3: Attention (fusion barrier)
        entry(OpKind::Attention, &[104], 105),
        // 4-5: Matmul -> BiasAdd (output projection)
        matmul_entry(105, 106, 107),
        entry(OpKind::BiasAdd, &[107], 108),
        // 6-7: ElemAdd -> LayerNorm (residual + LN2)
        entry(OpKind::ElemAdd, &[108, 100], 109),
        entry(OpKind::LayerNorm, &[109], 110),
        // 8-10: Matmul -> BiasAdd -> Gelu (FFN up)
        matmul_entry(110, 111, 112),
        entry(OpKind::BiasAdd, &[112], 113),
        entry(OpKind::Gelu, &[113], 114),
        // 11-12: Matmul -> BiasAdd (FFN down)
        matmul_entry(114, 115, 116),
        entry(OpKind::BiasAdd, &[116], 117),
        // 13: ElemAdd (residual, standalone)
        entry(OpKind::ElemAdd, &[117, 101], 118),
    ];

    let optimizer = FusionOptimizer::new();
    let plan = optimizer.analyze(&tape);

    print_plan(&plan);

    // Verify expected fusion groups
    assert_eq!(plan.len(), 5, "Expected 5 fusion groups");
    assert_eq!(plan.launches_saved(), 6, "Expected 6 launches saved");
    println!("\n  14 ops -> {} fused ops (saved {} kernel launches)\n", 14 - plan.launches_saved(), plan.launches_saved());

    // -----------------------------------------------------------------------
    // Demo 2: Elementwise chain detection (ElemAdd -> Relu)
    // -----------------------------------------------------------------------
    println!("--- Demo 2: Elementwise Chain Detection ---\n");
    println!("Tape: ElemAdd -> Relu -> Sigmoid (3 elementwise ops chained)\n");

    let elem_tape = vec![
        entry(OpKind::ElemAdd, &[0, 1], 2),
        entry(OpKind::Relu, &[2], 3),
        entry(OpKind::Sigmoid, &[3], 4),
    ];

    let plan2 = optimizer.analyze(&elem_tape);
    print_plan(&plan2);

    assert_eq!(plan2.len(), 1, "Expected 1 elementwise chain");
    assert_eq!(
        plan2.groups[0].end - plan2.groups[0].start,
        3,
        "Chain should cover all 3 ops"
    );
    println!("  3 ops fused into 1 kernel launch\n");

    // -----------------------------------------------------------------------
    // Demo 3: JIT compile and execute a fused kernel on GPU
    // -----------------------------------------------------------------------
    println!("--- Demo 3: Fused Kernel Execution (GPU) ---\n");

    let dev = Arc::new(CudaDevice::new(0).expect("CUDA device initialization failed"));
    println!("CUDA device initialized");

    let codegen = FusionCodegen::new();

    // Fuse ElemAdd + Gelu into a single kernel
    let ops = vec![OpKind::ElemAdd, OpKind::Gelu];
    println!("Compiling fused kernel: ElemAdd -> Gelu ...");

    let (module_name, func_name) = codegen
        .get_or_compile(&ops, &[], &dev)
        .expect("Failed to compile fused kernel");
    println!("  Module: {module_name}");
    println!("  Function: {func_name}");

    // Prepare test data
    let n = 4096usize;
    let input: Vec<f32> = (0..n).map(|i| (i as f32 * 0.002) - 4.0).collect();
    let addend: Vec<f32> = (0..n).map(|i| (i as f32 * 0.001) - 2.0).collect();

    let d_input = dev.htod_sync_copy(&input).expect("upload input");
    let d_addend = dev.htod_sync_copy(&addend).expect("upload addend");
    let mut d_output = dev.alloc_zeros::<f32>(n).expect("alloc output");

    let n_u32 = n as u32;
    let threads = 256u32;
    let total_threads = (n_u32 + 3) / 4;
    let grid = ((total_threads + threads - 1) / threads, 1, 1);
    let config = LaunchConfig {
        grid_dim: grid,
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // Launch fused kernel
    let func = dev
        .get_func(&module_name, &func_name)
        .expect("get fused function");
    unsafe {
        func.launch(config, (&d_input, &mut d_output, &d_addend, n_u32))
            .expect("launch fused kernel");
    }
    dev.synchronize().expect("synchronize");

    let result = dev.dtoh_sync_copy(&d_output).expect("download result");

    // CPU reference
    let expected: Vec<f32> = input
        .iter()
        .enumerate()
        .map(|(i, &x)| cpu_gelu(x + addend[i]))
        .collect();

    let max_err = result
        .iter()
        .zip(&expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, f32::max);

    println!("\n  Elements:  {n}");
    println!("  Max error: {max_err:.6e} (tolerance: 1e-4)");
    assert!(max_err < 1e-4, "Fused kernel error exceeds tolerance");
    println!("  PASSED: Fused ElemAdd+Gelu matches CPU reference\n");

    // -----------------------------------------------------------------------
    // Demo 4: Performance — fused vs unfused comparison
    // -----------------------------------------------------------------------
    println!("--- Demo 4: Fused vs Unfused Performance ---\n");

    let perf_n = 1024 * 1024; // 1M elements
    let perf_input: Vec<f32> = (0..perf_n).map(|i| (i as f32 * 0.001) - 0.5).collect();
    let perf_addend: Vec<f32> = (0..perf_n).map(|i| (i as f32 * 0.0005) + 0.1).collect();

    let d_perf_input = dev.htod_sync_copy(&perf_input).expect("upload");
    let d_perf_addend = dev.htod_sync_copy(&perf_addend).expect("upload");
    let mut d_perf_output = dev.alloc_zeros::<f32>(perf_n).expect("alloc");

    let perf_n_u32 = perf_n as u32;
    let perf_total_threads = (perf_n_u32 + 3) / 4;
    let perf_grid = ((perf_total_threads + threads - 1) / threads, 1, 1);
    let perf_config = LaunchConfig {
        grid_dim: perf_grid,
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    };

    // Compile unfused kernels
    let add_src = r#"
extern "C" __global__ void elem_add(
    const float* __restrict__ input,
    const float* __restrict__ addend,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
        float4 a = *reinterpret_cast<const float4*>(&addend[idx]);
        v.x += a.x; v.y += a.y; v.z += a.z; v.w += a.w;
        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            output[i] = input[i] + addend[i];
        }
    }
}
"#;
    let gelu_src = r#"
extern "C" __global__ void gelu_act(
    const float* __restrict__ input,
    float* __restrict__ output,
    unsigned int n
) {
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int idx = tid * 4;
    if (idx + 3 < n) {
        float4 v = *reinterpret_cast<const float4*>(&input[idx]);
        const float S = 0.7978845608f;
        const float C = 0.044715f;
        float4 t;
        t.x = S * (v.x + C * v.x * v.x * v.x);
        t.y = S * (v.y + C * v.y * v.y * v.y);
        t.z = S * (v.z + C * v.z * v.z * v.z);
        t.w = S * (v.w + C * v.w * v.w * v.w);
        v.x = 0.5f * v.x * (1.0f + tanhf(t.x));
        v.y = 0.5f * v.y * (1.0f + tanhf(t.y));
        v.z = 0.5f * v.z * (1.0f + tanhf(t.z));
        v.w = 0.5f * v.w * (1.0f + tanhf(t.w));
        *reinterpret_cast<float4*>(&output[idx]) = v;
    } else {
        for (unsigned int i = idx; i < n && i < idx + 4; i++) {
            float x = input[i];
            const float S = 0.7978845608f;
            const float C = 0.044715f;
            float t = S * (x + C * x * x * x);
            output[i] = 0.5f * x * (1.0f + tanhf(t));
        }
    }
}
"#;

    let ptx_add = cudarc::nvrtc::compile_ptx(add_src).expect("compile elem_add");
    let ptx_gelu = cudarc::nvrtc::compile_ptx(gelu_src).expect("compile gelu_act");
    dev.load_ptx(ptx_add, "unfused_add", &["elem_add"])
        .expect("load add");
    dev.load_ptx(ptx_gelu, "unfused_gelu", &["gelu_act"])
        .expect("load gelu");

    let mut d_tmp = dev.alloc_zeros::<f32>(perf_n).expect("alloc tmp");

    // Warm up
    for _ in 0..5 {
        let f = dev.get_func(&module_name, &func_name).unwrap();
        unsafe {
            f.launch(
                perf_config,
                (&d_perf_input, &mut d_perf_output, &d_perf_addend, perf_n_u32),
            )
            .unwrap();
        }
        dev.synchronize().unwrap();
    }
    for _ in 0..5 {
        let fa = dev.get_func("unfused_add", "elem_add").unwrap();
        let fg = dev.get_func("unfused_gelu", "gelu_act").unwrap();
        unsafe {
            fa.launch(
                perf_config,
                (&d_perf_input, &d_perf_addend, &mut d_tmp, perf_n_u32),
            )
            .unwrap();
            fg.launch(perf_config, (&d_tmp, &mut d_perf_output, perf_n_u32))
                .unwrap();
        }
        dev.synchronize().unwrap();
    }

    // Benchmark fused
    let iters = 100;
    let start = Instant::now();
    for _ in 0..iters {
        let f = dev.get_func(&module_name, &func_name).unwrap();
        unsafe {
            f.launch(
                perf_config,
                (&d_perf_input, &mut d_perf_output, &d_perf_addend, perf_n_u32),
            )
            .unwrap();
        }
    }
    dev.synchronize().unwrap();
    let fused_us = start.elapsed().as_micros() as f64 / iters as f64;

    // Benchmark unfused
    let start = Instant::now();
    for _ in 0..iters {
        let fa = dev.get_func("unfused_add", "elem_add").unwrap();
        let fg = dev.get_func("unfused_gelu", "gelu_act").unwrap();
        unsafe {
            fa.launch(
                perf_config,
                (&d_perf_input, &d_perf_addend, &mut d_tmp, perf_n_u32),
            )
            .unwrap();
            fg.launch(perf_config, (&d_tmp, &mut d_perf_output, perf_n_u32))
                .unwrap();
        }
    }
    dev.synchronize().unwrap();
    let unfused_us = start.elapsed().as_micros() as f64 / iters as f64;

    let speedup = unfused_us / fused_us;
    println!("  Elements:    {perf_n}");
    println!("  Iterations:  {iters}");
    println!("  Fused:       {fused_us:.1} us/iter");
    println!("  Unfused:     {unfused_us:.1} us/iter (2 separate kernels)");
    println!("  Speedup:     {speedup:.2}x");

    assert!(
        speedup >= 1.0,
        "Fused kernel should not be slower than unfused"
    );
    println!("  PASSED: Fused kernel is faster than unfused\n");

    // -----------------------------------------------------------------------
    // Demo 5: Fan-out detection (fusion blocker)
    // -----------------------------------------------------------------------
    println!("--- Demo 5: Fan-Out Blocks Fusion ---\n");
    println!("Tape: Matmul -> BiasAdd + Relu (output consumed twice)\n");

    let fanout_tape = vec![
        matmul_entry(0, 1, 2),
        entry(OpKind::BiasAdd, &[2], 3),
        entry(OpKind::Relu, &[2], 4), // also consumes tensor 2 (fan-out)
    ];

    let plan_fanout = optimizer.analyze(&fanout_tape);
    print_plan(&plan_fanout);

    assert!(
        plan_fanout.is_empty(),
        "Fan-out should block fusion"
    );
    println!("  PASSED: Fan-out correctly detected, no fusion applied\n");

    println!("=== All demos complete! ===");
}
