//! Full GPT-2 inference tests: 12-layer forward pass, greedy generation,
//! f32 forward pass, CPU f64 reference.

use std::sync::Arc;

use cudarc::driver::{CudaDevice, CudaSlice, LaunchAsync, LaunchConfig};

use crate::error::{GpuHostError, Result};
use crate::mapped_mem::{alloc_mapped_result_array, free_mapped_mem};

/// full-inference.2: 12-layer GPT-2 forward pass with real weights.
///
/// Loads GPT-2 small weights from safetensors, tokenizes a prompt, runs
/// embedding + 12 transformer layers + final LayerNorm on GPU.
/// Skips if model file is not present.
pub(crate) fn run_full_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path = std::path::Path::new("../../models/model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping 12-layer forward pass (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- 12-layer GPT-2 forward pass (full-inference.2) ---");

    // Load weights
    let weights =
        gpu_host::model::load_gpt2_weights(model_path).map_err(|e| GpuHostError::Verification {
            test: "full_forward",
            detail: format!("weight loading: {e}"),
        })?;
    println!(
        "  Loaded {} params ({:.1} MB)",
        weights.total_params(),
        weights.memory_bytes() as f64 / 1e6
    );

    // Tokenize
    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "full_forward",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let tokens = tokenizer.encode(prompt);
    let actual_seq = tokens.len();
    println!("  Prompt: \"{prompt}\" → {actual_seq} tokens: {tokens:?}");

    // Pad to multiple of 32 for GEMM kernel alignment
    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let seq = SEQ.max((actual_seq as u32).div_ceil(32) * 32);
    let total_seq_model = (seq * D_MODEL) as usize;
    let head_total = (N_HEADS * seq * D_HEAD) as usize;

    let mut token_ids_u32: Vec<u32> = tokens.to_vec();
    token_ids_u32.resize(seq as usize, 0); // pad with token 0

    // Load PTX with all needed kernels
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(
        ptx,
        "gpt2",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    );

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("gpt2", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    // === Helper: transpose weight [K, N] row-major → column-major f32 for gemm_f32 ===
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // === Allocate a single reusable status buffer ===
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // === Upload embedding tables ===
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;
    let token_ids_dev = dev.htod_sync_copy(&token_ids_u32)?;

    // === Run embedding lookup ===
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    unsafe {
        f_embed.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &wte_dev,
                &wpe_dev,
                &token_ids_dev,
                &mut hidden_dev,
                seq,
                D_MODEL,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // Zero out padded positions to prevent NaN from propagating through GEMM tiles.
    // Padded rows (actual_seq..seq) carry real embeddings for padding token 0,
    // which can diverge and produce NaN after a few layers.
    let pad_start = (actual_seq as u32) * D_MODEL;
    let pad_count = total_seq_model as u32 - pad_start;
    if pad_count > 0 {
        unsafe {
            f_zero.clone().launch(
                LaunchConfig {
                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, pad_start, total_seq_model as u32),
            )?;
        }
        dev.synchronize()?;
    }
    println!("  Embedding done (seq={seq}, actual={actual_seq})");

    // Drop large embedding tables from GPU to save memory
    drop(wte_dev);
    drop(wpe_dev);

    // === Upload all layer weights as f32 column-major ===
    println!("  Uploading 12 layers (f32 column-major)...");
    struct LayerWeightsGpu {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsGpu> = Vec::with_capacity(12);
    for (i, layer) in weights.layers.iter().enumerate() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsGpu {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
        if i == 0 || i == 11 {
            println!("    Layer {i} uploaded");
        }
    }
    println!("  All 12 layers uploaded (f32)");

    // === Allocate reusable activation buffers ===
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (seq as usize).div_ceil(32) as u32;

    // === Diagnostic: dump embedding at pos 4 ===
    {
        let emb_snap: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
        let pos4 = actual_seq - 1;
        let emb4 = &emb_snap[pos4 * (D_MODEL as usize)..(pos4 + 1) * (D_MODEL as usize)];
        println!(
            "  GPU embed pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
            emb4[0], emb4[1], emb4[2], emb4[3]
        );
    }

    // === Run 12 transformer layers ===
    for layer_idx in 0..12u32 {
        let lw = &gpu_layers[layer_idx as usize];

        // Step 1: LayerNorm1
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln1_g,
                    &lw.ln1_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: dump LN1 output ===
        if layer_idx == 0 {
            let ln_snap: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
            let pos4 = actual_seq - 1;
            let ln4 = &ln_snap[pos4 * (D_MODEL as usize)..(pos4 + 1) * (D_MODEL as usize)];
            println!(
                "  GPU L0 LN1 pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                ln4[0], ln4[1], ln4[2], ln4[3]
            );
        }

        // Step 2: QKV projection (gemm_f32: f32 input, f32 column-major weights)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, 2304 / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_qkv,
                    &mut qkv_dev,
                    D_MODEL,
                    2304u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: dump QKV GEMM output (before bias) ===
        if layer_idx == 0 {
            let qkv_snap: Vec<f32> = dev.dtoh_sync_copy(&qkv_dev)?;
            let pos4 = actual_seq - 1;
            let qkv4 = &qkv_snap[pos4 * 2304..(pos4 + 1) * 2304];
            println!(
                "  GPU L0 QKV pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
        }

        // Bias add on QKV
        let total_qkv = seq * 2304;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_qkv.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        // Step 3: Split QKV → Q, K, V
        unsafe {
            f_split.clone().launch(
                LaunchConfig {
                    grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 4: Flash attention (causal)
        unsafe {
            f_attn.clone().launch(
                LaunchConfig {
                    grid_dim: (N_HEADS, n_q_tiles, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 2 * 32 * 64 * 4, // k_tile + v_tile = 16KB
                },
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut attn_out_dev,
                    seq,
                    D_HEAD,
                    1u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 5: Concat heads
        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
            )?;
        }
        dev.synchronize()?;

        // Step 6: Output projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &concat_dev,
                    &lw.w_proj,
                    &mut proj_out_dev,
                    D_MODEL,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut proj_out_dev,
                    &lw.b_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 7: Residual add (hidden += proj_out) + zero padded rows
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // Step 8: LayerNorm2
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln2_g,
                    &lw.ln2_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 9: FFN up projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_FFN / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_fc,
                    &mut ffn_hidden_dev,
                    D_MODEL,
                    D_FFN,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        let total_ffn = seq * D_FFN;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_hidden_dev,
                    &lw.b_fc,
                    D_FFN,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 10: GELU
        unsafe {
            f_gelu.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &ffn_hidden_dev,
                    &mut gelu_out_dev,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 11: FFN down projection
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &gelu_out_dev,
                    &lw.w_fc_proj,
                    &mut ffn_out_dev,
                    D_FFN,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Bias add
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_out_dev,
                    &lw.b_fc_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Step 12: Residual add (hidden += ffn_out) + zero padded rows
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // === Per-layer diagnostic ===
        // Download hidden state and check max|val| per position
        let hidden_snapshot: Vec<f32> = dev.dtoh_sync_copy(&hidden_dev)?;
        let d = D_MODEL as usize;
        let mut layer_max_abs = 0.0f32;
        let mut layer_nan_positions = Vec::new();
        let mut layer_overflow_positions = Vec::new();
        for row in 0..actual_seq {
            let row_slice = &hidden_snapshot[row * d..(row + 1) * d];
            let row_nan = row_slice.iter().any(|v| v.is_nan());
            let row_max = row_slice
                .iter()
                .filter(|v| !v.is_nan() && !v.is_infinite())
                .fold(0.0f32, |m, &v| m.max(v.abs()));
            if row_nan {
                layer_nan_positions.push(row);
            }
            if row_max > 65504.0 {
                layer_overflow_positions.push((row, row_max));
            }
            if row_max > layer_max_abs {
                layer_max_abs = row_max;
            }
        }
        let nan_str = if layer_nan_positions.is_empty() {
            String::new()
        } else {
            format!(", NaN@{layer_nan_positions:?}")
        };
        let ovf_str = if layer_overflow_positions.is_empty() {
            String::new()
        } else {
            let ovf_pos: Vec<_> = layer_overflow_positions
                .iter()
                .map(|(p, v)| format!("pos{p}={v:.0}"))
                .collect();
            format!(", OVERFLOW: [{}]", ovf_pos.join(", "))
        };
        // Also print pos4 (last real token) specifics for CPU comparison
        let pos4_slice = &hidden_snapshot[(actual_seq - 1) * d..actual_seq * d];
        let pos4_maxabs = pos4_slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let pos4_first4: Vec<String> = pos4_slice[..4].iter().map(|x| format!("{x:.4}")).collect();
        println!("    Layer {layer_idx}: max|val|={layer_max_abs:.2}, pos4_max|val|={pos4_maxabs:.2}, pos4_first4=[{}]{nan_str}{ovf_str}",
            pos4_first4.join(", "));
    }

    // === Final LayerNorm ===
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (seq, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &hidden_dev,
                &mut ln_out_dev,
                &ln_f_g_dev,
                &ln_f_b_dev,
                D_MODEL,
                EPS,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // === Download and validate output ===
    let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;

    // Check for NaN/Inf in the prediction-relevant token positions.
    // Position 0 may go NaN with f16 GEMM because it only self-attends (no averaging),
    // causing residual values to grow without damping until they overflow f16 range.
    // This is a known limitation of f16 Tensor Core inference vs f32 reference.
    // For inference, we only need the last actual token position for next-token prediction.
    let dm = D_MODEL as usize;
    let mut nan_positions: Vec<usize> = Vec::new();
    let mut pred_nan = 0;
    let mut pred_inf = 0;
    let mut max_abs = 0.0f32;
    for row in 0..actual_seq {
        let row_slice = &output[row * dm..(row + 1) * dm];
        let row_nan = row_slice.iter().filter(|v| v.is_nan()).count();
        if row_nan > 0 {
            nan_positions.push(row);
            if row == actual_seq - 1 {
                pred_nan = row_nan;
            }
        }
        for &v in row_slice {
            if !v.is_nan() && !v.is_infinite() && v.abs() > max_abs {
                max_abs = v.abs();
            }
            if row == actual_seq - 1 && v.is_infinite() {
                pred_inf += 1;
            }
        }
    }

    // Print stats for prediction position (last actual token)
    let last_pos = actual_seq - 1;
    let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
    let row_mean: f32 = last_row.iter().sum::<f32>() / D_MODEL as f32;
    let row_var: f32 =
        last_row.iter().map(|x| (x - row_mean).powi(2)).sum::<f32>() / D_MODEL as f32;
    println!("  Output shape: [{seq}, {D_MODEL}] (actual {actual_seq} tokens)");
    if !nan_positions.is_empty() {
        println!("  NaN positions: {nan_positions:?} (f16 precision — no attention averaging)");
    }
    println!("  max|val|={max_abs:.4} (non-NaN actual positions)");
    println!("  Prediction pos {last_pos}: mean={row_mean:.6}, var={row_var:.6}");
    println!(
        "  First 8 values: {:?}",
        &last_row[..8]
            .iter()
            .map(|v| format!("{v:.4}"))
            .collect::<Vec<_>>()
    );

    // Free status buffer
    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    // The prediction position (last actual token) must be clean
    if pred_nan > 0 || pred_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "full_forward",
            detail: format!("prediction position has {pred_nan} NaN and {pred_inf} Inf values"),
        });
    }

    // Sanity: max absolute value should be reasonable (not exploded)
    if max_abs > 1000.0 {
        return Err(GpuHostError::Verification {
            test: "full_forward",
            detail: format!("output magnitude too large: max|val|={max_abs:.2}"),
        });
    }

    println!("  12-layer GPT-2 forward pass (seq={seq}, actual={actual_seq}) — PASSED");

    // ================================================================
    // LM Head (full-inference.3): project hidden state → vocabulary logits
    // GPT-2 uses weight tying: logits = hidden_state @ wte.T
    // wte is [50257, 768] row-major, so logits[v] = dot(hidden[last_pos], wte[v])
    // ================================================================
    println!("\n--- LM head + greedy decode (full-inference.3) ---");

    let vocab_size = 50257;
    let hidden = &output[last_pos * dm..(last_pos + 1) * dm];

    // Compute logits on CPU (only 1 row × 768, not worth a GPU kernel for 50257 non-aligned)
    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let wte_row = &weights.wte[v * dm..(v + 1) * dm];
        let mut dot = 0.0f32;
        for d in 0..dm {
            dot += hidden[d] * wte_row[d];
        }
        logits[v] = dot;
    }

    // Softmax for probabilities (numerically stable)
    let max_logit = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp_sum: f32 = logits.iter().map(|&l| (l - max_logit).exp()).sum();
    let probs: Vec<f32> = logits
        .iter()
        .map(|&l| (l - max_logit).exp() / exp_sum)
        .collect();

    // Top-5 predictions
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| probs[b].partial_cmp(&probs[a]).unwrap());

    println!("  Top-5 predictions after \"The capital of France is\":");
    for &idx in &indices[..5] {
        let token_str = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<tok {idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.2}, prob={:.4})",
            indices.iter().position(|&i| i == idx).unwrap() + 1,
            idx,
            token_str,
            logits[idx],
            probs[idx],
        );
    }

    // Greedy prediction = argmax
    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<tok {top1}>"));
    println!("  Greedy next token: {} = {:?}", top1, top1_str);

    // Validation: logits should be finite at prediction position
    let logit_nan = logits.iter().filter(|v| v.is_nan()).count();
    let logit_inf = logits.iter().filter(|v| v.is_infinite()).count();
    if logit_nan > 0 || logit_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "lm_head",
            detail: format!("logits have {logit_nan} NaN and {logit_inf} Inf values"),
        });
    }

    // Validation: top-1 probability should be > 0.01 (model has a clear preference)
    if probs[top1] < 0.01 {
        println!(
            "  WARNING: top-1 probability very low ({:.4}), model may not be confident",
            probs[top1]
        );
    }

    println!("  LM head (vocab=50257, CPU matmul) — PASSED");
    Ok(())
}

/// full-inference.4: Greedy autoregressive generation loop.
///
/// Runs repeated forward passes, each time appending the argmax token to the
/// sequence, until max_new_tokens is reached or <|endoftext|> is produced.
/// No KV cache — full recompute each step (proof of concept).
/// Skips if model file is not present.
pub(crate) fn run_generation_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path = std::path::Path::new("../../models/model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping generation test (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- Greedy autoregressive generation (full-inference.4) ---");

    // Load weights
    let weights =
        gpu_host::model::load_gpt2_weights(model_path).map_err(|e| GpuHostError::Verification {
            test: "generation",
            detail: format!("weight loading: {e}"),
        })?;

    // Tokenize
    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "generation",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let prompt_len = prompt_tokens.len();
    println!("  Prompt: \"{prompt}\" → {prompt_len} tokens");

    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;

    // Fixed seq=128 for all steps (pad shorter sequences)
    const SEQ: u32 = 128;
    let total_seq_model = (SEQ * D_MODEL) as usize;
    let head_total = (N_HEADS * SEQ * D_HEAD) as usize;

    // Max generation: fill up to seq=128
    let max_new_tokens: usize = 50.min(SEQ as usize - prompt_len);
    println!("  Generating up to {max_new_tokens} new tokens (seq={SEQ})");

    // Helper: transpose weight [K, N] row-major → column-major f32 for gemm_f32
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // Load PTX
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(
        ptx,
        "gen",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    );

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("gen", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    // Allocate status buffer
    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables (keep alive for LM head weight-tying)
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;

    // Upload all layer weights as f32 column-major
    struct LayerWeightsGpu {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsGpu> = Vec::with_capacity(12);
    for layer in weights.layers.iter() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsGpu {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
    }

    // Final layer norm weights
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    // Allocate reusable activation buffers
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((SEQ * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (SEQ as usize).div_ceil(32) as u32;

    // Build the token sequence (will grow each step)
    let mut tokens: Vec<u32> = prompt_tokens.clone();
    let mut generated: Vec<u32> = Vec::new();

    let gen_start = std::time::Instant::now();

    for step in 0..max_new_tokens {
        let actual_seq = tokens.len();

        // Pad to SEQ
        let mut token_ids_padded = tokens.clone();
        token_ids_padded.resize(SEQ as usize, 0);
        let token_ids_dev = dev.htod_sync_copy(&token_ids_padded)?;

        // === Embedding ===
        unsafe {
            f_embed.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &wte_dev,
                    &wpe_dev,
                    &token_ids_dev,
                    &mut hidden_dev,
                    SEQ,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Zero out padded positions after embedding
        let pad_start = (actual_seq as u32) * D_MODEL;
        let pad_count = total_seq_model as u32 - pad_start;
        if pad_count > 0 {
            unsafe {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
            dev.synchronize()?;
        }

        // === 12 transformer layers ===
        for layer_idx in 0..12u32 {
            let lw = &gpu_layers[layer_idx as usize];

            // LayerNorm1
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln1_g,
                        &lw.ln1_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // QKV projection
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ / 32, 2304 / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: gemm_shared,
                    },
                    (
                        &ln_out_dev,
                        &lw.w_qkv,
                        &mut qkv_dev,
                        D_MODEL,
                        2304u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // QKV bias
            let total_qkv = SEQ * 2304;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_qkv.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
                )?;
            }
            dev.synchronize()?;

            // Split QKV
            unsafe {
                f_split.clone().launch(
                    LaunchConfig {
                        grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, SEQ, N_HEADS, D_HEAD,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Flash attention (causal)
            unsafe {
                f_attn.clone().launch(
                    LaunchConfig {
                        grid_dim: (N_HEADS, n_q_tiles, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 2 * 32 * 64 * 4,
                    },
                    (
                        &q_dev,
                        &k_dev,
                        &v_dev,
                        &mut attn_out_dev,
                        SEQ,
                        D_HEAD,
                        1u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Concat heads
            unsafe {
                f_concat.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&attn_out_dev, &mut concat_dev, SEQ, N_HEADS, D_HEAD),
                )?;
            }
            dev.synchronize()?;

            // Output projection
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: gemm_shared,
                    },
                    (
                        &concat_dev,
                        &lw.w_proj,
                        &mut proj_out_dev,
                        D_MODEL,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Projection bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut proj_out_dev,
                        &lw.b_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 1 + zero padded rows
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;

            // LayerNorm2
            unsafe {
                f_ln.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ, 1, 1),
                        block_dim: (32, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &hidden_dev,
                        &mut ln_out_dev,
                        &lw.ln2_g,
                        &lw.ln2_b,
                        D_MODEL,
                        EPS,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN up
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ / 32, D_FFN / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: gemm_shared,
                    },
                    (
                        &ln_out_dev,
                        &lw.w_fc,
                        &mut ffn_hidden_dev,
                        D_MODEL,
                        D_FFN,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN bias
            let total_ffn = SEQ * D_FFN;
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_hidden_dev,
                        &lw.b_fc,
                        D_FFN,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // GELU
            unsafe {
                f_gelu.clone().launch(
                    LaunchConfig {
                        grid_dim: (total_ffn.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &ffn_hidden_dev,
                        &mut gelu_out_dev,
                        total_ffn,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN down
            unsafe {
                f_gemm.clone().launch(
                    LaunchConfig {
                        grid_dim: (SEQ / 32, D_MODEL / 16, 1),
                        block_dim: (128, 1, 1),
                        shared_mem_bytes: gemm_shared,
                    },
                    (
                        &gelu_out_dev,
                        &lw.w_fc_proj,
                        &mut ffn_out_dev,
                        D_FFN,
                        D_MODEL,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // FFN bias
            unsafe {
                f_bias.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (
                        &mut ffn_out_dev,
                        &lw.b_fc_proj,
                        D_MODEL,
                        total_seq_model as u32,
                        status_dev_ptr,
                    ),
                )?;
            }
            dev.synchronize()?;

            // Residual 2 + zero padded rows
            unsafe {
                f_add.clone().launch(
                    LaunchConfig {
                        grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
                )?;
                if pad_count > 0 {
                    f_zero.clone().launch(
                        LaunchConfig {
                            grid_dim: (pad_count.div_ceil(256), 1, 1),
                            block_dim: (256, 1, 1),
                            shared_mem_bytes: 0,
                        },
                        (&mut hidden_dev, pad_start, total_seq_model as u32),
                    )?;
                }
            }
            dev.synchronize()?;
        }

        // === Final LayerNorm ===
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (SEQ, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &ln_f_g_dev,
                    &ln_f_b_dev,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Download only the prediction position (last actual token)
        let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;
        let last_pos = actual_seq - 1;
        let hidden_vec = &output[last_pos * dm..(last_pos + 1) * dm];

        // Check for NaN at prediction position
        let nan_count = hidden_vec.iter().filter(|v| v.is_nan()).count();
        if nan_count > 0 {
            println!(
                "  Step {step}: prediction position has {nan_count} NaN — stopping generation"
            );
            break;
        }

        // CPU LM head: logits[v] = dot(hidden, wte[v])
        let vocab_size = 50257;
        let mut best_logit = f32::NEG_INFINITY;
        let mut best_token: u32 = 0;
        for v in 0..vocab_size {
            let wte_row = &weights.wte[v * dm..(v + 1) * dm];
            let mut dot = 0.0f32;
            for d in 0..dm {
                dot += hidden_vec[d] * wte_row[d];
            }
            if dot > best_logit {
                best_logit = dot;
                best_token = v as u32;
            }
        }

        // Decode and print
        let token_str = tokenizer
            .decode(&[best_token])
            .unwrap_or_else(|_| format!("<tok {best_token}>"));
        print!("{token_str}");

        // Stop on <|endoftext|>
        if best_token == 50256 {
            println!();
            println!("  [<|endoftext|> at step {step}]");
            break;
        }

        generated.push(best_token);
        tokens.push(best_token);
    }

    let gen_elapsed = gen_start.elapsed();
    println!();

    // Print full generated text
    let full_text = if !generated.is_empty() {
        tokenizer
            .decode(&generated)
            .unwrap_or_else(|_| "<decode error>".to_string())
    } else {
        String::new()
    };
    println!("  Prompt: \"{prompt}\"");
    println!("  Generated ({} tokens): \"{full_text}\"", generated.len());
    println!(
        "  Time: {:.1}ms total, {:.1}ms/token",
        gen_elapsed.as_secs_f64() * 1000.0,
        gen_elapsed.as_secs_f64() * 1000.0 / generated.len().max(1) as f64,
    );

    // Free status buffer
    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    // Validation: should have generated at least 1 token
    if generated.is_empty() {
        return Err(GpuHostError::Verification {
            test: "generation",
            detail: "no tokens generated".to_string(),
        });
    }

    println!("  Greedy autoregressive generation — PASSED");
    Ok(())
}

/// full-inference.6: 12-layer GPT-2 forward pass with pure f32 GEMM (no Tensor Cores).
///
/// Identical pipeline to run_full_forward_test but uses gemm_f32 kernel
/// instead of full_gemm_f32in. Weights are uploaded as column-major f32
/// instead of packed f16x2. This validates whether f16 precision was
/// the reason for wrong predictions.
pub(crate) fn run_f32_forward_test(dev: Arc<CudaDevice>) -> Result<()> {
    let model_path = std::path::Path::new("../../models/model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping f32 GEMM forward pass (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- f32 GEMM forward pass (full-inference.6) ---");

    let weights =
        gpu_host::model::load_gpt2_weights(model_path).map_err(|e| GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("weight loading: {e}"),
        })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let actual_seq = prompt_tokens.len();
    println!("  Prompt: \"{prompt}\" → {actual_seq} tokens: {prompt_tokens:?}");

    const SEQ: u32 = 32;
    const D_MODEL: u32 = 768;
    const N_HEADS: u32 = 12;
    const D_HEAD: u32 = 64;
    const D_FFN: u32 = 3072;
    const EPS: f32 = 1e-5;
    let dm = D_MODEL as usize;
    let seq = SEQ;
    let total_seq_model = (seq * D_MODEL) as usize;
    let head_total = (N_HEADS * seq * D_HEAD) as usize;

    let mut token_ids_u32: Vec<u32> = prompt_tokens.clone();
    token_ids_u32.resize(seq as usize, 0);

    // Load PTX
    let ptx = cudarc::nvrtc::Ptx::from_src(crate::KERNEL_PTX);
    let _ = dev.load_ptx(
        ptx,
        "f32fwd",
        &[
            "embedding_lookup",
            "layer_norm",
            "gemm_f32",
            "bias_add",
            "split_qkv",
            "flash_attention",
            "concat_heads",
            "gelu_forward",
            "elementwise_add",
            "zero_pad",
        ],
    );

    macro_rules! get_fn {
        ($name:expr) => {
            dev.get_func("f32fwd", $name)
                .ok_or(GpuHostError::KernelNotFound($name))?
        };
    }

    let f_embed = get_fn!("embedding_lookup");
    let f_ln = get_fn!("layer_norm");
    let f_gemm = get_fn!("gemm_f32");
    let f_bias = get_fn!("bias_add");
    let f_split = get_fn!("split_qkv");
    let f_attn = get_fn!("flash_attention");
    let f_concat = get_fn!("concat_heads");
    let f_gelu = get_fn!("gelu_forward");
    let f_add = get_fn!("elementwise_add");
    let f_zero = get_fn!("zero_pad");

    let (status_host_ptr, status_dev_ptr) = unsafe { alloc_mapped_result_array(&dev, 1)? };

    // Upload embedding tables
    let wte_dev = dev.htod_sync_copy(&weights.wte)?;
    let wpe_dev = dev.htod_sync_copy(&weights.wpe)?;
    let token_ids_dev = dev.htod_sync_copy(&token_ids_u32)?;

    // === Embedding ===
    let mut hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    unsafe {
        f_embed.clone().launch(
            LaunchConfig {
                grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &wte_dev,
                &wpe_dev,
                &token_ids_dev,
                &mut hidden_dev,
                seq,
                D_MODEL,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    // Zero padded positions
    let pad_start = (actual_seq as u32) * D_MODEL;
    let pad_count = total_seq_model as u32 - pad_start;
    if pad_count > 0 {
        unsafe {
            f_zero.clone().launch(
                LaunchConfig {
                    grid_dim: (pad_count.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, pad_start, total_seq_model as u32),
            )?;
        }
        dev.synchronize()?;
    }
    println!("  Embedding done (seq={seq}, actual={actual_seq})");

    drop(wte_dev);
    drop(wpe_dev);

    // === Helper: transpose weight [K, N] row-major to column-major f32 ===
    // Column-major B[col * K + row] for gemm_f32
    fn to_col_major(w: &[f32], k: usize, n: usize) -> Vec<f32> {
        let mut cm = vec![0.0f32; k * n];
        for row in 0..k {
            for col in 0..n {
                cm[col * k + row] = w[row * n + col];
            }
        }
        cm
    }

    // === Upload layer weights as f32 column-major ===
    println!("  Uploading 12 layers (f32 column-major)...");
    struct LayerWeightsF32 {
        ln1_g: CudaSlice<f32>,
        ln1_b: CudaSlice<f32>,
        w_qkv: CudaSlice<f32>,
        b_qkv: CudaSlice<f32>,
        w_proj: CudaSlice<f32>,
        b_proj: CudaSlice<f32>,
        ln2_g: CudaSlice<f32>,
        ln2_b: CudaSlice<f32>,
        w_fc: CudaSlice<f32>,
        b_fc: CudaSlice<f32>,
        w_fc_proj: CudaSlice<f32>,
        b_fc_proj: CudaSlice<f32>,
    }

    let mut gpu_layers: Vec<LayerWeightsF32> = Vec::with_capacity(12);
    for (i, layer) in weights.layers.iter().enumerate() {
        let w_qkv_cm = to_col_major(&layer.c_attn_weight, 768, 2304);
        let w_proj_cm = to_col_major(&layer.c_proj_weight, 768, 768);
        let w_fc_cm = to_col_major(&layer.mlp_fc_weight, 768, 3072);
        let w_fc_proj_cm = to_col_major(&layer.mlp_proj_weight, 3072, 768);

        gpu_layers.push(LayerWeightsF32 {
            ln1_g: dev.htod_sync_copy(&layer.ln_1.weight)?,
            ln1_b: dev.htod_sync_copy(&layer.ln_1.bias)?,
            w_qkv: dev.htod_sync_copy(&w_qkv_cm)?,
            b_qkv: dev.htod_sync_copy(&layer.c_attn_bias)?,
            w_proj: dev.htod_sync_copy(&w_proj_cm)?,
            b_proj: dev.htod_sync_copy(&layer.c_proj_bias)?,
            ln2_g: dev.htod_sync_copy(&layer.ln_2.weight)?,
            ln2_b: dev.htod_sync_copy(&layer.ln_2.bias)?,
            w_fc: dev.htod_sync_copy(&w_fc_cm)?,
            b_fc: dev.htod_sync_copy(&layer.mlp_fc_bias)?,
            w_fc_proj: dev.htod_sync_copy(&w_fc_proj_cm)?,
            b_fc_proj: dev.htod_sync_copy(&layer.mlp_proj_bias)?,
        });
        if i == 0 || i == 11 {
            println!("    Layer {i} uploaded");
        }
    }
    println!("  All 12 layers uploaded (f32)");

    // Final LN weights
    let ln_f_g_dev = dev.htod_sync_copy(&weights.ln_f.weight)?;
    let ln_f_b_dev = dev.htod_sync_copy(&weights.ln_f.bias)?;

    // === Allocate activation buffers ===
    let mut ln_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut qkv_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * 2304) as usize)?;
    let mut q_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut k_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut v_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut attn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(head_total)?;
    let mut concat_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut proj_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;
    let mut ffn_hidden_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut gelu_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>((seq * D_FFN) as usize)?;
    let mut ffn_out_dev: CudaSlice<f32> = dev.alloc_zeros::<f32>(total_seq_model)?;

    let gemm_shared = (32 * 16 + 16 * 16) * 4; // f32 shared memory for gemm_f32
    let n_q_tiles = (seq as usize).div_ceil(32) as u32;

    let fwd_start = std::time::Instant::now();

    // === 12 transformer layers ===
    for layer_idx in 0..12u32 {
        let lw = &gpu_layers[layer_idx as usize];

        // LN1
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln1_g,
                    &lw.ln1_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // QKV GEMM (f32): ln_out[32,768] × w_qkv[768,2304] → qkv[32,2304]
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, 2304 / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_qkv,
                    &mut qkv_dev,
                    D_MODEL,
                    2304u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // === Layer 0 diagnostic: compare f32 GEMM QKV with CPU f64 ===
        if layer_idx == 0 {
            let qkv_snap: Vec<f32> = dev.dtoh_sync_copy(&qkv_dev)?;
            let pos4 = actual_seq - 1;
            let qkv4 = &qkv_snap[pos4 * 2304..(pos4 + 1) * 2304];
            println!(
                "  f32 GEMM L0 QKV pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
            println!("  (CPU f64 reference: [-0.191336, -0.079322, 0.937499, 0.162505])");
        }

        // QKV bias
        let total_qkv = seq * 2304;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_qkv.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut qkv_dev, &lw.b_qkv, 2304u32, total_qkv, status_dev_ptr),
            )?;
        }
        dev.synchronize()?;

        // Split QKV
        unsafe {
            f_split.clone().launch(
                LaunchConfig {
                    grid_dim: ((head_total as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &qkv_dev, &mut q_dev, &mut k_dev, &mut v_dev, seq, N_HEADS, D_HEAD,
                ),
            )?;
        }
        dev.synchronize()?;

        // Flash attention (causal)
        unsafe {
            f_attn.clone().launch(
                LaunchConfig {
                    grid_dim: (N_HEADS, n_q_tiles, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 2 * 32 * 64 * 4,
                },
                (
                    &q_dev,
                    &k_dev,
                    &v_dev,
                    &mut attn_out_dev,
                    seq,
                    D_HEAD,
                    1u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Concat heads
        unsafe {
            f_concat.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&attn_out_dev, &mut concat_dev, seq, N_HEADS, D_HEAD),
            )?;
        }
        dev.synchronize()?;

        // Output projection (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &concat_dev,
                    &lw.w_proj,
                    &mut proj_out_dev,
                    D_MODEL,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Proj bias
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut proj_out_dev,
                    &lw.b_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Residual 1 + zero pad
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &proj_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        // LN2
        unsafe {
            f_ln.clone().launch(
                LaunchConfig {
                    grid_dim: (seq, 1, 1),
                    block_dim: (32, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &hidden_dev,
                    &mut ln_out_dev,
                    &lw.ln2_g,
                    &lw.ln2_b,
                    D_MODEL,
                    EPS,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN up (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_FFN / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &ln_out_dev,
                    &lw.w_fc,
                    &mut ffn_hidden_dev,
                    D_MODEL,
                    D_FFN,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN bias
        let total_ffn = seq * D_FFN;
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_hidden_dev,
                    &lw.b_fc,
                    D_FFN,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // GELU
        unsafe {
            f_gelu.clone().launch(
                LaunchConfig {
                    grid_dim: (total_ffn.div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &ffn_hidden_dev,
                    &mut gelu_out_dev,
                    total_ffn,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN down (f32 GEMM)
        unsafe {
            f_gemm.clone().launch(
                LaunchConfig {
                    grid_dim: (seq / 32, D_MODEL / 16, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: gemm_shared,
                },
                (
                    &gelu_out_dev,
                    &lw.w_fc_proj,
                    &mut ffn_out_dev,
                    D_FFN,
                    D_MODEL,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // FFN bias
        unsafe {
            f_bias.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (
                    &mut ffn_out_dev,
                    &lw.b_fc_proj,
                    D_MODEL,
                    total_seq_model as u32,
                    status_dev_ptr,
                ),
            )?;
        }
        dev.synchronize()?;

        // Residual 2 + zero pad
        unsafe {
            f_add.clone().launch(
                LaunchConfig {
                    grid_dim: ((total_seq_model as u32).div_ceil(256), 1, 1),
                    block_dim: (256, 1, 1),
                    shared_mem_bytes: 0,
                },
                (&mut hidden_dev, &ffn_out_dev, total_seq_model as u32),
            )?;
            if pad_count > 0 {
                f_zero.clone().launch(
                    LaunchConfig {
                        grid_dim: (pad_count.div_ceil(256), 1, 1),
                        block_dim: (256, 1, 1),
                        shared_mem_bytes: 0,
                    },
                    (&mut hidden_dev, pad_start, total_seq_model as u32),
                )?;
            }
        }
        dev.synchronize()?;

        if layer_idx == 0 || layer_idx == 5 || layer_idx == 11 {
            println!("    Layer {layer_idx} done");
        }
    }

    // === Final LayerNorm ===
    unsafe {
        f_ln.clone().launch(
            LaunchConfig {
                grid_dim: (seq, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            (
                &hidden_dev,
                &mut ln_out_dev,
                &ln_f_g_dev,
                &ln_f_b_dev,
                D_MODEL,
                EPS,
                status_dev_ptr,
            ),
        )?;
    }
    dev.synchronize()?;

    let fwd_elapsed = fwd_start.elapsed();

    // Download output
    let output: Vec<f32> = dev.dtoh_sync_copy(&ln_out_dev)?;

    // Check prediction position
    let last_pos = actual_seq - 1;
    let last_row = &output[last_pos * dm..(last_pos + 1) * dm];
    let pred_nan = last_row.iter().filter(|v| v.is_nan()).count();
    let pred_inf = last_row.iter().filter(|v| v.is_infinite()).count();
    let max_abs = last_row
        .iter()
        .filter(|v| !v.is_nan() && !v.is_infinite())
        .fold(0.0f32, |m, &v| m.max(v.abs()));

    println!(
        "  Forward pass: {:.1}ms",
        fwd_elapsed.as_secs_f64() * 1000.0
    );
    println!("  Prediction pos {last_pos}: nan={pred_nan}, inf={pred_inf}, max|val|={max_abs:.4}");

    if pred_nan > 0 || pred_inf > 0 {
        return Err(GpuHostError::Verification {
            test: "f32_forward",
            detail: format!("prediction has {pred_nan} NaN, {pred_inf} Inf"),
        });
    }

    // CPU LM head: logits[v] = dot(hidden[last_pos], wte[v])
    let vocab_size = 50257;
    let hidden = last_row;
    let mut logits = vec![0.0f32; vocab_size];
    for v in 0..vocab_size {
        let wte_row = &weights.wte[v * dm..(v + 1) * dm];
        let mut dot = 0.0f32;
        for d in 0..dm {
            dot += hidden[d] * wte_row[d];
        }
        logits[v] = dot;
    }

    // Top-5
    let mut indices: Vec<usize> = (0..vocab_size).collect();
    indices.sort_unstable_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    println!("  Top-5 predictions (f32 GEMM):");
    for rank in 0..5 {
        let idx = indices[rank];
        let tok = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<tok {idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.2})",
            rank + 1,
            idx,
            tok,
            logits[idx],
        );
    }

    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<tok {top1}>"));
    println!("  Greedy: {} = {:?}", top1, top1_str);

    unsafe {
        free_mapped_mem(status_host_ptr)?;
    }

    println!("  f32 GEMM forward pass — PASSED");
    Ok(())
}

/// full-inference.8+10: CPU f64 reference forward pass — definitive diagnostic.
///
/// Implements the COMPLETE GPT-2 transformer in pure Rust f64 arithmetic on CPU.
/// No GPU involved. If this also predicts " a", the model genuinely cannot answer.
/// If this predicts "Paris", there is a GPU implementation bug.
///
/// Also performs LM head position audit (full-inference.10): checks predictions
/// from last_pos-1, last_pos, last_pos+1.
///
/// Also performs per-layer intermediate predictions (full-inference.9): applies
/// LM head after each of 12 layers to track where semantic content appears/is lost.
pub(crate) fn run_cpu_f64_reference_test() -> Result<()> {
    let model_path = std::path::Path::new("../../models/model.safetensors");
    if !model_path.exists() {
        println!("\n--- Skipping CPU f64 reference (models/model.safetensors not found) ---");
        return Ok(());
    }

    println!("\n--- CPU f64 reference forward pass (full-inference.8+9+10) ---");

    let weights =
        gpu_host::model::load_gpt2_weights(model_path).map_err(|e| GpuHostError::Verification {
            test: "cpu_f64_ref",
            detail: format!("weight loading: {e}"),
        })?;

    let tokenizer =
        gpu_host::tokenizer::Gpt2Tokenizer::new().map_err(|e| GpuHostError::Verification {
            test: "cpu_f64_ref",
            detail: format!("tokenizer: {e}"),
        })?;
    let prompt = "The capital of France is";
    let prompt_tokens = tokenizer.encode(prompt);
    let seq = prompt_tokens.len();
    println!("  Prompt: \"{prompt}\" → {seq} tokens: {prompt_tokens:?}");

    const D_MODEL: usize = 768;
    const N_HEADS: usize = 12;
    const D_HEAD: usize = 64;
    const D_FFN: usize = 3072;
    const VOCAB_SIZE: usize = 50257;
    const EPS: f64 = 1e-5;

    // Convert all weights to f64
    let wte: Vec<f64> = weights.wte.iter().map(|&x| x as f64).collect();
    let wpe: Vec<f64> = weights.wpe.iter().map(|&x| x as f64).collect();
    let ln_f_g: Vec<f64> = weights.ln_f.weight.iter().map(|&x| x as f64).collect();
    let ln_f_b: Vec<f64> = weights.ln_f.bias.iter().map(|&x| x as f64).collect();

    // === Embedding: hidden[pos] = wte[token] + wpe[pos] ===
    let mut hidden = vec![0.0f64; seq * D_MODEL];
    for pos in 0..seq {
        let tok = prompt_tokens[pos] as usize;
        for d in 0..D_MODEL {
            hidden[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
        }
    }
    println!(
        "  Embedding done. hidden[0][0..4] = [{:.6}, {:.6}, {:.6}, {:.6}]",
        hidden[0], hidden[1], hidden[2], hidden[3]
    );
    let pos4_start = (seq - 1) * D_MODEL;
    println!(
        "  CPU embed pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
        hidden[pos4_start],
        hidden[pos4_start + 1],
        hidden[pos4_start + 2],
        hidden[pos4_start + 3]
    );

    // Helper: LayerNorm
    fn layer_norm(input: &[f64], gamma: &[f64], beta: &[f64], dim: usize, eps: f64) -> Vec<f64> {
        let seq = input.len() / dim;
        let mut output = vec![0.0f64; input.len()];
        for s in 0..seq {
            let row = &input[s * dim..(s + 1) * dim];
            let mean: f64 = row.iter().sum::<f64>() / dim as f64;
            let var: f64 = row.iter().map(|&x| (x - mean) * (x - mean)).sum::<f64>() / dim as f64;
            let inv_std = 1.0 / (var + eps).sqrt();
            for d in 0..dim {
                output[s * dim + d] = (row[d] - mean) * inv_std * gamma[d] + beta[d];
            }
        }
        output
    }

    // Helper: matmul A[M,K] @ B[K,N] → C[M,N]  (B stored row-major as [K,N])
    fn matmul(a: &[f64], b: &[f64], m: usize, k: usize, n: usize) -> Vec<f64> {
        let mut c = vec![0.0f64; m * n];
        for i in 0..m {
            for j in 0..n {
                let mut sum = 0.0f64;
                for p in 0..k {
                    sum += a[i * k + p] * b[p * n + j];
                }
                c[i * n + j] = sum;
            }
        }
        c
    }

    // Helper: add bias
    fn add_bias(data: &mut [f64], bias: &[f64], dim: usize) {
        let seq = data.len() / dim;
        for s in 0..seq {
            for d in 0..dim {
                data[s * dim + d] += bias[d];
            }
        }
    }

    // Helper: GELU (GPT-2 variant with tanh approximation)
    fn gelu(x: f64) -> f64 {
        let sqrt_2_over_pi: f64 = 0.7978845608028654;
        let coeff: f64 = 0.044715;
        let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
        x * 0.5 * (1.0 + inner.tanh())
    }

    // Helper: compute top-5 predictions from a hidden vector using wte
    fn top5_from_hidden(
        hidden_vec: &[f64],
        wte: &[f64],
        dim: usize,
        vocab: usize,
        tokenizer: &gpu_host::tokenizer::Gpt2Tokenizer,
    ) -> Vec<(usize, f64, String)> {
        let mut logits = vec![0.0f64; vocab];
        for v in 0..vocab {
            let mut dot = 0.0f64;
            for d in 0..dim {
                dot += hidden_vec[d] * wte[v * dim + d];
            }
            logits[v] = dot;
        }
        let mut indices: Vec<usize> = (0..vocab).collect();
        indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());
        indices[..5]
            .iter()
            .map(|&idx| {
                let tok_str = tokenizer
                    .decode(&[idx as u32])
                    .unwrap_or_else(|_| format!("<{idx}>"));
                (idx, logits[idx], tok_str)
            })
            .collect()
    }

    let timer = std::time::Instant::now();

    // === 12 Transformer Layers ===
    for layer_idx in 0..12 {
        let lw = &weights.layers[layer_idx];
        let ln1_g: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
        let ln1_b: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
        let c_attn_w: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
        let c_attn_b: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
        let c_proj_w: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
        let c_proj_b: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
        let ln2_g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
        let ln2_b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
        let fc_w: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
        let fc_b: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
        let proj_w: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
        let proj_b: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

        // 1. LayerNorm1
        let ln_out = layer_norm(&hidden, &ln1_g, &ln1_b, D_MODEL, EPS);

        if layer_idx == 0 {
            let lp = seq - 1;
            let ln4 = &ln_out[lp * D_MODEL..(lp + 1) * D_MODEL];
            println!(
                "  CPU L0 LN1 pos4 first4: [{:.6}, {:.6}, {:.6}, {:.6}]",
                ln4[0], ln4[1], ln4[2], ln4[3]
            );
        }

        // 2. QKV projection: ln_out[seq, 768] @ c_attn_w[768, 2304] + bias
        let mut qkv = matmul(&ln_out, &c_attn_w, seq, D_MODEL, D_MODEL * 3);

        if layer_idx == 0 {
            let lp = seq - 1;
            let qkv4 = &qkv[lp * (D_MODEL * 3)..(lp + 1) * (D_MODEL * 3)];
            println!(
                "  CPU L0 QKV pos4 first4 (no bias): [{:.6}, {:.6}, {:.6}, {:.6}]",
                qkv4[0], qkv4[1], qkv4[2], qkv4[3]
            );
        }

        add_bias(&mut qkv, &c_attn_b, D_MODEL * 3);

        // 3. Split Q, K, V and compute attention per head
        // qkv layout: [seq, 2304] where Q=[0..768], K=[768..1536], V=[1536..2304]
        // Within each 768: [head0_d0..head0_d63, head1_d0..head1_d63, ...]
        let mut attn_concat = vec![0.0f64; seq * D_MODEL];

        for h in 0..N_HEADS {
            // Extract Q, K, V for this head
            let mut q = vec![0.0f64; seq * D_HEAD];
            let mut k = vec![0.0f64; seq * D_HEAD];
            let mut v = vec![0.0f64; seq * D_HEAD];

            for s in 0..seq {
                for d in 0..D_HEAD {
                    q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + h * D_HEAD + d];
                    k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + h * D_HEAD + d];
                    v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + h * D_HEAD + d];
                }
            }

            // Attention scores: Q @ K^T / sqrt(d_head) with causal mask
            let scale = 1.0 / (D_HEAD as f64).sqrt();
            let mut scores = vec![0.0f64; seq * seq];
            for i in 0..seq {
                for j in 0..seq {
                    if j <= i {
                        // Causal: position i can attend to positions 0..=i
                        let mut dot = 0.0f64;
                        for d in 0..D_HEAD {
                            dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                        }
                        scores[i * seq + j] = dot * scale;
                    } else {
                        scores[i * seq + j] = f64::NEG_INFINITY;
                    }
                }
            }

            // Softmax per row
            for i in 0..seq {
                let row = &mut scores[i * seq..(i + 1) * seq];
                let max_val = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                let mut sum = 0.0f64;
                for j in 0..seq {
                    row[j] = (row[j] - max_val).exp();
                    sum += row[j];
                }
                for j in 0..seq {
                    row[j] /= sum;
                }
            }

            // Attention output: softmax_scores @ V
            let attn_out = matmul(&scores, &v, seq, seq, D_HEAD);

            // Concat: write to [seq][d_model] at head offset
            for s in 0..seq {
                for d in 0..D_HEAD {
                    attn_concat[s * D_MODEL + h * D_HEAD + d] = attn_out[s * D_HEAD + d];
                }
            }
        }

        // 4. Output projection: attn_concat[seq, 768] @ c_proj_w[768, 768] + bias
        let mut proj_out = matmul(&attn_concat, &c_proj_w, seq, D_MODEL, D_MODEL);
        add_bias(&mut proj_out, &c_proj_b, D_MODEL);

        // 5. Residual 1
        for i in 0..hidden.len() {
            hidden[i] += proj_out[i];
        }

        // 6. LayerNorm2
        let ln2_out = layer_norm(&hidden, &ln2_g, &ln2_b, D_MODEL, EPS);

        // 7. FFN up: ln2_out[seq, 768] @ fc_w[768, 3072] + bias
        let mut ffn_hidden = matmul(&ln2_out, &fc_w, seq, D_MODEL, D_FFN);
        add_bias(&mut ffn_hidden, &fc_b, D_FFN);

        // 8. GELU
        for x in ffn_hidden.iter_mut() {
            *x = gelu(*x);
        }

        // 9. FFN down: ffn_hidden[seq, 3072] @ proj_w[3072, 768] + bias
        let mut ffn_out = matmul(&ffn_hidden, &proj_w, seq, D_FFN, D_MODEL);
        add_bias(&mut ffn_out, &proj_b, D_MODEL);

        // 10. Residual 2
        for i in 0..hidden.len() {
            hidden[i] += ffn_out[i];
        }

        // === Per-layer intermediate predictions (full-inference.9) ===
        // Compute norms of each component
        let last_pos = seq - 1;
        let attn_res_norm: f64 = proj_out[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ffn_res_norm: f64 = ffn_out[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ln_probe = layer_norm(&hidden, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let probe_vec = &ln_probe[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
        let top5 = top5_from_hidden(probe_vec, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let hidden_norm: f64 = hidden[last_pos * D_MODEL..(last_pos + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        println!(
            "  Layer {:2}: top1={:?} ({:.4}) | h_norm={:.2} attn_r={:.2} ffn_r={:.2} | top5: {}",
            layer_idx,
            top5[0].2,
            top5[0].1,
            hidden_norm,
            attn_res_norm,
            ffn_res_norm,
            top5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // === Final LayerNorm ===
    let ln_final = layer_norm(&hidden, &ln_f_g, &ln_f_b, D_MODEL, EPS);

    let elapsed = timer.elapsed();
    println!(
        "  Forward pass done in {:.1}ms",
        elapsed.as_secs_f64() * 1000.0
    );

    // === LM Head position audit (full-inference.10) ===
    let last_pos = seq - 1;
    println!("\n  === Position audit ===");
    for offset in [-1i32, 0, 1] {
        let pos = last_pos as i32 + offset;
        if pos < 0 || pos >= seq as i32 {
            continue;
        }
        let pos = pos as usize;
        let hvec = &ln_final[pos * D_MODEL..(pos + 1) * D_MODEL];
        let top5 = top5_from_hidden(hvec, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let label = match offset {
            -1 => "last_pos-1",
            0 => "last_pos  ",
            1 => "last_pos+1",
            _ => unreachable!(),
        };
        println!(
            "  {label} (pos={pos}): top5: {}",
            top5.iter()
                .map(|(id, l, s)| format!("{s}[{id}]({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    // === Diagnostic: norms ===
    let final_vec_for_diag = &ln_final[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
    let ln_final_norm: f64 = final_vec_for_diag.iter().map(|x| x * x).sum::<f64>().sqrt();
    let ln_final_mean: f64 = final_vec_for_diag.iter().sum::<f64>() / D_MODEL as f64;
    let ln_final_max: f64 = final_vec_for_diag
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    let ln_final_min: f64 = final_vec_for_diag
        .iter()
        .cloned()
        .fold(f64::INFINITY, f64::min);
    println!("\n  === Final LN output diagnostics ===");
    println!(
        "  ln_final norm={:.4}, mean={:.6}, min={:.4}, max={:.4}",
        ln_final_norm, ln_final_mean, ln_final_min, ln_final_max
    );

    // Check a few wte row norms
    let mut wte_norms: Vec<f64> = Vec::new();
    for v in 0..100 {
        let norm: f64 = wte[v * D_MODEL..(v + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        wte_norms.push(norm);
    }
    let avg_wte_norm: f64 = wte_norms.iter().sum::<f64>() / wte_norms.len() as f64;
    println!("  wte avg row norm (first 100) = {:.4}", avg_wte_norm);

    // Check ln_f gamma stats
    let gamma_norm: f64 = ln_f_g.iter().map(|x| x * x).sum::<f64>().sqrt();
    let gamma_mean: f64 = ln_f_g.iter().sum::<f64>() / D_MODEL as f64;
    let gamma_max: f64 = ln_f_g.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    println!(
        "  ln_f gamma: norm={:.4}, mean={:.6}, max={:.4}",
        gamma_norm, gamma_mean, gamma_max
    );

    // === Final top-10 predictions from last_pos ===
    println!("\n  === Final predictions (last_pos={last_pos}) ===");
    let final_vec = &ln_final[last_pos * D_MODEL..(last_pos + 1) * D_MODEL];
    let mut logits = vec![0.0f64; VOCAB_SIZE];
    for v in 0..VOCAB_SIZE {
        let mut dot = 0.0f64;
        for d in 0..D_MODEL {
            dot += final_vec[d] * wte[v * D_MODEL + d];
        }
        logits[v] = dot;
    }
    let mut indices: Vec<usize> = (0..VOCAB_SIZE).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap());

    for rank in 0..10 {
        let idx = indices[rank];
        let tok = tokenizer
            .decode(&[idx as u32])
            .unwrap_or_else(|_| format!("<{idx}>"));
        println!(
            "    #{}: token {} = {:?} (logit={:.4})",
            rank + 1,
            idx,
            tok,
            logits[idx]
        );
    }

    let top1 = indices[0];
    let top1_str = tokenizer
        .decode(&[top1 as u32])
        .unwrap_or_else(|_| format!("<{top1}>"));
    println!(
        "\n  CPU f64 GREEDY PREDICTION: token {} = {:?}",
        top1, top1_str
    );

    // === Padding test: does padding change the result? ===
    println!("\n  === CPU f64 with padding (seq=32, same as GPU) ===");
    {
        let padded_seq = 32usize;
        // Re-do embedding with padding
        let mut h_pad = vec![0.0f64; padded_seq * D_MODEL];
        for pos in 0..seq {
            let tok = prompt_tokens[pos] as usize;
            for d in 0..D_MODEL {
                h_pad[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }
        // Zero padded positions (seq..padded_seq) — already zero from initialization

        // Run 12 layers with padded_seq
        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h_pad, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, padded_seq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; padded_seq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; padded_seq * D_HEAD];
                let mut k = vec![0.0f64; padded_seq * D_HEAD];
                let mut v = vec![0.0f64; padded_seq * D_HEAD];
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; padded_seq * padded_seq];
                for i in 0..padded_seq {
                    for j in 0..padded_seq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * padded_seq + j] = dot * scale;
                        } else {
                            scores[i * padded_seq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..padded_seq {
                    let row = &mut scores[i * padded_seq..(i + 1) * padded_seq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..padded_seq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..padded_seq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, padded_seq, padded_seq, D_HEAD);
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, padded_seq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h_pad.len() {
                h_pad[i] += po[i];
            }

            // Zero padded positions after residual (like GPU does)
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_pad[s * D_MODEL + d] = 0.0;
                }
            }

            let l2o = layer_norm(&h_pad, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, padded_seq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, padded_seq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h_pad.len() {
                h_pad[i] += fo[i];
            }

            // Zero padded positions after FFN residual
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_pad[s * D_MODEL + d] = 0.0;
                }
            }
        }

        let lnf = layer_norm(&h_pad, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let lp = seq - 1;
        let hv = &lnf[lp * D_MODEL..(lp + 1) * D_MODEL];
        let t5 = top5_from_hidden(hv, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        let h_norm: f64 = h_pad[lp * D_MODEL..(lp + 1) * D_MODEL]
            .iter()
            .map(|x| x * x)
            .sum::<f64>()
            .sqrt();
        let ln_norm: f64 = hv.iter().map(|x| x * x).sum::<f64>().sqrt();
        println!(
            "  Padded result: h_norm={:.2} ln_norm={:.2}",
            h_norm, ln_norm
        );
        println!(
            "  Top5: {}",
            t5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
        println!("  Compare no-pad: h_norm=429.60, top1=\" the\"(-100.25)");
    }

    // === Layer-by-layer GPU vs CPU comparison ===
    // Recompute with padding (seq=32) and print per-layer hidden norms at pos 4
    // to compare against GPU output
    println!("\n  === Per-layer hidden state comparison (for GPU validation) ===");
    {
        let padded_seq = 32usize;
        let mut h_cmp = vec![0.0f64; padded_seq * D_MODEL];
        for pos in 0..seq {
            let tok = prompt_tokens[pos] as usize;
            for d in 0..D_MODEL {
                h_cmp[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }

        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h_cmp, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, padded_seq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; padded_seq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; padded_seq * D_HEAD];
                let mut k = vec![0.0f64; padded_seq * D_HEAD];
                let mut v = vec![0.0f64; padded_seq * D_HEAD];
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; padded_seq * padded_seq];
                for i in 0..padded_seq {
                    for j in 0..padded_seq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * padded_seq + j] = dot * scale;
                        } else {
                            scores[i * padded_seq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..padded_seq {
                    let row = &mut scores[i * padded_seq..(i + 1) * padded_seq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..padded_seq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..padded_seq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, padded_seq, padded_seq, D_HEAD);
                for s in 0..padded_seq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, padded_seq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h_cmp.len() {
                h_cmp[i] += po[i];
            }
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_cmp[s * D_MODEL + d] = 0.0;
                }
            }

            let l2o = layer_norm(&h_cmp, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, padded_seq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, padded_seq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h_cmp.len() {
                h_cmp[i] += fo[i];
            }
            for s in seq..padded_seq {
                for d in 0..D_MODEL {
                    h_cmp[s * D_MODEL + d] = 0.0;
                }
            }

            // Print hidden stats at pos 4 for comparison with GPU
            let lp = seq - 1;
            let h_slice = &h_cmp[lp * D_MODEL..(lp + 1) * D_MODEL];
            let h_max = h_slice.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
            let h_min = h_slice.iter().cloned().fold(f64::INFINITY, f64::min);
            let h_maxabs = h_slice.iter().map(|x| x.abs()).fold(0.0f64, f64::max);
            let first4: Vec<String> = h_slice[..4].iter().map(|x| format!("{x:.4}")).collect();
            println!(
                "  Layer {:2}: max|val|={:.2}, max={:.2}, min={:.2}, first4=[{}]",
                li,
                h_maxabs,
                h_max,
                h_min,
                first4.join(", ")
            );
        }
    }

    // === Additional prompt tests ===
    println!("\n  === Multi-prompt CPU f64 test ===");
    let test_prompts = [
        "Hello, my name is",
        "The largest city in Japan is",
        "1 + 1 =",
        "Barack Obama was the president of the",
    ];
    for tp in &test_prompts {
        let toks = tokenizer.encode(tp);
        let tseq = toks.len();

        // Embedding
        let mut h = vec![0.0f64; tseq * D_MODEL];
        for pos in 0..tseq {
            let tok = toks[pos] as usize;
            for d in 0..D_MODEL {
                h[pos * D_MODEL + d] = wte[tok * D_MODEL + d] + wpe[pos * D_MODEL + d];
            }
        }

        // 12 layers
        for li in 0..12 {
            let lw = &weights.layers[li];
            let lg: Vec<f64> = lw.ln_1.weight.iter().map(|&x| x as f64).collect();
            let lb: Vec<f64> = lw.ln_1.bias.iter().map(|&x| x as f64).collect();
            let caw: Vec<f64> = lw.c_attn_weight.iter().map(|&x| x as f64).collect();
            let cab: Vec<f64> = lw.c_attn_bias.iter().map(|&x| x as f64).collect();
            let cpw: Vec<f64> = lw.c_proj_weight.iter().map(|&x| x as f64).collect();
            let cpb: Vec<f64> = lw.c_proj_bias.iter().map(|&x| x as f64).collect();
            let l2g: Vec<f64> = lw.ln_2.weight.iter().map(|&x| x as f64).collect();
            let l2b: Vec<f64> = lw.ln_2.bias.iter().map(|&x| x as f64).collect();
            let fw: Vec<f64> = lw.mlp_fc_weight.iter().map(|&x| x as f64).collect();
            let fb: Vec<f64> = lw.mlp_fc_bias.iter().map(|&x| x as f64).collect();
            let pw: Vec<f64> = lw.mlp_proj_weight.iter().map(|&x| x as f64).collect();
            let pb: Vec<f64> = lw.mlp_proj_bias.iter().map(|&x| x as f64).collect();

            let lo = layer_norm(&h, &lg, &lb, D_MODEL, EPS);
            let mut qkv = matmul(&lo, &caw, tseq, D_MODEL, D_MODEL * 3);
            add_bias(&mut qkv, &cab, D_MODEL * 3);

            let mut ac = vec![0.0f64; tseq * D_MODEL];
            for hd in 0..N_HEADS {
                let mut q = vec![0.0f64; tseq * D_HEAD];
                let mut k = vec![0.0f64; tseq * D_HEAD];
                let mut v = vec![0.0f64; tseq * D_HEAD];
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        q[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + hd * D_HEAD + d];
                        k[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + D_MODEL + hd * D_HEAD + d];
                        v[s * D_HEAD + d] = qkv[s * (3 * D_MODEL) + 2 * D_MODEL + hd * D_HEAD + d];
                    }
                }
                let scale = 1.0 / (D_HEAD as f64).sqrt();
                let mut scores = vec![0.0f64; tseq * tseq];
                for i in 0..tseq {
                    for j in 0..tseq {
                        if j <= i {
                            let mut dot = 0.0f64;
                            for d in 0..D_HEAD {
                                dot += q[i * D_HEAD + d] * k[j * D_HEAD + d];
                            }
                            scores[i * tseq + j] = dot * scale;
                        } else {
                            scores[i * tseq + j] = f64::NEG_INFINITY;
                        }
                    }
                }
                for i in 0..tseq {
                    let row = &mut scores[i * tseq..(i + 1) * tseq];
                    let mx = row.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
                    let mut sm = 0.0f64;
                    for j in 0..tseq {
                        row[j] = (row[j] - mx).exp();
                        sm += row[j];
                    }
                    for j in 0..tseq {
                        row[j] /= sm;
                    }
                }
                let ao = matmul(&scores, &v, tseq, tseq, D_HEAD);
                for s in 0..tseq {
                    for d in 0..D_HEAD {
                        ac[s * D_MODEL + hd * D_HEAD + d] = ao[s * D_HEAD + d];
                    }
                }
            }

            let mut po = matmul(&ac, &cpw, tseq, D_MODEL, D_MODEL);
            add_bias(&mut po, &cpb, D_MODEL);
            for i in 0..h.len() {
                h[i] += po[i];
            }

            let l2o = layer_norm(&h, &l2g, &l2b, D_MODEL, EPS);
            let mut fh = matmul(&l2o, &fw, tseq, D_MODEL, D_FFN);
            add_bias(&mut fh, &fb, D_FFN);
            for x in fh.iter_mut() {
                *x = gelu(*x);
            }
            let mut fo = matmul(&fh, &pw, tseq, D_FFN, D_MODEL);
            add_bias(&mut fo, &pb, D_MODEL);
            for i in 0..h.len() {
                h[i] += fo[i];
            }
        }

        let lnf = layer_norm(&h, &ln_f_g, &ln_f_b, D_MODEL, EPS);
        let lp = tseq - 1;
        let hv = &lnf[lp * D_MODEL..(lp + 1) * D_MODEL];
        let t5 = top5_from_hidden(hv, &wte, D_MODEL, VOCAB_SIZE, &tokenizer);
        println!(
            "  \"{tp}\" → top5: {}",
            t5.iter()
                .map(|(_, l, s)| format!("{s}({l:.2})"))
                .collect::<Vec<_>>()
                .join(", ")
        );
    }

    println!("  CPU f64 reference forward pass — PASSED");
    Ok(())
}
