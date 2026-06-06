//! Transparent Data — GpuArray<T> with automatic host-device sync.
//!
//! Demonstrates GpuArray's residency state machine:
//! 1. Create from Vec — data lives on host only (HostOnly)
//! 2. bind_gpu_array() — auto H2D transfer (-> Synced)
//! 3. Kernel writes to output buffer
//! 4. mark_device_dirty() — host copy is stale (-> DeviceOnly)
//! 5. Deref reads results — auto D2H sync (-> Synced)
//!
//! The user never writes cudaMemcpy, htod, or dtoh.

use gpu_host::gpu;
use gpu_host::gpu_array::{GpuArray, Residency};

/// Minimal PTX kernel: f(x) = x * 2.0 + 1.0
///
/// Signature: fn(input: *const f32, output: *mut f32, n: u32)
/// Grid-stride loop for multi-block correctness.
const DOUBLE_PLUS_ONE_PTX: &str = "\
.version 7.8\n\
.target sm_75\n\
.address_size 64\n\
\n\
.visible .entry double_plus_one(\n\
    .param .u64 input,\n\
    .param .u64 output,\n\
    .param .u32 n\n\
)\n\
{\n\
    .reg .u32  %r<10>;\n\
    .reg .u64  %rd<6>;\n\
    .reg .f32  %f<3>;\n\
    .reg .pred %p;\n\
\n\
    ld.param.u64    %rd0, [input];\n\
    ld.param.u64    %rd1, [output];\n\
    ld.param.u32    %r0,  [n];\n\
\n\
    mov.u32         %r1,  %tid.x;\n\
    mov.u32         %r2,  %ntid.x;\n\
    mov.u32         %r3,  %ctaid.x;\n\
    mov.u32         %r4,  %nctaid.x;\n\
\n\
    mad.lo.u32      %r5,  %r3, %r2, %r1;\n\
    mul.lo.u32      %r6,  %r4, %r2;\n\
\n\
LOOP:\n\
    setp.ge.u32     %p, %r5, %r0;\n\
    @%p bra         DONE;\n\
\n\
    mul.wide.u32    %rd2, %r5, 4;\n\
    add.u64         %rd3, %rd0, %rd2;\n\
    ld.global.f32   %f0,  [%rd3];\n\
\n\
    fma.rn.f32      %f1,  %f0, 0f40000000, 0f3F800000;\n\
\n\
    add.u64         %rd4, %rd1, %rd2;\n\
    st.global.f32   [%rd4], %f1;\n\
\n\
    add.u32         %r5, %r5, %r6;\n\
    bra             LOOP;\n\
\n\
DONE:\n\
    ret;\n\
}\n\
";

fn main() {
    println!("=== Transparent Data: GpuArray<T> Demo ===\n");

    // -----------------------------------------------------------------------
    // Step 1: Create data as normal Rust Vecs, wrap in GpuArray
    // -----------------------------------------------------------------------
    let input_data = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let n = input_data.len();

    let input = GpuArray::from_vec(input_data);
    let output = GpuArray::<f32>::zeroed(n);

    println!("Step 1: Created GpuArray from Vec<f32> ({n} elements)");
    println!("  Input residency:  {:?}", input.residency());
    println!("  Output residency: {:?}", output.residency());
    assert_eq!(input.residency(), Residency::HostOnly);
    assert_eq!(output.residency(), Residency::HostOnly);

    // -----------------------------------------------------------------------
    // Step 2: Deref reads host data — no GPU involvement
    // -----------------------------------------------------------------------
    println!("\nStep 2: Read input via Deref (no GPU transfer)");
    println!("  input[0] = {}", input[0]);
    println!("  input[7] = {}", input[7]);
    assert_eq!(input.residency(), Residency::HostOnly);

    // -----------------------------------------------------------------------
    // Step 3: Prepare and launch kernel — auto H2D transfers
    // -----------------------------------------------------------------------
    println!("\nStep 3: Launch kernel — auto host-to-device transfer");

    let ctx = gpu::custom("double_plus_one")
        .ptx(DOUBLE_PLUS_ONE_PTX)
        .threads(256)
        .elements(n as u32)
        .prepare()
        .expect("Failed to prepare GPU context");

    // bind_gpu_array() calls ensure_device() internally
    // This triggers the HostOnly -> Synced transition (auto H2D copy)
    let input_ptr = ctx.bind_gpu_array(&input).expect("Failed to bind input");
    let output_ptr = ctx.bind_gpu_array(&output).expect("Failed to bind output");

    println!("  Input residency after bind:  {:?}", input.residency());
    println!("  Output residency after bind: {:?}", output.residency());
    assert_eq!(input.residency(), Residency::Synced);
    assert_eq!(output.residency(), Residency::Synced);

    // Launch: f(x) = x * 2.0 + 1.0
    let _result = unsafe {
        ctx.launch((input_ptr, output_ptr, n as u32))
            .expect("Kernel launch failed")
    };

    // -----------------------------------------------------------------------
    // Step 4: Mark output as device-dirty (kernel wrote to it)
    // -----------------------------------------------------------------------
    output.mark_device_dirty();
    println!("\nStep 4: Marked output as device-dirty");
    println!("  Output residency: {:?}", output.residency());
    assert_eq!(output.residency(), Residency::DeviceOnly);

    // -----------------------------------------------------------------------
    // Step 5: Read results via Deref — auto D2H sync
    // -----------------------------------------------------------------------
    println!("\nStep 5: Read results — auto device-to-host sync");
    println!("  Output residency before read: {:?}", output.residency());

    // This Deref triggers DeviceOnly -> Synced (auto D2H copy)
    let results: &[f32] = &output;

    println!("  Output residency after read:  {:?}", output.residency());
    assert_eq!(output.residency(), Residency::Synced);

    println!("\n  Results (f(x) = x * 2.0 + 1.0):");
    for i in 0..n {
        let expected = input[i] * 2.0 + 1.0;
        println!("    f({:.1}) = {:.1} (expected {:.1})", input[i], results[i], expected);
        assert!(
            (results[i] - expected).abs() < 1e-5,
            "Mismatch at index {i}: got {}, expected {expected}",
            results[i]
        );
    }

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("\n=== Summary ===");
    println!("GpuArray residency transitions:");
    println!("  HostOnly  -> (bind_gpu_array) -> Synced");
    println!("  Synced    -> (mark_device_dirty) -> DeviceOnly");
    println!("  DeviceOnly -> (Deref read) -> Synced");
    println!();
    println!("Zero explicit cudaMemcpy, htod, or dtoh calls.");
    println!("Data flows automatically based on access patterns.");
    println!("\n=== All assertions passed ===");
}
