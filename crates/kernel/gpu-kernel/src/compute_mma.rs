// gpu-compute.3 & .4: Tensor Core MMA + shared memory tests

use crate::helpers::{bar_sync, get_dynamic_smem_ptr};
use core::arch::nvptx;

// ============================================================
// gpu-compute.3: Tensor Core MMA via inline PTX
// ============================================================

/// gpu-compute.3: Test Tensor Core MMA instruction via inline PTX.
///
/// Uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on SM80+.
/// Each thread in the warp holds a fragment of A, B, C matrices.
/// Test: A=0, B=0, C=known -> D should equal C (0*0 + C = C).
///
/// Parameters:
/// - c_vals: pointer to 4 f32 values per thread = 128 f32 total (as u32 bits)
/// - d_out:  pointer to 4 f32 values per thread = 128 f32 output (as u32 bits)
/// - status: 0 on entry, set to 1 on success
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_mma_m16n8k16(
    c_vals: *const u32,
    d_out: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    // Each thread reads its 4 C fragment registers (f32 as u32 bits)
    let base = (tid * 4) as usize;
    let c0 = *c_vals.add(base);
    let c1 = *c_vals.add(base + 1);
    let c2 = *c_vals.add(base + 2);
    let c3 = *c_vals.add(base + 3);

    // A = 0 (f16x2), B = 0 (f16x2) -> D = 0*0 + C = C
    let a0: u32 = 0;
    let a1: u32 = 0;
    let a2: u32 = 0;
    let a3: u32 = 0;
    let b0: u32 = 0;
    let b1: u32 = 0;

    let d0: u32;
    let d1: u32;
    let d2: u32;
    let d3: u32;

    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        d0 = c0;
        d1 = c1;
        d2 = c2;
        d3 = c3;
    }

    // Write D fragment back
    *d_out.add(base) = d0;
    *d_out.add(base + 1) = d1;
    *d_out.add(base + 2) = d2;
    *d_out.add(base + 3) = d3;

    // Lane 0 sets status
    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-compute.4: Shared memory access + bar.sync
// ============================================================

/// gpu-compute.4: Test shared memory access + bar.sync from Rust inline PTX.
///
/// Each thread writes its thread ID to shared memory, synchronizes,
/// then reads its neighbor's value (tid XOR 1) and writes to output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_shared_memory(output: *mut u32, n: u32, status: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid >= n {
        return;
    }

    #[cfg(target_arch = "nvptx64")]
    {
        // Get shared memory base (generic address space pointer)
        let smem = get_dynamic_smem_ptr() as *mut u32;

        // Each thread writes its tid to shared memory
        *smem.add(tid as usize) = tid + 1; // +1 so we can distinguish from zero-init

        // Synchronize all threads in the block
        bar_sync();

        // Each thread reads its neighbor's value (XOR with 1 for pair swap)
        let neighbor = tid ^ 1;
        let val = if neighbor < n {
            *smem.add(neighbor as usize)
        } else {
            0
        };

        // Write to global output
        *output.add(tid as usize) = val;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (output, n);
    }

    // Lane 0 sets status
    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

/// gpu-pipeline.1: MMA with proper fragment-to-matrix index mapping.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_mma_mapped(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [16][8] = 128 u32
        let b_smem = smem.add(128); // [16][4] = 64 u32

        // Cooperative load: 32 threads load 128 u32s of A (4 each)
        for i in 0..4u32 {
            let idx = (tid * 4 + i) as usize;
            *a_smem.add(idx) = *a_global.add(idx);
        }
        // 32 threads load 64 u32s of B (2 each)
        for i in 0..2u32 {
            let idx = (tid * 2 + i) as usize;
            *b_smem.add(idx) = *b_global.add(idx);
        }
        bar_sync();

        // Fragment indexing for m16n8k16:
        let group = tid / 4; // 0..7
        let lane = tid % 4; // 0..3

        let a0 = *a_smem.add((group * 8 + lane) as usize);
        let a1 = *a_smem.add((group * 8 + lane + 4) as usize);
        let a2 = *a_smem.add(((group + 8) * 8 + lane) as usize);
        let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);

        let b0 = *b_smem.add((group * 4 + lane) as usize);
        let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

        // C = 0 (f32 accumulator)
        let c0: u32 = 0;
        let c1: u32 = 0;
        let c2: u32 = 0;
        let c3: u32 = 0;

        let d0: u32;
        let d1: u32;
        let d2: u32;
        let d3: u32;
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );

        // Write D fragments to output (thread-indexed)
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = d0;
        *d_global.add(out_base + 1) = d1;
        *d_global.add(out_base + 2) = d2;
        *d_global.add(out_base + 3) = d3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}
