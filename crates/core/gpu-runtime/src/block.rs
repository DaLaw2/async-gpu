/// Block-level barrier synchronization. PTX: `bar.sync 0`.
///
/// All threads in the block must reach this point before any can proceed.
///
/// # Safety
/// All threads in the block must call this function (or the block will deadlock).
#[inline(always)]
pub unsafe fn sync() {
    #[cfg(target_arch = "nvptx64")]
    core::arch::asm!("bar.sync 0;");
}

/// Get pointer to dynamically-allocated shared memory.
///
/// The kernel must declare shared memory via `global_asm!` and the
/// host must set `shared_mem_bytes > 0` in the launch config.
///
/// # Safety
/// Shared memory must be allocated in the launch config.
#[inline(always)]
pub unsafe fn shared_mem_ptr() -> *mut u8 {
    #[cfg(target_arch = "nvptx64")]
    {
        let ptr: u64;
        core::arch::asm!(
            "cvta.shared.u64 {out}, dynamic_smem;",
            out = out(reg64) ptr,
        );
        ptr as *mut u8
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        core::ptr::null_mut()
    }
}

/// Cast shared memory pointer to a typed pointer at a byte offset.
///
/// # Safety
/// - Shared memory must be allocated and large enough
/// - Alignment of T must be satisfied at the offset
#[inline(always)]
pub unsafe fn shared_mem_at<T>(offset: usize) -> *mut T {
    shared_mem_ptr().add(offset) as *mut T
}

/// Block-level parallel reduction: sum of `val` across `block_size` threads.
///
/// Uses shared memory with halving-stride tree reduction. Result is available
/// in thread 0 and broadcast to all threads via shared memory.
///
/// Requires `block_size * 4` bytes of shared memory at `smem_offset`.
///
/// # Safety
/// - All `block_size` threads in the block must call this function
/// - Shared memory must be allocated (at least `smem_offset + block_size * 4` bytes)
/// - `block_size` must be a power of 2
#[inline(always)]
pub unsafe fn reduce_sum_f32(val: f32, tid: u32, block_size: u32, smem_offset: usize) -> f32 {
    let smem = shared_mem_at::<f32>(smem_offset);
    *smem.add(tid as usize) = val;
    sync();

    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            let a = *smem.add(tid as usize);
            let b = *smem.add((tid + stride) as usize);
            *smem.add(tid as usize) = a + b;
        }
        sync();
        stride /= 2;
    }
    let result = *smem.add(0);
    sync();
    result
}

/// Block-level parallel reduction: maximum of `val` across `block_size` threads.
///
/// Uses shared memory with halving-stride tree reduction. Result is broadcast to all threads.
///
/// # Safety
/// Same requirements as [`reduce_sum_f32`].
#[inline(always)]
pub unsafe fn reduce_max_f32(val: f32, tid: u32, block_size: u32, smem_offset: usize) -> f32 {
    let smem = shared_mem_at::<f32>(smem_offset);
    *smem.add(tid as usize) = val;
    sync();

    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            let a = *smem.add(tid as usize);
            let b = *smem.add((tid + stride) as usize);
            if b > a {
                *smem.add(tid as usize) = b;
            }
        }
        sync();
        stride /= 2;
    }
    let result = *smem.add(0);
    sync();
    result
}

/// Block-level parallel reduction: minimum of `val` across `block_size` threads.
///
/// # Safety
/// Same requirements as [`reduce_sum_f32`].
#[inline(always)]
pub unsafe fn reduce_min_f32(val: f32, tid: u32, block_size: u32, smem_offset: usize) -> f32 {
    let smem = shared_mem_at::<f32>(smem_offset);
    *smem.add(tid as usize) = val;
    sync();

    let mut stride = block_size / 2;
    while stride > 0 {
        if tid < stride {
            let a = *smem.add(tid as usize);
            let b = *smem.add((tid + stride) as usize);
            if b < a {
                *smem.add(tid as usize) = b;
            }
        }
        sync();
        stride /= 2;
    }
    let result = *smem.add(0);
    sync();
    result
}
