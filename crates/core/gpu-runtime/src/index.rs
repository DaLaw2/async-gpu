/// Thread index within block (x dimension).
#[inline(always)]
pub fn thread_idx_x() -> u32 {
    crate::nvptx_shim::thread_idx_x()
}

/// Thread index within block (y dimension).
#[inline(always)]
pub fn thread_idx_y() -> u32 {
    crate::nvptx_shim::thread_idx_y()
}

/// Thread index within block (z dimension).
#[inline(always)]
pub fn thread_idx_z() -> u32 {
    crate::nvptx_shim::thread_idx_z()
}

/// Block index within grid (x dimension).
#[inline(always)]
pub fn block_idx_x() -> u32 {
    crate::nvptx_shim::block_idx_x()
}

/// Block index within grid (y dimension).
#[inline(always)]
pub fn block_idx_y() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_block_idx_y() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        0
    }
}

/// Block index within grid (z dimension).
#[inline(always)]
pub fn block_idx_z() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_block_idx_z() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        0
    }
}

/// Block dimension (x — threads per block in x).
#[inline(always)]
pub fn block_dim_x() -> u32 {
    crate::nvptx_shim::block_dim_x()
}

/// Block dimension (y — threads per block in y).
#[inline(always)]
pub fn block_dim_y() -> u32 {
    crate::nvptx_shim::block_dim_y()
}

/// Block dimension (z — threads per block in z).
#[inline(always)]
pub fn block_dim_z() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_block_dim_z() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        1
    }
}

/// Grid dimension (x — blocks per grid in x).
#[inline(always)]
pub fn grid_dim_x() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_grid_dim_x() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        1
    }
}

/// Grid dimension (y — blocks per grid in y).
#[inline(always)]
pub fn grid_dim_y() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_grid_dim_y() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        1
    }
}

/// Grid dimension (z — blocks per grid in z).
#[inline(always)]
pub fn grid_dim_z() -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        unsafe { core::arch::nvptx::_grid_dim_z() }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        1
    }
}

/// Global 1D thread index: `block_idx_x * block_dim_x + thread_idx_x`.
#[inline(always)]
pub fn global_thread_idx() -> u32 {
    block_idx_x() * block_dim_x() + thread_idx_x()
}

/// Total number of threads in a 1D grid: `grid_dim_x * block_dim_x`.
#[inline(always)]
pub fn global_thread_count() -> u32 {
    grid_dim_x() * block_dim_x()
}

/// Read the GPU global timer (nanoseconds since GPU reset).
#[inline(always)]
pub fn clock_nanos() -> u64 {
    #[cfg(target_arch = "nvptx64")]
    {
        let nanos: u64;
        unsafe {
            core::arch::asm!("mov.u64 {out}, %globaltimer;", out = out(reg64) nanos);
        }
        nanos
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        0
    }
}
