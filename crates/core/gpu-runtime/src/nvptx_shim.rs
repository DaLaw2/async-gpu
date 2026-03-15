//! GPU intrinsic wrappers — stubs on non-nvptx targets for doc builds.

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn block_idx_x() -> u32 {
    unsafe { core::arch::nvptx::_block_idx_x() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn block_idx_x() -> u32 {
    0
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn thread_idx_x() -> u32 {
    unsafe { core::arch::nvptx::_thread_idx_x() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn thread_idx_x() -> u32 {
    0
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn thread_idx_y() -> u32 {
    unsafe { core::arch::nvptx::_thread_idx_y() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn thread_idx_y() -> u32 {
    0
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn thread_idx_z() -> u32 {
    unsafe { core::arch::nvptx::_thread_idx_z() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn thread_idx_z() -> u32 {
    0
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn block_dim_x() -> u32 {
    unsafe { core::arch::nvptx::_block_dim_x() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn block_dim_x() -> u32 {
    1
}

#[cfg(target_arch = "nvptx64")]
#[inline(always)]
pub fn block_dim_y() -> u32 {
    unsafe { core::arch::nvptx::_block_dim_y() }
}

#[cfg(not(target_arch = "nvptx64"))]
#[inline(always)]
pub fn block_dim_y() -> u32 {
    1
}
