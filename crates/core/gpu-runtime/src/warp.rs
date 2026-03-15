/// Butterfly reduction: sum of `val` across all 32 lanes.
/// Result is available in ALL lanes (convergent).
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn reduce_sum_f32(mut val: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: f32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            val += other;
            offset /= 2;
        }
    }
    val
}

/// Butterfly reduction: sum of `val` (u32) across all 32 lanes.
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn reduce_sum_u32(mut val: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: u32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            val += other;
            offset /= 2;
        }
    }
    val
}

/// Butterfly reduction: maximum of `val` across all 32 lanes.
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn reduce_max_f32(mut val: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: f32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            if other > val {
                val = other;
            }
            offset /= 2;
        }
    }
    val
}

/// Butterfly reduction: minimum of `val` across all 32 lanes.
///
/// # Safety
/// All 32 lanes must be active and call this function.
#[inline(always)]
#[allow(unused_mut)]
pub unsafe fn reduce_min_f32(mut val: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: f32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            if other < val {
                val = other;
            }
            offset /= 2;
        }
    }
    val
}

/// Butterfly shuffle: exchange `val` with lane at `(lane_id ^ offset)`.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn shfl_bfly_u32(mask: u32, val: u32, offset: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
            dst = out(reg32) result,
            src = in(reg32) val,
            off = in(reg32) offset,
            mask = in(reg32) mask,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (mask, offset);
        val
    }
}

/// Shuffle down: read `val` from `(lane_id + delta)`.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn shfl_down_u32(mask: u32, val: u32, delta: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "shfl.sync.down.b32 {dst}, {src}, {off}, 0x1f, {mask};",
            dst = out(reg32) result,
            src = in(reg32) val,
            off = in(reg32) delta,
            mask = in(reg32) mask,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (mask, delta);
        val
    }
}

/// Shuffle up: read `val` from `(lane_id - delta)`.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn shfl_up_u32(mask: u32, val: u32, delta: u32) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let result: u32;
        core::arch::asm!(
            "shfl.sync.up.b32 {dst}, {src}, {off}, 0, {mask};",
            dst = out(reg32) result,
            src = in(reg32) val,
            off = in(reg32) delta,
            mask = in(reg32) mask,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (mask, delta);
        val
    }
}

/// Warp vote: ballot — returns bitmask of lanes where `predicate` is true.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn ballot(mask: u32, predicate: bool) -> u32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let pred_u32 = predicate as u32;
        let result: u32;
        core::arch::asm!(
            "{{ .reg .pred %p; setp.ne.u32 %p, {pred}, 0; vote.sync.ballot.b32 {out}, %p, {mask}; }}",
            pred = in(reg32) pred_u32,
            out = out(reg32) result,
            mask = in(reg32) mask,
            options(nostack),
        );
        result
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (mask, predicate);
        0
    }
}

/// Warp vote: true if `predicate` is true for ALL lanes in `mask`.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn all(mask: u32, predicate: bool) -> bool {
    let b = ballot(mask, predicate);
    b == mask
}

/// Warp vote: true if `predicate` is true for ANY lane in `mask`.
///
/// # Safety
/// All lanes in `mask` must call this function.
#[inline(always)]
pub unsafe fn any(mask: u32, predicate: bool) -> bool {
    let b = ballot(mask, predicate);
    b != 0
}
