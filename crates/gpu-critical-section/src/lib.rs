//! No-op critical-section implementation for GPU (nvptx64).
//!
//! On GPU, each thread (lane) runs its own independent executor instance.
//! There is no preemption and no shared mutable state between lanes for
//! executor internals, so critical sections are a no-op.
//!
//! This implementation satisfies Embassy's `critical-section` dependency
//! for the per-thread executor case (ADR-4).
//!
//! # Safety
//!
//! This is safe ONLY when:
//! - Each GPU thread has its own executor (no cross-thread sharing)
//! - The critical section only protects executor-internal state
//! - No ISR/preemption exists (true for GPU compute kernels)
//!
//! For multi-warp shared state, use `gpu-atomics` directly instead.

#![no_std]

use critical_section::RawRestoreState;

struct GpuCriticalSection;
critical_section::set_impl!(GpuCriticalSection);

unsafe impl critical_section::Impl for GpuCriticalSection {
    unsafe fn acquire() -> RawRestoreState {
        // No-op: GPU threads cannot be preempted.
        // Per-thread executor has no shared mutable state.
        // Returns unit () as RawRestoreState (nothing to restore).
    }

    unsafe fn release(_restore_state: RawRestoreState) {
        // No-op: nothing to restore.
    }
}
