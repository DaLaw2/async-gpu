//! Par-Iter kernel — GPU parallel iterators in pure Rust.
//!
//! This kernel crate demonstrates the par_iter API from gpu-runtime.
//! Each function shows a different combinator chain:
//! - map, enumerate, zip, filter, fold (sum), collect_into
//!
//! All chains fuse at compile time via Rust monomorphization.
//! See crates/kernel/gpu-kernel-compute/src/par_iter_demo.rs for the
//! full set of kernel demos — this crate re-exports the API patterns
//! for standalone documentation purposes.

#![no_std]
#![feature(abi_gpu_kernel)]

// Note: The actual par_iter kernel functions are compiled into the
// unified kernel PTX (KERNEL_STD) from gpu-kernel-compute. This kernel
// crate exists to show the API patterns. The host side loads kernels
// from the pre-compiled PTX, not from this crate.
//
// See: crates/kernel/gpu-kernel-compute/src/par_iter_demo.rs
// for the actual kernel implementations.
