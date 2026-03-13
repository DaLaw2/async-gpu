//! gpu-host — host-side GPU runtime library.
//!
//! Provides a high-level SDK for GPU kernel management, hostcall communication,
//! and pinned memory allocation.
//!
//! # Modules
//! - [`runtime`] — `GpuRuntime` for device init, PTX loading, kernel launch
//! - [`memory`] — `MappedBuffer<T>` for RAII pinned device-mapped memory
//! - [`hostcall`] — `HostcallBuffer` for GPU-host RPC communication
//! - [`error`] — Error types (`GpuHostError`, `HostcallError`)

#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod hostcall;
pub mod memory;
pub mod runtime;
