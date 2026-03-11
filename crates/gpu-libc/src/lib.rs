//! GPU libc shim layer for nvptx64.
//!
//! Provides `#[no_mangle] extern "C"` functions matching the libc API surface
//! required by Rust std. Functions are categorized into three tiers:
//!
//! - **Device-side**: Implemented directly on GPU (memcpy, memset, malloc, etc.)
//! - **Hostcall**: Requires GPU→host RPC (write, read, open, close)
//! - **Stub**: Returns ENOSYS or panics (threading, networking, process, etc.)
//!
//! This crate is `#![no_std]` and compiles for `nvptx64-nvidia-cuda`.

#![no_std]
#![feature(asm_experimental_arch)]
#![feature(c_variadic)]
#![allow(non_camel_case_types)]

mod types;
pub mod memory;
pub mod string;
pub mod stub;
mod errno;

pub use types::*;
pub use errno::*;

// Re-export key functions for direct use by GPU kernels
pub use memory::{malloc, calloc, free, realloc, posix_memalign, gpu_heap_init};
pub use memory::{memcpy, memset, memcmp, memmove};
pub use string::{strlen, strcmp, strncmp};
pub use stub::{abort, write, read, open, close};
