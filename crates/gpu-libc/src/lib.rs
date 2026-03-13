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

mod errno;
pub mod hostcall_io;
pub mod memory;
pub mod string;
pub mod stub;
mod types;

pub use errno::*;
pub use types::*;

// Re-export key functions for direct use by GPU kernels
pub use hostcall_io::{close, gpu_libc_io_init, open, read, write};
pub use memory::{calloc, free, gpu_heap_init, malloc, posix_memalign, realloc};
pub use memory::{memcmp, memcpy, memmove, memset};
pub use string::{strcmp, strlen, strncmp};
pub use stub::abort;
