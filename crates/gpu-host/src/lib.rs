//! gpu-host library — host-side hostcall infrastructure.
//!
//! Re-exports the `hostcall` and `error` modules for use by external crates.

#![allow(clippy::needless_range_loop)]

pub mod error;
pub mod hostcall;
