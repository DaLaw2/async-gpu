//! C type aliases for the GPU libc shim.
//!
//! These match the standard libc type definitions for a 64-bit target.

pub type c_void = core::ffi::c_void;
pub type c_char = i8;
pub type c_uchar = u8;
pub type c_short = i16;
pub type c_ushort = u16;
pub type c_int = i32;
pub type c_uint = u32;
pub type c_long = i64; // LP64 model (matches nvptx64)
pub type c_ulong = u64;
pub type c_longlong = i64;
pub type c_ulonglong = u64;

pub type size_t = usize;
pub type ssize_t = isize;
pub type off_t = i64;
pub type mode_t = u32;
pub type pid_t = i32;
pub type uid_t = u32;
pub type gid_t = u32;
pub type clockid_t = i32;
pub type time_t = i64;

/// File descriptor type (just an integer — actual fds live on the host).
pub type fd_t = c_int;

// File open flags (matching Linux definitions)
pub const O_RDONLY: c_int = 0;
pub const O_WRONLY: c_int = 1;
pub const O_RDWR: c_int = 2;
pub const O_CREAT: c_int = 0o100;
pub const O_EXCL: c_int = 0o200;
pub const O_TRUNC: c_int = 0o1000;
pub const O_APPEND: c_int = 0o2000;

// Standard file descriptors
pub const STDIN_FILENO: c_int = 0;
pub const STDOUT_FILENO: c_int = 1;
pub const STDERR_FILENO: c_int = 2;

// Seek whence
pub const SEEK_SET: c_int = 0;
pub const SEEK_CUR: c_int = 1;
pub const SEEK_END: c_int = 2;

// Minimum alignment for malloc (CUDA guarantees 256-byte alignment for
// device global memory, but 16 is sufficient for std's MIN_ALIGN).
pub const MALLOC_ALIGN: usize = 16;
