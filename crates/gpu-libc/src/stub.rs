//! Stub implementations for libc functions that are not supported on GPU.
//!
//! These return ENOSYS (Function not implemented) or abort.
//! They exist solely to satisfy linker requirements when compiling
//! Rust std for GPU.

use crate::errno::*;
use crate::types::*;

// ============================================================
// Process control stubs
// ============================================================

/// Abort the GPU thread. Uses PTX trap instruction.
#[no_mangle]
pub unsafe extern "C" fn abort() -> ! {
    core::arch::asm!("trap;", options(noreturn));
}

/// Exit — not meaningful on GPU. Abort instead.
#[no_mangle]
pub unsafe extern "C" fn exit(_status: c_int) -> ! {
    abort();
}

/// _exit — same as exit on GPU.
#[no_mangle]
pub unsafe extern "C" fn _exit(_status: c_int) -> ! {
    abort();
}

// ============================================================
// I/O stubs (remaining stubs — open/write/read/close moved to hostcall_io.rs)
// ============================================================

/// Seek in a file. Stub returns ENOSYS.
#[no_mangle]
pub unsafe extern "C" fn lseek(_fd: c_int, _offset: off_t, _whence: c_int) -> off_t {
    set_errno(ENOSYS);
    -1
}

/// Get file status. Stub returns ENOSYS.
#[no_mangle]
pub unsafe extern "C" fn fstat(_fd: c_int, _buf: *mut c_void) -> c_int {
    set_errno(ENOSYS);
    -1
}

/// Get file status by path. Stub returns ENOSYS.
#[no_mangle]
pub unsafe extern "C" fn stat(_path: *const c_char, _buf: *mut c_void) -> c_int {
    set_errno(ENOSYS);
    -1
}

/// File control. Stub returns ENOSYS.
#[no_mangle]
pub unsafe extern "C" fn fcntl(_fd: c_int, _cmd: c_int) -> c_int {
    set_errno(ENOSYS);
    -1
}

/// ioctl. Stub returns ENOSYS.
#[no_mangle]
pub unsafe extern "C" fn ioctl(_fd: c_int, _request: c_ulong) -> c_int {
    set_errno(ENOSYS);
    -1
}

// ============================================================
// Threading stubs (not supported on GPU)
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn pthread_create(
    _thread: *mut c_void,
    _attr: *const c_void,
    _start: *const c_void,
    _arg: *mut c_void,
) -> c_int {
    ENOSYS
}

#[no_mangle]
pub unsafe extern "C" fn pthread_join(_thread: c_ulong, _retval: *mut *mut c_void) -> c_int {
    ENOSYS
}

#[no_mangle]
pub unsafe extern "C" fn pthread_detach(_thread: c_ulong) -> c_int {
    ENOSYS
}

#[no_mangle]
pub unsafe extern "C" fn pthread_self() -> c_ulong {
    0
}

#[no_mangle]
pub unsafe extern "C" fn pthread_attr_init(_attr: *mut c_void) -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn pthread_attr_destroy(_attr: *mut c_void) -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn pthread_attr_setstacksize(_attr: *mut c_void, _size: size_t) -> c_int {
    0
}

#[no_mangle]
pub unsafe extern "C" fn sched_yield() -> c_int {
    // On GPU, we can use nanosleep as a yield hint
    0
}

// ============================================================
// Signal stubs
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn signal(_signum: c_int, _handler: *const c_void) -> *mut c_void {
    core::ptr::null_mut() // SIG_DFL
}

#[no_mangle]
pub unsafe extern "C" fn sigaction(
    _signum: c_int,
    _act: *const c_void,
    _oldact: *mut c_void,
) -> c_int {
    set_errno(ENOSYS);
    -1
}

// ============================================================
// Process stubs
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn fork() -> pid_t {
    set_errno(ENOSYS);
    -1
}

#[no_mangle]
pub unsafe extern "C" fn getpid() -> pid_t {
    1 // Return a fake PID
}

// ============================================================
// Environment stubs
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn getenv(_name: *const c_char) -> *mut c_char {
    core::ptr::null_mut()
}

#[no_mangle]
pub unsafe extern "C" fn getcwd(_buf: *mut c_char, _size: size_t) -> *mut c_char {
    set_errno(ENOSYS);
    core::ptr::null_mut()
}

// ============================================================
// Time stubs
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn clock_gettime(_clk_id: clockid_t, _tp: *mut c_void) -> c_int {
    set_errno(ENOSYS);
    -1
}

#[no_mangle]
pub unsafe extern "C" fn nanosleep(_req: *const c_void, _rem: *mut c_void) -> c_int {
    // Could use PTX nanosleep here
    0
}

// ============================================================
// Synchronization stubs (futex-based sync in std)
// ============================================================

// Linux futex syscall — std uses this for Mutex/Condvar on Linux.
// On GPU, we'd use spinlocks from gpu-atomics instead.
#[no_mangle]
pub unsafe extern "C" fn syscall(_number: c_long, _args: ...) -> c_long {
    set_errno(ENOSYS);
    -1
}

// ============================================================
// Miscellaneous stubs
// ============================================================

#[no_mangle]
pub unsafe extern "C" fn sysconf(_name: c_int) -> c_long {
    set_errno(ENOSYS);
    -1
}

#[no_mangle]
pub unsafe extern "C" fn getpwuid_r(
    _uid: uid_t,
    _pwd: *mut c_void,
    _buf: *mut c_char,
    _buflen: size_t,
    _result: *mut *mut c_void,
) -> c_int {
    ENOSYS
}

#[no_mangle]
pub unsafe extern "C" fn poll(_fds: *mut c_void, _nfds: c_ulong, _timeout: c_int) -> c_int {
    set_errno(ENOSYS);
    -1
}

#[no_mangle]
pub unsafe extern "C" fn pipe(_pipefd: *mut c_int) -> c_int {
    set_errno(ENOSYS);
    -1
}

#[no_mangle]
pub unsafe extern "C" fn dup(_oldfd: c_int) -> c_int {
    set_errno(ENOSYS);
    -1
}
