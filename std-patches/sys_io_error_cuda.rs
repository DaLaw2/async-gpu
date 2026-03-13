//! CUDA (nvptx64) error handling — reads errno from gpu-libc's __errno_location.

unsafe extern "C" {
    fn __errno_location() -> *mut i32;
}

pub fn errno() -> i32 {
    unsafe { *__errno_location() }
}

pub fn is_interrupted(errno: i32) -> bool {
    errno == 4 // EINTR
}

pub fn decode_error_kind(errno: i32) -> crate::io::ErrorKind {
    use crate::io::ErrorKind;
    match errno {
        1 => ErrorKind::PermissionDenied,    // EPERM
        2 => ErrorKind::NotFound,            // ENOENT
        5 => ErrorKind::Other,               // EIO
        9 => ErrorKind::Other,               // EBADF
        12 => ErrorKind::OutOfMemory,        // ENOMEM
        13 => ErrorKind::PermissionDenied,   // EACCES
        17 => ErrorKind::AlreadyExists,      // EEXIST
        22 => ErrorKind::InvalidInput,       // EINVAL
        38 => ErrorKind::Unsupported,        // ENOSYS
        _ => ErrorKind::Uncategorized,
    }
}

pub fn error_string(errno: i32) -> String {
    match errno {
        0 => "success".to_string(),
        1 => "operation not permitted".to_string(),
        2 => "no such file or directory".to_string(),
        5 => "input/output error".to_string(),
        9 => "bad file descriptor".to_string(),
        12 => "out of memory".to_string(),
        13 => "permission denied".to_string(),
        17 => "file exists".to_string(),
        22 => "invalid argument".to_string(),
        38 => "function not implemented".to_string(),
        _ => format!("unknown error {errno}"),
    }
}
