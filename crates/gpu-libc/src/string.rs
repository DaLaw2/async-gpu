//! String functions (strlen, etc.) — device-side implementations.

use crate::types::*;

/// Return the length of a null-terminated C string.
#[no_mangle]
pub unsafe extern "C" fn strlen(s: *const c_char) -> size_t {
    let mut len: size_t = 0;
    while *s.add(len) != 0 {
        len += 1;
    }
    len
}

/// Compare two null-terminated strings.
/// Returns 0 if equal, negative if s1 < s2, positive if s1 > s2.
#[no_mangle]
pub unsafe extern "C" fn strcmp(s1: *const c_char, s2: *const c_char) -> c_int {
    let mut i: usize = 0;
    loop {
        let c1 = *s1.add(i) as c_uchar;
        let c2 = *s2.add(i) as c_uchar;
        if c1 != c2 {
            return (c1 as c_int) - (c2 as c_int);
        }
        if c1 == 0 {
            return 0;
        }
        i += 1;
    }
}

/// Compare at most `n` bytes of two strings.
#[no_mangle]
pub unsafe extern "C" fn strncmp(s1: *const c_char, s2: *const c_char, n: size_t) -> c_int {
    let mut i: usize = 0;
    while i < n {
        let c1 = *s1.add(i) as c_uchar;
        let c2 = *s2.add(i) as c_uchar;
        if c1 != c2 {
            return (c1 as c_int) - (c2 as c_int);
        }
        if c1 == 0 {
            return 0;
        }
        i += 1;
    }
    0
}
