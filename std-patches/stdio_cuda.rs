/// CUDA PAL stdio implementation: routes stdout/stderr through an external
/// write function provided by the GPU kernel crate.
///
/// Design: The PAL calls `gpu_stdout_write()`, an extern "Rust" function that
/// the kernel crate defines using its own hostcall infrastructure. This avoids
/// putting inline PTX asm or global state inside std, which would cause LLVM
/// NVPTX backend crashes from circular global variable dependencies.

use crate::io::{self, IoSlice, IoSliceMut, BorrowedCursor};

// External I/O functions provided by the GPU kernel crate via Fat LTO.
// The kernel crate implements these using the hostcall PRINT/STDIN services.
unsafe extern "Rust" {
    /// Write bytes to stdout. Returns number of bytes written.
    fn gpu_stdout_write(buf: *const u8, len: usize) -> usize;
    /// Read bytes from stdin into buf. Returns number of bytes read, or 0 on EOF/error.
    fn gpu_stdin_read(buf: *mut u8, max_len: usize) -> usize;
}

pub struct Stdin;
pub struct Stdout;
pub type Stderr = Stdout;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Call the external read function provided by the kernel crate.
        // Safety: gpu_stdin_read is linked via Fat LTO from the kernel crate.
        let n = unsafe { gpu_stdin_read(buf.as_mut_ptr(), buf.len()) };
        Ok(n)
    }

    fn read_vectored(&mut self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // Read into the first non-empty buffer.
        for buf in bufs {
            if !buf.is_empty() {
                return self.read(buf);
            }
        }
        Ok(0)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        false
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        // Call the external write function provided by the kernel crate.
        // Safety: gpu_stdout_write is linked via Fat LTO from the kernel crate.
        let written = unsafe { gpu_stdout_write(buf.as_ptr(), buf.len()) };
        Ok(written)
    }

    fn write_vectored(&mut self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        let mut total = 0;
        for buf in bufs {
            total += self.write(buf)?;
        }
        Ok(total)
    }

    #[inline]
    fn is_write_vectored(&self) -> bool {
        false
    }

    #[inline]
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

pub const STDIN_BUF_SIZE: usize = 56; // Matches STDIN_MAX_READ_LEN from hostcall protocol

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<Vec<u8>> {
    None
}
