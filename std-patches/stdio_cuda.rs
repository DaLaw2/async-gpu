/// CUDA PAL stdio implementation: routes stdout/stderr through an external
/// write function provided by the GPU kernel crate.
///
/// Design: The PAL calls `gpu_stdout_write()`, an extern "Rust" function that
/// the kernel crate defines using its own hostcall infrastructure. This avoids
/// putting inline PTX asm or global state inside std, which would cause LLVM
/// NVPTX backend crashes from circular global variable dependencies.

use crate::io::{self, IoSlice, IoSliceMut, BorrowedCursor};

// External write function provided by the GPU kernel crate via Fat LTO.
// The kernel crate implements this using the hostcall PRINT service.
// Returns the number of bytes written, or 0 on failure.
unsafe extern "Rust" {
    fn gpu_stdout_write(buf: *const u8, len: usize) -> usize;
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
    #[inline]
    fn read(&mut self, _buf: &mut [u8]) -> io::Result<usize> {
        // TODO: implement via hostcall SERVICE_STDIN in std-pal.2
        Ok(0)
    }

    #[inline]
    fn read_buf(&mut self, _cursor: BorrowedCursor<'_>) -> io::Result<()> {
        Ok(())
    }

    #[inline]
    fn read_vectored(&mut self, _bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        Ok(0)
    }

    #[inline]
    fn is_read_vectored(&self) -> bool {
        false
    }

    #[inline]
    fn read_exact(&mut self, buf: &mut [u8]) -> io::Result<()> {
        if !buf.is_empty() { Err(io::Error::READ_EXACT_EOF) } else { Ok(()) }
    }

    #[inline]
    fn read_buf_exact(&mut self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        if cursor.capacity() != 0 { Err(io::Error::READ_EXACT_EOF) } else { Ok(()) }
    }

    #[inline]
    fn read_to_end(&mut self, _buf: &mut Vec<u8>) -> io::Result<usize> {
        Ok(0)
    }

    #[inline]
    fn read_to_string(&mut self, _buf: &mut String) -> io::Result<usize> {
        Ok(0)
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

pub const STDIN_BUF_SIZE: usize = 0;

pub fn is_ebadf(_err: &io::Error) -> bool {
    true
}

pub fn panic_output() -> Option<Vec<u8>> {
    None
}
