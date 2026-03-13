// CUDA PAL: stdio routing for nvptx64 GPU targets.
//
// Routes stdout/stdin through extern functions provided by the GPU kernel crate.
// These extern functions bridge to the hostcall protocol for host-side I/O.

use crate::io;

// Extern functions provided by the kernel crate (e.g., gpu-kernel-std).
// These must be #[no_mangle] in the kernel crate.
unsafe extern "C" {
    fn gpu_stdout_write(buf: *const u8, len: usize) -> usize;
    fn gpu_stdin_read(buf: *mut u8, max_len: usize) -> usize;
}

pub struct Stdin;
pub struct Stdout;
pub struct Stderr;

impl Stdin {
    pub const fn new() -> Stdin {
        Stdin
    }
}

impl io::Read for Stdin {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let n = unsafe { gpu_stdin_read(buf.as_mut_ptr(), buf.len()) };
        Ok(n)
    }
}

impl Stdout {
    pub const fn new() -> Stdout {
        Stdout
    }
}

impl io::Write for Stdout {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let n = unsafe { gpu_stdout_write(buf.as_ptr(), buf.len()) };
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

impl Stderr {
    pub const fn new() -> Stderr {
        Stderr
    }
}

impl io::Write for Stderr {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        // Route stderr through stdout on GPU (single output channel).
        let n = unsafe { gpu_stdout_write(buf.as_ptr(), buf.len()) };
        Ok(n)
    }

    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}

// BufReader capacity for stdin. Must be non-zero or BufReader::fill_buf()
// reads into an empty slice and immediately returns EOF.
// 128 bytes is reasonable — hostcall transfers max 56 bytes per request,
// so BufReader will issue multiple underlying reads for longer lines.
pub const STDIN_BUF_SIZE: usize = 128;

pub fn is_ebadf(_err: &io::Error) -> bool {
    false
}

pub fn panic_output() -> Option<impl io::Write> {
    Some(Stderr::new())
}
