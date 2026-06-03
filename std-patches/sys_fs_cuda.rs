//! CUDA (nvptx64) filesystem PAL — routes through gpu-libc's hostcall I/O.
//!
//! Supports: open, read, write, close via hostcall protocol.
//! Unsupported: stat, readdir, symlink, permissions, timestamps.

use crate::ffi::{CStr, OsString};
use crate::fmt;
use crate::fs::TryLockError;
use crate::io::{self, BorrowedCursor, IoSlice, IoSliceMut, SeekFrom};
use crate::path::{Path, PathBuf};
pub use crate::sys::fs::common::Dir;
use crate::sys::helpers::run_path_with_cstr;
use crate::sys::unsupported;

// libc functions provided by gpu-libc's hostcall_io module
unsafe extern "C" {
    fn open(pathname: *const i8, flags: i32, mode: u32) -> i32;
    fn read(fd: i32, buf: *mut u8, count: usize) -> isize;
    fn write(fd: i32, buf: *const u8, count: usize) -> isize;
    fn close(fd: i32) -> i32;
}

// Open flags (must match gpu-libc/src/types.rs)
const O_RDONLY: i32 = 0;
const O_WRONLY: i32 = 1;
const O_RDWR: i32 = 2;
const O_CREAT: i32 = 0o100;
const O_EXCL: i32 = 0o200;
const O_TRUNC: i32 = 0o1000;
const O_APPEND: i32 = 0o2000;

fn last_os_error() -> io::Error {
    io::Error::from_raw_os_error(crate::sys::io::errno())
}

pub struct File {
    fd: i32,
}

#[derive(Clone, Debug)]
pub struct OpenOptions {
    read: bool,
    write: bool,
    append: bool,
    truncate: bool,
    create: bool,
    create_new: bool,
}

// Stub types — not supported on GPU but required by the PAL interface.

pub struct FileAttr {
    len: u64,
}

pub struct ReadDir(!);

pub struct DirEntry(!);

#[derive(Copy, Clone, Debug, Default)]
pub struct FileTimes {}

pub struct FilePermissions {
    readonly: bool,
}

pub struct FileType {
    is_file: bool,
}

#[derive(Debug)]
pub struct DirBuilder {}

// --- OpenOptions ---

impl OpenOptions {
    pub fn new() -> OpenOptions {
        OpenOptions {
            read: false,
            write: false,
            append: false,
            truncate: false,
            create: false,
            create_new: false,
        }
    }

    pub fn read(&mut self, read: bool) { self.read = read; }
    pub fn write(&mut self, write: bool) { self.write = write; }
    pub fn append(&mut self, append: bool) { self.append = append; }
    pub fn truncate(&mut self, truncate: bool) { self.truncate = truncate; }
    pub fn create(&mut self, create: bool) { self.create = create; }
    pub fn create_new(&mut self, create_new: bool) { self.create_new = create_new; }

    fn to_libc_flags(&self) -> i32 {
        let mut flags = if self.read && self.write {
            O_RDWR
        } else if self.write || self.append {
            O_WRONLY
        } else {
            O_RDONLY
        };

        if self.append {
            flags |= O_APPEND;
        }
        if self.create {
            flags |= O_CREAT;
        }
        if self.truncate {
            flags |= O_TRUNC;
        }
        if self.create_new {
            flags |= O_CREAT | O_EXCL;
        }

        flags
    }
}

// --- File ---

impl File {
    pub fn open(path: &Path, opts: &OpenOptions) -> io::Result<File> {
        run_path_with_cstr(path, &|cstr: &CStr| {
            let flags = opts.to_libc_flags();
            let mode: u32 = 0o666; // default permissions
            let fd = unsafe { open(cstr.as_ptr() as *const i8, flags, mode) };
            if fd < 0 {
                Err(last_os_error())
            } else {
                Ok(File { fd })
            }
        })
    }

    pub fn file_attr(&self) -> io::Result<FileAttr> {
        // stat not supported — return a minimal FileAttr
        unsupported()
    }

    pub fn fsync(&self) -> io::Result<()> {
        Ok(()) // no-op on GPU
    }

    pub fn datasync(&self) -> io::Result<()> {
        Ok(()) // no-op on GPU
    }

    pub fn lock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn lock_shared(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn try_lock(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(crate::sys::unsupported_err()))
    }

    pub fn try_lock_shared(&self) -> Result<(), TryLockError> {
        Err(TryLockError::Error(crate::sys::unsupported_err()))
    }

    pub fn unlock(&self) -> io::Result<()> {
        unsupported()
    }

    pub fn truncate(&self, _size: u64) -> io::Result<()> {
        unsupported()
    }

    pub fn read(&self, buf: &mut [u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let ret = unsafe { read(self.fd, buf.as_mut_ptr(), buf.len()) };
        if ret < 0 {
            Err(last_os_error())
        } else {
            Ok(ret as usize)
        }
    }

    pub fn read_vectored(&self, bufs: &mut [IoSliceMut<'_>]) -> io::Result<usize> {
        // Simple: read into first non-empty buffer
        for buf in bufs {
            if !buf.is_empty() {
                return self.read(buf);
            }
        }
        Ok(0)
    }

    pub fn is_read_vectored(&self) -> bool {
        false
    }

    pub fn read_buf(&self, cursor: BorrowedCursor<'_>) -> io::Result<()> {
        crate::io::default_read_buf(|buf| self.read(buf), cursor)
    }

    pub fn write(&self, buf: &[u8]) -> io::Result<usize> {
        if buf.is_empty() {
            return Ok(0);
        }
        let ret = unsafe { write(self.fd, buf.as_ptr(), buf.len()) };
        if ret < 0 {
            Err(last_os_error())
        } else {
            Ok(ret as usize)
        }
    }

    pub fn write_vectored(&self, bufs: &[IoSlice<'_>]) -> io::Result<usize> {
        // Simple: write first non-empty buffer
        for buf in bufs {
            if !buf.is_empty() {
                return self.write(buf);
            }
        }
        Ok(0)
    }

    pub fn is_write_vectored(&self) -> bool {
        false
    }

    pub fn flush(&self) -> io::Result<()> {
        Ok(()) // no buffering at PAL level
    }

    pub fn seek(&self, _pos: SeekFrom) -> io::Result<u64> {
        unsupported() // lseek not implemented
    }

    pub fn size(&self) -> Option<io::Result<u64>> {
        None // stat not available
    }

    pub fn tell(&self) -> io::Result<u64> {
        unsupported()
    }

    pub fn duplicate(&self) -> io::Result<File> {
        unsupported()
    }

    pub fn set_permissions(&self, _perm: FilePermissions) -> io::Result<()> {
        unsupported()
    }

    pub fn set_times(&self, _times: FileTimes) -> io::Result<()> {
        unsupported()
    }
}

impl Drop for File {
    fn drop(&mut self) {
        unsafe { close(self.fd); }
    }
}

impl fmt::Debug for File {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("File").field("fd", &self.fd).finish()
    }
}

// --- FileAttr ---

impl FileAttr {
    pub fn size(&self) -> u64 { self.len }

    pub fn perm(&self) -> FilePermissions {
        FilePermissions { readonly: false }
    }

    pub fn file_type(&self) -> FileType {
        FileType { is_file: true }
    }

    pub fn modified(&self) -> io::Result<crate::sys::time::SystemTime> {
        unsupported()
    }

    pub fn accessed(&self) -> io::Result<crate::sys::time::SystemTime> {
        unsupported()
    }

    pub fn created(&self) -> io::Result<crate::sys::time::SystemTime> {
        unsupported()
    }
}

impl Clone for FileAttr {
    fn clone(&self) -> FileAttr {
        FileAttr { len: self.len }
    }
}

// --- FilePermissions ---

impl FilePermissions {
    pub fn readonly(&self) -> bool { self.readonly }
    pub fn set_readonly(&mut self, readonly: bool) { self.readonly = readonly; }
}

impl Clone for FilePermissions {
    fn clone(&self) -> FilePermissions {
        FilePermissions { readonly: self.readonly }
    }
}

impl PartialEq for FilePermissions {
    fn eq(&self, other: &FilePermissions) -> bool {
        self.readonly == other.readonly
    }
}

impl Eq for FilePermissions {}

impl fmt::Debug for FilePermissions {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FilePermissions").field("readonly", &self.readonly).finish()
    }
}

// --- FileTimes ---

impl FileTimes {
    pub fn set_accessed(&mut self, _t: crate::sys::time::SystemTime) {}
    pub fn set_modified(&mut self, _t: crate::sys::time::SystemTime) {}
}

// --- FileType ---

impl FileType {
    pub fn is_dir(&self) -> bool { !self.is_file }
    pub fn is_file(&self) -> bool { self.is_file }
    pub fn is_symlink(&self) -> bool { false }
}

impl Clone for FileType {
    fn clone(&self) -> FileType { FileType { is_file: self.is_file } }
}

impl Copy for FileType {}

impl PartialEq for FileType {
    fn eq(&self, other: &FileType) -> bool { self.is_file == other.is_file }
}

impl Eq for FileType {}

impl core::hash::Hash for FileType {
    fn hash<H: core::hash::Hasher>(&self, h: &mut H) {
        self.is_file.hash(h);
    }
}

impl fmt::Debug for FileType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FileType").field("is_file", &self.is_file).finish()
    }
}

// --- ReadDir ---

impl fmt::Debug for ReadDir {
    fn fmt(&self, _f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.0
    }
}

impl Iterator for ReadDir {
    type Item = io::Result<DirEntry>;
    fn next(&mut self) -> Option<io::Result<DirEntry>> {
        self.0
    }
}

// --- DirEntry ---

impl DirEntry {
    pub fn path(&self) -> PathBuf { self.0 }
    pub fn file_name(&self) -> OsString { self.0 }
    pub fn metadata(&self) -> io::Result<FileAttr> { self.0 }
    pub fn file_type(&self) -> io::Result<FileType> { self.0 }
}

// --- DirBuilder ---

impl DirBuilder {
    pub fn new() -> DirBuilder { DirBuilder {} }
    pub fn mkdir(&self, _p: &Path) -> io::Result<()> { unsupported() }
}

// --- Free functions ---

pub fn readdir(_p: &Path) -> io::Result<ReadDir> { unsupported() }
pub fn unlink(_p: &Path) -> io::Result<()> { unsupported() }
pub fn rename(_old: &Path, _new: &Path) -> io::Result<()> { unsupported() }
pub fn set_perm(_p: &Path, _perm: FilePermissions) -> io::Result<()> { unsupported() }
pub fn set_times(_p: &Path, _times: FileTimes) -> io::Result<()> { unsupported() }
pub fn set_times_nofollow(_p: &Path, _times: FileTimes) -> io::Result<()> { unsupported() }
pub fn rmdir(_p: &Path) -> io::Result<()> { unsupported() }
pub fn remove_dir_all(_path: &Path) -> io::Result<()> { unsupported() }
pub fn exists(_path: &Path) -> io::Result<bool> { unsupported() }
pub fn readlink(_p: &Path) -> io::Result<PathBuf> { unsupported() }
pub fn symlink(_original: &Path, _link: &Path) -> io::Result<()> { unsupported() }
pub fn link(_src: &Path, _dst: &Path) -> io::Result<()> { unsupported() }
pub fn stat(_p: &Path) -> io::Result<FileAttr> { unsupported() }
pub fn lstat(_p: &Path) -> io::Result<FileAttr> { unsupported() }
pub fn canonicalize(_p: &Path) -> io::Result<PathBuf> { unsupported() }
pub fn copy(_from: &Path, _to: &Path) -> io::Result<u64> { unsupported() }
