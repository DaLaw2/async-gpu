# gpu-std.1: Analyze Rust std's dependency graph on libc
**Cycle**: 16 | **Theme**: gpu-std | **Kind**: investigation | **Status**: done

## Summary

Rust std's platform abstraction layer (PAL) routes all OS interactions through `sys/`
modules that call libc functions. For our GPU target, the critical call chains for
println!, File I/O, and memory allocation converge on a small set of ~15 must-implement
libc functions. The full libc surface used by std on unix is ~120+ functions, but a
minimal facade for our three use cases needs only: write, read, open, close, malloc,
free, realloc, plus a handful of supporting functions.

## Findings

### Q: Which std modules directly call into libc?

A: Rust std's `sys/` directory contains all platform-specific code. The modules that
make direct `libc::` calls are:

| Module path | Category | Key libc functions |
|---|---|---|
| `sys/fd/unix.rs` | File descriptors | read, readv, write, writev, pread64, pwrite64, ioctl, fcntl |
| `sys/fs/unix.rs` | Filesystem | open64, stat64, lstat64, fstat64, lseek64, chmod, chown, mkdir, rmdir, unlink, rename, symlink, readlink, realpath, truncate64, fsync, fdatasync, opendir, readdir64, closedir, link, flock |
| `sys/alloc/unix.rs` | Memory allocation | malloc, calloc, free, realloc, posix_memalign (or memalign) |
| `sys/stdio/unix.rs` | Standard I/O | Uses FileDesc with STDOUT_FILENO/STDIN_FILENO/STDERR_FILENO constants |
| `sys/thread/unix.rs` | Threading | pthread_create, pthread_join, pthread_detach, pthread_attr_init/destroy/setstacksize, pthread_self, nanosleep, sched_yield |
| `sys/pal/unix/mod.rs` | PAL init | abort, poll, open, dup, signal, fcntl |
| `sys/pal/unix/os.rs` | OS queries | getcwd, chdir, getpid, getpwuid_r, sysconf |
| `sys/pal/unix/time.rs` | Time | clock_gettime |
| `sys/env/unix.rs` | Env vars | getenv, setenv, unsetenv |
| `sys/sync/mutex/` | Synchronization | futex_wait/futex_wake (Linux) or pthread_mutex_* |
| `sys/process/unix/` | Process mgmt | fork, exec*, waitpid, pipe, exit |

**Confidence**: high (verified against rust-lang/rust master source)

### Q: What libc functions does a minimal facade need to implement?

A: Prioritized into three tiers based on our use cases (println!, File I/O, allocation):

**Tier 1 — MUST implement (critical path, blocks all use cases):**

| Function | Used by | Implementation |
|---|---|---|
| `write(fd, buf, len)` | println!, File write | hostcall → host write() |
| `read(fd, buf, len)` | File read, stdin | hostcall → host read() |
| `open(path, flags, mode)` | File::open/create | hostcall → host open() |
| `close(fd)` | File Drop | hostcall → host close() |
| `malloc(size)` | GlobalAlloc::alloc | GPU heap allocator (device-side) |
| `free(ptr)` | GlobalAlloc::dealloc | GPU heap allocator (device-side) |
| `realloc(ptr, size)` | GlobalAlloc::realloc | GPU heap allocator (device-side) |
| `abort()` | panic handler | trap instruction / device-side halt |
| `memcpy` | compiler builtins | device-side implementation |
| `memset` | compiler builtins | device-side implementation |
| `memcmp` | compiler builtins | device-side implementation |
| `strlen` | CStr operations | device-side implementation |

**Tier 2 — SHOULD implement (needed for robust std usage):**

| Function | Used by | Implementation |
|---|---|---|
| `lseek(fd, off, whence)` | File seek | hostcall → host lseek() |
| `fstat(fd, buf)` | File::metadata | hostcall → host fstat() |
| `stat(path, buf)` | fs::metadata | hostcall → host stat() |
| `posix_memalign` | aligned alloc | GPU heap with alignment support |
| `fcntl(fd, cmd, ...)` | fd flags | hostcall → host fcntl() |
| `getcwd(buf, size)` | env::current_dir | hostcall → host getcwd() |
| `clock_gettime` | SystemTime, Instant | hostcall → host OR PTX %globaltimer |
| `getenv(name)` | env::var | hostcall → host getenv() |
| `errno` (thread-local) | Error handling | GPU thread-local variable |

**Tier 3 — STUB with error (not needed for MVP):**

All remaining functions (threading, process, signals, networking, etc.) can return
`ENOSYS` or panic. These include: pthread_*, fork, exec, pipe, socket, signal, etc.

**Minimum viable count: ~12 real implementations + ~8 useful additions + stubs for everything else.**

**Confidence**: high

### Q: What is the full call chain for println!?

A: The complete call chain from `println!("Hello")` to the libc level:

```
println!("Hello")
  → format_args!("Hello")
  → std::io::_print(args: Arguments)
    → print_to(args, stdout_used: &AtomicBool)
      → stdout().write_fmt(args)  // or falls back to stderr on failure
        → Stdout::lock()
          → ReentrantLock::lock()  // needs futex or atomic-based sync
        → StdoutLock::write_fmt(args)
          → <LineWriter<StdoutRaw> as Write>::write_fmt(args)
            → core::fmt::write(&mut self, args)  // formats into buffer
              → <T as Display>::fmt() for each argument
              → Writer.write_str(s) → Writer.write_all(s.as_bytes())
                → LineWriter::write(buf)  // buffers until newline
                  → BufWriter::write(buf) // when newline found, flush
                    → StdoutRaw::write(buf)
                      → handle_ebadf(self.0.write(buf))
                        → FileDesc::write(buf)
                          ★ libc::write(STDOUT_FILENO, buf.as_ptr(), buf.len())
```

**Key libc functions needed for println!:**
1. `write(fd=1, buf, len)` — the actual output
2. `malloc/free/realloc` — String formatting may allocate (for Display impls, though simple &str doesn't)
3. Synchronization primitive for `ReentrantLock` — futex or atomic spinlock
4. `abort()` — if panic occurs in the print path

**For our GPU implementation:**
- The hostcall mechanism already handles print (hostcall.4 verified this)
- The libc shim needs to route `write(1, ...)` to a hostcall PRINT request
- The formatting happens entirely device-side (core::fmt is no_std compatible)
- The lock can use our gpu-atomics spinlock (no OS futex needed)

**Confidence**: high

### Q: What is the full call chain for File I/O?

A: For `File::create("test.txt")` followed by `write!(file, "data")`:

```
File::create("test.txt")
  → OpenOptions::new().write(true).create(true).truncate(true).open("test.txt")
    → OpenOptions::_open(path)
      → sys::fs::File::open(path, opts)
        → run_path_with_cstr(path, |p| ...)
          → cvt_r(|| libc::open64(p.as_ptr(), flags, mode))
            ★ libc::open64(path, O_WRONLY|O_CREAT|O_TRUNC, 0o666)
          → FileDesc::from_raw_fd(fd)
          → if O_CLOEXEC not available: libc::fcntl(fd, F_SETFD, FD_CLOEXEC)

write!(file, "data")
  → <File as Write>::write(buf)
    → self.inner.write(buf)
      → FileDesc::write(buf)
        ★ libc::write(fd, buf.as_ptr(), buf.len())

drop(file)
  → FileDesc::drop()
    ★ libc::close(fd)

File::open("test.txt") for reading:
  → same open chain with O_RDONLY
  → <File as Read>::read(buf)
    → FileDesc::read(buf)
      ★ libc::read(fd, buf.as_mut_ptr(), buf.len())
```

**Key libc functions for File I/O:**
1. `open` / `open64` (path, flags, mode) → returns fd
2. `read(fd, buf, len)` → returns bytes read
3. `write(fd, buf, len)` → returns bytes written
4. `close(fd)` → cleanup
5. `lseek` / `lseek64` → for seek operations
6. `fstat` / `fstat64` → for metadata()
7. `fcntl` → for fd flags (can be stubbed initially)

**For our GPU implementation:**
- Each of these becomes a hostcall request
- The host opens/reads/writes real files and returns results
- File descriptors are host-side; GPU just stores the integer fd
- Need to serialize path strings in the hostcall payload
- Can reuse the existing two-stack hostcall protocol with new opcode types

**Confidence**: high

### Q: What is the call chain for memory allocation (GlobalAlloc)?

A: Rust's allocator goes through the `#[global_allocator]` which defaults to `System`:

```
Box::new(value) / Vec::push() / String::from() / etc.
  → alloc::alloc::Global.allocate(layout)
    → __rust_alloc(size, align)
      → <System as GlobalAlloc>::alloc(layout)
        → if align <= MIN_ALIGN && align <= size:
            ★ libc::malloc(size)
          → else:
            → aligned_malloc(layout)
              ★ libc::posix_memalign(&mut out, align, size)

alloc_zeroed:
  → if simple alignment:
      ★ libc::calloc(size, 1)
    → else: alloc() + write_bytes(0)

dealloc:
  ★ libc::free(ptr)

realloc:
  → if simple alignment:
      ★ libc::realloc(ptr, new_size)
    → else: alloc new + copy + dealloc old
```

**Key libc functions for allocation:**
1. `malloc(size)` — basic allocation
2. `free(ptr)` — deallocation
3. `realloc(ptr, new_size)` — resize
4. `calloc(nmemb, size)` — zeroed allocation
5. `posix_memalign(&ptr, align, size)` — aligned allocation

**For our GPU implementation, there are two approaches:**

**Approach A: Device-side allocator (preferred for performance)**
- Implement a simple bump/freelist allocator in GPU global memory
- No hostcall needed — allocations happen at device speed
- `malloc` → bump pointer or freelist pop
- `free` → freelist push (or no-op for bump allocator)
- This is critical because `core::fmt` formatting may allocate

**Approach B: Host-side allocator via hostcall**
- Every malloc/free becomes a hostcall round-trip
- Extremely slow — format_args + println alone could trigger dozens of allocations
- Only viable for rare large allocations

**Recommended: Approach A for general allocation, with hostcall fallback for very large allocations or when GPU memory is exhausted.**

Note: MIN_ALIGN on most platforms is 8 or 16 bytes. On nvptx64, we should use 16-byte
alignment (matches CUDA's malloc alignment guarantee).

**Confidence**: high

## Unexpected Discoveries

1. **core::fmt is no_std compatible**: The entire formatting machinery (`format_args!`,
   `Display`, `Debug` traits, `core::fmt::write`) works without std or libc. Only the
   final output step (`write` syscall) needs the libc shim. This means printf-style
   formatting on GPU is essentially free.

2. **Synchronization is needed even for println!**: `Stdout` wraps in a `ReentrantLock`,
   which on Linux uses futex. On GPU we need our own sync primitive — the gpu-atomics
   spinlock should work here. Alternatively, we can provide a custom `Stdout` that
   bypasses the lock (since GPU print already serializes through hostcall).

3. **errno is pervasive**: Almost every libc wrapper in std checks errno after syscalls
   via `io::Error::last_os_error()`. Our libc shim must provide a thread-local errno
   or we need to embed error codes in hostcall responses. On GPU, "thread-local" means
   per-warp or per-lane storage.

4. **open64/stat64/lseek64 vs open/stat/lseek**: std uses the `*64` variants on
   32-bit platforms. On nvptx64 (64-bit), the regular versions should suffice since
   off_t is already 64-bit.

5. **VectorWare uses host-side std, not libc 1:1**: Their hostcall handlers use Rust
   std on the host side, meaning they don't need to exactly replicate libc semantics —
   they can implement higher-level operations. We should consider the same approach.

## Open Questions

1. **Allocator strategy**: Should we use a simple bump allocator (fast, no fragmentation
   concern for short-lived kernels) or a proper freelist (supports long-running kernels)?
   The hostcall buffer already uses global memory — allocator needs its own region.

2. **errno storage**: Where to store errno on GPU? Options:
   - Per-lane in registers (compiler may spill)
   - Per-warp in shared memory (need warp-id indexing)
   - Per-block in shared memory
   - Skip errno entirely and encode errors in return values

3. **File descriptor table**: Host-side fds are just integers, but do we need a
   GPU-side mapping table? Probably not — just pass the integer through hostcall.

4. **Buffering**: `LineWriter` for stdout buffers until newline. This happens device-side
   using allocated memory. Is this desirable, or should we bypass buffering and send
   each write directly via hostcall?

5. **Which std modules to disable**: Threading, networking, process spawning — these
   should return errors. Need to verify that std compiles when these modules' libc
   functions all return ENOSYS.

## Impact on Downstream Tasks

- **gpu-std.2 (libc shim)**: Now has a concrete function list. Start with Tier 1 (12
  functions), expand to Tier 2 as needed. The shim crate should provide `#[no_mangle]`
  extern "C" functions that either handle locally (malloc, memcpy, abort) or dispatch
  via hostcall (write, read, open, close).

- **gpu-std.3 (File I/O experiment)**: Needs open/read/write/close/fstat hostcall
  opcodes added to the existing protocol. The hostcall.4 PRINT mechanism proves
  the pattern works — File I/O is structurally identical.

- **async-runtime.2 (executor design)**: Executor needs an allocator for task queues.
  Device-side malloc is prerequisite. The gpu-atomics spinlock can serve as the
  sync primitive for stdout locking.

- **New potential task**: Implement a device-side bump/freelist allocator as a
  prerequisite for gpu-std.2.
