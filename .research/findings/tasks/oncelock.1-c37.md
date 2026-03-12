# oncelock.1-c37: OnceLock Dependency Chain Analysis for CUDA PAL

- **Task**: Trace OnceLock's full call chain on the nvptx64-nvidia-cuda path
- **Status**: Done
- **Cycle**: 37

## Summary

OnceLock on the CUDA PAL path uses the `no_threads` backend for both `Once` and `Mutex`. This backend is `Cell`-based (no atomics), which is fundamentally single-threaded. The problem is NOT complex atomic ops or futex calls -- it is that `Cell` is not thread-safe, and on GPU where multiple warps/blocks execute concurrently, the `no_threads` assumption is violated.

## 1. Full Call Chain: `println!()` to PAL

### 1.1 Stdout Path

```
println!("...")
  -> std::io::_print(args)
    -> print_to(args, stdout, "stdout")
      -> stdout()                         // constructs Stdout handle
        -> STDOUT.get_or_init(|| ...)     // OnceLock<ReentrantLock<RefCell<LineWriter<StdoutRaw>>>>
          -> OnceLock::get_or_try_init()
            -> self.get()                 // fast path: Once::is_completed()
            -> self.initialize(f)         // slow path
              -> self.once.call_once_force(|p| ...)
                -> Once(poison)::call_once_force()
                  -> sys::Once::call()    // no_threads backend
                    -> Cell::get() / Cell::set()  // state machine via Cell
      -> stdout.write_fmt(args)
        -> Stdout::lock()
          -> ReentrantLock::lock()
            -> thread::current_id()       // TLS access for ThreadId
            -> self.owner.contains(id)    // AtomicU64::load(Relaxed)
            -> sys::Mutex::lock()         // no_threads: Cell<bool> based
          -> StdoutLock.inner.borrow_mut().write(buf)
            -> LineWriter::write()
              -> StdoutRaw::write()
                -> stdio::Stdout::write()
                  -> gpu_stdout_write()   // extern "Rust" -> hostcall
```

### 1.2 Stdin Path

```
io::stdin()
  -> static INSTANCE: OnceLock<Mutex<BufReader<StdinRaw>>> = OnceLock::new()
  -> INSTANCE.get_or_init(|| Mutex::new(BufReader::with_capacity(..., stdin_raw())))
     [Same OnceLock chain as above]
```

### 1.3 Stderr Path (different -- no OnceLock!)

```
io::stderr()
  -> static INSTANCE: ReentrantLock<RefCell<StderrRaw>> = ReentrantLock::new(...)
     // Const-initialized, NO OnceLock involved
  -> Stderr { inner: &INSTANCE }
```

**Key insight**: `stderr()` avoids OnceLock entirely by using a const-initialized `ReentrantLock` directly. This is possible because `StderrRaw` wraps `stdio::Stderr` which has `const fn new()`. Stdout can't do this because `LineWriter` requires allocation (buffer).

## 2. The `no_threads` Backend (What Actually Runs on NVPTX)

Since `nvptx64-nvidia-cuda` doesn't match any OS-specific target in the `cfg_select!` chains, **all** sync primitives fall through to the `no_threads` fallback:

### 2.1 sys::sync::once::no_threads::Once

**File**: `patched-std/library/std/src/sys/sync/once/no_threads.rs`

```rust
pub struct Once {
    state: Cell<State>,  // NOT atomic -- just Cell<State>
}
```

State machine: `Incomplete -> Running -> Complete` (or `Poisoned`)

- `is_completed()`: `self.state.get() == State::Complete` (plain Cell read)
- `call()`: `self.state.get()` then `self.state.set()` (plain Cell mutation)
- `wait()`: `panic!("not implementable on this target")`

**No atomics at all.** No futex. No thread parking. Just `Cell` get/set.

### 2.2 sys::sync::mutex::no_threads::Mutex

**File**: `patched-std/library/std/src/sys/sync/mutex/no_threads.rs`

```rust
pub struct Mutex {
    locked: Cell<bool>,  // NOT atomic
}
```

- `lock()`: `assert_eq!(self.locked.replace(true), false)` -- panics on recursive lock
- `unlock()`: `self.locked.set(false)`
- `try_lock()`: `self.locked.replace(true) == false`

### 2.3 sys::sync::thread_parking::unsupported::Parker

No-op stubs. `park()` and `unpark()` do nothing.

## 3. GPU-Incompatible Operations

### 3.1 `Cell` is NOT thread-safe on GPU

The `no_threads` backend assumes `unsafe impl Sync for Once` is safe because "threads are not supported on this platform." On GPU:
- Multiple warps within a block execute concurrently (true parallelism)
- Multiple blocks may execute concurrently
- `Cell` reads/writes are NOT atomic on GPU -- concurrent access = UB

**If two GPU threads call `stdout()` simultaneously, the `Cell<State>` in `Once` can be torn, leading to double-initialization, data races, or worse.**

### 3.2 `ReentrantLock::lock()` calls `thread::current_id()`

`current_id()` uses TLS (`#[thread_local]` or `local_pointer!`) to store ThreadId. On GPU:
- `#[thread_local]` may compile to a global (not per-thread) on NVPTX
- Thread ID generation uses `static COUNTER: AtomicU64` which may work (we've verified u64 atomics in atomics.4), but TLS semantics are broken
- If all GPU threads see the same ThreadId, the ReentrantLock will incorrectly think the lock is already held by "this thread" and skip locking

### 3.3 `RefCell` borrow checking is non-atomic

Inside `ReentrantLock<RefCell<...>>`, the `RefCell` borrow counter is a plain `Cell<isize>`. If two GPU threads pass the ReentrantLock (due to broken ThreadId), they'll both `borrow_mut()` the RefCell simultaneously = UB.

### 3.4 `LineWriter` buffer allocation

`LineWriter::new(stdout_raw())` allocates a `Vec<u8>` buffer. This happens inside OnceLock init, so it calls the global allocator on GPU. If the allocator isn't set up, this panics or segfaults.

### 3.5 Output capture uses `thread_local!`

`print_to_buffer_if_capture_used()` reads `OUTPUT_CAPTURE_USED: AtomicBool` (fine on GPU with our atomics work), then accesses `OUTPUT_CAPTURE` thread_local. TLS on GPU is broken, but this path is only taken during testing, so it's not a blocker.

## 4. Other std Internals Depending on OnceLock/Once/LazyLock

| Component | Primitive | Risk |
|-----------|-----------|------|
| `io::stdin()` | `OnceLock<Mutex<BufReader<StdinRaw>>>` | Same issues as stdout |
| `io::stdout()` | `OnceLock<ReentrantLock<RefCell<LineWriter<StdoutRaw>>>>` | Primary problem |
| `io::stderr()` | `ReentrantLock<RefCell<StderrRaw>>` (const) | No OnceLock, but ReentrantLock still uses broken ThreadId |
| `rt::cleanup()` | `static CLEANUP: Once` | Used at shutdown, not a GPU concern |
| `panicking::HOOK` | `RwLock<Hook>` | Panic hook dispatch -- uses sys::RwLock (also no_threads backend) |
| `backtrace` | `LazyLock` (inside Capture) | Only triggered on panic backtrace |
| `sys::alloc::uefi` | `OnceLock<u32>` | UEFI only, not relevant |

## 5. Replacement Strategies

### 5.1 Strategy A: Spin-based OnceLock (replace `no_threads` Once with atomics)

Replace `Cell<State>` with `AtomicU8` and use a CAS spin loop:

```rust
// Conceptual replacement for sys::sync::once on CUDA
pub struct Once {
    state: AtomicU8,  // 0=Incomplete, 1=Running, 2=Complete, 3=Poisoned
}

impl Once {
    pub fn call(&self, ignore_poisoning: bool, f: &mut impl FnMut(&OnceState)) {
        // CAS loop: try to transition Incomplete->Running
        loop {
            match self.state.compare_exchange(INCOMPLETE, RUNNING, Acquire, Acquire) {
                Ok(_) => { /* we won, run f */ break; }
                Err(COMPLETE) => return,  // already done
                Err(RUNNING) => { /* spin wait */ core::hint::spin_loop(); }
                Err(POISONED) if !ignore_poisoning => panic!("poisoned"),
                Err(POISONED) => { /* try CAS Poisoned->Running */ }
                _ => unreachable!()
            }
        }
        // Run closure, then store Complete
        f(&state);
        self.state.store(COMPLETE, Release);
    }
}
```

**Safety**:
- **Single-block (single warp)**: Safe. Only one warp can execute at a time (assuming no warp-level parallelism within a single call). Actually, even within a block, warps execute concurrently, so spin-waiting is needed.
- **Single-block (multi-warp)**: Safe IF atomics work across warps (they do -- shared memory atomics within a block are guaranteed).
- **Multi-block**: Risky. Atomics on global memory across blocks use `atom.global` instructions. These work but starvation is possible if many blocks compete. Since OnceLock is init-once, contention is brief.
- **Deadlock risk**: If the thread running the init closure gets preempted (GPU does NOT preempt within a block in classic CUDA), other spinning warps will eventually proceed. Across blocks, the scheduler might not schedule the "winner" block, causing deadlock on pre-Volta architectures without independent thread scheduling.

### 5.2 Strategy B: Bypass OnceLock entirely (pre-initialize statics)

Make stdout/stdin use const-initialized statics like stderr already does:

```rust
// Instead of:
static STDOUT: OnceLock<ReentrantLock<RefCell<LineWriter<StdoutRaw>>>> = OnceLock::new();

// Use unbuffered direct writes (no LineWriter, no allocation):
static STDOUT: ReentrantLock<RefCell<StdoutRaw>> =
    ReentrantLock::new(RefCell::new(StdoutRaw(stdio::Stdout::new())));
```

**Tradeoff**: Loses line-buffering. On GPU, buffering is actually harmful (adds complexity, memory overhead, and the buffer would be shared across all threads). Direct writes via hostcall are already "buffered" by the hostcall protocol.

### 5.3 Strategy C: `static mut` with unsafe init (simplest, most fragile)

```rust
static mut STDOUT_INNER: MaybeUninit<LineWriter<StdoutRaw>> = MaybeUninit::uninit();
static STDOUT_INIT: AtomicBool = AtomicBool::new(false);
```

Not recommended -- requires manual lifetime management and is easy to get wrong.

### 5.4 Strategy D: GPU-specific `stdout()` that skips OnceLock entirely

Add a `#[cfg(target_arch = "nvptx64")]` path in `stdio.rs` that returns an unbuffered, unsynchronized handle:

```rust
#[cfg(target_arch = "nvptx64")]
pub fn stdout() -> Stdout {
    // No OnceLock, no ReentrantLock, no buffering
    // Just wrap StdoutRaw directly
    Stdout { inner: StdoutRaw(stdio::Stdout::new()) }
}
```

This is essentially what `writeln!(std::io::stdout(), ...)` already achieves via `io::stdout()` returning a raw handle.

## 6. Recommended Approach

**Strategy B (const-init like stderr) + fix ReentrantLock** is the cleanest path:

1. **Remove OnceLock from stdout/stdin**: Use const-initialized `ReentrantLock<RefCell<StdoutRaw>>` (drop `LineWriter` buffering on CUDA)
2. **Fix ReentrantLock for GPU**: Replace `thread::current_id()` with a GPU thread identifier (e.g., `%laneid` + `%smid` + `%ctaid` encoded as u64, or just use a spin-lock Mutex and drop re-entrancy)
3. **Replace `no_threads` Mutex with spin-lock**: Use `AtomicU8` CAS instead of `Cell<bool>`

Alternatively, **Strategy D** is simplest if we don't need `println!()` to go through the standard `Stdout` type at all -- just make `_print()` call `gpu_stdout_write()` directly on CUDA targets, bypassing the entire sync stack.

## 7. Why `writeln!(std::io::stdout(), ...)` Works Today

The current workaround works because `io::stdout()` on the PAL returns a plain `stdio::Stdout` struct (zero-sized, const-constructible). The `Write` impl on `StdoutRaw` calls `gpu_stdout_write()` directly. The problematic code is in the **standard library's wrapping layer** (`OnceLock`, `ReentrantLock`, `LineWriter`), not in the PAL itself.

`writeln!(std::io::stdout(), ...)` calls `<Stdout as Write>::write_fmt()`, which goes through the OnceLock path. If this actually works, it means either:
- OnceLock initialization happens to succeed because only one GPU thread calls it first (lucky timing)
- Or the user is calling a different `stdout()` that returns the raw PAL handle

The direct `gpu_stdin_read()` / `gpu_stdout_write()` extern calls bypass all of this.

## Key Files Examined

- `patched-std/library/std/src/io/stdio.rs` -- stdin/stdout/stderr handles, `_print()`, OnceLock usage
- `patched-std/library/std/src/sync/once_lock.rs` -- OnceLock wraps Once + UnsafeCell
- `patched-std/library/std/src/sync/poison/once.rs` -- Once wraps sys::Once
- `patched-std/library/std/src/sys/sync/once/no_threads.rs` -- Cell-based Once (no atomics!)
- `patched-std/library/std/src/sys/sync/mutex/no_threads.rs` -- Cell-based Mutex
- `patched-std/library/std/src/sync/reentrant_lock.rs` -- uses sys::Mutex + ThreadId (TLS)
- `patched-std/library/std/src/sys/stdio/cuda.rs` -- PAL: extern calls to gpu_stdout_write/gpu_stdin_read
- `patched-std/library/std/src/thread/current.rs` -- current_id() uses TLS
