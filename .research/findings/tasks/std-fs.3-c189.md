# std-fs.3: Fix OnceLock for stdin path (spinlock replacement)
**Cycle**: 189 | **Theme**: std-fs | **Kind**: experiment | **Status**: done

## Summary
Investigated whether OnceLock and Mutex block stdin() on GPU. Found that both fall into
`no_threads` variants (Cell-based) on `target_os = "cuda"`, requiring no code changes.
stdin() path is already functional.

## Findings

### Q: Can OnceLock use GPU atomic spinlock instead of futex?
A: Not needed. On cuda, `Once` (backing OnceLock) uses the `no_threads` variant from
`std/src/sys/sync/once/no_threads.rs` — a simple `Cell<State>` state machine
(Incomplete→Running→Complete). No futex, no atomics. This was already configured by
the `target_os = "cuda"` patch to `thread_local/mod.rs` which puts cuda in the
`no_threads` group.

**Confidence**: high

### Q: Does fixing OnceLock unblock std::io::stdin().read_line()?
A: Yes — no fix needed. The full stdin() path works:
1. `OnceLock` → `Once` (no_threads, Cell-based) ✓
2. `Mutex<BufReader<StdinRaw>>` → Mutex is also `no_threads` variant (Cell<bool>) ✓
3. `StdinRaw` → cuda PAL routes through `extern fn gpu_stdin_read` ✓
4. `STDIN_BUF_SIZE = 0` → BufReader won't buffer, direct pass-through ✓

The Mutex implementation at `sys/sync/mutex/mod.rs` has a cfg_select! where cuda falls
through to the catch-all `_ => { mod no_threads; }` — same as Once. The no_threads
Mutex is just `Cell<bool>` with lock/unlock/try_lock.

**Confidence**: high

## Unexpected Discoveries
- All std sync primitives (Once, Mutex, RwLock, Condvar) likely use no_threads variants
  on cuda, meaning std::sync::Mutex etc. would compile but only be safe for single-thread
  GPU usage (which matches our current model).

## Open Questions
- None — stdin path is clear.

## Impact on Downstream Tasks
- std-fs.4 (end-to-end File::create + write_all + read): UNBLOCKED. All prerequisites met.
