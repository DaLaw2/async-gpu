# oncelock.3: Verify no cascading breakage from OnceLock bypass
**Cycle**: 44 | **Theme**: oncelock | **Kind**: experiment | **Status**: done

## Summary
Verified that the OnceLock bypass in `_print()` and `_eprint()` causes no cascading
breakage across the entire test suite. All 25+ tests pass, including async hostcall,
file I/O, stdin, multi-block, allocator, and the new println!() direct test.

## Findings

### Q: Does thread::panicking() still work?
A: **Yes, by design.** `thread::panicking()` uses `PANIC_COUNT` (a TLS variable on normal
targets, a simple `Cell` on `no_threads` targets). The OnceLock bypass only affects
`_print()` and `_eprint()` — it does not touch the panic machinery at all.

**Confidence**: high (design analysis + all tests pass including panic-inducing paths)

### Q: Does the panic hook still function correctly?
A: **Yes.** The panic hook calls `_eprint()` which now has the same bypass. If a panic
occurs on GPU, `_eprint()` will write to `Stderr::new()` directly, which routes through
the hostcall PAL layer. This is correct behavior.

**Confidence**: high (design analysis)

### Q: Are there any other std subsystems that break?
A: **No.** The full test suite passes:
- Basic kernels (write_thread_idx, vector_add) — PASSED
- Atomic operations (u32, u64, CAS, spin-load) — PASSED
- Hostcall (single, multi-block 128, multi-block 512) — PASSED
- Embassy async (immediate, countdown, two-task) — PASSED
- File I/O (open, write, read, close) — PASSED
- Async hostcall (single, two concurrent, futures::join) — PASSED
- Std time (Instant, SystemTime) — PASSED
- Build-std (Vec, String, format!) — PASSED
- Dynamic allocation (Vec push, grow, multi-Vec) — PASSED
- PAL stdout (writeln!) — PASSED
- PAL stdin (read) — PASSED
- Multi-warp sync (32 threads) — PASSED
- Pipeline (4-step async) — PASSED
- Showcase demo (Vec + format + stdin + stdout) — PASSED
- Multi-block (4×32=128, 8×64=512) — PASSED
- Slab allocator (dealloc, 32-thread concurrent) — PASSED
- **println!() direct (NEW)** — PASSED

**Confidence**: high (full test suite execution)

### Q: Is the OnceLock bypass safe for single-block and multi-block?
A: **Yes.** The bypass creates `Stdout::new()` on each `_print()` call — there is no
shared singleton state. Each thread gets its own `Stdout` instance, which writes to
the hostcall buffer via the PAL layer (already proven concurrent-safe with 512 threads).

**Confidence**: high (512-thread multi-block test passes)

## Test Results
| Test | Config | Expected | Result |
|------|--------|----------|--------|
| Full test suite | 25+ tests | All pass | **ALL PASSED** |

## Files Modified
None — this was a verification-only task.

## Impact
- oncelock theme is now COMPLETE (3/3 tasks done)
- println!() and all other std I/O work correctly on GPU
- No regressions from the OnceLock bypass
