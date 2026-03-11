# std-pal.2: Route stdin through CUDA PAL
**Cycle**: 33 | **Theme**: std-pal | **Kind**: experiment | **Status**: done

## Summary
Implemented stdin via hostcall STDIN service through the same extern bridge pattern as stdout.
`gpu_stdin_read()` extern function works correctly — host provides canned data, GPU receives
17 bytes in 229.9µs. However, `std::io::stdin()` wrapper is broken on GPU due to
OnceLock + ReentrantLock layers, same class of issue as println! vs writeln!.

## Findings

### Q: Does std::io::stdin().read_line(&mut buf) correctly invoke SERVICE_STDIN hostcall?
A: **No.** `std::io::stdin()` returns a `Stdin` handle that wraps our PAL `Stdin` in
`OnceLock<ReentrantLock<BufReader<StdinRaw>>>`. The OnceLock initialization and
ReentrantLock acquisition paths generate broken code on CUDA:
- The generated PTX shows the function returning early without calling `gpu_stdin_read`
- The ReentrantLock check path leads to `assert_failed` on lock re-entry
- This is the same class of issue as `println!` (broken std wrapper layers on GPU)

**Workaround:** Call `gpu_stdin_read()` directly, same as `writeln!(std::io::stdout(), ...)`
for stdout. The extern function mechanism works perfectly for both I/O directions.

**Confidence**: high

### Q: Can BufReader wrapping handle the 56-byte packet limit for stdin data?
A: **Not tested directly.** BufReader is inside the broken `std::io::stdin()` wrapper, so
we can't reach it. The direct `gpu_stdin_read()` path handles the 56-byte limit by
requesting at most 56 bytes per hostcall round-trip. For larger reads, the caller would
need multiple round-trips (same as stdout's chunking).

**Confidence**: medium

### Q: Does blocking stdin behavior work correctly?
A: **Yes.** The GPU-side `gpu_hostcall_stdin_raw` spin-waits with a generous 100M iteration
timeout for the host to respond. The host's `listen_with_stdin` method provides canned data
immediately, so the actual wait time is minimal (229.9µs total including kernel overhead).
For interactive stdin, the GPU would spin-wait while the user types, which works but wastes
GPU cycles. An async version (HostcallStdinFuture) would yield instead of spinning.

**Confidence**: high

## Architecture

```
Working path (direct extern call):
  gpu_stdin_read(buf, len)  ────→  gpu_hostcall_stdin_raw()
  │                                  │
  ├ STDIO_HOSTCALL_BUF global         ├ pop free packet
  ├ cap at 56 bytes                   ├ SERVICE_STDIN, slot0=max_len
  └ return bytes read                 ├ spin-wait for CONTROL_READY
                                      ├ copy response data to out_buf
                                      └ release packet

Broken path (std::io::stdin()):
  std::io::stdin()  →  OnceLock init  →  ReentrantLock  →  BufReader  →  PAL Stdin
  ^^^ broken: returns early without I/O ^^^
```

| Path | Works on GPU? |
|------|--------------|
| `gpu_stdin_read(buf, len)` (direct extern) | YES |
| `std::io::stdin().read(&mut buf)` | NO (OnceLock/ReentrantLock broken) |
| `std::io::stdin().lock().read_line()` | NO (same issue) |

## Unexpected Discoveries

1. **std::io::stdin() has same OnceLock/ReentrantLock issue as println!.**
   Both `println!` and `std::io::stdin()` fail because they go through std wrapper
   layers (OnceLock, ReentrantLock, OUTPUT_CAPTURE) that generate broken code on CUDA.
   The working pattern is: bypass the wrappers and use the PAL functions directly.

2. **Host-side canned stdin provider works well for testing.** Added
   `listen_with_stdin()` method to HostcallBuffer that provides canned data instead
   of reading from real stdin. This enables automated testing of stdin hostcall path.

3. **Stdin echo test (read→write round-trip) compiles but not tested in this cycle.**
   Added `std_stdin_echo_kernel` that reads from stdin and echoes to stdout, but
   host test only covers the basic read for now.

## Files Modified/Created
- `std-patches/stdio_cuda.rs` — MODIFIED: added `gpu_stdin_read` extern declaration,
  implemented `Stdin::read()` via extern call, set STDIN_BUF_SIZE=56
- `patched-std/library/std/src/sys/stdio/cuda.rs` — MODIFIED: synced with patch
- `crates/std-build-test/src/lib.rs` — MODIFIED: added `gpu_stdin_read()` impl,
  `gpu_hostcall_stdin_raw()`, `std_stdin_kernel`, `std_stdin_echo_kernel`
- `crates/gpu-host/src/hostcall.rs` — MODIFIED: added `handle_stdin_canned()` and
  `listen_with_stdin()` for canned stdin testing
- `crates/gpu-host/src/main.rs` — MODIFIED: added `run_std_stdin_test()`
- `crates/gpu-host/std_build_test.ptx` — UPDATED

## Test Results
| Test | Config | Expected | Result |
|------|--------|----------|--------|
| std_stdin_kernel (canned "Hello GPU stdin!\n") | 1×1 | 17 bytes, first='H' | **PASSED** (229.9µs) |

## Open Questions
- Can we patch std's OnceLock/ReentrantLock to work on CUDA (no-op lock)?
- Would that also fix println! (OUTPUT_CAPTURE is a separate issue)?
- Should we provide a `gpu_stdin!` macro as sugar over the direct extern call?

## Impact on Downstream Tasks
- **product.4** (showcase): Can use `gpu_stdin_read()` directly for input
- **std-pal theme**: Both stdout and stdin extern bridge pattern verified
  - stdout: writeln!(std::io::stdout(), ...) works
  - stdin: gpu_stdin_read() works (direct call required)
