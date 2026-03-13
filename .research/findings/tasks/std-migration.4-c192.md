# std-migration.4: Verify stdin().read_line() end-to-end on GPU
**Cycle**: 192 | **Theme**: std-migration | **Kind**: experiment | **Status**: done

## Summary
Implemented and verified `stdin().read_line()` on GPU via `gpu-kernel-std`. Fixed two issues:
(1) `STDIN_BUF_SIZE = 0` caused `BufReader` to immediately return EOF, and (2) the default
10M spin timeout was insufficient for blocking stdin hostcall. Both fixed, test passes.

## Findings
### Q: Does std::io::stdin().read_line() work via hostcall STDIN?
A: **Yes**, after fixing `STDIN_BUF_SIZE`. The full chain works:
`stdin().lock().read_line()` → `BufReader::fill_buf()` → `Stdin::read()` →
`gpu_stdin_read()` → `gpu_hostcall_request_with_timeout(SERVICE_STDIN)` →
host `listen_with_stdin()` → `CannedStdin` → data returned to GPU.

Test output: "Read 18 bytes: Hello from stdin!" — correct.
**Confidence**: high

### Q: Does gpu_stdin_read need to be wired to SERVICE_STDIN?
A: **Yes**, the placeholder in gpu-kernel-std returned 0. Implemented using
`gpu_runtime::hostcall::gpu_hostcall_request_with_timeout()` with 100M spin count.
**Confidence**: high

## Unexpected Discoveries
1. **STDIN_BUF_SIZE = 0 causes silent EOF**: When `BufReader::with_capacity(0, ...)` is used,
   `fill_buf()` reads into a zero-length slice, which returns `Ok(0)` — interpreted as EOF.
   This is standard Rust behavior but was an unexpected trap. Fixed by setting
   `STDIN_BUF_SIZE = 128`.

2. **Blocking hostcall needs longer timeout**: Added `gpu_hostcall_request_with_timeout()`
   to gpu-runtime as a public API. SERVICE_STDIN routes through the host I/O thread which
   adds latency beyond the default 10M spin cycles. Using 100M spins (matching std-build-test).

## Changes Made
- `patched-std/library/std/src/sys/stdio/cuda.rs`: `STDIN_BUF_SIZE: 0 → 128`
- `crates/gpu-runtime/src/lib.rs`: Added `gpu_hostcall_request_with_timeout()`
- `crates/gpu-kernel-std/src/lib.rs`: Implemented `gpu_stdin_read()` + `std_stdin_test` kernel
- `crates/gpu-host/src/tests_std.rs`: Added `run_std_stdin_readline_test()`
- `crates/gpu-host/src/main.rs`: Wired test into full suite + `ONLY_TEST=std_stdin`

## Open Questions
None — stdin works end-to-end.

## Impact on Downstream Tasks
- **real-std epic criterion 4**: "std::io::stdin().read_line() works on GPU via hostcall" — **MET**
- All 5 real-std criteria now satisfied (pending user confirmation to close epic)
