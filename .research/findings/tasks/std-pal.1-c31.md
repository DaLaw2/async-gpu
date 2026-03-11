# std-pal.1: Implement CUDA PAL for stdout with hostcall routing
**Cycle**: 31 | **Theme**: std-pal | **Kind**: experiment | **Status**: done

## Summary
Successfully routed `std::io::stdout()` through hostcall on GPU. Created a CUDA PAL stdio
module in the vendored std source that delegates writes to an extern function `gpu_stdout_write()`
implemented by the kernel crate. `writeln!(std::io::stdout(), ...)` works correctly with
runtime values, multiple calls, and Vec/format! data. `println!` itself crashes LLVM's NVPTX
backend due to a "Circular dependency found in global variable set" bug in the `_print` path.

## Findings

### Q: Can we add a cuda PAL variant to route Stdout::write() through hostcall?
A: **Yes.** Created `sys/stdio/cuda.rs` in the vendored std source. The PAL defines:
- `Stdout::write()` that calls `extern "Rust" fn gpu_stdout_write(buf, len) -> usize`
- `Stdin::read()` stubs (returns Ok(0), for std-pal.2)
- `panic_output() -> None` (no stderr for panic output on GPU)

The extern function is resolved by Fat LTO from the kernel crate, which implements
`gpu_stdout_write` using the hostcall PRINT protocol.

**Confidence**: high (verified by compilation + GPU execution)

### Q: How to pass the hostcall buffer pointer to the PAL layer?
A: **Inverted the dependency.** Instead of the PAL holding the hostcall buffer pointer
(which caused LLVM crashes from global variable cycles), the kernel crate holds it:
- Kernel crate has `static STDIO_HOSTCALL_BUF: AtomicU64`
- Kernel calls `stdio_init(buf)` at entry
- `gpu_stdout_write()` (in kernel crate) reads this global and sends via hostcall

The PAL itself has ZERO global state and ZERO inline PTX — it's a pure extern call bridge.

**Confidence**: high

### Q: Does BufWriter wrapping in std's Stdout interfere with per-packet message semantics?
A: **Not tested directly.** The `writeln!` path goes through `Write::write_fmt` →
`Stdout::write()` which calls our PAL directly. The `write_fmt` call passes the full
formatted output as a single `write()` call in most cases. For messages >56 bytes,
our `gpu_stdout_write` implementation splits into multiple hostcall packets.

**Confidence**: medium (works for all tested cases, but edge cases possible)

### Q: Does println!("{}", runtime_value) produce a correct hostcall?
A: **`println!` CRASHES the LLVM NVPTX backend.** The crash is:
```
LLVM ERROR: Circular dependency found in global variable set
Running pass 'NVPTX Assembly Printer' on function '@begin_panic{{reify_shim}}'
```

This is because `println!` → `std::io::_print()` → `print_to()` → references `OUTPUT_CAPTURE`
global (for test capture) → which involves `ReentrantLock` and panic infrastructure →
which references stderr → creating a circular dependency in the global variable graph.

**`writeln!(std::io::stdout(), ...)` is the working alternative.** It goes through
`Write::write_fmt()` directly, bypassing the `_print()` → `print_to()` → `OUTPUT_CAPTURE`
path entirely.

| Macro | Path | Works on GPU? |
|-------|------|--------------|
| `println!` | `_print()` → `print_to()` → Stdout | NO (LLVM crash) |
| `writeln!(std::io::stdout(), ...)` | `Write::write_fmt()` → Stdout | YES |
| `write!(std::io::stdout(), ...)` | `Write::write_fmt()` → Stdout | YES |
| `stdout().write_all(bytes)` | `Write::write_all()` → Stdout | YES |

**Confidence**: high

## Architecture

```
Kernel Code                     std's CUDA PAL              Hostcall
─────────────                   ──────────────              ────────
writeln!(stdout(), ...)  ──→  Write::write_fmt()
                              │
                              Stdout::write(buf)
                              │
                              extern gpu_stdout_write(ptr, len)  ←── Fat LTO resolves
                              │
gpu_stdout_write() impl  ←───┘
│
STDIO_HOSTCALL_BUF global
│
hostcall_print_raw()  ──────────────────────────────→  Host listener
│                                                       │
inline PTX atomics                                      prints to host stdout
```

## Files Modified/Created
- `patched-std/library/std/src/sys/stdio/cuda.rs` — NEW: CUDA PAL stdio (extern bridge)
- `patched-std/library/std/src/sys/stdio/mod.rs` — MODIFIED: added `target_os = "cuda"` case
- `std-patches/stdio_cuda.rs` — NEW: PAL source for apply.sh
- `std-patches/stdio_mod.patch` — NEW: patch for stdio mod.rs
- `std-patches/apply.sh` — MODIFIED: added stdio patching steps
- `crates/std-build-test/src/lib.rs` — MODIFIED: added println test kernels + gpu_stdout_write impl

## Test Results
| Test | Config | Expected | Result |
|------|--------|----------|--------|
| std_println_kernel(value=42) | 1×1 | `"Hello from GPU println! value = 42\n"` | **PASSED** |
| std_println_multi_kernel([100,200,300]) | 1×1 | 3 messages | **PASSED** (3 messages received) |
| std_println_vec_kernel([10,20,30,40,50]) | 1×1 | `"GPU Vec: 5 elements, sum = 150\n"` | **PASSED** |

## Unexpected Discoveries

1. **`println!` crashes LLVM NVPTX backend.** The `_print()` → `print_to()` path involves
   `OUTPUT_CAPTURE` (test output capture) and `ReentrantLock`, creating circular global
   variable dependencies that the NVPTX backend's global variable ordering pass can't resolve.
   This is an LLVM bug, not a Rust or std issue.

2. **`writeln!` is a complete replacement.** `writeln!(std::io::stdout(), "{}", x)` is
   functionally identical to `println!("{}", x)` — same output, same formatting. The only
   difference is the import (`use std::io::Write`) and explicit `stdout()`.

3. **Zero PAL state works.** The PAL doesn't need any global variables — it just calls
   the extern function. This avoids all LLVM global dependency issues and keeps the std
   patches minimal.

4. **VectorWare likely has a patched `_print()`.** Their `println!` works because they
   probably modified the `_print()` function to avoid `OUTPUT_CAPTURE` on GPU, or their
   custom rustc fork doesn't include the test capture infrastructure.

## Open Questions
- Can we patch `_print()` to bypass `OUTPUT_CAPTURE` on `target_os = "cuda"`?
- Would that fix the LLVM circular dependency crash?
- Should we pursue `println!` support or accept `writeln!` as sufficient?

## Impact on Downstream Tasks
- **std-pal.2** (stdin): Same extern-bridge pattern will work for `Stdin::read()`
- **product.4** (showcase): Use `writeln!(std::io::stdout(), ...)` instead of `println!`
- **VectorWare parity**: 95%+ with writeln!, 100% would need println! (LLVM fix or _print patch)
