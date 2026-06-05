# split-execute — Execute kernel crate split

**Epic**: kernel-split (T0)
**Status**: active

## Completed tasks

### split-execute.1 — Extract stdio to gpu-runtime (c614)
Moved 3 statics + 5 functions from gpu-kernel-std/lib.rs to gpu-runtime/stdio.rs.
Force-link via `#[used]` confirmed working through LTO. Functions marked `unsafe`
to satisfy clippy in gpu-runtime. PTX symbols verified present. Zero regressions.

## Pattern established

Migration pattern for future extractions (panic, entry, etc.):
1. Create module in gpu-runtime with moved code
2. Register in lib.rs + add prelude re-exports
3. Add `#[used]` force-link in kernel crate for PAL-called `#[no_mangle]` symbols
4. Mark functions `unsafe` if they take/deref raw pointers (clippy requirement)
5. Verify PTX symbols + clippy + fmt
