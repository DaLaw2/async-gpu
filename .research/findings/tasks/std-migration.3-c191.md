# std-migration.3: Port async pipeline kernels to gpu-kernel-std with Result types
**Cycle**: 191 | **Theme**: std-migration | **Kind**: experiment | **Status**: done

## Summary
Created `std_pipeline_test` kernel in gpu-kernel-std that demonstrates the full combination
of real Rust std + idiomatic error handling on GPU. The kernel uses Vec, format!(),
std::fs::File, std::io::{Read,Write} traits, and the `?` operator for Result propagation —
all running on GPU via hostcall protocol. Multi-step pipeline: generate data → write file →
read back → verify content → report.

## Findings

### Q: Can async pipeline kernels use std::io + Result error handling?
A: YES. A kernel function returning `Result<(), std::io::Error>` works perfectly on GPU.
The `?` operator desugars to normal match/return which is handled by LLVM for nvptx64.
`std::io::Error::new()` with custom messages works. `std::io::Error` from libc errno
(via File::create, write_all, read_to_end) propagates correctly.

Key pattern:
```rust
fn pipeline() -> Result<(), std::io::Error> {
    let mut f = File::create(path)?;  // ? propagates io::Error
    f.write_all(&data)?;              // ? propagates io::Error
    // ...
    Ok(())
}
```

**Confidence**: high (tested end-to-end)

### Q: Does the combined std + error propagation work end-to-end?
A: YES. The full chain works:
1. Vec::new() + format!() for data generation (heap allocation on GPU)
2. File::create() → libc open() → hostcall OPEN → host opens file
3. write_all() → libc write() → hostcall WRITE (multiple chunks, 48 bytes each)
4. File drop → libc close() → hostcall CLOSE
5. File::open() → libc open() → hostcall OPEN
6. read_to_end() → libc read() → hostcall READ (multiple chunks until EOF)
7. Byte-by-byte verification using Vec comparison
8. str::from_utf8() + lines().count() for text processing
9. println!() for status reporting via hostcall PRINT

All with proper error propagation via ? at each step.

**Confidence**: high

## Changes

### Modified: `crates/gpu-kernel-std/src/lib.rs`
- Added `std_pipeline_inner()` → `Result<(), std::io::Error>` helper function
  - Uses Vec, format!, File::create, write_all, File::open, read_to_end
  - ? operator throughout for error propagation
  - Content verification with byte comparison
- Added `std_pipeline_test` kernel entry point
  - Calls std_pipeline_inner() and reports Ok/Err via println!

### Modified: `crates/gpu-host/src/tests_std.rs`
- Added `run_std_pipeline_test()` host-side test

### Modified: `crates/gpu-host/src/main.rs`
- Added std_pipeline ONLY_TEST entry
- Added test to main sequence

## Impact on Downstream Tasks
- std-migration theme approaching completion (criteria 1+3 met, criterion 2 already met)
- Demonstrates the full real-std + gpu-error epic intersection
