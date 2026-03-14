# data-iter.1: Implement convergence-loop kernel — Newton's method sqrt on GPU
**Cycle**: 263 | **Theme**: data-iter | **Kind**: experiment | **Status**: done

## Summary

Implemented `newton_sqrt_kernel` — a GPU kernel that autonomously iterates Newton's method for computing square roots until convergence, without host intervention. Host test verifies 5 test cases with varying inputs. Demonstrates the data-dependent iteration pattern for gpu-autonomous criterion 3.

## Changes

### crates/gpu-kernel/src/hostcall_kernels.rs
Added `newton_sqrt_kernel(input, output, iterations, max_iter)`:
- Reads f32 input S from mapped memory
- Initial guess: S/2
- Iterates: x_{n+1} = (x_n + S/x_n) / 2
- Convergence: |x_{n+1} - x_n| < 1e-6 or max_iter reached
- Writes result (f32) and iteration count (u32) to mapped memory
- Handles edge case: S <= 0 returns 0

### crates/gpu-host/src/tests_pipeline.rs
Added `run_newton_sqrt_test()`:
- 5 test cases: sqrt(4)=2, sqrt(2)=1.414..., sqrt(100)=10, sqrt(0.25)=0.5, sqrt(1e6)=1000
- Uses mapped memory (alloc_mapped_u32 for f32 via to_bits/from_bits)
- Verifies each result within tolerance and iteration count > 0

### crates/gpu-host/src/main.rs
Registered `run_newton_sqrt_test` in test suite.

## Architecture

```
Host                          GPU (newton_sqrt_kernel)
  │                             │
  ├─ write S to mapped mem      │
  ├─ launch kernel ─────────────┤
  │                             ├─ read S
  │                             ├─ x = S/2 (initial guess)
  │                             ├─ loop {
  │                             │    x_new = (x + S/x) / 2
  │                             │    if |x_new - x| < ε: break
  │                             │    x = x_new
  │                             │  }
  │                             ├─ write result + iter count
  ├─ synchronize ◄──────────────┤
  ├─ read result                │
  └─ verify                     │
```

Key property: **iteration count is data-dependent** — not known at launch time. The kernel autonomously decides when to stop.

## Impact on Downstream Tasks

- data-iter theme criterion 1: DONE (autonomous convergence loop)
- data-iter theme criterion 2: DONE (host receives iteration count + result)
- data-iter theme criterion 3: DONE (test verifies convergence)
- gpu-autonomous criterion 3 ("Data-dependent iteration... pattern not yet demonstrated"): NOW DEMONSTRATED
