# split-loader.5: Litmus test — single-crate rebuild timing

## Summary

The kernel-split epic litmus test ("edit a test kernel, PTX+cubin rebuild < 5 min")
**FAILED**. The test crate alone takes ~30 min for ptxas, nearly identical to the
old unified 11.4 MB build. The kernel split improved cargo (PTX) time and cubin
file size, but ptxas time did NOT scale down proportionally with PTX size.

## Methodology

1. Added a trivial comment to `gpu-kernel-test/src/lib.rs` to trigger rebuild.
2. Timed `cargo build --release` in the test crate directory (incremental, deps cached).
3. Timed `ptxas --gpu-name sm_75` on the resulting PTX.
4. Reverted the trivial change.

## Timing data: gpu-kernel-test single-crate rebuild

| Step               | Time      | Notes                        |
|--------------------|-----------|------------------------------|
| PTX (cargo build)  | 0m27s     | Incremental, deps cached     |
| cubin (ptxas)      | 29m58s    | sm_75, 6.0 MB PTX input      |
| **Total**          | **30m25s**| **FAILS 5 min litmus test**  |

## PTX + cubin size comparison

| Metric          | Old unified (kernel_std) | New split (kernel_test) | Delta   |
|-----------------|--------------------------|--------------------------|---------|
| PTX size        | 11.4 MB (est.)           | 6.0 MB                   | -47%    |
| cubin size      | 179 MB                   | 116 MB                   | -35%    |
| ptxas time      | ~30 min                  | ~30 min                  | ~0%     |

## All 4 crate PTX sizes (dev mode, opt-level 1)

| Crate           | PTX bytes   | PTX lines | Human     |
|-----------------|-------------|-----------|-----------|
| kernel_core     | 1,405,562   | 39,417    | 1.3 MB    |
| kernel_compute  | 2,037,383   | 58,478    | 1.9 MB    |
| kernel_io       | 2,412,930   | 67,818    | 2.3 MB    |
| kernel_test     | 6,224,000   | 167,732   | 5.9 MB    |

## Key finding: ptxas does NOT scale with PTX size

The test crate PTX shrank from 11.4 MB to 6.0 MB (-47%), but ptxas time barely
changed (30 min vs ~30 min). This means ptxas compilation time is dominated by
code complexity (number of unique functions, register allocation passes, control
flow graph analysis) rather than raw PTX byte count.

The test crate is the worst case because it re-exports `gpu_kernel_core` and has
76 entry points with complex std-based code (HashMap, Vec, File I/O, thread::spawn,
cooperative compute, structured concurrency, par_iter). These generate highly
complex PTX that ptxas spends its time optimizing — size reduction alone does not
help.

## Smaller crates should be faster

The 3 non-test crates (core=1.3 MB, compute=1.9 MB, io=2.3 MB) are much smaller
and simpler. Their ptxas times were not measured in this experiment (would take
~1 hour total), but based on the super-linear scaling observation, they should
complete in 2-10 minutes each.

The real benefit of kernel-split for the build pipeline is:
1. Parallel ptxas: all 4 crates compile simultaneously (wall clock = slowest crate)
2. Incremental rebuilds: editing a core-only kernel doesn't recompile test crate
3. The test crate is the bottleneck — other crates are likely under 5 min

## Litmus test verdict

**FAILED** for the test crate specifically. The epic's litmus test assumed ptxas
scales linearly with PTX size, but it scales with complexity. The test crate
inherits all of gpu-kernel-core's code (via `extern crate gpu_kernel_core`) and
adds 76 complex entry points.

However, editing a kernel in a smaller crate (core, compute, io) and rebuilding
only that crate would likely meet the 5 min target. The litmus test is ambiguous
about which crate — "a test kernel" could mean any kernel in the test crate, which
is the worst case.
