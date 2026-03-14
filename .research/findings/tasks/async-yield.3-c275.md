# async-yield.3: Data pipeline demo — read→compute→write with async I/O
**Cycle**: 275 | **Theme**: async-yield | **Kind**: experiment | **Status**: done

## Summary
Built and verified a complete async data pipeline demo using `#[warp_cooperative] async fn` with real hostcall Futures. The GPU kernel reads a file, transforms data (byte+1), and writes the result — all using `GpuOpenFuture`, `GpuReadFuture`, `GpuWriteFuture`, `GpuCloseFuture` with `.await` syntax. Compiled with patched rustc 1.96.0-dev. PTX output has 7x `bar.warp.sync` + 1x `shfl.sync.idx` confirming warp-cooperative MIR pass.

## Findings

### Q: Does the async pipeline with real hostcall I/O work end-to-end?
A: **Yes.** The demo at `examples/async-pipeline/` successfully:
1. Opens input file via `GpuOpenFuture::new(buf, b"pipeline_input.txt", FILE_OPEN_READ).await`
2. Reads 30 bytes via `GpuReadFuture::new(buf, fd, &mut data).await`
3. Closes input via `GpuCloseFuture::new(buf, fd).await`
4. Transforms data on GPU (each byte += 1)
5. Opens output file via `GpuOpenFuture::new(buf, b"pipeline_output.txt", FILE_OPEN_WRITE_CREATE).await`
6. Writes 30 bytes via `GpuWriteFuture::new(buf, out_fd, &out[..n]).await`
7. Closes output via `GpuCloseFuture::new(buf, out_fd).await`

Host verification confirms output = input with each byte incremented by 1.

**Confidence**: high

### Q: Does the MIR pass produce correct warp-cooperative PTX?
A: **Yes.** The patched rustc's `WarpCooperativeTransform` pass reports:
```
warp_cooperative: `data_pipeline::{closure#0}` — 0 yield(s), 6 poll(s), 6 suspension(s), 7 return(s)
```
PTX output contains:
- 1x `shfl.sync.idx.b32` — discriminant broadcast (lane 0 → all lanes)
- 7x `bar.warp.sync` — convergence barriers at each suspension point

**Confidence**: high

### Key design decisions
1. **Single thread execution**: The kernel runs with 1 thread because each hostcall Future allocates from a shared packet pool. Multi-lane requires warp-batched hostcall (one lane submits, others wait) — this is future work.
2. **nanosleep between polls**: Without nanosleep, the GPU poll loop (200 iterations in ns) finishes before the host listener processes any request. Adding `nanosleep.u32 1000` between polls yields the SM and gives the host time to respond. This is the actual yield mechanism that makes async I/O practical.
3. **No NUL terminator in paths**: The hostcall protocol uses `path_len` for boundary, not NUL. Including `\0` causes Windows `File::open` to fail.
4. **register_attr → register_tool**: rustc 1.96 removed `#![feature(register_attr)]`; use `#![feature(register_tool)]` + `#![register_tool(warp_cooperative)]` instead.

## Unexpected Discoveries
- The `register_attr` feature was removed between the version our patches were developed against and 1.96. The MIR pass still works because it checks for `warp_cooperative` tool attribute.

## Open Questions
- Multi-lane warp-cooperative I/O: how to batch 32 lanes into 1 hostcall? Lane 0 submits, others wait on `bar.warp.sync`, result broadcast via `shfl.sync`.

## Impact on Downstream Tasks
- **async-std epic criterion 3**: SATISFIED — practical demo runs data pipeline with async I/O yielding
- **async-yield theme**: all 3 criteria met (async hostcall Future, PAL bridge, demo)
