# ml-workload.1: Feasibility analysis — async pipeline for tensor computation
**Cycle**: 93 | **Theme**: ml-workload | **Kind**: investigation | **Status**: done

## Summary
All f32 math operations needed for vector similarity search work natively on nvptx64. `sqrt.approx.f32` confirmed via inline PTX. Sideband buffer (1MB default) fits ~1,800 128-dim vectors. Infrastructure is complete — no gaps to fill before ml-workload.2.

## Findings

### Q: Can the existing async pipeline infrastructure support tensor computation?
A: Yes. f32 add, mul, div compile to native PTX ops. `sqrt` requires inline PTX (`sqrt.approx.f32`, ~1 ULP precision), implemented as `gpu_sqrtf()` helper. Dot product, norm, cosine similarity all verified on hardware.
**Confidence**: high (hardware verified)

### Q: What are the sideband buffer size limits for realistic vector databases?
A: 128 dimensions × 4 bytes = 512 bytes per vector. In 1MB sideband:
- Raw capacity: 2,048 vectors
- With query + results overhead: ~1,800 database vectors
- For larger databases: `new_with_sideband(n, size)` already supports custom sizes, or streaming via `sideband_reset()` between chunks
**Confidence**: high

### Q: What infrastructure gaps need filling?
A: None significant:
- `gpu_sqrtf()`: Added (inline PTX, 1 ULP)
- f32 arithmetic: Native
- Sideband bulk I/O: Existing `warp_hostcall_submit` + `warp_hostcall_wait_u64`
- Data layout: AoS (contiguous 512-byte vectors) — simple, adequate for mapped memory
**Confidence**: high

### Q: What is the maximum practical database size given 1MB sideband?
A: ~1,800 vectors at 128 dimensions. For 10K+ vectors, increase sideband to 8MB (trivial host-side change). Streaming approach (multiple bulk reads with sideband reset) theoretically unlimited but adds complexity.
**Confidence**: high

## Design Specification for ml-workload.2

### Data Layout (sideband, single-round)
```
Offset 0:            query vector        (128 × f32 = 512 bytes)
Offset 512:          database vectors     (N × 512 bytes)
Offset 512 + N*512:  result buffer        (K × 8 bytes: {id:u32, score_bits:u32})
```

### File Format (binary)
- **Database file** (`vecdb.bin`): `[N:u32][dim:u32][v0_d0:f32]...[v0_d127:f32][v1_d0:f32]...`
- **Query file** (`query.bin`): `[dim:u32][q_d0:f32]...[q_d127:f32]`
- **Output file** (`results.bin`): `[K:u32][id0:u32][score0:f32]...[idK-1:u32][scoreK-1:f32]`

### WarpFuture State Machine (12 states)
```
OPEN_DB(0) → READ_DB(1) → CLOSE_DB(2) →
OPEN_QUERY(3) → READ_QUERY(4) → CLOSE_QUERY(5) →
COMPUTE(6) → MERGE_TOPK(7) →
OPEN_OUTPUT(8) → WRITE_RESULTS(9) → CLOSE_OUTPUT(10) → DONE(11)
```

### Work Distribution
- **Per-lane**: each of 32 lanes handles ceil(N/32) database vectors
- Each lane computes full 128-dim dot product for its assigned vectors
- Each lane maintains local top-K (K=10, insertion sort)
- MERGE_TOPK: lane 0 collects 32 × 10 = 320 candidates via shfl.sync, selects global top-10

### Parameters
- DIM = 128, K = 10, N = up to 1500 (demo default: 100)
- Host creates random database + known query with planted similar vector for verification

## Open Questions
None — ready to implement ml-workload.2.

## Impact on Downstream Tasks
- ml-workload.2 can proceed with the design above
- `gpu_sqrtf()` helper is already in gpu-kernel/src/lib.rs
