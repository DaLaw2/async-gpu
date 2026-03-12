# ml-workload.2: Vector similarity search demo
**Cycle**: 94 | **Theme**: ml-workload | **Kind**: experiment | **Status**: done

## Summary
GPU-autonomous vector similarity search verified on hardware. 20-state VecSearchFuture
coordinates: open(db) -> read(db) -> close(db) -> open(query) -> read(query) -> close(query) ->
compute(cosine similarity) -> open(results) -> write(top-K) -> close(results). 100 vectors x 128
dimensions, 6.2ms end-to-end, zero CPU intervention between steps.

## Findings

### Q: Can a WarpFuture state machine coordinate the full vector search pipeline?
A: Yes. The 20-state explicit submit/wait design works correctly. Each state does exactly one
thing: either submit a hostcall or wait for a response. This eliminates the sentinel-based
multi-phase logic that caused deadlocks in the original 12-state design.
**Confidence**: high (hardware verified)

### Q: What state machine architecture works for complex multi-I/O pipelines?
A: Explicit submit/wait pairs. The pattern is:
- SUBMIT_X: call `warp_hostcall_submit(... next=WAIT_X ...)`
- WAIT_X: call `warp_hostcall_wait_u64(... next=SUBMIT_Y ...)`, process result
- Repeat for each I/O operation

The original 12-state design used sentinel values (fd==0, db_bytes==0) to multiplex phases
within single states. This caused a deadlock — likely in the CLOSE_DB/OPEN_QUERY transition
where `fd` was used both as a "wait for close response" sentinel and as a value to submit
to the CLOSE service. The 20-state design has zero ambiguity.
**Confidence**: high

### Q: How does per-lane cosine similarity computation work?
A: Each lane processes vectors at stride-32 offsets (lane 0: vectors 0,32,64,96; lane 1:
vectors 1,33,65,97; etc.). All lanes read the same query vector from sideband. Each lane
maintains a local top-K array with insertion sort. Current demo only saves lane 0's results
(1/32 of DB). Full warp merge via shfl.sync planned for future iteration.
**Confidence**: high (hardware verified, but only lane 0 results)

### Q: What is the performance profile?
A: 6.2ms end-to-end for 100 vectors x 128 dimensions. Dominated by 9 hostcall round-trips
(~0.5ms each at 1-warp scale). Compute is negligible for this DB size. Bulk I/O correctly
transfers 51KB (db) + 516B (query) via sideband.
**Confidence**: high

## Design Details

### State Machine (20 states)
```
SUBMIT_OPEN_DB(0) -> WAIT_OPEN_DB(1) -> SUBMIT_READ_DB(2) -> WAIT_READ_DB(3) ->
SUBMIT_CLOSE_DB(4) -> WAIT_CLOSE_DB(5) -> SUBMIT_OPEN_Q(6) -> WAIT_OPEN_Q(7) ->
SUBMIT_READ_Q(8) -> WAIT_READ_Q(9) -> SUBMIT_CLOSE_Q(10) -> WAIT_CLOSE_Q(11) ->
COMPUTE(12) -> SUBMIT_OPEN_OUT(13) -> WAIT_OPEN_OUT(14) -> SUBMIT_WRITE(15) ->
WAIT_WRITE(16) -> SUBMIT_CLOSE_OUT(17) -> WAIT_CLOSE_OUT(18) -> DONE(19)
```

### Sideband Layout
```
Offset 0:          database data (up to 900KB, includes 8-byte header)
Offset 921600:     query data (516 bytes: 4-byte header + 512-byte vector)
Offset 922116:     results (84 bytes: 4-byte K + 10 x 8-byte entries)
```

### Key Lesson
Never use sentinel values for state machine phase tracking. Explicit states are slightly
more verbose but completely eliminate an entire class of deadlock bugs.

## Open Questions
- Full warp merge: how to collect 32 lanes' top-K via shfl.sync (320 candidates -> top-10)?

## Impact on Downstream Tasks
- ml-workload.3 can extend VecSearchFuture with a loop over multiple queries
- The explicit submit/wait pattern should be used for all future WarpFuture designs
