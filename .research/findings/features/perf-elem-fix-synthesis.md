# perf-elem-fix: elementwise fix
**Epic**: kernel-perf | **Status**: completed | **Updated**: 2026-06-04

## Progress
- Implemented elementwise_add_out (out-of-place) in reshape.rs via NVRTC
- Benchmarked out-of-place vs in-place: in-place is faster (160 vs 119 GB/s)
- Conclusion: in-place already meets target, no fix needed

## Verified Conclusions
- In-place elementwise_add: 160 GB/s (83% of peak 192 GB/s) — meets >= 160 GB/s target
- Out-of-place elementwise_add_out: 119 GB/s — RW separation actually slower
- float4 vectorized loads already in use for both variants

## Rejected Approaches
- Out-of-place add: slower due to extra memory write (119 vs 160 GB/s)

## Open Questions
- None — theme goals met with existing in-place approach

## Key Metrics
- In-place elementwise_add: 160 GB/s (83% peak)
- Out-of-place elementwise_add_out: 119 GB/s (62% peak)

## Next Steps
- None — theme completed
