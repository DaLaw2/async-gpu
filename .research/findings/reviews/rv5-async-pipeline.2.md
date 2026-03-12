# rv5: Review Synthesis — async-pipeline.1 + async-pipeline.2
**Task**: async-pipeline.1 (move helpers) + async-pipeline.2 (generalize macro)
**Level**: Full (2-agent team: proposer + skeptic)

## Review Sources
- `rv5-async-pipeline.2-proposer.md` — deep technical review (correctness, soundness, API, edge cases, performance)
- `rv5-async-pipeline.2-skeptic.md` — adversarial challenge (variable capture, convergence, payload, missing features, breaking changes)

## Issues Found and Resolution

### Blocking Issues (all fixed)

| # | Source | Issue | Fix Applied |
|---|--------|-------|------------|
| P1 | Proposer | PRINT buffer overflow: msg bytes not clamped to PRINT_MAX_MSG_LEN (56) | Added `copy_len = min(msg.len(), PRINT_MAX_MSG_LEN)` clamp |
| P2 | Proposer | OPEN buffer overflow: path bytes not clamped to FILE_MAX_PATH_LEN (56) | Added `path_len = min(path.len(), FILE_MAX_PATH_LEN)` clamp |
| P3 | Proposer | WRITE magic number `48` instead of FILE_MAX_WRITE_LEN constant | Replaced `48` with `gpu_protocol::FILE_MAX_WRITE_LEN` |
| S1 | Skeptic | Variable shadowing: duplicate `let fd = ...` creates duplicate struct fields | Added compile-time duplicate detection with clear error message |
| S2 | Skeptic | Return type locked to `true`: `-> ()` or `-> u32` fails | Restricted to `-> bool` and `-> ()` with proper ready values; other types get clear error |
| S6 | Skeptic | Expression cast fragility: `#expr as u64` → `(#expr) as u64` | Parenthesized all expression casts |

### Non-blocking Issues (documented, not fixed)

| # | Source | Issue | Decision |
|---|--------|-------|----------|
| S3 | Skeptic | Sideband ops impossible inside macro | By design: pass pre-allocated offsets via params. Document in macro doc. |
| S4 | Skeptic | No error propagation (FILE_ERROR_SENTINEL stored as normal value) | Accepted: matches hand-written WarpFuture behavior. Error checking is user's responsibility. |
| P6 | Proposer | No control=0 reset in warp_hostcall_submit | Low risk: recycled packets already have control cleared by host. |

## Consensus Findings

Both reviewers agreed on:
1. **Warp convergence is correct** — all generated match arms maintain SIMT convergence via broadcast_u32 + syncwarp
2. **Payload layouts match host parser** — verified against gpu-protocol constants
3. **No stale pkt_idx issue** — warp_hostcall_submit only writes pkt_idx_cell after successful pop
4. **Runtime migration is clean** — #[inline(always)] preserves PTX output
5. **The macro is significantly simpler than hand-written WarpFuture** — ~15 lines input → ~200 lines generated code (13x reduction)

## Verdict: PASS (after fixes applied)

All 6 blocking issues resolved. The macro is production-ready for the supported service set (open, close, read, write, bulk_read, bulk_write, print). Variable capture model is correct. Generated state machines maintain warp convergence. Buffer bounds are enforced.
