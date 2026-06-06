# audit-api — Feature Synthesis
## Status: active (2/N tasks complete)

## Task 1: API Surface Review (audit-api.1) — DONE
Systematic review found 6 issues. API is clean overall.

## Task 2: Fix Quick Wins (audit-api.2) — DONE
- HostcallBuffer: 9 pub fields → pub(crate) + accessor methods (9 files updated)
- Facade: added GpuVec + Scheduler/CpuScheduler/GpuScheduler/AutoScheduler re-exports

## Remaining Tech Debt
1. **GpuHostError::Verification overload** — ~32 error modes in one variant (large scope)
2. **onnx_rt missing docs** — #[allow(missing_docs)] on public module (large scope)
3. **No thiserror** — hand-rolled Error impls, functional, low priority
4. **Pipeline hardcoded sleep** — 100ms in Pipeline::run(), needs investigation
