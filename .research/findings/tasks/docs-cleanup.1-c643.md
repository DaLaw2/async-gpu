# Task: docs-cleanup.1 — Remove stale docs, fix .gitignore, re-track docs/

## Status: done

## Findings

### Stale docs reviewed and removed

**docs/DESIGN-executor.md** (636 lines) — Historical design document for the GPU-side
async task spawning executor. The executor described (GpuExecutor, WorkQueue, TaskSlot,
spawn(), type-erased poll functions, slot recycling) is fully implemented in
`crates/core/gpu-runtime/src/executor.rs` (54 references to the design concepts).
The document opens with "no ability to launch additional work dynamically" as the
problem statement — this has been solved. **Removed as stale.**

**docs/VALIDATION.md** (305 lines) — First-run hardware validation checklist. Opens
with "All development to date has been compile-verified only — no actual GPU execution
has been performed." This is no longer true — GPU tests run on real hardware. The
checklist references SM target patching and example structures that may have drifted.
The getting-started.md (585 lines) now covers running examples with up-to-date
instructions. **Removed as stale.**

No other orphaned docs found (checked for BLOG_DRAFT.md, DRAFT*.md, TODO*.md, NOTES*.md).

### .gitignore fixed

Removed the `docs/` entry (line 44) from `.gitignore`. The comment "# Local docs (not tracked)"
and the `docs/` pattern were deleted. The three rewritten docs are now tracked by git.

### Verification

After `git add docs/`:
- docs/ARCHITECTURE.md — staged (new file)
- docs/CHANGELOG.md — staged (new file)
- docs/getting-started.md — staged (new file)
- docs/DESIGN-executor.md — gone (removed)
- docs/VALIDATION.md — gone (removed)

## Files changed
- `.gitignore` — removed docs/ exclusion
- `docs/DESIGN-executor.md` — deleted
- `docs/VALIDATION.md` — deleted
