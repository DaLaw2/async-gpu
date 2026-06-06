# Feature synthesis: docs-cleanup

## What shipped
- Removed 2 stale docs: DESIGN-executor.md (pre-implementation design, now fully shipped)
  and VALIDATION.md (pre-hardware-validation checklist, now superseded by getting-started.md)
- Fixed .gitignore: removed `docs/` exclusion so docs are version-controlled
- Staged 3 rewritten docs: ARCHITECTURE.md, CHANGELOG.md, getting-started.md

## Outcome
docs/ now contains only current, accurate documentation tracked in git.
No useful content was lost — both removed files were pre-implementation artifacts.

## Files changed
- `.gitignore` (modified)
- `docs/DESIGN-executor.md` (deleted)
- `docs/VALIDATION.md` (deleted)
