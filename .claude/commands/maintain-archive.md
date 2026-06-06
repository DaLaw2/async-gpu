# /maintain archive — Archive completed items from state.toml

Move completed/done/skipped items out of state.toml into archive files.
Goal: keep state.toml under ~300 lines (active/pending/parked items only).

## Language
- Conversation: 繁體中文 | Files: English

## What to archive
- **Stories** with `status = "completed"` → `.research/archive/stories-archived.toml`
- **Features** with `status = "completed"` → `.research/archive/features-archived.toml`
- **Tasks** with `status = "done"` or `"skipped"` → `.research/archive/tasks-archived.toml`
- **Brainstorms**: keep last 3, archive the rest → `.research/archive/brainstorms-archived.toml`

When a completed story is present, archive it together with ALL its child features and tasks as a single batch. ROUTE's cascade close guarantees all children have archivable status before this runs. If any child still has non-archivable status (bug in cascade), log a warning and skip that child.

Epics are NOT archived by maintain-archive — they are permanent strategic milestones.

## Steps

1. **Read** `.research/state.toml`
2. **Count** archivable items
3. If nothing to archive → print `[OK] state.toml is lean` and stop
4. **Archive** — for each completed story, process as a batch (story + all children). For standalone completed features and done tasks (no parent story being archived), process individually.
   - Read existing archive file (if any)
   - Append items with a header comment: `# Archived at cycle {N}` (or `# Archived at cycle {N} — story {story_id} completed` for batches)
   - In state.toml, replace archived items with a one-line comment listing their IDs
5. **Update** state.toml: remove archived entries, keep `completed_tasks` count accurate
6. **Report**: `[FIX] Archived N stories, M features, K tasks, J brainstorms (state.toml: X lines)`

## Rules
- NEVER archive active/pending/parked items
- NEVER modify `[meta]` fields other than counts
- Preserve TOML formatting and comments
- Add separator comments in archive files between batches
