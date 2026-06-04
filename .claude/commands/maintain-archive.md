# /maintain archive — Archive completed items from state.toml

Move completed/done/skipped items out of state.toml into archive files.
Goal: keep state.toml under ~300 lines (active/pending/parked items only).

## Language
- Conversation: 繁體中文 | Files: English

## What to archive
- **Epics** with `status = "completed"` → `.research/archive/epics-archived.toml`
- **Themes** with `status = "completed"` → `.research/archive/themes-archived.toml`
- **Tasks** with `status = "done"` or `"skipped"` → `.research/archive/tasks-archived.toml`
- **Brainstorms**: keep last 3, archive the rest → `.research/archive/brainstorms-archived.toml`

When a completed epic is present, archive it together with ALL its child themes and tasks as a single batch. ROUTE's cascade close guarantees all children have archivable status before this runs. If any child still has non-archivable status (bug in cascade), log a warning and skip that child.

## Steps

1. **Read** `.research/state.toml`
2. **Count** archivable items
3. If nothing to archive → print `[OK] state.toml is lean` and stop
4. **Archive** — for each completed epic, process as a batch (epic + all children). For standalone completed themes and done tasks (no parent epic being archived), process individually.
   - Read existing archive file (if any)
   - Append items with a header comment: `# Archived at cycle {N}` (or `# Archived at cycle {N} — epic {epic_id} completed` for batches)
   - In state.toml, replace archived items with a one-line comment listing their IDs
5. **Update** state.toml: remove archived entries, keep `completed_tasks` count accurate
6. **Report**: `[FIX] Archived N epics, M themes, K tasks, J brainstorms (state.toml: X lines)`

## Rules
- NEVER archive active/pending/parked items
- NEVER modify `[meta]` fields other than counts
- Preserve TOML formatting and comments
- Add separator comments in archive files between batches
