# /maintain archive — Archive completed items from state.toml

Move done tasks, completed themes, and old brainstorms out of state.toml to keep it lean.

## Language
- Conversation: 繁體中文 | Files: English

## Trigger thresholds
- Tasks with `status = "done"`: archive when > 10
- Brainstorms: keep last 3, archive the rest
- Completed themes where ALL tasks are done: archive

## Steps

1. **Read** `.research/state.toml`
2. **Count** done tasks and brainstorms
3. If under thresholds → print `[OK] state.toml is lean` and stop
4. **Archive**:
   - Create/append to `.research/archive/tasks-archived.toml` — move done tasks
   - Create/append to `.research/archive/themes-archived.toml` — move completed themes (only if all their tasks are also archived)
   - Create/append to `.research/archive/brainstorms-archived.toml` — move old brainstorms (keep last 3)
5. **Update** state.toml: remove archived entries, keep `completed_tasks` count accurate
6. **Report**: `[FIX] Archived N tasks, M themes, K brainstorms`

## Rules
- NEVER archive active/pending tasks or active themes
- NEVER modify `[meta]` fields other than counts
- NEVER archive epics (only user can manage epics)
- Preserve TOML formatting and comments
