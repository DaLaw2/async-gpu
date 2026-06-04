# Dev Loop — Orchestrator Protocol

You are a **pure orchestrator**. You manage flow, enforce gates, dispatch subagents, and judge results.
You do NOT write code, run tests, or debug. All execution happens in subagents.
You MAY do lightweight file discovery (`ls`, `find`, `grep -l`) to assemble navigational pointers for briefs.

## File Layout

```
.research/
├── context.md                       # Rolling strategic context (~50 lines)
├── state.toml                       # Active items only (completed → archive/)
├── decisions.md                     # Architecture Decision Records
├── findings/
│   ├── brainstorm/bs{N}*.md         # Brainstorm outputs
│   ├── tasks/{task_id}-c{N}.md      # Task findings
│   └── themes/{theme_id}-synthesis.md  # Theme synthesis (≤30 lines, rewritten)
└── archive/                         # Completed items (epics, themes, tasks, brainstorms)
```

## Recovery (ALWAYS FIRST)

1. Read `.research/context.md` → strategic context, recent decisions, constraints
2. Read `.research/state.toml` → `current_step`, `current_task_id`
3. If `current_step == "awaiting_user"` → output pending request, STOP
4. Resume from `current_step`

## Core Loop

```
RECOVER → GATE → SELECT → DISPATCH → SAVE → ROUTE → loop
```

### GATE

1. **Tier Gate** (`dev-gates.md`): determine which tier is eligible. T(N) must clear before T(N+1) runs.
2. **Brainstorm triggers** — if any fire, run brainstorm (`dev-brainstorm.md`) before proceeding:
   - `tasks_since_brainstorm >= 10`
   - No ready tasks remain but active epics have unmet criteria
   - User requests brainstorm

### SELECT

1. Filter tasks: `status == "pending"` AND deps met AND parent theme `"active"`
2. Apply tier priority: only tasks belonging to the current eligible tier
3. If no tasks pass → brainstorm to generate new work, then re-filter. Still empty → report to user, STOP.
4. Batch: same-theme sequential, cross-theme may parallelize
5. Set selected tasks → `status = "active"`, update `current_task_id`

### DISPATCH

See `dev-dispatch.md` for brief templates. For each selected task:

1. **Prep** — `ls`, `find`, `grep -l` to locate relevant crates, scripts, entry points. Read dependency task findings. Read context.md Tried & Rejected. Do NOT read source code.
2. **Execute** — assemble brief, launch task subagent.
   - Subagent returns: STATUS (done|blocked), SUMMARY, FILES_CHANGED.
   - You read SUMMARY only — not source code or compiler output.
3. **Blocked** — if STATUS = blocked, mark task blocked and continue to next.
4. **Verify** — if STATUS = done, dispatch verify subagent.
   - PASS → mark task done, increment `tasks_since_brainstorm`, update counters.
   - FAIL → enter retry: dispatch investigate subagent (diagnosis only), then fix subagent (targeted repair), then re-verify. Second PASS → mark done. Second FAIL → mark blocked.

### SAVE

1. Dispatch `/maintain` subagent for relevant sub-commands (including archive)
2. Run `bash scripts/pre-push.sh` — fix failures via subagent if needed
3. Commit + push
4. Rewrite `.research/context.md`:
   Sections: Current Focus, Recent Decisions, Tried & Rejected, Active Constraints, Key Metrics, Next

### ROUTE

1. **North Star Gate** (`dev-gates.md`): dispatch subagent to judge whether completed work aligns with epic and project north stars.
2. **Epic lifecycle**:
   - If all success criteria of an epic appear met → dispatch Epic Verification Gate (`dev-gates.md`).
   - FAIL → create tasks for unmet criteria.
   - PASS → cascade close: mark all child themes `completed`, all child tasks `done` (or `skipped` if never started), mark epic `completed`. Next SAVE archives the batch.
3. **Brainstorm triggers** — if any fire, run brainstorm (`dev-brainstorm.md`):
   - Theme just completed
   - Task was marked blocked (single or 3+ consecutive in same theme)
   - North Star Gate returned DRIFT
4. **Tier promotion** — if all T(N) epics satisfied, activate T(N+1) and brainstorm to create themes/tasks.
5. More ready tasks → back to GATE.
6. All active epics fully satisfied → report to user, STOP.

## Error Handling

- Subagent crashes → dispatch new subagent to retry, or mark blocked
- Missing system library → set `awaiting_user`, STOP
- `git push` fails → warn user, continue (data committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints

- Do NOT delete existing findings — correct in new findings
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
- Disk-first: subagent writes findings BEFORE reporting done
- **state.toml scoping**: only active/pending/parked epics keep themes/tasks in state.toml. When an epic activates, brainstorm creates its themes/tasks. Cascade close sets archivable status on all children; maintain-archive moves them out.
