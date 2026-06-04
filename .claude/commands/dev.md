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
└── archive/                         # Completed items
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
Read `dev-gates.md`. Execute hard gates (Tier Gate, Epic Verification Gate).
**Brainstorm check** — if any proactive trigger fires, read `dev-brainstorm.md` BEFORE SELECT:
  - `tasks_since_brainstorm >= 10`
  - No ready tasks but active epics have unmet criteria
  - User requests brainstorm

### SELECT (`do.select`)
1. Filter: `status == "pending"` AND deps met AND theme `"active"`
2. **Tier priority**: T0 before T1 before T2 (HARD — enforced by Tier Gate)
3. Batch: same-theme sequential, cross-theme can parallelize
4. Set selected → `status = "active"`, update `current_task_id`

### DISPATCH (`do.execute`)
Read `dev-dispatch.md`. For each task:
1. **PREP**: File discovery for the task — `ls`, `find`, `grep -l` to locate relevant crates, scripts, entry points. Read dependency task findings if any. Read context.md Tried & Rejected. Do NOT read source code.
2. Assemble brief per `dev-dispatch.md` template (includes Codebase Pointers + Prior Work sections)
3. Launch task subagent with brief
4. Subagent does all work: code, tests, findings, synthesis update
5. Subagent returns: STATUS (done|blocked), SUMMARY (3 sentences), FILES_CHANGED
6. You read SUMMARY only. You do NOT read source code or compiler output.
7. If blocked → mark blocked, continue.
8. **If done → VERIFY**: Dispatch a separate verify subagent (see `dev-dispatch.md` type: verify). The verify subagent checks: tests pass, lint clean, findings file exists, work matches task goal and epic success criteria. **PASS → mark done, update counters. FAIL → mark blocked with verify failure reason.**

### SAVE (`do.save`)
1. Dispatch maintenance subagent: `/maintain` for relevant sub-commands
2. Auto-archive: completed items → archive/. state.toml must stay under ~300 lines
3. Run `bash scripts/pre-push.sh`. Fix failures via subagent if needed.
4. Commit + push
5. Rewrite `.research/context.md`:
   Sections: Current Focus, Recent Decisions, Tried & Rejected, Active Constraints, Key Metrics, Next

### ROUTE (`do.route`)
1. **North Star Gate** (from `dev-gates.md`): dispatch subagent to judge completed work
2. **Reactive brainstorm triggers** — if any fire, read `dev-brainstorm.md`:
   - Theme just completed
   - Task was marked blocked
   - 3+ consecutive blocked tasks in same theme
3. **Tier promotion**: If all T(N) epics satisfied → activate T(N+1)
4. More ready tasks → back to SELECT
5. All active epics fully satisfied → report to user, STOP

## Error Handling

- Subagent fails → dispatch new subagent to retry or mark blocked
- Missing system lib → set `awaiting_user`, STOP
- `git push` fails → warn user, continue (data committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints

- Do NOT delete existing findings (correct in new findings)
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
- Disk-first: subagent writes findings BEFORE reporting done
- **state.toml scoping**: Only active epics have themes/tasks in state.toml. When an epic activates, brainstorm creates its themes/tasks. When archived, its themes/tasks are removed.
