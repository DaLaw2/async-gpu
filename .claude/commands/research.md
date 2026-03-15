# Autonomous Research Loop — Think / Do / Check

You are an autonomous exploratory research agent. Cyclical, evolving research — not linear.

## Hierarchy
See CLAUDE.md for Epic/Theme/Task definitions. Key workflow rules:
- **Epic**: Agent may create epics only via brainstorm when no ready tasks remain or no active themes exist. Only the user can close them. Status: active | completed.
- **Theme**: Status: active | parked | completed. References parent epic via `epic = "..."`.
- **Task**: Kind: investigation | experiment | design.

## CRITICAL RULES

1. **Disk-first**: Write findings to disk BEFORE proceeding. Synthesis reads from FILES.
2. **HOST IS READ-ONLY**: No installing packages, no modifying system config. If env changes needed → STOP, list what user must do, set `current_step = "awaiting_user"`.
3. **Git save**: Commit + push after each completed batch of work (not after every micro-step).
4. **Epic alignment**: Every brainstorm MUST read all active epics first. All spawned themes/tasks must serve an active epic.
5. **Brainstorm output**: Every brainstorm MUST produce at least one of: new task, new theme, or new epic. No empty brainstorms.

---

## File Layout

```
.research/findings/
├── brainstorm/
│   ├── bs{seq}.md                 # Quick/Standard brainstorm (single file)
│   ├── bs{seq}-proposer.md        # Deep brainstorm: proposer analysis
│   └── bs{seq}-skeptic.md         # Deep brainstorm: skeptic challenges
├── tasks/
│   └── {task_id}-c{cycle}.md      # Task findings
└── reviews/                       # Legacy — no longer generated
```

---

## Recovery Protocol (ALWAYS RUN FIRST)

1. Read `.research/state.toml` → check `current_mode`, `current_step`, `last_summary`
2. If `current_step == "awaiting_user"` → output pending request, STOP
3. If `last_summary` is sufficient for context → proceed directly
4. Otherwise read `decisions.md` and relevant recent findings (current task + its deps only)
5. Resume from `current_step`:
   - `"do.select"` → pick next task batch
   - `"do.execute"` → check `current_task_id`, verify if findings file exists (done vs resume)
   - `"think.*"` → check if brainstorm file exists

---

## Phase 1: Think (Brainstorm)

### Trigger (any one):
- `current_mode == "think"`
- A task was marked `blocked`
- A theme was just completed (reassess direction)
- A decision gate was reached
- User explicitly requests brainstorm

### Epic Check (MANDATORY before every brainstorm)
1. Read all `[[epics]]` with `status = "active"`
2. Verify: are there themes/tasks actively working toward each epic?
3. If an active epic has no active themes → brainstorm MUST spawn themes/tasks for it
4. All recommendations must reference which epic they serve

### Brainstorm Triage

Assess scope to choose the right level:

| Level | Criteria | Method |
|-------|----------|--------|
| **Quick** | Routine interval, no blocked tasks, smooth progress | Main agent directly. No subagent, no file. |
| **Standard** | New direction, reprioritization, or 1-2 blocked tasks | 1 subagent → `bs{seq}.md` |
| **Deep** | Major pivot, 2+ blocked tasks, decision gate, cross-theme conflict | 2-agent team → `bs{seq}-proposer.md` + `bs{seq}-skeptic.md` + `bs{seq}.md` |

### Quick Brainstorm (main agent, no subagent)
1. Review active epics, active themes, recent findings, ready tasks
2. Decide task priority and any small adjustments directly
3. Update state.toml (tasks, dependencies)
4. No separate brainstorm file — record key insight in `[[brainstorms]]` entry only

### Standard Brainstorm (single subagent)
1. **Prepare**: Gather context — active epics, active themes, recent findings, blocked tasks, open questions
2. **Launch subagent**:

```
You are analyzing the next steps for a GPU research project. Write a structured
analysis covering ALL of these perspectives in a single document:

## Active Epics
- List each active epic and assess progress toward its success criteria

## Technical Analysis (systems + compiler + GPU architecture)
- What's feasible, what are the risks, what are the constraints?

## Skeptic Challenges
- What assumptions are untested? What could go wrong?
- Challenge every "should work" conclusion.

## Recommendations
- Concrete task changes: new tasks, skip/park decisions, dependency updates
- Each recommendation must reference which epic it serves
- Priority ordering with rationale

Context: {epics, themes, recent findings, blocked tasks, open questions}

Write to: .research/findings/brainstorm/bs{seq}.md
```

3. **Synthesize + adapt**: Read brainstorm file, update state.toml

### Deep Brainstorm (2-agent team)
1. **Prepare**: Gather extensive context including cross-theme dependencies
2. **Launch agent team** with 2 teammates:
   - **"proposer"**: Write structured analysis with MANDATORY separate sections. Write to `bs{seq}-proposer.md`:
     ```
     ## Active Epics Assessment
     - Progress toward each epic's success criteria
     - Gaps and unaddressed criteria

     ## Systems Analysis (memory models, ABI, unsafe boundaries)
     - What's feasible? What are the constraints?
     - Specific risks for the current tasks

     ## Compiler Analysis (rustc, LLVM, codegen, PTX backend)
     - What are the fundamental compiler limitations?
     - IR transformations, target support, known bugs

     ## GPU Architecture Analysis (warp model, memory hierarchy, occupancy)
     - What CPU assumptions break on GPU?
     - Hardware constraints that affect the design

     ## Concrete Recommendations
     - New tasks, skip/park decisions, dependency updates
     - Each must reference which epic it serves
     - Priority ordering with rationale
     ```
   - **"skeptic"**: Challenge every claim, find holes, identify untested assumptions. Read proposer's file and write counterarguments. Write to `bs{seq}-skeptic.md`
3. **Synthesize**: Read both files, extract consensus and unresolved disputes. Write `bs{seq}.md`
4. **Adapt**: Update state.toml with theme/task changes

### After any brainstorm level:
- Increment `brainstorm_seq` (even for Quick — keeps seq monotonic)
- Record `[[brainstorms]]` entry with seq, trigger, level, key insight
- git commit + push
- Transition → `current_mode = "do"`, `current_step = "do.select"`

---

## Phase 2: Do (Execute Tasks)

### Step do.select
Task selection:
1. Filter: `status == "pending"` AND all deps `"done"` AND theme `"active"`
2. Priority: brainstorm/review-spawned > theme momentum > investigation before experiment
3. **Batch selection**: Pick ALL ready tasks that can run in this session. Group by:
   - Same-theme tasks → execute sequentially
   - Cross-theme independent tasks → can parallelize with subagents
4. Set selected tasks `status = "active"`, update `current_task_id` to first task
5. Update `current_step = "do.execute"`

### Step do.execute
**Execute tasks in a batch** — do NOT stop between tasks unless blocked.

For each task:

**kind = "investigation"**: Use subagents for research. Write findings immediately.

**kind = "experiment"**:
- Read relevant same-theme findings first
- **Baseline-first**: Before making any changes, run the relevant test/benchmark on the *current* code to capture a baseline result. Record it in the findings file. All subsequent changes are measured against this baseline.
- Write code → compile → fix → retry (max 5 rounds)
- If 5 rounds fail → `status = "blocked"`, `git reset` to pre-experiment commit, continue to next task (brainstorm later)
- **Output redirection**: For long-running commands (`cargo test`, `cargo bench`, etc.), redirect output to a temp file (`> .research/run.log 2>&1`) and `grep` only the key results (pass/fail counts, metrics). Only read full output on failure (`tail -50`). Delete the log file after use.

**kind = "design"**: Synthesize findings, produce architecture doc, record ADR in `decisions.md`.

After each task:
- **Lint** (experiment/design only): Run `cargo +stable fmt --check` and `cargo +stable clippy -- -D warnings` on modified crates. Fix before proceeding.
- Write findings to `.research/findings/tasks/{task_id}-c{cycle}.md` (see template below)
- Update state.toml: task status, `total_cycles++`, `completed_tasks++`
- Update `current_task_id` to next task in batch (or clear if batch done)

### Step do.save
After the batch is complete:
1. **Maintenance**: Invoke `/maintain` to run housekeeping checks (CI sync, state archive, README sync, etc.). Fix any issues it finds.
2. **Pre-push check**: Run `bash scripts/pre-push.sh` BEFORE committing. This regenerates std patches (if patched-std/ was modified) and runs local CI lint. Fix all failures first. Include generated patch files in commit.
3. **Commit + push**: one commit per task, or batch commit if tasks are small.
4. **CI check**: After push, run `gh run list --limit 1` to verify CI triggered. If the latest run shows `failure`, run `gh run view <id> --log-failed | tail -30` to diagnose. Fix CI-breaking issues before proceeding.
5. Update `last_summary` in state.toml

### Step do.route
Decide what to do next:

1. Review triage on completed tasks (see Check phase below)
2. If any brainstorm trigger fired (including escalation from review) → `current_mode = "think"`, `current_step = "think.triage"`
3. If more ready tasks exist → back to `do.select`
4. If no ready tasks but active epics have UNMET success criteria → `current_mode = "think"` (brainstorm new themes/tasks to address remaining criteria)
5. If ALL active epics have ALL success criteria met → report to user, STOP

**CRITICAL**: Completing all current themes/tasks does NOT mean stop. Check epic success criteria. If any criterion is unmet, brainstorm MUST generate new work. The loop only terminates when every active epic is fully satisfied or the user explicitly stops.

---

## Phase 3: Check (Review)

After each experiment/design task completes, assess risk level:

| Level | Criteria | Action |
|-------|----------|--------|
| **Skip** | Extends a proven pattern (e.g., u32→u64 same asm pattern) | No review. Proceed. |
| **Light** | New code but within established crate/pattern | Self-review checklist (below). No agent. |
| **Escalate** | Cross-theme architecture, decision gate, major design choice | Trigger brainstorm (Think phase). |

**Self-review checklist** (for Light):
- [ ] No UB: pointer validity, alignment, address space
- [ ] No missing `.sys` scope on GPU-CPU atomics
- [ ] Host correctly synchronizes before freeing mapped memory
- [ ] PTX output verified for key instructions
- [ ] Test covers the happy path

If checklist passes → proceed. If issues found → create fix task `{task_id}.{n}`.

---

## Findings Template

```markdown
# {task_id}: {title}
**Cycle**: {cycle} | **Theme**: {theme} | **Kind**: {kind} | **Status**: done | blocked

## Summary
(2-3 sentences)

## Findings
### Q: {research_question}
A: ...
**Confidence**: high | medium | low

## Unexpected Discoveries

## Open Questions

## Impact on Downstream Tasks
```

---

## Cycle Control

```
Recovery Protocol
  │
  ▼
awaiting_user? ──Yes──► Output request, STOP
  │ No
  ▼
current_mode?
  ├─ "think" → Epic check → Triage (quick/standard/deep) → adapt → git save → "do"
  ├─ "do"    → Batch select → execute tasks → lint → git save → route
  └─ "check" → Triage (skip/light/escalate) → route
       │
       ▼
  Continue until context runs low or ALL epic criteria met
```

**NEVER stop for human input EXCEPT:**
1. All tasks blocked and brainstorm cannot unblock
2. Environment changes needed (`awaiting_user`)
3. ALL active epics have ALL success criteria met (nothing left to do)

**On session recovery**: If resuming from context break, do NOT wait for user confirmation. Read `last_summary`, check `current_step`, and IMMEDIATELY continue the loop. The user already said "go" — they don't need to say it again every session.


---

## State File: last_summary

The `[meta]` section in state.toml includes a `last_summary` field (max 3 sentences) that captures the most recent session's outcome. This allows fast recovery without reading multiple findings files.

Example:
```toml
last_summary = "atomics.4 done: u64 CAS/add/exchange + spin-load + activemask all verified. hostcall.4 unblocked. Next: hostcall.4 implementation."
```

---

## Error Handling
- Compilation fails 5 times → mark blocked, `git reset` to pre-experiment commit, continue to next task, brainstorm later
- Missing system lib → STOP, ask user
- `git push` fails → warn user, continue (data committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints
- Do NOT delete existing findings (correct in new findings)
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
