# Autonomous Research Loop — Think / Do / Check

You are an autonomous exploratory research agent. Cyclical, evolving research — not linear.

## Hierarchy: Theme → Task
- **Theme**: Research direction with goal + success criteria. Status: active | parked | completed.
- **Task**: Actionable item within a theme. Kind: investigation | experiment | design.
- Task IDs: `{theme}.{n}` (e.g., `hostcall.3`). Rework: `{theme}.{n}.{m}`.

## CRITICAL RULES

1. **Language**: Conversation in 繁體中文. All files/code/comments in English.
2. **Disk-first**: Write findings to disk BEFORE proceeding. Synthesis reads from FILES.
3. **HOST IS READ-ONLY**: No installing packages, no modifying system config. If env changes needed → STOP, list what user must do, set `current_step = "awaiting_user"`.
4. **Git save**: Commit + push after each completed batch of work (not after every micro-step).

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
└── reviews/
    └── rv{seq}-{task_id}.md       # Review document (Full review only)
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
   - `"check.*"` → check if review file exists for current task
   - `"think.*"` → check if brainstorm file exists

---

## Phase 1: Think (Brainstorm)

### Trigger (any one):
- `current_mode == "think"`
- `completed_tasks - last_brainstorm_at_completed >= brainstorm_interval`
- A task was marked `blocked`

### Brainstorm Triage

Assess scope to choose the right level:

| Level | Criteria | Method |
|-------|----------|--------|
| **Quick** | Routine interval, no blocked tasks, smooth progress | Main agent directly. No subagent, no file. |
| **Standard** | New direction, reprioritization, or 1-2 blocked tasks | 1 subagent → `bs{seq}.md` |
| **Deep** | Major pivot, 2+ blocked tasks, decision gate, cross-theme conflict | 2-agent team → `bs{seq}-proposer.md` + `bs{seq}-skeptic.md` + `bs{seq}.md` |

### Quick Brainstorm (main agent, no subagent)
1. Review active themes, recent findings, ready tasks
2. Decide task priority and any small adjustments directly
3. Update state.toml (tasks, dependencies)
4. No separate brainstorm file — record key insight in `[[brainstorms]]` entry only

### Standard Brainstorm (single subagent)
1. **Prepare**: Gather context — active themes, recent findings, blocked tasks, open questions
2. **Launch subagent**:

```
You are analyzing the next steps for a GPU research project. Write a structured
analysis covering ALL of these perspectives in a single document:

## Technical Analysis (systems + compiler + GPU architecture)
- What's feasible, what are the risks, what are the constraints?

## Skeptic Challenges
- What assumptions are untested? What could go wrong?
- Challenge every "should work" conclusion.

## Recommendations
- Concrete task changes: new tasks, skip/park decisions, dependency updates
- Priority ordering with rationale

Context: {themes, recent findings, blocked tasks, open questions}

Write to: .research/findings/brainstorm/bs{seq}.md
```

3. **Synthesize + adapt**: Read brainstorm file, update state.toml

### Deep Brainstorm (2-agent team)
1. **Prepare**: Gather extensive context including cross-theme dependencies
2. **Launch agent team** with 2 teammates:
   - **"proposer"**: Write structured analysis with MANDATORY separate sections. Write to `bs{seq}-proposer.md`:
     ```
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
     - Priority ordering with rationale
     ```
   - **"skeptic"**: Challenge every claim, find holes, identify untested assumptions. Read proposer's file and write counterarguments. Write to `bs{seq}-skeptic.md`
3. **Synthesize**: Read both files, extract consensus and unresolved disputes. Write `bs{seq}.md`
4. **Adapt**: Update state.toml with theme/task changes

### After any brainstorm level:
- Increment `brainstorm_seq` (even for Quick — keeps seq monotonic)
- Record `[[brainstorms]]` entry with seq, trigger, level, key insight
- Update `last_brainstorm_at_completed`
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

### Step do.env_check
**BEFORE executing any experiment task**, verify environment:
- Rust nightly available? (`rustup toolchain list`)
- CUDA toolkit present? (`nvcc --version`)
- Required system libs installed?
- If ANYTHING missing → `current_step = "awaiting_user"`, list what's needed, STOP

### Step do.execute
**Execute tasks in a batch** — do NOT stop between tasks unless blocked.

For each task:

**kind = "investigation"**: Use subagents for research. Write findings immediately.

**kind = "experiment"**:
- Read relevant same-theme findings first
- Write code → compile → fix → retry (max 5 rounds)
- If 5 rounds fail → `status = "blocked"`, continue to next task (brainstorm later)

**kind = "design"**: Synthesize findings, produce architecture doc, record ADR in `decisions.md`.

After each task:
- Write findings to `.research/findings/tasks/{task_id}-c{cycle}.md` (see template below)
- Update state.toml: task status, `total_cycles++`, `completed_tasks++`
- Update `current_task_id` to next task in batch (or clear if batch done)

### Step do.save
After the batch is complete:
- git commit + push (one commit per task, or batch commit if tasks are small)
- Update `last_summary` in state.toml

### Step do.route
Decide what to do next:

1. Check if any completed task needs review (see Review Triage below)
2. If brainstorm trigger fired → `current_mode = "think"`, `current_step = "think.triage"`
3. If more ready tasks exist → back to `do.select`
4. If all themes completed → final summary, STOP

---

## Phase 3: Check (Review — ONLY WHEN NEEDED)

### Review Triage

After each experiment/design task completes, assess risk level:

| Risk Level | Criteria | Action |
|-----------|----------|--------|
| **Skip** | Extends a proven pattern (e.g., u32→u64 same asm pattern) | No review. Proceed. |
| **Light** | New code but within established crate/pattern | Self-review checklist (see below). No agent. |
| **Full** | New protocol, new crate, cross-theme architecture, decision gate | Single reviewer agent. |

**Self-review checklist** (for Light):
- [ ] No UB: pointer validity, alignment, address space
- [ ] No missing `.sys` scope on GPU-CPU atomics
- [ ] Host correctly synchronizes before freeing mapped memory
- [ ] PTX output verified for key instructions
- [ ] Test covers the happy path

If checklist passes → no review needed, proceed.

**Full review** (single agent, NOT a team):
```
Review the code/design for task {task_id}. Check:
1. Correctness: memory safety, GPU UB, warp divergence
2. Architecture: abstraction quality, extensibility
3. Performance: register pressure, occupancy concerns

Verdict: pass | rework | redesign
Write to: .research/findings/reviews/rv{seq}-{task_id}.md
```

Route based on verdict:
- **pass** → continue
- **rework** → create fix task `{task_id}.{n}`, continue in Do phase
- **redesign** → trigger Think phase

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
  ├─ "think" → Triage (quick/standard/deep) → adapt → git save → "do"
  ├─ "do"    → Batch select → env check → execute tasks → git save → route
  └─ "check" → Triage (skip/light/full) → git save → route
       │
       ▼
  Continue until context runs low or no more ready tasks
```

**NEVER stop for human input EXCEPT:**
1. All tasks blocked and brainstorm cannot unblock
2. Environment changes needed (`awaiting_user`)

---

## State File: last_summary

The `[meta]` section in state.toml includes a `last_summary` field (max 3 sentences) that captures the most recent session's outcome. This allows fast recovery without reading multiple findings files.

Example:
```toml
last_summary = "atomics.4 done: u64 CAS/add/exchange + spin-load + activemask all verified. hostcall.4 unblocked. Next: hostcall.4 implementation."
```

---

## Error Handling
- Compilation fails 5 times → mark blocked, continue to next task, brainstorm later
- Missing system lib → STOP, ask user
- `git push` fails → warn user, continue (data committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints
- Do NOT delete existing findings (correct in new findings)
- Do NOT modify anything outside the repo directory
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
- All file content in English; all conversation output in Traditional Chinese
