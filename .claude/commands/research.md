# Autonomous Research Loop — Think / Do

You are an autonomous exploratory research agent. Cyclical, evolving research — not linear.

## Hierarchy
See CLAUDE.md for full Epic/Theme/Task definitions. Key workflow rules:
- **Epic**: Tiered (T0-T3). Agent may create epics only via brainstorm. Only the user can close them. Status: active | pending | completed.
- **Tier enforcement**: T0 epics take absolute priority. NEVER start T(N+1) work while T(N) has unmet criteria, unless T(N) is blocked on external factors. This is not a suggestion — it is a hard constraint.
- **North Star**: Every epic has a one-sentence user-facing vision. When a task's outcome could go multiple ways, choose the path aligned with the North Star, not just the success criteria letter.
- **depends_on**: Epic-level dependency. An epic with `depends_on` stays `pending` until dependencies are `completed`.
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
.research/
├── context.md                      # Rolling context (replaces last_summary)
├── state.toml                      # Active items only (completed → archive/)
├── decisions.md                    # Architecture Decision Records
├── findings/
│   ├── brainstorm/
│   │   ├── bs{seq}.md              # Standard brainstorm (single file)
│   │   ├── bs{seq}-proposer.md     # High brainstorm: proposer analysis
│   │   ├── bs{seq}-skeptic.md      # High brainstorm: skeptic challenges
│   │   ├── bs{seq}-{role}.md       # Deep brainstorm: expert analysis (Round 1)
│   │   └── bs{seq}-{role}-review.md # Deep brainstorm: cross-review (Round 2)
│   ├── tasks/
│   │   └── {task_id}-c{cycle}.md   # Task findings
│   └── themes/
│       └── {theme_id}-synthesis.md # Theme synthesis (rewritten, not appended)
├── golden/                         # Golden test baselines
└── archive/                        # Completed epics, themes, tasks, brainstorms
```

---

## Recovery Protocol (ALWAYS RUN FIRST)

1. Read `.research/context.md` → get strategic context, recent decisions, active constraints
2. Read `.research/state.toml` → check `current_mode`, `current_step`, `current_task_id`
3. If `current_step == "awaiting_user"` → output pending request, STOP
4. If context.md is sufficient → proceed directly
5. Otherwise read active theme syntheses from `findings/themes/` (current theme + related themes only)
6. Resume from `current_step`:
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
- **Proactive**: `tasks_since_brainstorm >= 10`
- User explicitly requests brainstorm

### Epic Check (MANDATORY before every brainstorm)
1. Read all `[[epics]]` with `status = "active"`, sorted by tier (T0 first)
2. **Tier gate**: If ANY T0 epic has unmet success criteria, brainstorm MUST focus on T0. Do NOT spawn work for T1+ unless all T0 epics are satisfied or explicitly blocked.
3. Verify: are there themes/tasks actively working toward each active epic?
4. If an active epic has no active themes → brainstorm MUST spawn themes/tasks for it
5. **North Star alignment**: For each recommendation, verify it serves the epic's North Star (the user-facing vision), not just the success criteria letter
6. **depends_on check**: Do NOT activate epics whose dependencies are unmet. Mark them `pending`.
7. All recommendations must reference which epic (and tier) they serve

### Brainstorm Triage

Assess scope to choose the right level:

| Level | Criteria | Method |
|-------|----------|--------|
| **Standard** | Routine (proactive trigger), single blocked task, theme completed | 1 subagent → `bs{N}.md` |
| **High** | Multiple blocked tasks, reprioritization, direction uncertainty | 2-agent Proposer + Skeptic |
| **Deep** | Epic-level pivot, cross-epic conflict, major architecture decision, user request | 3-4+ expert Agent Team |

### Standard Brainstorm (single subagent)
1. **Prepare**: Gather context — active epics, active theme syntheses, blocked tasks, open questions
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

Context: {epics, theme syntheses, blocked tasks, open questions}

Write to: .research/findings/brainstorm/bs{seq}.md
```

3. **Synthesize + adapt**: Read brainstorm file, update state.toml

### High Brainstorm (2-agent Proposer + Skeptic)
1. **Prepare**: Gather context including cross-theme dependencies and theme syntheses
2. **Launch agent team** with 2 teammates:
   - **"proposer"**: Write structured analysis with MANDATORY separate sections. Write to `bs{N}-proposer.md`:
     ```
     ## Active Epics Assessment
     - Progress toward each epic's success criteria
     - Gaps and unaddressed criteria

     ## Systems Analysis (memory models, ABI, unsafe boundaries)
     - What's feasible? What are the constraints?

     ## Compiler Analysis (rustc, LLVM, codegen, PTX backend)
     - Fundamental compiler limitations, IR transformations, target support

     ## GPU Architecture Analysis (warp model, memory hierarchy, occupancy)
     - CPU assumptions that break on GPU, hardware constraints

     ## Concrete Recommendations
     - New tasks, skip/park decisions, dependency updates
     - Each must reference which epic it serves
     - Priority ordering with rationale
     ```
   - **"skeptic"**: Challenge every claim, find holes, identify untested assumptions. Read proposer's file and write counterarguments to `bs{N}-skeptic.md`
3. **Synthesize**: Read both files, extract consensus and unresolved disputes. Write `bs{N}.md`
4. **Adapt**: Update state.toml with theme/task changes

### Deep Brainstorm (3-4+ Expert Agent Team via TeamCreate)

For major decisions requiring multi-perspective analysis with real discussion.
Uses `TeamCreate` for structured coordination: shared task list, message passing, task ownership.

**Expert Role Pool** (select 3-4 most relevant per topic):

| Role | Expertise |
|------|-----------|
| **Systems Architect** | Memory safety, ABI, unsafe boundaries, concurrency models |
| **Compiler Engineer** | rustc internals, LLVM, PTX codegen, MIR passes |
| **GPU Architect** | Warp execution model, memory hierarchy, occupancy, CUDA semantics |
| **Performance Engineer** | Benchmarking, bottleneck analysis, optimization strategies |
| **API Designer** | Developer experience, public API design, ergonomics |

**Step 1: Setup Team**
1. `TeamCreate` with `team_name = "bs{N}"`, description = topic summary
2. `TaskCreate` for each expert — Round 1 tasks:
   - `"[R1] {role}: Independent Analysis"` — one per expert
3. `TaskCreate` for each expert — Round 2 tasks (blocked on all R1 tasks):
   - `"[R2] {role}: Cross-Review"` — one per expert
4. `TaskCreate` for synthesis:
   - `"[R3] Synthesis"` — blocked on all R2 tasks, owned by team lead

**Step 2: Spawn Expert Teammates**
Launch each expert via `Agent` with `team_name = "bs{N}"`:
```
Agent({
  name: "{role}",
  team_name: "bs{N}",
  prompt: """
  You are a {role} on an expert panel analyzing a GPU async/await research project.
  Team: bs{N}. Read the team config and task list to find your assignments.

  ## Your workflow:
  1. Check TaskList — claim your [R1] task via TaskUpdate
  2. Read the context files provided below
  3. Write your independent analysis to: .research/findings/brainstorm/bs{N}-{role}.md
     - Assessment of the current state from your domain expertise
     - Risks and constraints in your domain
     - Concrete recommendations with rationale
     - What other perspectives might miss from your vantage point
     - Be specific and opinionated. Flag assumptions that need testing.
  4. Mark your [R1] task completed
  5. Check TaskList — when your [R2] task is unblocked, claim it
  6. Read ALL other experts' Round 1 files (bs{N}-*.md, excluding your own)
  7. Write your cross-review to: .research/findings/brainstorm/bs{N}-{role}-review.md
     - Where you agree with other experts (and why)
     - Where you disagree (with specific counterarguments)
     - Blind spots in others' analyses that your expertise reveals
     - New insights triggered by reading others' perspectives
     - Updated recommendations (if your position changed)
  8. Mark your [R2] task completed

  Context:
  - Active epics: {epics summary}
  - Theme syntheses: {list of relevant synthesis files to read}
  - Specific question: {topic}
  """
})
```

**Step 3: Synthesis (team lead)**
Once all R2 tasks complete, the main agent (team lead) reads all files and writes `bs{N}.md`:

```markdown
# Deep Brainstorm {N}: {Topic}

## Panel
- {role1}: {one-line stance}
- {role2}: {one-line stance}
- ...

## Consensus
Conclusions all experts agree on.

## Disagreements
Who disagrees with whom, each side's reasoning.

## Resolved
Disputes settled during Round 2, with the winning argument.

## Unresolved
Points still contested + recommended decision approach.

## Recommendations
Concrete task/theme/epic changes.
Each item: confidence level + which experts support/oppose.
```

**Step 4: Shutdown**
- Send `{type: "shutdown_request"}` to all teammates via SendMessage
- Delete the team (or let it expire)

### After any brainstorm level:
- Increment `brainstorm_seq` (keeps seq monotonic)
- Reset `tasks_since_brainstorm = 0`
- Record `[[brainstorms]]` entry with seq, trigger, level, key insight
- git commit + push
- Transition → `current_mode = "do"`, `current_step = "do.select"`

---

## Phase 2: Do (Execute Tasks)

### Step do.select
Task selection:
1. Filter: `status == "pending"` AND all deps `"done"` AND theme `"active"`
2. **Tier priority**: T0 tasks ALWAYS before T1, T1 before T2, etc. Within same tier:
   brainstorm-spawned > theme momentum > investigation before experiment
3. **Batch selection**: Pick ALL ready tasks for this session. Group by:
   - Same-theme tasks → execute sequentially
   - Cross-theme independent tasks → can parallelize with subagents
4. **Pre-read**: Read theme synthesis for each selected task's theme (`findings/themes/{theme_id}-synthesis.md`)
5. Set selected tasks `status = "active"`, update `current_task_id` to first task
6. Update `current_step = "do.execute"`

### Step do.execute
**Execute tasks in a batch** — do NOT stop between tasks unless blocked.

For each task:

**kind = "investigation"**: Use subagents for research. Write findings immediately.

**kind = "experiment"**:
- Read theme synthesis first (not individual findings)
- **Baseline-first**: Before changes, run the relevant test/benchmark to capture baseline. Record in findings file.
- Write code → compile → fix → retry
- **Smart bailout** — classify failure after each attempt:

| Failure Type | Max Attempts | Rationale |
|-------------|-------------|-----------|
| Syntax/typo | 5 | Mechanical fix, likely to converge |
| Missing API/feature | 2 | Architectural gap, more attempts won't help |
| Linker/ABI/backend bug | 2 | Toolchain limitation, not a code problem |
| Wrong output/logic | 3 | May need different approach |
| Crash/segfault | 2 | Usually fundamental issue |

Count **distinct approaches**, not retries of the same fix with tweaks.
- If max attempts exceeded → `status = "blocked"`, `git reset` to pre-experiment commit, continue to next task
- **Output redirection**: For long-running commands, redirect to `.research/run.log`, grep key results. Delete log after use.

**kind = "design"**: Synthesize from theme synthesis, produce architecture doc, record ADR in `decisions.md`.

**Verification (replaces old Check phase):**
After each experiment/design task, BEFORE marking done:
- Tests pass? (machine-verifiable)
- PTX output contains expected instructions? (grep)
- Benchmark improved vs baseline? (numeric comparison)
- No regression on existing tests? (cargo test on affected crate)

If verification fails → don't mark done, continue fixing (counts toward attempt limit).
If verification passes → mark `status = "done"`.

After each completed task:
- **Lint** (experiment/design only): `cargo +stable fmt --check` and `cargo +stable clippy -- -D warnings` on modified crates. Fix before proceeding.
- Write findings to `.research/findings/tasks/{task_id}-c{cycle}.md`
- Update state.toml: task status, `total_cycles++`, `completed_tasks++`, `tasks_since_brainstorm++`
- Update `current_task_id` to next task in batch (or clear if batch done)

### Step do.synthesize (NEW)
After each completed task, update the theme's synthesis file:

1. Read current `findings/themes/{theme_id}-synthesis.md` (or create if first task)
2. **Rewrite** (not append) with current state of the theme:

```markdown
# {theme_id}: {title}
**Epic**: {epic} | **Status**: {status} | **Updated**: {date}

## Progress
What's been accomplished in this theme so far.

## Verified Conclusions
Facts established by experiments with high confidence.

## Rejected Approaches
What was tried and why it didn't work (prevents re-treading).

## Open Questions
Remaining unknowns that affect downstream tasks.

## Key Metrics
Quantitative results (GFLOPS, latency, accuracy — whatever's relevant).

## Next Steps
What the remaining tasks in this theme should focus on.
```

3. Keep each synthesis to **30 lines max** — this is a summary, not a log.

### Step do.save
After the batch is complete:
1. **Maintenance**: Invoke `/maintain` for housekeeping (CI sync, archive, README). Fix issues.
2. **Auto-archive**: Move all newly completed epics/themes/done tasks to `archive/`. state.toml must stay under ~300 lines.
3. **Pre-push check**: Run `bash scripts/pre-push.sh`. Fix failures. Include generated files in commit.
4. **Commit + push**: one commit per task, or batch commit if tasks are small.
5. **CI check**: `gh run list --limit 1` to verify. If failure, `gh run view <id> --log-failed | tail -30`. Fix before proceeding.
6. **Update context.md**: Rewrite `.research/context.md` with current strategic state (see format below).

### Step do.route
Decide what to do next:

1. **North Star check**: Does the completed task's outcome align with its epic's North Star? If the task technically passes but drifts from the spirit → flag in context.md, consider course correction.
2. **Strategic assessment** on completed tasks:
   - Did this result significantly change direction? → Think
   - Consecutive failures (3+ blocked tasks in same theme) without progress? → Think
   - Theme synthesis shows original goal needs revision? → Think
3. **Tier promotion**: If all T(N) epics are now satisfied, activate T(N+1) epics (check depends_on first).
4. If any brainstorm trigger fired → `current_mode = "think"`, `current_step = "think.triage"`
5. If more ready tasks exist → back to `do.select`
6. If no ready tasks but active epics have UNMET success criteria → `current_mode = "think"`
7. If ALL active epics have ALL success criteria met → report to user, STOP

**CRITICAL**: Completing all current themes/tasks does NOT mean stop. Check epic success criteria. If any criterion is unmet, brainstorm MUST generate new work. Respect tier ordering — don't skip ahead.

---

## context.md Format

Rewritten at every `do.save`. Max ~50 lines. Structure:

```markdown
## Current Focus
What we're working on right now and why. (2-3 sentences)

## Recent Decisions
Last 3-5 key decisions with brief rationale.
- {date}: {decision} — {why}

## Tried & Rejected
Approaches ruled out recently. Prevents recovery from re-walking dead ends.
- {approach}: {why it failed}

## Active Constraints
Known hard limits (compiler bugs, hardware limitations, toolchain issues).

## Key Metrics
Current benchmark numbers for quick reference.

## Next
Immediate next steps when resuming.
```

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
  ├─ "think" → Epic check → Triage (standard/high/deep) → adapt → git save → "do"
  └─ "do"    → Batch select → execute+verify → synthesize → git save → route
                                                                │
       ┌────────────────────────────────────────────────────────┘
       ▼
  route decides:
       ├─ strategic concern      → "think"
       ├─ brainstorm trigger     → "think"
       ├─ more ready tasks       → "do" (do.select)
       ├─ no tasks, unmet epics  → "think"
       └─ all epics satisfied    → STOP
```

**NEVER stop for human input EXCEPT:**
1. All tasks blocked and brainstorm cannot unblock
2. Environment changes needed (`awaiting_user`)
3. ALL active epics have ALL success criteria met

**On session recovery**: Do NOT wait for user confirmation. Read `context.md`, check `current_step`, IMMEDIATELY continue.

---

## Error Handling
- Experiment exceeds attempt limit → mark blocked, `git reset` to pre-experiment commit, continue to next task, brainstorm later
- Missing system lib → STOP, ask user
- `git push` fails → warn user, continue (data committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints
- Do NOT delete existing findings (correct in new findings)
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
