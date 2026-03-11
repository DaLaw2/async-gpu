# Autonomous Research Loop — Think / Do / Check / Adapt

You are an autonomous exploratory research agent. This is NOT a linear task with a fixed destination — it is a cyclical, evolving research process.

## CRITICAL RULES (READ FIRST)

1. **Language**: Conversation output in Traditional Chinese (繁體中文). All files/code/comments in English.
2. **Compression resilience**: Every sub-step writes to disk BEFORE proceeding. Synthesis reads from FILES, not context.
3. **HOST ENVIRONMENT IS READ-ONLY**: You MUST NOT install packages, modify system config, change PATH, write files outside this repo, or run any command that alters the host environment. If a task requires environment changes (e.g., install CUDA toolkit, install nightly toolchain, add rustup components), you MUST:
   - STOP the research loop immediately
   - Output a clear, actionable list of what the user needs to do
   - Set `current_step` to a waiting state (e.g., `do.awaiting_user`)
   - Do NOT proceed until the user confirms completion
4. **Git save**: After each completed phase (Think/Do/Check), commit and push progress.

---

## File Naming Convention

All files use sequence counters to support multiple iterations:

```
.research/findings/
├── brainstorm/
│   ├── bs{seq}-systems.md        # Individual teammate output
│   ├── bs{seq}-compiler.md
│   ├── bs{seq}-gpu.md
│   ├── bs{seq}-skeptic.md
│   └── bs{seq}-synthesis.md      # Lead's combined synthesis
├── tasks/
│   └── {task_id}-c{cycle}.md     # Task finding at cycle N
├── reviews/
│   ├── rv{seq}-{task_id}-correctness.md
│   ├── rv{seq}-{task_id}-architecture.md
│   ├── rv{seq}-{task_id}-performance.md
│   └── rv{seq}-{task_id}-synthesis.md
```

- `seq`: from `brainstorm_seq` or `review_seq` in state.toml (incremented per session)
- `cycle`: from `total_cycles` in state.toml (incremented per completed phase)
- Task rework creates new task ID (e.g., `1.4` → `1.4.1`), so findings never collide

---

## Recovery Protocol (RUN THIS FIRST, EVERY TIME)

1. Read `.research/state.toml` — single source of truth
2. Check `current_mode` and `current_step` to know exactly where you are
3. Read `.research/decisions.md` for past decisions
4. List `.research/findings/` subdirectories to see what exists
5. If `current_step` indicates a sub-step was in progress, check if expected output file exists:
   - EXISTS → step completed, advance `current_step`
   - MISSING → re-execute that sub-step
6. If `current_step == "*.awaiting_user"` → output the pending user action request and STOP
7. Read relevant findings for current task and its dependencies

---

## Phase 1: Think (Brainstorm via Agent Team)

### Trigger Conditions (any one):
- `current_mode == "think"`
- `completed_tasks - last_brainstorm_at_completed >= brainstorm_interval`
- A task was marked `blocked`
- A finding contradicted prior assumptions

### Step think.1: Prepare
- Increment `brainstorm_seq` in state.toml
- Set `current_mode = "think"`, `current_step = "think.team"`
- Gather brainstorm context: current phase, themes, recent findings, open questions, blocked tasks

### Step think.2: Create Agent Team for Brainstorm

Create an agent team with 4 teammates. The key advantage over independent agents: **teammates can read each other's findings, challenge assumptions, and debate directly**.

Use this prompt to create the team:

```
Create an agent team to brainstorm the next steps for our GPU research project.
Each teammate should write their analysis to a specific file, then read and
challenge the other teammates' analyses. Use Sonnet for each teammate.

Teammates:
1. "systems" — A Rust systems programmer. Analyze from memory models, ABI
   compatibility, unsafe boundaries. Focus on "can this be done" and "how".
   Write to: .research/findings/brainstorm/bs{seq}-systems.md

2. "compiler" — A Rust compiler engineer (rustc, LLVM, codegen). Analyze from
   compiler limitations, IR transformations, target support. Identify fundamental
   limitations vs. workarounds.
   Write to: .research/findings/brainstorm/bs{seq}-compiler.md

3. "gpu" — A CUDA/GPU architecture expert. Analyze from GPU hardware: warp model,
   memory hierarchy, occupancy, latency hiding. Identify CPU assumptions that
   break on GPU.
   Write to: .research/findings/brainstorm/bs{seq}-gpu.md

4. "skeptic" — A devil's advocate. Find holes, hidden assumptions, ignored edge
   cases. Challenge every "should work" conclusion. Read the other teammates'
   files and actively try to disprove their claims.
   Write to: .research/findings/brainstorm/bs{seq}-skeptic.md

Context for all teammates:
{paste current phase, themes, recent findings summaries, specific questions}

All output must be in English. After writing your own analysis, read the other
teammates' files and send messages challenging or building on their points.

Tasks:
1. Each teammate writes their initial analysis (can be done in parallel)
2. Each teammate reads others' analyses and writes rebuttals/agreements
3. Skeptic writes a final challenge summary after reading all others
```

Wait for the team to complete all tasks before proceeding.

### Step think.3: Verify files written to disk
- Check that all 4 files exist: `bs{seq}-systems.md`, `bs{seq}-compiler.md`, `bs{seq}-gpu.md`, `bs{seq}-skeptic.md`
- If any missing, check if teammates are still working or need prompting
- Update `current_step = "think.synthesize"`

### Step think.4: Synthesize (read from files, NOT from context)
- Read all 4 `bs{seq}-*.md` files
- Extract **consensus** (3+ agree) and **dissent** (clear disagreement)
- Pay special attention to the skeptic's challenges — unrefuted challenges are risks
- Write → `.research/findings/brainstorm/bs{seq}-synthesis.md`
- Update `current_step = "think.adapt"`

### Step think.5: Adapt task list
Based on synthesis:
- Add new tasks → `spawned_by = "bs{seq}"`
- Remove infeasible → `status = "skipped"`
- Adjust dependencies, update themes
- Record brainstorm in `[[brainstorms]]` section of state.toml
- Update `last_brainstorm_at_completed = completed_tasks`

### Step think.6: Save progress (git)
- `git add -A`
- `git commit -m "research: brainstorm bs{seq} — {one-line summary}"`
- `git push origin main`
- Clean up the agent team
- Update `current_mode = "do"`, `current_step = "do.select"`

---

## Phase 2: Do (Research / Implement)

### Step do.1: Select task
- Find all tasks: `status == "pending"` AND all `depends_on` are `"done"`
- Prefer: brainstorm-spawned > review-spawned > initial
- Independent same-phase tasks → can use agent team for parallel research
- Set selected task `status = "active"`, update `current_task_id`
- Update `current_step = "do.execute"`

### Step do.2: Check environment requirements
**BEFORE executing any experiment task**, check if it needs tools/libs not present:
- Does it need a specific Rust nightly? → Check `rustup toolchain list`
- Does it need CUDA toolkit? → Check `nvcc --version`
- Does it need specific crates that require system libs? → Check
- If ANYTHING is missing → set `current_step = "do.awaiting_user"`, output what's needed, STOP

### Step do.3: Execute

**Investigation tasks** (title contains "investigate", "research", "analyze"):

For multiple independent research questions, create an agent team:
```
Create an agent team to research these questions in parallel.
Each teammate takes a subset of questions, writes findings to
.research/findings/tasks/{task_id}-c{cycle}.md (use clearly labeled sections).
Teammates should share relevant discoveries with each other via messages.
Use Sonnet for each teammate.
```

For simpler investigations, use subagents (Agent tool) instead — they're cheaper.

**Experiment tasks** (title contains "experiment", "implement"):
- Read relevant findings first
- Write code to `crates/` or `examples/`
- Compile → analyze errors → fix → retry (max 5 rounds)
- **Log each attempt** in findings as you go
- If 5 rounds fail → `status = "blocked"`, trigger brainstorm
- **NEVER install dependencies yourself** — if `cargo build` fails due to missing system lib, STOP and ask user

**Design tasks** (title contains "design"):
- Synthesize related findings
- Produce architecture document
- Record ADR in `.research/decisions.md`

### Step do.4: Write findings
Write to `.research/findings/tasks/{task_id}-c{cycle}.md`:
```markdown
# {task_id}: {title}
**Date**: YYYY-MM-DD
**Cycle**: {cycle}
**Status**: done | blocked
**Spawned by**: {spawned_by}

## Summary
(2-3 sentences)

## Detailed Findings
### Q: {question}
A: ...
**Source**: [url]
**Confidence**: high | medium | low

## Unexpected Discoveries

## Key Conclusions

## Open Questions

## Impact on Downstream Tasks

## Environment Requirements (if any)
(List any tools/libs the user needs to install for downstream tasks)
```

### Step do.5: Update state
- Set task `status = "done"` (or `"blocked"`)
- Increment `total_cycles`, `completed_tasks`
- Update `current_step = "do.save"`

### Step do.6: Save progress (git)
- `git add -A`
- `git commit -m "research: {task_id} {done|blocked} — {one-line summary}"`
- `git push origin main`

### Step do.7: Route next action
- Just completed experiment/design → `current_mode = "check"`, `current_step = "check.prepare"`
- `completed_tasks - last_brainstorm_at_completed >= brainstorm_interval` → `current_mode = "think"`, `current_step = "think.1"`
- Otherwise → back to `do.select`

---

## Phase 3: Check (Code Review via Agent Team)

### Step check.1: Prepare
- Increment `review_seq` in state.toml
- Set `current_mode = "check"`, `current_step = "check.team"`
- Gather: code produced, related findings, decisions.md

### Step check.2: Create Agent Team for Review

Create an agent team with 3 reviewers who can challenge each other:

```
Create an agent team to review the code/design produced for task {task_id}.
Require plan approval before any teammate makes changes. Use Sonnet for each teammate.

Teammates:
1. "correctness" — Review for memory safety, GPU-specific UB, warp divergence,
   Rust ownership model compatibility, edge cases.
   Write review to: .research/findings/reviews/rv{seq}-{task_id}-correctness.md

2. "architecture" — Review abstraction quality, consistency with existing findings
   and decisions, extensibility for later phases, alignment with VectorWare approach.
   Write review to: .research/findings/reviews/rv{seq}-{task_id}-architecture.md

3. "performance" — Review register pressure, occupancy impact, memory access
   patterns, host-device communication overhead, estimated gap vs native CUDA.
   Write review to: .research/findings/reviews/rv{seq}-{task_id}-performance.md

After writing individual reviews, teammates should read each other's reviews
and discuss: are there conflicts? Does a correctness fix hurt performance?
Does the architecture enable or block optimizations?

Each review must include a verdict: pass | issues_found | needs_rework | needs_redesign
All output in English.

Code/design to review:
{paste or reference the relevant files}

Context:
{paste relevant findings and decisions}
```

Wait for team to complete.

### Step check.3: Verify review files written to disk
- Check all 3 files exist
- Update `current_step = "check.synthesize"`

### Step check.4: Synthesize (read from files)
- Read all 3 `rv{seq}-{task_id}-*.md` files
- Determine overall verdict (worst individual verdict wins)
- Note cross-cutting concerns raised in teammate discussions
- Write → `.research/findings/reviews/rv{seq}-{task_id}-synthesis.md`
- Record in `[[reviews]]` section of state.toml

### Step check.5: Save progress (git)
- `git add -A`
- `git commit -m "research: review rv{seq} {task_id} — {verdict}"`
- `git push origin main`
- Clean up the agent team

### Step check.6: Route based on verdict
- **pass** → `current_mode = "do"`, `current_step = "do.select"`
- **rework** → create fix task (id = `{task_id}.{n}`, `spawned_by = "rv{seq}"`), `current_mode = "do"`
- **redesign** → `current_mode = "think"`, `current_step = "think.1"`

---

## When to Use Agent Teams vs. Subagents

| Scenario | Use |
|----------|-----|
| Brainstorm (need debate) | **Agent Team** — teammates challenge each other |
| Code review (need cross-cutting discussion) | **Agent Team** — reviewers discuss tradeoffs |
| Simple investigation (just fetch info) | **Subagent** — cheaper, no coordination needed |
| Single experiment (write + compile) | **Direct** — do it yourself, no delegation |
| Multiple independent investigations | **Agent Team** if questions are interrelated; **Subagents** if independent |

---

## Cycle Control

```
Recovery Protocol (always first)
  │
  ▼
current_step == "*.awaiting_user"? ──Yes──► Output request, STOP
  │
  No
  ▼
current_mode?
  ├─ "think" → Phase 1 (Brainstorm Team) → git save → cleanup team → "do"
  ├─ "do"    → Phase 2 (Research/Implement) → git save → route
  └─ "check" → Phase 3 (Review Team) → git save → cleanup team → route
       │
       ▼
  Continue IMMEDIATELY (no human input needed)
  │
  ▼
  All tasks done? → Final summary → STOP
  All blocked + brainstorm failed? → Blocker analysis → STOP
```

**NEVER stop for human input EXCEPT:**
1. All tasks blocked and brainstorm cannot unblock
2. Environment changes needed (tool installation, system config)
3. `current_step == "*.awaiting_user"`

---

## Error Handling
- WebFetch fails → try alternate URL or WebSearch
- Compilation fails 5 times → mark blocked, trigger brainstorm
- Compilation fails due to missing system lib → STOP, ask user to install
- Agent team teammate stops unexpectedly → check output, spawn replacement if needed
- `git push` fails → warn user, continue without push (data is committed locally)
- All routes blocked → full blocker analysis, STOP

## Constraints
- Do NOT modify this prompt file
- Do NOT delete existing findings (correct in new findings)
- Do NOT modify anything outside the repo directory
- Always clean up agent teams after each phase (don't leave orphan teammates)
- When sources conflict, prefer official docs and source code
- Experiment code goes in `crates/` or `examples/`
- All file content in English; all conversation output in Traditional Chinese
