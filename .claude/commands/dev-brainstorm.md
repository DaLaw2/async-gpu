# Dev Brainstorm — Think Phase

Dispatched from `dev.md` when a brainstorm trigger fires.
Triggers checked at GATE (proactive, before work) and ROUTE (reactive, after work).

## Context Gathering (MANDATORY)

1. Read all `[[epics]]` for strategic context (north stars, success criteria)
2. Read all `[[stories]]` with `status = "active"`, sorted by priority (high first)
3. Read `.research/archive/stories-archived.toml` — flag any archived story whose value has changed due to: new capabilities landed since archival, dependency stories that completed and unlock new approaches, or technology shifts. These are reopen candidates.
4. Prepare context: epics + stories summary, feature syntheses, blocked tasks, open questions, reopen candidates

All recommendations must reference which epic and story they serve.

## Level Selection

| Level | When | Method |
|-------|------|--------|
| **Standard** | Proactive trigger (tasks>=10), single blocked task, feature completed | 1 subagent |
| **High** | Multiple blocked tasks, reprioritization, direction uncertainty | 2 subagents |
| **Deep** | Epic-level pivot, cross-epic conflict, major architecture, user request | 3-4+ experts via TeamCreate |

## Standard (1 subagent)

Dispatch subagent with context (epics, stories, feature syntheses, blocked tasks, open questions, reopen candidates).
Subagent writes structured analysis to `.research/findings/brainstorm/bs{N}.md`:
- Epic progress assessment, technical feasibility, risks, skeptic challenges, recommendations.
Read output. Orchestrator updates state.toml based on recommendations.

## High (Proposer + Skeptic)

1. Dispatch **proposer** subagent → `bs{N}-proposer.md`
   Epic Assessment + Systems/Compiler/GPU Architecture Analysis + Concrete Recommendations
2. Dispatch **skeptic** subagent (reads proposer's file) → `bs{N}-skeptic.md`
   Challenge every claim, find holes, identify untested assumptions
3. Read both files. Write synthesis → `bs{N}.md`. Orchestrator updates state.toml.

## Deep (Agent Team via TeamCreate)

Select 3-4 experts from: Systems Architect, Compiler Engineer, GPU Architect, Performance Engineer, API Designer.

1. `TeamCreate` with `team_name = "bs{N}"`
2. Create tasks: [R1] Independent Analysis per expert, [R2] Cross-Review per expert (blocked on R1s), [R3] Synthesis
3. Spawn each expert as teammate. Each expert:
   - R1: Write independent analysis → `bs{N}-{role}.md`
   - R2: Read all R1 files, write cross-review → `bs{N}-{role}-review.md`
4. Main agent reads all files, writes synthesis → `bs{N}.md`
   Structure: Panel (stances), Consensus, Disagreements, Resolved, Unresolved, Recommendations
5. Shutdown team via SendMessage `{type: "shutdown_request"}`

## Post-brainstorm (all levels)

Update state:
- Increment `brainstorm_seq`, reset `tasks_since_brainstorm = 0`
- Record `[[brainstorms]]` entry:
  `seq`, `at_cycle`, `trigger`, `level`, `key_insight`,
  `new_tasks_spawned`, `new_features_spawned`, `new_stories_spawned`,
  `features_parked`, `features_completed`, `stories_completed`
- Transition → `current_step = "gate"` (re-enter loop from top)

### Possible outputs

Brainstorm may produce any combination of:
- **Create** new tasks, features, or stories (within existing epics)
- **Reprioritize** existing tasks or park features
- **Reopen** an archived story — move entry from archive back to state.toml, set `status = "active"`, assign to appropriate epic. Do NOT restore old features/tasks; a follow-up brainstorm creates fresh ones from current codebase state. Record in brainstorm entry: `stories_reopened = ["story-id"]`. Record reason in `context.md` Recent Decisions.

- **Record** strategic insight (no-task brainstorms must still record `key_insight`)

Brainstorm can NOT create new epics — the strategic epics are fixed.

### Output rules by trigger
- **No ready tasks** → **MUST produce** at least one new task (loop is stuck without new work)
- **All other triggers** → outputs are optional. No-task brainstorms must still record `key_insight`.
