# Dev Brainstorm — Think Phase

Dispatched from `dev.md` ROUTE when a brainstorm trigger fires.

## Pre-brainstorm (MANDATORY)

1. Read all `[[epics]]` with `status = "active"`, sorted by tier (T0 first)
2. **Tier gate**: If ANY T0 epic has unmet success criteria → brainstorm MUST focus on T0 only
3. Prepare context: active epics summary, theme syntheses, blocked tasks, open questions
4. All recommendations must reference which epic (and tier) they serve

## Level Selection

| Level | When | Method |
|-------|------|--------|
| **Standard** | Proactive trigger (tasks>=10), single blocked task, theme completed | 1 subagent |
| **High** | Multiple blocked tasks, reprioritization, direction uncertainty | 2 subagents |
| **Deep** | Epic-level pivot, cross-epic conflict, major architecture, user request | 3-4+ experts via TeamCreate |

## Standard (1 subagent)

Dispatch subagent with context (epics, theme syntheses, blocked tasks, open questions).
Subagent writes structured analysis to `.research/findings/brainstorm/bs{N}.md`:
- Epic progress assessment, technical feasibility, risks, skeptic challenges, recommendations.
Read output. Update state.toml with task/theme changes.

## High (Proposer + Skeptic)

1. Dispatch **proposer** subagent → `bs{N}-proposer.md`
   Epic Assessment + Systems/Compiler/GPU Architecture Analysis + Concrete Recommendations
2. Dispatch **skeptic** subagent (reads proposer's file) → `bs{N}-skeptic.md`
   Challenge every claim, find holes, identify untested assumptions
3. Read both files. Write synthesis → `bs{N}.md`. Update state.toml.

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

- Increment `brainstorm_seq`, reset `tasks_since_brainstorm = 0`
- Record `[[brainstorms]]` entry with seq, trigger, level, key_insight
- Commit + push
- Transition → `current_step = "do.select"`
- **MUST produce** at least one new task, theme, or epic. No empty brainstorms.
