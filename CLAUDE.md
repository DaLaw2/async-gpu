# async_gpu — Rust Async/Await on GPU

## Language Convention
- **Conversation**: Always use Traditional Chinese (繁體中文)
- **Files, docs, comments, code**: Always use English
- This applies to ALL generated content without exception

## Dependency Policy
- **anyhow is PROHIBITED** — do not use `anyhow` in any crate
- Use `thiserror` for custom error types, or plain `Result<T, E>` with concrete error types
- Existing `anyhow` usage must be migrated when touched

## CRITICAL: Host Environment Policy
- The research workflow MUST NOT modify anything outside this repository
- No installing packages, toolchains, or system libraries
- No modifying PATH, environment variables, or system config
- If environment changes are needed → STOP and ask the user to perform them

## Goal
Reproduce VectorWare's technology: Rust std + async/await on GPU.
Exploratory research with no fixed endpoint.

## Reference Articles
- https://www.vectorware.com/blog/rust-std-on-gpu/
- https://www.vectorware.com/blog/async-await-on-gpu/

## Hierarchy: Epic → Theme → Task
- **Epic**: User-defined high-level goal. Only the user can create or close epics. Brainstorm CANNOT park, complete, or delete epics.
- **Theme**: A research direction with goal + success criteria. Can be active/parked/completed. References parent epic.
- **Task**: An actionable item within a theme (investigation/experiment/design).
- Task IDs prefixed with theme: `toolchain.1`, `hostcall.3`
- Rework suffix: `toolchain.4.1`
- Brainstorm can add new themes or park existing ones, but must align with active epics.

## Project Structure
```
async_gpu/
├── CLAUDE.md
├── .claude/
│   ├── settings.json
│   └── commands/
│       └── research.md                    # /research loop engine
├── .research/
│   ├── state.toml                         # State machine (SSOT) + last_summary
│   │   ├── [[epics]]                      #   User-defined high-level goals
│   │   ├── [[themes]]                     #   Research directions (under epics)
│   │   └── [[tasks]]                      #   Actionable items per theme
│   ├── decisions.md                       # Architecture Decision Records
│   └── findings/
│       ├── brainstorm/bs{seq}.md          # Brainstorm document
│       ├── tasks/{task_id}-c{cycle}.md    # Task findings
│       └── reviews/rv{seq}-{task_id}.md   # Review document
├── crates/                                # Implementation code
└── examples/
```

## Autonomous Research Workflow

### Core Loop: Think → Do → Check

| Phase | What | How |
|-------|------|-----|
| **Think** | Brainstorm (tiered) | Quick (inline) / Standard (1 agent) / Deep (2-agent team) |
| **Do** | Research/Experiment | Batch execute multiple tasks per session |
| **Check** | Review (tiered) | Skip / self-checklist / single-agent review based on risk |

### Review Triage
- **Skip**: Extends proven pattern (e.g., u32→u64 same asm) → no review
- **Light**: New code in established crate → self-review checklist
- **Full**: New protocol/crate/architecture/decision gate → single reviewer agent

### Task Selection
1. Only from `active` themes
2. Priority: brainstorm-spawned > review-spawned > initial
3. Within theme: investigation before experiment
4. Batch: execute all ready tasks in one session, don't stop between tasks

### Launch
```
/research
```

## Git Remote
- Origin: https://github.com/DaLaw2/async-gpu.git
- Commit + push after each batch of work
