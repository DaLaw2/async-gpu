# async_gpu — Rust Async/Await on GPU

## Language Convention
- **Conversation**: Always use Traditional Chinese (繁體中文)
- **Files, docs, comments, code**: Always use English
- This applies to ALL generated content without exception

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

## Hierarchy: Theme → Task
- **Theme**: A research direction with goal + success criteria. Can be active/parked/completed.
- **Task**: An actionable item within a theme (investigation/experiment/design).
- Task IDs prefixed with theme: `toolchain.1`, `hostcall.3`
- Rework suffix: `toolchain.4.1`
- Brainstorm can add new themes or park existing ones.

## Project Structure
```
async_gpu/
├── CLAUDE.md
├── .claude/
│   ├── settings.json                      # Agent Teams enabled
│   └── commands/
│       └── research.md                    # /research loop engine
├── .research/
│   ├── state.toml                         # State machine (SSOT)
│   │   ├── [[themes]]                     #   Research directions
│   │   └── [[tasks]]                      #   Actionable items per theme
│   ├── decisions.md                       # Architecture Decision Records
│   └── findings/
│       ├── brainstorm/bs{seq}-*.md        # Brainstorm outputs
│       ├── tasks/{task_id}-c{cycle}.md    # Task findings
│       └── reviews/rv{seq}-{id}-*.md      # Review outputs
├── crates/                                # Implementation code
└── examples/
```

## Autonomous Research Workflow

### Core Loop: Think → Do → Check → Adapt

| Phase | What | How |
|-------|------|-----|
| **Think** | Brainstorm | Agent Team: 4 teammates debate; can add/park themes, add/skip tasks |
| **Do** | Research/Experiment | Select from active themes; parallel across themes |
| **Check** | Code Review | Agent Team: 3 reviewers cross-discuss |
| **Adapt** | Update Plan | Embedded in Think (themes+tasks) and Check (rework/redesign) |
| **Save** | Git commit+push | After every completed phase |

### Task Selection (Theme-Aware)
1. Only from `active` themes
2. Priority: brainstorm-spawned > review-spawned > initial
3. Within theme: investigation before experiment (research before build)
4. Cross-theme parallelism when tasks are independent

### Launch
```
/research
```

## Git Remote
- Origin: https://github.com/DaLaw2/async-gpu.git
- Auto-commit and push after each phase completion
