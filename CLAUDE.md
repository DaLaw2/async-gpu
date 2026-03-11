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
- This is a hard rule — violating it can break the user's system

## Goal
Reproduce VectorWare's technology: Rust std + async/await on GPU.
Exploratory research with no fixed endpoint.

## Reference Articles
- https://www.vectorware.com/blog/rust-std-on-gpu/
- https://www.vectorware.com/blog/async-await-on-gpu/

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
│   ├── decisions.md                       # Architecture Decision Records
│   └── findings/
│       ├── brainstorm/
│       │   ├── bs{seq}-{role}.md          # Individual teammate output
│       │   └── bs{seq}-synthesis.md       # Lead's combined synthesis
│       ├── tasks/
│       │   └── {task_id}-c{cycle}.md      # Task findings at cycle N
│       └── reviews/
│           ├── rv{seq}-{id}-{role}.md     # Individual reviewer output
│           └── rv{seq}-{id}-synthesis.md  # Combined verdict
├── crates/                                # Implementation code
└── examples/
```

## Autonomous Research Workflow

### Agent Teams (Experimental Feature)
- Enabled via `.claude/settings.json`
- Used for **Think** (brainstorm) and **Check** (code review) phases
- Teammates can read each other's output, challenge assumptions, and debate directly
- This is fundamentally different from independent subagents — teammates collaborate
- Always clean up teams after each phase

### Core Loop: Think → Do → Check → Adapt

| Phase | What | How |
|-------|------|-----|
| **Think** | Brainstorm | Agent Team: 4 teammates (Systems/Compiler/GPU/Skeptic) debate each other |
| **Do** | Research/Experiment | Agent Team for parallel research; direct for experiments |
| **Check** | Code Review | Agent Team: 3 reviewers (Correctness/Architecture/Performance) cross-discuss |
| **Adapt** | Update Plan | Lead synthesizes from teammate files, updates task list |
| **Save** | Git commit+push | After every completed phase |

### When to Use Agent Teams vs. Subagents
- **Agent Team**: When debate/challenge is valuable (brainstorm, review)
- **Subagent**: When just fetching info independently (simple lookup)
- **Direct**: When doing it yourself is faster (single experiment)

### Compression Resilience
- `current_step` tracks micro-state within each phase
- Teammate results → individual files → synthesis reads from files (never from context)
- Recovery protocol runs first every time: re-reads state.toml + checks file existence

### Multi-Iteration Naming
- Brainstorms: `bs{seq}` — seq increments per session
- Reviews: `rv{seq}` — seq increments per session
- Tasks: `{task_id}-c{cycle}` — cycle is when it was completed
- Rework: new task ID (e.g., `1.4.1`), so history is never overwritten

### Launch
```
/research
```

## Git Remote
- Origin: https://github.com/DaLaw2/async-gpu.git
- Auto-commit and push after each phase completion
