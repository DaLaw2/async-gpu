# async_gpu — Rust Async/Await on GPU

## Language Convention
- **Conversation**: Always use Traditional Chinese (繁體中文)
- **Files, docs, comments, code**: Always use English
- This applies to ALL generated content without exception

## Code Quality
- **anyhow is PROHIBITED** — do not use `anyhow` in any crate. Use `thiserror` for custom error types, or plain `Result<T, E>` with concrete error types. Existing `anyhow` usage must be migrated when touched.
- **No `#[allow(dead_code)]`** — remove unused code, don't hide it with `#[allow]`. If it appears unused but is used cross-crate, fix visibility so the compiler sees the usage.

## Hierarchy: Epic → Story → Feature → Task

### Epics
- Strategic milestone with a "before/after" quality shift. Only 3-4 total.
- All strategic epics are active simultaneously (parallel pillars, not sequential).
- Only the user can create or close epics.
- **North Star**: Each epic has a one-sentence vision of what the USER gets.

### Stories
- User need within an epic — "As a Rust dev, I want..."
- Priority (high/medium/low) + depends_on determines execution order within each epic.
- Agent may create stories only via brainstorm when: (a) no ready tasks remain, or (b) no active features exist.
- **Success criteria**: Must be **user-observable outcomes**, not implementation details.
- **Litmus Test**: A concrete "how do you know it's done" check, phrased as a user-observable outcome.

### Features
- Technical deliverable within a story. Status: active | parked | completed.
- References parent story via `story = "..."`.

### Tasks
- Actionable item within a feature. Kind: investigation | experiment | design.
- Task IDs: `{feature}.{n}` (e.g., `cost-warnings.2`). Rework: `{feature}.{n}.{m}`.

## CRITICAL Rules
1. **Host Environment Policy** — The dev workflow MUST NOT modify anything outside this repository. No installing packages, toolchains, or system libraries. No modifying PATH, environment variables, or system config. If environment changes are needed → STOP and ask the user.
2. **Infrastructure Serves Real Use Cases** — Build for practical value, NOT for its own sake. Every feature must answer: "How does this help a user build real GPU programs?" Demos must showcase COMPUTE, not just I/O. Always ask: "Would a user care about this?"
3. **Keep the Repo Clean** — Do NOT litter the repo with temp files, one-off scripts, stale outputs, or misplaced artifacts. Clean up after yourself.
4. **GPU is Available — Always Run Tests** — This machine has a CUDA-capable NVIDIA GPU. **NEVER** say "I cannot run GPU" or "needs GPU hardware to test" — just run it. CI lacks GPU, but the development machine always has one.
5. **No Publishing** — **NEVER** do crates.io publish, docs site, blog posts, or any public-facing release actions. These are external-visibility actions that only the user decides to do. Do not create tasks/themes/epics related to publishing.

## Dev Workflow
- Autonomous orchestrator loop. Launch with `/dev`. Details in `.claude/commands/dev.md`.
- Main agent manages flow only — all execution via subagents. See `dev-gates.md`, `dev-dispatch.md`, `dev-brainstorm.md`.
- Always run `bash scripts/ci-lint.sh` before `git push`. Only push when it passes.

## Maintenance
- `/maintain` dispatches housekeeping sub-commands (ci, archive, readme, nightly, patches, gitignore).
- At `do.save` in the dev loop, dispatch maintenance as a subagent.
- Auto-archive completed items to keep state.toml under ~300 lines.
