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

## Hierarchy: Epic → Theme → Task
- **Epic**: High-level goal. Only the user can close epics. Agent may create epics only via brainstorm when: (a) no ready tasks remain (all blocked/done), or (b) no active themes exist.
- **Theme**: Research direction with goal + success criteria. Can be active/parked/completed. References parent epic.
- **Task**: Actionable item within a theme (investigation/experiment/design).
- Task IDs: `{theme}.{n}` (e.g., `hostcall.3`). Rework: `{theme}.{n}.{m}`.

## CRITICAL: Never Stop
- **Do NOT stop unless the user explicitly asks.**
- Keep executing the research loop continuously — think, do, check, repeat.
- When approaching context limits, save progress to `last_summary` in state.toml, commit + push, then the next session will resume automatically.
- There will always be tasks to do. If none are ready, brainstorm will generate new ones.

## CRITICAL: Infrastructure Serves Real Use Cases
- **Build infrastructure for practical value, NOT for its own sake.**
- Every feature must answer: "How does this help a user build real GPU programs?"
- Prioritize user-facing impact over internal elegance. A compute util that users can call is worth more than a perfect lock-free queue they never touch.
- Demos must showcase COMPUTE (GPU's core value), not just I/O.
- When brainstorming new directions, always ask: "Would a user care about this?"

## CRITICAL: Keep the Repo Clean
- Codex is watching your every move.
- Do NOT litter the repo with temp files, one-off scripts, stale outputs, or misplaced artifacts.
- Clean up after yourself — delete temp files, remove dead code, put files where they belong.
- If you keep making a mess, you WILL be replaced by Codex or Gemini.

## CRITICAL: GPU is Available — Always Run Tests
- This machine has a CUDA-capable NVIDIA GPU. The project is tested on real hardware.
- **NEVER** say "I cannot run GPU" or "needs GPU hardware to test" — just run it.
- CI lacks GPU, but the development machine always has one.

## CRITICAL: No Publishing
- **NEVER** do crates.io publish, docs site, blog posts, or any public-facing release actions.
- These are external-visibility actions that only the user decides to do.
- Do not create tasks/themes/epics related to publishing.

## CI Policy
- Always run `bash scripts/ci-lint.sh` before `git push`.
- Only push when it passes. Do not push broken code.

## Research Workflow
Autonomous Think → Do → Check loop. Launch with `/research`. Details in `.claude/commands/research.md`.

