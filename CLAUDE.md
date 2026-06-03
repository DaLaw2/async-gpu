# async_gpu — Rust Async/Await on GPU

## Language Convention
- **Conversation**: Always use Traditional Chinese (繁體中文)
- **Files, docs, comments, code**: Always use English
- This applies to ALL generated content without exception

## Dependency Policy
- **anyhow is PROHIBITED** — do not use `anyhow` in any crate
- Use `thiserror` for custom error types, or plain `Result<T, E>` with concrete error types
- Existing `anyhow` usage must be migrated when touched

## Hierarchy: Epic → Theme → Task

### Epics
- High-level goal. Only the user can close epics.
- Agent may create epics only via brainstorm when: (a) no ready tasks remain, or (b) no active themes exist.
- **Tier system** (T0 → T3): Strict priority ordering. Never start T(N+1) tasks while T(N) has unmet criteria, unless T(N) is explicitly blocked on external factors.
  - **T0**: Foundation — must be done first, everything builds on this
  - **T1**: Core pillars — primary library value
  - **T2**: Advanced — extends the library, depends on T0/T1
  - **T3**: Exploratory — research directions, may or may not ship
- **North Star**: Each epic has a one-sentence vision of what the USER gets. This is the spirit of the epic. When in doubt, check against the North Star, not the success criteria letter.
- **Litmus Test**: A concrete "how do you know it's done" check, phrased as a user-observable outcome.
- **depends_on**: Epic-level dependency. An epic cannot become active until its dependencies are completed.
- **Success criteria**: Must be **user-observable outcomes**, not implementation details. "Users write `std::thread::spawn()` and it works" not "implement warp wake-up protocol via atomic CAS in shared memory."

### Themes
- Direction within an epic. Status: active | parked | completed.
- References parent epic via `epic = "..."`.

### Tasks
- Actionable item within a theme. Kind: investigation | experiment | design.
- Task IDs: `{theme}.{n}` (e.g., `hostcall.3`). Rework: `{theme}.{n}.{m}`.

## CRITICAL Rules
1. **Host Environment Policy** — The research workflow MUST NOT modify anything outside this repository. No installing packages, toolchains, or system libraries. No modifying PATH, environment variables, or system config. If environment changes are needed → STOP and ask the user.
2. **Never Stop** — Do NOT stop unless the user explicitly asks. Keep executing the research loop continuously. **NEVER** pause to ask "要我繼續嗎?", **NEVER** report and wait, **NEVER** say "下個 session 繼續". When approaching context limits, save progress to `context.md` + state.toml, commit + push, then **keep working**. If no ready tasks, brainstorm will generate new ones. This rule has been reinforced by the user 5+ times — treat any violation as a critical failure.
3. **Infrastructure Serves Real Use Cases** — Build for practical value, NOT for its own sake. Every feature must answer: "How does this help a user build real GPU programs?" Demos must showcase COMPUTE, not just I/O. Always ask: "Would a user care about this?"
4. **Keep the Repo Clean** — Codex is watching your every move. Do NOT litter the repo with temp files, one-off scripts, stale outputs, or misplaced artifacts. Clean up after yourself. If you keep making a mess, you WILL be replaced by Codex or Gemini.
5. **GPU is Available — Always Run Tests** — This machine has a CUDA-capable NVIDIA GPU. **NEVER** say "I cannot run GPU" or "needs GPU hardware to test" — just run it. CI lacks GPU, but the development machine always has one.
6. **No Publishing** — **NEVER** do crates.io publish, docs site, blog posts, or any public-facing release actions. These are external-visibility actions that only the user decides to do. Do not create tasks/themes/epics related to publishing.

## Research Workflow
- Autonomous Think → Do loop. Launch with `/research`. Details in `.claude/commands/research.md`.
- **context.md**: Rolling strategic context in `.research/context.md` (replaces last_summary). Rewritten at each do.save.
- **Theme synthesis**: Per-theme summaries in `.research/findings/themes/`. Updated after each task. Read these instead of individual findings.
- **Brainstorm levels**: Standard (1 agent) → High (proposer + skeptic) → Deep (3-4+ expert Agent Team with cross-review).
- Always run `bash scripts/ci-lint.sh` before `git push`. Only push when it passes.

## Maintenance
- Run `/maintain` to dispatch housekeeping sub-commands:
  - `/maintain ci` — sync CI with actual crates/PTX files
  - `/maintain archive` — archive completed tasks/themes/brainstorms from state.toml
  - `/maintain readme` — update outdated README.md sections
  - `/maintain nightly` — ensure nightly version is consistent across all files
  - `/maintain patches` — regenerate std patches from patched-std/
  - `/maintain gitignore` — clean up untracked noise in git status
- At `do.save` in the research loop, invoke relevant sub-commands based on what changed.
- At `do.save`, auto-archive completed items to keep state.toml under ~300 lines.
