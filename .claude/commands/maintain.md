# Codebase Maintenance — Router

Dispatch to the appropriate maintenance sub-command.

## Usage

- `/maintain ci` — Sync CI workflow + local lint script with actual crates/PTX files
- `/maintain archive` — Archive completed tasks/features/brainstorms from state.toml
- `/maintain readme` — Update outdated README.md sections
- `/maintain nightly` — Ensure nightly version is consistent across all files
- `/maintain patches` — Regenerate std patches from patched-std/
- `/maintain gitignore` — Clean up untracked noise in git status

## Language Convention
- **Conversation**: 繁體中文
- **Files/code/comments**: English

## Rules
- **No state.toml task entries** — maintenance is invisible to the research loop
- **No findings files** — fixes are self-evident from the git diff
- **Minimal output** — one line per check, details only on fixes
- **Safe fixes only** — if unsure, print `[WARN]` and let user decide
- **Never delete user code** — only add/update config, docs, gitignore

## For `/research` integration

At `do.save`, invoke only the relevant sub-commands based on what changed:
- Modified crates or PTX → `/maintain ci`
- Modified patched-std → `/maintain patches`
- Completed a feature → `/maintain readme`
- state.toml bloated → `/maintain archive`
