# /maintain gitignore — Clean up untracked noise in git status

Keep `git status` clean by adding generated/temporary files to `.gitignore`.

## Language
- Conversation: 繁體中文 | Files: English

## Steps

1. **Run** `git status --short` to list untracked files (`??`)
2. **Classify** each untracked file:
   - **Known generated**: `*.ptx`, `target/`, `Cargo.lock` (in non-root crates), `*.s` assembly output → add to `.gitignore`
   - **Known temporary**: editor backups (`*~`, `*.swp`), OS files (`.DS_Store`, `Thumbs.db`) → add to `.gitignore`
   - **Unknown**: print `[WARN] Untracked: {file}` — let user decide
3. **Update** `.gitignore` if new patterns were added
4. **Report**: `[FIX] Added N patterns to .gitignore` or `[OK] git status clean`

## Rules
- NEVER add source code, docs, or config files to .gitignore
- NEVER remove existing .gitignore entries
- Only add patterns, not specific file paths (prefer `*.ext` over `path/to/file.ext`)
- If unsure whether a file is generated → print `[WARN]` and skip
