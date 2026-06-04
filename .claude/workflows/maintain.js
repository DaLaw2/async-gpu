export const meta = {
  name: 'maintain',
  description: 'Codebase maintenance — ci, archive, readme, nightly, patches, gitignore',
  phases: [
    { title: 'Maintain', detail: 'Run relevant sub-commands in parallel' },
  ],
}

const ALL_COMMANDS = ['ci', 'archive', 'readme', 'nightly', 'patches', 'gitignore']

const raw = args && args.commands
const requested = raw === 'all'
  ? ALL_COMMANDS
  : (Array.isArray(raw) ? raw : typeof raw === 'string' ? raw.split(/[\s,]+/).filter(Boolean) : ALL_COMMANDS)

phase('Maintain')
log(`Running: ${requested.join(', ')}`)

const RESULT_SCHEMA = {
  type: "object",
  properties: {
    command: { type: "string" },
    status: { type: "string", enum: ["ok", "fixed", "skipped", "error"] },
    detail: { type: "string" }
  },
  required: ["command", "status", "detail"]
}

const COMMANDS = {
  ci: `Sync CI with actual crate and PTX state.
1. Scan all crates/*/Cargo.toml — classify each crate:
   - Stable-compatible (no #![feature(...)] in lib.rs): eligible for fmt + clippy
   - Nightly-only (has #![feature(...)]): fmt only, no stable clippy
   - GPU kernel (has .cargo/config.toml with nvptx64 target): PTX build list
2. Grep crates/*/src/ for include_str!("*.ptx") — extract PTX filenames
3. Compare against scripts/ci-lint.sh (CRATES_FMT, CRATES_CLIPPY, PTX_KERNELS) and .github/workflows/build.yml
4. Fix mismatches by updating the CI files
5. Report what was added/removed, or [OK] if already in sync
Rules: don't add non-existent crates, don't remove intentionally excluded ones (check for comments).
Return command="ci".`,

  archive: `Archive completed items from .research/state.toml to keep it lean.
1. Read .research/state.toml
2. Archive these categories:
   - Completed epics (status="completed") → .research/archive/epics-archived.toml
   - Done/skipped tasks (status="done" or "skipped") → .research/archive/tasks-archived.toml
   - Completed themes (status="completed") → .research/archive/themes-archived.toml
   - Brainstorms: keep last 3, archive the rest → .research/archive/brainstorms-archived.toml
3. In state.toml, replace archived items with a one-line comment listing their IDs
4. Keep completed_tasks count accurate in [meta]
5. Report how many items archived and new line count
Rules: NEVER archive active/pending/parked items. Preserve TOML formatting and comments.
Return command="archive".`,

  readme: `Check README.md against current project state and fix stale sections.
1. Read README.md
2. Check each section:
   - Quick Start: do referenced script paths exist?
   - Capabilities: does it mention all working features?
   - Limitations: are resolved limitations still listed?
   - Examples/Demos: are all examples/ subdirectories represented?
   - Architecture: does crate list match actual crates/?
   - Performance numbers: are they still accurate?
3. Fix only stale sections — do NOT rewrite the whole README
4. Report which sections were updated, or [OK] if current
Rules: keep existing style and tone. Only update factually wrong or outdated content.
Return command="readme".`,

  nightly: `Sync nightly toolchain version across all files.
1. Read rust-toolchain.toml → extract channel (e.g., nightly-2026-06-03)
2. Check these files for hardcoded nightly versions:
   - scripts/ci-lint.sh → NIGHTLY= variable
   - .github/workflows/build.yml → toolchain: field
3. Fix any mismatches → update hardcoded values to match rust-toolchain.toml
4. Report which files were updated, or [OK] if consistent
Return command="nightly".`,

  patches: `Regenerate std patches from patched-std/ directory.
1. Check if patched-std/ directory exists. If not → return status="skipped"
2. Run: bash scripts/gen-std-patches.sh
3. Verify output:
   - std-patches/*.patch files exist
   - std-patches/PATCHES.md was generated
   - scripts/apply-std-patches.sh was regenerated
4. Check git diff on std-patches/ — report what changed
Rules: only run if patched-std/ exists. Do NOT modify patched-std/ — only regenerate patches FROM it.
Return command="patches".`,

  gitignore: `Clean up untracked noise in git status.
1. Run git status --short, find untracked (??) files
2. Classify each untracked file:
   - Known generated (*.ptx, target/, *.s, Cargo.lock in non-root crates) → add pattern to .gitignore
   - Known temporary (*~, *.swp, .DS_Store, Thumbs.db) → add pattern to .gitignore
   - Unknown → skip (do NOT add to .gitignore)
3. Update .gitignore if new patterns were added
4. Report how many patterns added, or [OK] if clean
Rules: NEVER add source code, docs, or config files. Only add patterns (*.ext), not specific file paths.
Return command="gitignore".`
}

const results = await parallel(requested.map(cmd => async () => {
  if (!COMMANDS[cmd]) {
    return { command: cmd, status: 'error', detail: 'Unknown sub-command: ' + cmd }
  }
  return await agent(COMMANDS[cmd], {
    label: 'maintain:' + cmd,
    phase: 'Maintain',
    schema: RESULT_SCHEMA
  })
}))

const report = results.filter(Boolean)
log(report.map(r => '[' + r.status.toUpperCase() + '] ' + r.command + ': ' + r.detail).join('\n'))
return report
