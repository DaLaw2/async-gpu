# Dev Dispatch — Subagent Brief Protocol

All execution happens in subagents. They receive a structured brief and do NOT read state.toml
or make strategic decisions. The orchestrator assembles the brief and judges results.

## Brief Template

```
## Task
{task_id}: {title}
Kind: {investigation | experiment | design}

## Context
Theme: {theme_id} — {theme_title}
Epic: {epic_id} — {epic_title}
Epic North Star: {one_sentence_from_state_toml}
Epic Success Criteria: {list from state.toml}
This task resolves: {which specific criterion or sub-goal}
Theme synthesis: {paste findings/themes/{theme_id}-synthesis.md, or "First task in this theme"}

## Prior Work
Tried & Rejected: {from context.md — what NOT to do and why}
Dependency findings: {paste findings from depends tasks, or "None"}

## Codebase Pointers
Relevant crates: {paths found by orchestrator via ls/find}
Relevant scripts: {paths found by orchestrator via ls/find}
Entry point: {specific file, script, or function to start from}

## Constraints
- {relevant active constraints from context.md}
- Experiment code goes in crates/ or examples/

## Deliverables
1. **Findings file**: .research/findings/tasks/{task_id}-c{cycle}.md
   Format: Summary (2-3 sentences), Findings (Q/A with confidence), Unexpected Discoveries,
   Open Questions, Impact on Downstream Tasks
2. **Theme synthesis update**: .research/findings/themes/{theme_id}-synthesis.md
   REWRITE (not append), ≤30 lines. Sections: Progress, Verified Conclusions,
   Rejected Approaches, Open Questions, Key Metrics, Next Steps
3. **Return to orchestrator**: STATUS (done|blocked), SUMMARY (3 sentences), FILES_CHANGED (list)
```

## Experiment Rules (include in brief when kind = experiment)

```
- Baseline-first: run the relevant test/benchmark BEFORE making changes. Record in findings.
- Smart bailout by failure type (count distinct approaches, not retries):
    Syntax/typo: 5 attempts | Missing API/feature: 2 | Linker/ABI/backend: 2
    Wrong output: 3 | Crash/segfault: 2
- If max exceeded: git reset to pre-experiment commit, return STATUS=blocked
- Lint before reporting done: cargo +stable fmt --check && cargo +stable clippy -- -D warnings
- Redirect long output to .research/run.log, grep key results, delete log after use
```

## Dispatch Types

### type: task (default)
Full brief above. Subagent writes code, runs tests, produces findings + synthesis.

### type: verify
Dispatched automatically after a task subagent reports STATUS=done. A separate subagent verifies.
```
## Verify: {task_id}
Task goal: {title}
Epic success criteria this task serves: {criterion}
Files changed: {FILES_CHANGED from task subagent}
Findings file: .research/findings/tasks/{task_id}-c{cycle}.md

## Checks (all must PASS)
1. Tests pass: run relevant tests for changed crates
2. Lint clean: cargo +stable fmt --check && cargo +stable clippy -- -D warnings
3. Findings file exists and has required sections (Summary, Findings, Open Questions)
4. Theme synthesis updated and ≤30 lines
5. Goal check: do the changes actually resolve the task goal? Read the diff and findings.

Return: PASS (all checks green) or FAIL (which check failed + evidence)
```

### type: north-star
```
Read these completed task findings: {paths}
Read this epic's North Star: {north_star_text}
Read the Project North Star from state.toml [meta] section.
Does this work advance BOTH?
Return: ALIGNED (1-sentence evidence) or DRIFT (1-sentence explanation)
```

### type: epic-verify
```
Epic: {epic_id} — {title}
Litmus Test: {litmus_test_text}
Success Criteria: {criteria_list}
Verify EACH criterion by actually running/checking the observable outcome described.
Return: PASS or FAIL with concrete evidence for each criterion.
```

### type: brainstorm
See `dev-brainstorm.md`.

### type: maintain
Dispatch `/maintain` with relevant sub-commands based on what changed in this cycle.

## Extensibility

To add a new dispatch type:
1. Create `dev-{type}.md` with the subagent brief template and return contract
2. Add trigger condition to `dev.md` ROUTE step
3. No changes to this file or the core loop needed
