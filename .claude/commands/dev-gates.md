# Dev Gates

Binary pass/fail. No exceptions. No judgment calls.

## Gate 1: Tier Gate

**When**: GATE step, every cycle.
**Check**: Read all `[[epics]]` sorted by tier. For the lowest active tier T(N):
- Are ALL success_criteria met for ALL epics at T(N)?
- If NO → only T(N) tasks are eligible. Skip all T(N+1)+ tasks.
- If YES → advance to next tier.
**Exception**: A T(N) epic explicitly blocked on external factors (noted in context.md) may be skipped.
**Fail action**: Filter task selection to current tier only. Log which criteria are unmet.

## Gate 2: North Star Gate

**When**: ROUTE step, after each completed task batch.
**Check**: Dispatch a subagent:
```
Read these completed task findings: {findings_paths}
Read this epic's North Star: {north_star_text}
Read the Project North Star from state.toml [meta] section.
Question: Does this work advance BOTH the epic North Star AND the Project North Star?
Answer: ALIGNED (with 1-sentence evidence) or DRIFT (with 1-sentence explanation of what drifted).
```
**Pass (ALIGNED)**: Continue to next cycle.
**Fail (DRIFT)**: Record drift in context.md. Triggers brainstorm in ROUTE.

## Gate 3: Epic Verification Gate

**When**: ROUTE step, when all success criteria of an epic appear met.
**Check**: Dispatch a verification subagent:
```
Epic: {epic_id} — {title}
Litmus Test: {litmus_test_text}
Success Criteria: {criteria_list}
Verify EACH criterion by actually running/checking the observable outcome described.
Return: PASS or FAIL with concrete evidence for each criterion.
```
**Pass**: ALL criteria PASS → orchestrator runs cascade close (see ROUTE in `dev.md`).
**Fail**: ANY criterion FAIL → epic stays active. Create tasks for unmet criteria.
