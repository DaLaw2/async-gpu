# Dev Gates

Binary pass/fail. No exceptions. No judgment calls.

## Gate 1: Story Priority Gate

**When**: SELECT step, every cycle.
**Check**: Read all `[[stories]]` across all epics.
- **Hard gate**: If ANY story with priority `high` has unmet criteria (status != completed) → stories with priority `medium` or `low` are NOT eligible.
- Once ALL `high` stories are completed → `medium` becomes eligible. Once ALL `medium` completed → `low` eligible.
- **Blocked exception**: A story explicitly blocked on external factors (noted in context.md) may be skipped — it does not block lower-priority stories.
- Within the same priority level, prefer stories with active features that have ready tasks.
**Fail action**: Filter out ineligible stories. Log which high/medium stories are blocking.
**Output**: Ordered list of eligible stories for task selection.

## Gate 2: North Star Gate

**When**: ROUTE step, after each completed task batch.
**Check**: Dispatch a subagent:
```
Read these completed task findings: {findings_paths}
Read this story's success criteria: {story_criteria}
Read the parent epic's North Star: {epic_north_star}
Read the Project North Star from state.toml [meta] section.
Question: Does this work advance the story criteria, the epic North Star, AND the Project North Star?
Answer: ALIGNED (with 1-sentence evidence) or DRIFT (with 1-sentence explanation of what drifted).
```
**Pass (ALIGNED)**: Continue to next cycle.
**Fail (DRIFT)**: Record drift in context.md. Triggers brainstorm in ROUTE.

## Gate 3: Story Verification Gate

**When**: ROUTE step, when all success criteria of a story appear met.
**Check**: Dispatch a verification subagent:
```
Story: {story_id} — {title}
Epic: {epic_id} — {epic_title}
Success Criteria: {criteria_list}
Verify EACH criterion by actually running/checking the observable outcome described.
Return: PASS or FAIL with concrete evidence for each criterion.
```
**Pass**: ALL criteria PASS → orchestrator runs cascade close (mark features completed, tasks done/skipped, story completed).
**Fail**: ANY criterion FAIL → story stays active. Create tasks for unmet criteria.

## Gate 4: Epic Verification Gate

**When**: ROUTE step, when all stories within a non-evergreen epic are completed.
**Skip**: Evergreen epics (e.g., codebase-health) are never verified or closed.
**Check**: Dispatch a verification subagent:
```
Epic: {epic_id} — {title}
North Star: {north_star_text}
Success Criteria: {epic_criteria_list}
All stories completed: {list of story IDs and their verified criteria}
Verify that the epic's overall success criteria are met by the aggregate of all story completions.
Return: PASS or FAIL with concrete evidence for each criterion.
```
**Pass**: ALL criteria PASS → mark epic completed.
**Fail**: ANY criterion FAIL → identify which story needs reopening or which new story is needed. Create story/tasks.
