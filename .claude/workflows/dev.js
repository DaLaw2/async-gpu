export const meta = {
  name: 'dev',
  description: 'Autonomous dev loop — GATE, SELECT, DISPATCH, SAVE, ROUTE',
  phases: [
    { title: 'Recover', detail: 'Read state.toml + context.md' },
    { title: 'Gate', detail: 'Tier gate + brainstorm triggers' },
    { title: 'Select', detail: 'Pick ready tasks, assemble briefs' },
    { title: 'Dispatch', detail: 'Execute tasks + verify (parallel cross-theme)' },
    { title: 'Save', detail: 'Update state, maintain, commit, push' },
    { title: 'Route', detail: 'North Star gate, epic completion, tier promotion' },
  ],
}

// ── Schemas ──────────────────────────────────────────────

const RECOVER_SCHEMA = {
  type: "object",
  properties: {
    meta: {
      type: "object",
      properties: {
        total_cycles: { type: "number" },
        completed_tasks: { type: "number" },
        status: { type: "string" },
        current_step: { type: "string" },
        current_task_id: { type: "string" },
        brainstorm_seq: { type: "number" },
        tasks_since_brainstorm: { type: "number" }
      },
      required: ["total_cycles", "completed_tasks", "brainstorm_seq", "tasks_since_brainstorm"]
    },
    project_north_star: { type: "string" },
    epics: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          title: { type: "string" },
          tier: { type: "number" },
          status: { type: "string" },
          north_star: { type: "string" },
          litmus_test: { type: "string" },
          success_criteria: { type: "array", items: { type: "string" } }
        },
        required: ["id", "status", "success_criteria"]
      }
    },
    themes: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          epic: { type: "string" },
          status: { type: "string" },
          title: { type: "string" },
          goal: { type: "string" }
        },
        required: ["id", "epic", "status"]
      }
    },
    tasks: {
      type: "array",
      items: {
        type: "object",
        properties: {
          id: { type: "string" },
          theme: { type: "string" },
          title: { type: "string" },
          kind: { type: "string" },
          status: { type: "string" },
          depends: { type: "array", items: { type: "string" } }
        },
        required: ["id", "theme", "title", "kind", "status"]
      }
    },
    context: {
      type: "object",
      properties: {
        current_focus: { type: "string" },
        tried_and_rejected: { type: "string" },
        active_constraints: { type: "string" },
        key_metrics: { type: "string" }
      }
    },
    awaiting_user: { type: "string" }
  },
  required: ["meta", "epics", "themes", "tasks"]
}

const SELECT_SCHEMA = {
  type: "object",
  properties: {
    selected_tasks: {
      type: "array",
      items: {
        type: "object",
        properties: {
          task_id: { type: "string" },
          theme_id: { type: "string" },
          epic_id: { type: "string" },
          title: { type: "string" },
          kind: { type: "string" },
          brief: { type: "string" }
        },
        required: ["task_id", "theme_id", "epic_id", "title", "kind", "brief"]
      }
    },
    no_ready_tasks: { type: "boolean" }
  },
  required: ["selected_tasks"]
}

const TASK_SCHEMA = {
  type: "object",
  properties: {
    task_id: { type: "string" },
    status: { type: "string", enum: ["done", "blocked"] },
    summary: { type: "string" },
    files_changed: { type: "array", items: { type: "string" } },
    blocked_reason: { type: "string" },
    findings_path: { type: "string" }
  },
  required: ["task_id", "status", "summary", "files_changed"]
}

const VERIFY_SCHEMA = {
  type: "object",
  properties: {
    task_id: { type: "string" },
    verdict: { type: "string", enum: ["PASS", "FAIL"] },
    tests_pass: { type: "boolean" },
    lint_clean: { type: "boolean" },
    findings_exist: { type: "boolean" },
    goal_resolved: { type: "boolean" },
    failure_evidence: { type: "string" }
  },
  required: ["task_id", "verdict"]
}

const NORTH_STAR_SCHEMA = {
  type: "object",
  properties: {
    epic_id: { type: "string" },
    verdict: { type: "string", enum: ["ALIGNED", "DRIFT"] },
    evidence: { type: "string" }
  },
  required: ["epic_id", "verdict", "evidence"]
}

// ── Brief Templates ─────────────────────────────────────

const TASK_BRIEF_TEMPLATE = [
  '## Task',
  '{task_id}: {title}',
  'Kind: {kind}',
  '',
  '## Context',
  'Theme: {theme_id} — {theme_title}',
  'Epic: {epic_id} — {epic_title}',
  'Epic North Star: {north_star}',
  'Epic Success Criteria:',
  '{criteria}',
  'This task resolves: {resolves}',
  'Theme synthesis: {synthesis}',
  '',
  '## Prior Work',
  'Tried & Rejected: {tried_rejected}',
  'Dependency findings: {dep_findings}',
  '',
  '## Codebase Pointers',
  '{codebase_pointers}',
  '',
  '## Constraints',
  '{constraints}',
  '- Experiment code goes in crates/ or examples/',
  '',
  '## Deliverables',
  '1. Findings file: .research/findings/tasks/{task_id}-c{cycle}.md',
  '   Format: Summary (2-3 sentences), Findings (Q/A with confidence), Unexpected Discoveries, Open Questions, Impact on Downstream Tasks',
  '2. Theme synthesis update: .research/findings/themes/{theme_id}-synthesis.md',
  '   REWRITE (not append), <=30 lines. Sections: Progress, Verified Conclusions, Rejected Approaches, Open Questions, Key Metrics, Next Steps',
  '3. Return to orchestrator: STATUS (done|blocked), SUMMARY (3 sentences), FILES_CHANGED (list)',
].join('\n')

const EXPERIMENT_RULES = [
  '## Experiment Rules',
  '- Baseline-first: run the relevant test/benchmark BEFORE making changes. Record in findings.',
  '- Smart bailout by failure type (count distinct approaches, not retries):',
  '    Syntax/typo: 5 attempts | Missing API/feature: 2 | Linker/ABI/backend: 2',
  '    Wrong output: 3 | Crash/segfault: 2',
  '- If max exceeded: git reset to pre-experiment commit, return STATUS=blocked',
  '- Lint before reporting done: cargo +stable fmt --check && cargo +stable clippy -- -D warnings',
  '- Redirect long output to .research/run.log, grep key results, delete log after use',
].join('\n')

const VERIFY_BRIEF_TEMPLATE = [
  '## Verify: {task_id}',
  'Task goal: {title}',
  'Epic success criteria this task serves: {criterion}',
  'Files changed: {files_changed}',
  'Findings file: .research/findings/tasks/{task_id}-c{cycle}.md',
  '',
  '## Checks (all must PASS)',
  '1. Tests pass: run relevant tests for changed crates',
  '2. Lint clean: cargo +stable fmt --check && cargo +stable clippy -- -D warnings',
  '3. Findings file exists and has required sections (Summary, Findings, Open Questions)',
  '4. Theme synthesis updated: .research/findings/themes/{theme_id}-synthesis.md exists and is <=30 lines',
  '5. Goal check: do the changes actually resolve the task goal? Read the diff and findings.',
  '',
  'Return: PASS (all checks green) or FAIL (which check failed + evidence)',
].join('\n')

// ── Phase 1: RECOVER ─────────────────────────────────────

phase('Recover')
log('Reading state.toml + context.md...')

const state = await agent(
  'Read .research/state.toml (parse as TOML) and .research/context.md.\n' +
  'Extract ALL data:\n' +
  '- [meta] section: all fields\n' +
  '- Project North Star: from comments in [meta] section (lines starting with # that mention North Star)\n' +
  '- [[epics]]: all fields + North Star from TOML comments (# North Star: ...)\n' +
  '- [[themes]]: all fields\n' +
  '- [[tasks]]: all fields including depends array\n' +
  '- context.md: parse into sections (Current Focus, Tried & Rejected, Active Constraints, Key Metrics)\n' +
  '- If current_step == "awaiting_user", set awaiting_user field to the pending request\n' +
  'Return as structured JSON matching the schema.',
  { label: 'recover', phase: 'Recover', schema: RECOVER_SCHEMA }
)

if (state.awaiting_user) {
  log('BLOCKED: awaiting user input — ' + state.awaiting_user)
  return { status: 'awaiting_user', message: state.awaiting_user }
}

// ── Main Loop ────────────────────────────────────────────

const MAX_CYCLES = 5
let cycle = 0

while (cycle < MAX_CYCLES) {
  cycle++
  const currentCycle = state.meta.total_cycles + cycle

  // ── Phase 2: GATE (deterministic JS logic) ─────────────

  phase('Gate')

  const activeEpics = state.epics.filter(function(e) {
    return e.status === 'active' && e.tier != null
  })
  const tiers = []
  activeEpics.forEach(function(e) {
    if (tiers.indexOf(e.tier) === -1) tiers.push(e.tier)
  })
  tiers.sort(function(a, b) { return a - b })
  const eligibleTier = tiers.length > 0 ? tiers[0] : 0

  log('Tier Gate: eligible T' + eligibleTier + ' (' + activeEpics.length + ' active epics)')

  // Proactive brainstorm triggers
  var brainstormNeeded = state.meta.tasks_since_brainstorm >= 10
  var brainstormReason = brainstormNeeded
    ? 'tasks_since_brainstorm = ' + state.meta.tasks_since_brainstorm
    : ''

  if (!brainstormNeeded && args && args.userMessage) {
    var msg = String(args.userMessage).toLowerCase()
    if (msg.indexOf('brainstorm') !== -1) {
      brainstormNeeded = true
      brainstormReason = 'User requested brainstorm'
    }
  }

  if (brainstormNeeded) {
    phase('Gate')
    log('Brainstorm triggered: ' + brainstormReason)
    await workflow('brainstorm', {
      trigger: brainstormReason,
      level: 'standard',
      seq: state.meta.brainstorm_seq + 1,
      eligibleTier: eligibleTier
    })
    var fresh = await agent(
      'Read .research/state.toml and .research/context.md. Return full structured state.',
      { label: 'recover-post-bs', schema: RECOVER_SCHEMA }
    )
    Object.assign(state, fresh)
    continue
  }

  // ── Phase 3: SELECT ─────────────────────────────────────

  phase('Select')
  log('Selecting ready tasks...')

  var eligibleEpics = state.epics.filter(function(e) {
    return (e.tier === eligibleTier && e.status === 'active') ||
           (e.tier == null && e.status === 'active')
  })

  var activeThemes = state.themes.filter(function(t) { return t.status === 'active' })
  var pendingTasks = state.tasks.filter(function(t) { return t.status === 'pending' })

  var selected = await agent(
    'You are the task selector for the dev loop.\n\n' +
    'Eligible tier: T' + eligibleTier + '\n' +
    'Eligible epics: ' + JSON.stringify(eligibleEpics.map(function(e) { return { id: e.id, title: e.title, tier: e.tier } })) + '\n' +
    'Active themes: ' + JSON.stringify(activeThemes.map(function(t) { return { id: t.id, epic: t.epic, title: t.title } })) + '\n' +
    'Pending tasks: ' + JSON.stringify(pendingTasks.map(function(t) { return { id: t.id, theme: t.theme, title: t.title, kind: t.kind, depends: t.depends } })) + '\n' +
    'All tasks (for dep resolution): ' + JSON.stringify(state.tasks.map(function(t) { return { id: t.id, status: t.status } })) + '\n' +
    'Context tried & rejected: ' + (state.context && state.context.tried_and_rejected || 'None') + '\n' +
    'Context constraints: ' + (state.context && state.context.active_constraints || 'None') + '\n\n' +
    'Rules:\n' +
    '1. Filter: status=="pending" AND all items in depends[] have status=="done" AND theme is "active"\n' +
    '2. Only tasks from themes belonging to eligible epics (T' + eligibleTier + ' or evergreen)\n' +
    '3. Same-theme: pick FIRST ready task only. Cross-theme: can select one from each theme.\n' +
    '4. For each selected task, do file discovery:\n' +
    '   - ls, find, grep -l to locate relevant crates, scripts, entry points\n' +
    '   - Read dependency task findings from .research/findings/tasks/ if depends are listed\n' +
    '   - Read theme synthesis from .research/findings/themes/{theme_id}-synthesis.md\n' +
    '   - Assemble the full brief using this template:\n\n' +
    TASK_BRIEF_TEMPLATE + '\n\n' +
    'For experiment tasks, also append:\n' + EXPERIMENT_RULES + '\n\n' +
    'Return the list of selected tasks with their assembled briefs.',
    { label: 'select', phase: 'Select', schema: SELECT_SCHEMA }
  )

  if (selected.no_ready_tasks || selected.selected_tasks.length === 0) {
    log('No ready tasks.')
    var activeCount = state.epics.filter(function(e) { return e.status === 'active' }).length
    if (activeCount === 0) {
      log('All epics completed. Workflow done.')
      return { status: 'all_complete', cycles: cycle }
    }
    log('Triggering brainstorm to create new tasks...')
    await workflow('brainstorm', {
      trigger: 'No ready tasks but active epics have unmet criteria',
      level: 'standard',
      seq: state.meta.brainstorm_seq + 1,
      eligibleTier: eligibleTier
    })
    var fresh2 = await agent(
      'Read .research/state.toml and .research/context.md. Return full structured state.',
      { label: 'recover-post-bs2', schema: RECOVER_SCHEMA }
    )
    Object.assign(state, fresh2)
    continue
  }

  log(selected.selected_tasks.length + ' task(s) selected: ' +
    selected.selected_tasks.map(function(t) { return t.task_id }).join(', '))

  // ── Phase 4: DISPATCH (parallel cross-theme) ────────────

  phase('Dispatch')

  var byTheme = {}
  selected.selected_tasks.forEach(function(t) {
    if (!byTheme[t.theme_id]) byTheme[t.theme_id] = []
    byTheme[t.theme_id].push(t)
  })

  var themeEntries = Object.keys(byTheme).map(function(k) { return [k, byTheme[k]] })

  var themeThunks = themeEntries.map(function(entry) {
    var themeId = entry[0]
    var tasks = entry[1]

    return async function() {
      var results = []
      for (var i = 0; i < tasks.length; i++) {
        var task = tasks[i]
        log('Task: ' + task.task_id)

        var brief = task.brief
        if (task.kind === 'experiment' && brief.indexOf('Experiment Rules') === -1) {
          brief = brief + '\n\n' + EXPERIMENT_RULES
        }

        var taskResult = await agent(brief, {
          label: 'task:' + task.task_id,
          phase: 'Dispatch',
          schema: TASK_SCHEMA
        })

        if (taskResult.status === 'blocked') {
          log('BLOCKED: ' + task.task_id + ' — ' + (taskResult.blocked_reason || ''))
          results.push(taskResult)
          continue
        }

        log('Verify: ' + task.task_id)
        var verifyBrief = VERIFY_BRIEF_TEMPLATE
          .replace(/\{task_id\}/g, task.task_id)
          .replace(/\{title\}/g, task.title)
          .replace(/\{files_changed\}/g, (taskResult.files_changed || []).join(', '))
          .replace(/\{cycle\}/g, String(currentCycle))
          .replace(/\{theme_id\}/g, task.theme_id)
          .replace(/\{criterion\}/g, '')

        var verifyResult = await agent(verifyBrief, {
          label: 'verify:' + task.task_id,
          phase: 'Dispatch',
          schema: VERIFY_SCHEMA
        })

        results.push({
          task_id: taskResult.task_id || task.task_id,
          theme_id: task.theme_id,
          epic_id: task.epic_id,
          status: verifyResult.verdict === 'PASS' ? 'done' : 'blocked',
          summary: taskResult.summary,
          files_changed: taskResult.files_changed,
          findings_path: taskResult.findings_path,
          verify: verifyResult.verdict,
          failure_evidence: verifyResult.failure_evidence
        })
      }
      return results
    }
  })

  var dispatchResults = (await parallel(themeThunks)).filter(Boolean)
  var allResults = []
  dispatchResults.forEach(function(arr) {
    if (Array.isArray(arr)) arr.forEach(function(r) { allResults.push(r) })
  })

  var doneTasks = allResults.filter(function(r) { return r.status === 'done' })
  var blockedTasks = allResults.filter(function(r) { return r.status === 'blocked' })

  // ── Phase 5: SAVE ───────────────────────────────────────

  phase('Save')
  log(doneTasks.length + ' done, ' + blockedTasks.length + ' blocked')

  // 5a. Update state.toml
  var stateUpdates = []
  doneTasks.forEach(function(t) {
    stateUpdates.push(t.task_id + ': status = "done"')
  })
  blockedTasks.forEach(function(t) {
    stateUpdates.push(t.task_id + ': status = "blocked", blocked_reason = "' +
      (t.failure_evidence || t.blocked_reason || 'verify failed') + '"')
  })

  await agent(
    'Read .research/state.toml and update:\n' +
    stateUpdates.map(function(s) { return '- ' + s }).join('\n') + '\n\n' +
    'Also update [meta]:\n' +
    '- total_cycles += 1\n' +
    '- completed_tasks += ' + doneTasks.length + '\n' +
    '- tasks_since_brainstorm += ' + doneTasks.length + '\n' +
    '- current_step = "route"\n' +
    '- current_task_id = ""',
    { label: 'save-state', phase: 'Save' }
  )

  // 5b. Run maintain sub-workflow
  var maintainCmds = ['archive']
  var cratesChanged = doneTasks.some(function(t) {
    return t.files_changed && t.files_changed.some(function(f) {
      return f.indexOf('crates/') !== -1
    })
  })
  if (cratesChanged) maintainCmds.push('ci')

  await workflow('maintain', { commands: maintainCmds })

  // 5c. CI lint + commit + push
  await agent(
    'Run bash scripts/ci-lint.sh. If there are lint failures, attempt to fix them:\n' +
    '- cargo +stable fmt for formatting issues\n' +
    '- Apply clippy suggestions where safe\n\n' +
    'Then commit and push:\n' +
    'git add the relevant changed files (source code, findings, state.toml, context.md)\n' +
    'git commit -m "cycle ' + currentCycle + ': ' + doneTasks.map(function(t) { return t.task_id }).join(', ') + '"\n' +
    'git push origin main\n\n' +
    'If push fails, warn but continue (data is committed locally).',
    { label: 'save-push', phase: 'Save' }
  )

  // 5d. Rewrite context.md
  await agent(
    'Rewrite .research/context.md with these sections:\n' +
    '- Current Focus: what T0/T1 epics are active, what just happened\n' +
    '- Recent Decisions: key decisions from this cycle\n' +
    '- Tried & Rejected: what approaches failed and why\n' +
    '- Active Constraints: hardware limits, toolchain issues, environment notes\n' +
    '- Key Metrics: latest performance numbers\n' +
    '- Next: what should happen in the next cycle\n\n' +
    'Read the current .research/state.toml for accurate data.',
    { label: 'save-context', phase: 'Save' }
  )

  // ── Phase 6: ROUTE ──────────────────────────────────────

  phase('Route')

  // 6a. North Star Gate
  var epicsWithWork = []
  doneTasks.forEach(function(t) {
    var theme = state.themes.find(function(th) { return th.id === t.theme_id })
    if (theme && epicsWithWork.indexOf(theme.epic) === -1) {
      epicsWithWork.push(theme.epic)
    }
  })

  var driftDetected = false

  if (epicsWithWork.length > 0) {
    log('North Star gate for ' + epicsWithWork.length + ' epic(s)...')
    var nsThunks = epicsWithWork.map(function(epicId) {
      return async function() {
        var epic = state.epics.find(function(e) { return e.id === epicId })
        var epicDoneTasks = doneTasks.filter(function(t) {
          var theme = state.themes.find(function(th) { return th.id === t.theme_id })
          return theme && theme.epic === epicId
        })
        var findingsPaths = epicDoneTasks.map(function(t) { return t.findings_path || '' }).filter(Boolean)

        return await agent(
          'Read these completed task findings:\n' +
          findingsPaths.join('\n') + '\n\n' +
          'Epic North Star: ' + (epic && epic.north_star || 'unknown') + '\n' +
          'Project North Star (from state.toml [meta] comments): read it from the file.\n\n' +
          'Does this work advance BOTH the epic North Star AND the Project North Star?\n' +
          'Return: ALIGNED (with 1-sentence evidence) or DRIFT (with 1-sentence explanation).',
          {
            label: 'ns:' + epicId,
            phase: 'Route',
            schema: NORTH_STAR_SCHEMA
          }
        )
      }
    })

    var nsResults = await parallel(nsThunks)
    nsResults.filter(Boolean).forEach(function(r) {
      if (r.verdict === 'DRIFT') {
        log('DRIFT: ' + r.epic_id + ' — ' + r.evidence)
        driftDetected = true
      }
    })
  }

  // 6b. Reactive brainstorm triggers
  var reactiveTriggered = driftDetected || blockedTasks.length >= 3
  if (reactiveTriggered) {
    var reactiveReason = driftDetected ? 'North Star drift detected' : '3+ tasks blocked'
    log('Reactive brainstorm: ' + reactiveReason)
    await workflow('brainstorm', {
      trigger: reactiveReason,
      level: driftDetected ? 'high' : 'standard',
      seq: state.meta.brainstorm_seq + 1,
      eligibleTier: eligibleTier
    })
  }

  // 6c. Re-read state for next cycle
  var refreshed = await agent(
    'Read .research/state.toml and .research/context.md. Return full structured state.',
    { label: 'recover-next', schema: RECOVER_SCHEMA }
  )
  Object.assign(state, refreshed)

  // 6d. Check for more ready tasks
  var ready = state.tasks.filter(function(t) {
    if (t.status !== 'pending') return false
    if (!t.depends || t.depends.length === 0) return true
    return t.depends.every(function(d) {
      var dep = state.tasks.find(function(x) { return x.id === d })
      return dep && dep.status === 'done'
    })
  })

  if (ready.length === 0 && !reactiveTriggered) {
    log('No more ready tasks. Loop ends.')
    break
  }

  log(ready.length + ' ready task(s). Continuing...')
}

log('Dev loop completed. Ran ' + cycle + ' cycle(s).')
return { status: 'completed', cycles: cycle }
