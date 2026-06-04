export const meta = {
  name: 'brainstorm',
  description: 'Dev brainstorm — Standard/High/Deep analysis for the dev loop',
  phases: [
    { title: 'Prepare', detail: 'Read state + context' },
    { title: 'Analyze', detail: 'Expert analysis (1-4 agents by level)' },
    { title: 'Synthesize', detail: 'Merge findings, update state.toml' },
  ],
}

const RECOVER_SCHEMA = {
  type: "object",
  properties: {
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
          success_criteria: { type: "array", items: { type: "string" } }
        },
        required: ["id", "status"]
      }
    },
    blocked_tasks: { type: "string" },
    context_summary: { type: "string" }
  },
  required: ["epics"]
}

const BRAINSTORM_SCHEMA = {
  type: "object",
  properties: {
    seq: { type: "number" },
    level: { type: "string", enum: ["standard", "high", "deep"] },
    trigger: { type: "string" },
    key_insight: { type: "string" },
    new_tasks: { type: "array", items: { type: "string" } },
    new_themes: { type: "array", items: { type: "string" } },
    new_epics: { type: "array", items: { type: "string" } },
    themes_parked: { type: "array", items: { type: "string" } },
    state_toml_updated: { type: "boolean" }
  },
  required: ["seq", "level", "trigger", "key_insight", "state_toml_updated"]
}

const trigger = args.trigger || 'unknown'
const level = args.level || 'standard'
const seq = args.seq || 0
const eligibleTier = args.eligibleTier != null ? args.eligibleTier : 0

phase('Prepare')
log('Brainstorm ' + seq + ' (' + level + ') — ' + trigger)

const state = await agent(
  'Read .research/state.toml and .research/context.md.\n' +
  'Return: all active epics (with tier, status, north_star, success_criteria),\n' +
  'list of blocked tasks (id + blocked_reason),\n' +
  'and a summary of context.md (current focus, tried & rejected, constraints).',
  { label: 'bs-recover', phase: 'Prepare', schema: RECOVER_SCHEMA }
)

const activeEpics = state.epics.filter(function(e) { return e.status === 'active' })
const tierConstraint = 'FOCUS: T' + eligibleTier + ' epics only. Any T0 epic with unmet criteria takes absolute priority.'
const epicsSummary = activeEpics.map(function(e) {
  return 'T' + (e.tier != null ? e.tier : '?') + ' ' + e.id + ': ' + e.title + ' — ' + (e.north_star || '')
}).join('\n')

const contextBlob = '## Active Epics (sorted by tier)\n' + epicsSummary +
  '\n\n## Blocked Tasks\n' + (state.blocked_tasks || 'None') +
  '\n\n## Context Summary\n' + (state.context_summary || '') +
  '\n\n' + tierConstraint

if (level === 'standard') {
  phase('Analyze')
  const result = await agent(
    'You are a brainstorm analyst for the dev loop.\n' +
    'Brainstorm seq: ' + seq + '. Trigger: ' + trigger + '.\n\n' +
    contextBlob + '\n\n' +
    'Read the theme synthesis files at .research/findings/themes/*-synthesis.md.\n' +
    'Read .research/context.md for tried & rejected approaches.\n\n' +
    'Write structured analysis to .research/findings/brainstorm/bs' + seq + '.md:\n' +
    '- Epic progress assessment\n' +
    '- Technical feasibility analysis\n' +
    '- Risks and mitigation\n' +
    '- Skeptic challenges (challenge your own recommendations)\n' +
    '- Concrete recommendations: new tasks, themes, reprioritization\n\n' +
    'Then update .research/state.toml based on your recommendations:\n' +
    '- Create new [[tasks]] entries if needed\n' +
    '- Create new [[themes]] entries if needed\n' +
    '- Park or complete themes if appropriate\n' +
    '- Set brainstorm_seq = ' + seq + ', tasks_since_brainstorm = 0\n' +
    '- Add [[brainstorms]] entry with seq, trigger, level, key_insight\n\n' +
    (trigger.includes('No ready tasks')
      ? 'CRITICAL: You MUST produce at least one new task — the loop is stuck without new work.\n'
      : 'You MAY reprioritize existing tasks instead of creating new ones. Record key_insight either way.\n') +
    'All task IDs follow format: {theme_id}.{N}.\n' +
    'All recommendations must reference which epic (and tier) they serve.',
    { label: 'bs' + seq, phase: 'Analyze', schema: BRAINSTORM_SCHEMA }
  )
  return result

} else if (level === 'high') {
  phase('Analyze')

  await agent(
    'You are the PROPOSER for brainstorm ' + seq + '.\n\n' +
    contextBlob + '\n\n' +
    'Read theme synthesis files at .research/findings/themes/*-synthesis.md.\n' +
    'Read .research/context.md.\n\n' +
    'Write your analysis to .research/findings/brainstorm/bs' + seq + '-proposer.md:\n' +
    '- Epic Assessment: progress, gaps, what needs to happen next\n' +
    '- Systems/Compiler/GPU Architecture Analysis: technical deep dive\n' +
    '- Concrete Recommendations: specific tasks, themes, priorities\n\n' +
    'Be bold. Think big. Propose the path forward.',
    { label: 'bs' + seq + '-proposer', phase: 'Analyze' }
  )

  await agent(
    'You are the SKEPTIC for brainstorm ' + seq + '.\n' +
    'Read the proposer analysis at .research/findings/brainstorm/bs' + seq + '-proposer.md.\n\n' +
    contextBlob + '\n\n' +
    'Write your critique to .research/findings/brainstorm/bs' + seq + '-skeptic.md:\n' +
    '- Challenge every claim the proposer makes\n' +
    '- Find holes in the reasoning\n' +
    '- Identify untested assumptions\n' +
    '- What could go wrong with each recommendation?\n' +
    '- What is the proposer NOT seeing?',
    { label: 'bs' + seq + '-skeptic', phase: 'Analyze' }
  )

  phase('Synthesize')
  const result = await agent(
    'Synthesize brainstorm ' + seq + ' from proposer and skeptic.\n' +
    'Read both files:\n' +
    '- .research/findings/brainstorm/bs' + seq + '-proposer.md\n' +
    '- .research/findings/brainstorm/bs' + seq + '-skeptic.md\n\n' +
    'Write synthesis to .research/findings/brainstorm/bs' + seq + '.md.\n' +
    'Structure: Points of Agreement, Points of Disagreement, Resolution, Final Recommendations.\n\n' +
    'Then update .research/state.toml:\n' +
    '- Implement final recommendations (new tasks/themes, reprioritization)\n' +
    '- Set brainstorm_seq = ' + seq + ', tasks_since_brainstorm = 0\n' +
    '- Add [[brainstorms]] entry\n' +
    (trigger.includes('No ready tasks')
      ? 'CRITICAL: MUST produce at least one new task.\n'
      : ''),
    { label: 'bs' + seq + '-synth', phase: 'Synthesize', schema: BRAINSTORM_SCHEMA }
  )
  return result

} else if (level === 'deep') {
  const experts = [
    { role: 'systems', title: 'Systems Architect' },
    { role: 'compiler', title: 'Compiler Engineer' },
    { role: 'gpu', title: 'GPU Architect' },
    { role: 'performance', title: 'Performance Engineer' }
  ]

  phase('Analyze')
  log('R1: Independent analysis by ' + experts.length + ' experts...')
  await parallel(experts.map(function(expert) {
    return async function() {
      return await agent(
        'You are a ' + expert.title + ' analyzing brainstorm ' + seq + '.\n\n' +
        contextBlob + '\n\n' +
        'Read theme synthesis files at .research/findings/themes/*-synthesis.md.\n' +
        'Read .research/context.md.\n\n' +
        'Write your independent analysis to .research/findings/brainstorm/bs' + seq + '-' + expert.role + '.md.\n' +
        'Focus on your domain expertise. Be specific and technical.\n' +
        'Structure: Assessment, Key Risks, Recommendations, Open Questions.',
        { label: 'bs' + seq + '-' + expert.role, phase: 'Analyze' }
      )
    }
  }))

  log('R2: Cross-review...')
  const allR1Files = experts.map(function(e) {
    return '.research/findings/brainstorm/bs' + seq + '-' + e.role + '.md'
  }).join(', ')

  await parallel(experts.map(function(expert) {
    return async function() {
      return await agent(
        'You are a ' + expert.title + ' cross-reviewing brainstorm ' + seq + '.\n' +
        'Read ALL R1 analysis files: ' + allR1Files + '\n\n' +
        'Write your cross-review to .research/findings/brainstorm/bs' + seq + '-' + expert.role + '-review.md.\n' +
        'Note: agreements with other experts, disagreements, new insights from combining perspectives.\n' +
        'Where you disagree, explain why from your domain expertise.',
        { label: 'bs' + seq + '-' + expert.role + '-review', phase: 'Analyze' }
      )
    }
  }))

  phase('Synthesize')
  log('R3: Synthesis...')
  const allFiles = experts.map(function(e) {
    return '.research/findings/brainstorm/bs' + seq + '-' + e.role + '.md'
  }).concat(experts.map(function(e) {
    return '.research/findings/brainstorm/bs' + seq + '-' + e.role + '-review.md'
  }))

  const result = await agent(
    'Synthesize deep brainstorm ' + seq + '.\n' +
    'Read ALL files:\n' + allFiles.join('\n') + '\n\n' +
    'Write synthesis to .research/findings/brainstorm/bs' + seq + '.md.\n' +
    'Structure:\n' +
    '- Panel: each expert\'s stance (1 sentence each)\n' +
    '- Consensus: what all experts agree on\n' +
    '- Disagreements: resolved (with reasoning) + unresolved\n' +
    '- Recommendations: concrete tasks/themes/priorities\n\n' +
    'Then update .research/state.toml:\n' +
    '- Implement recommendations\n' +
    '- Set brainstorm_seq = ' + seq + ', tasks_since_brainstorm = 0\n' +
    '- Add [[brainstorms]] entry\n' +
    (trigger.includes('No ready tasks')
      ? 'CRITICAL: MUST produce at least one new task.\n'
      : ''),
    { label: 'bs' + seq + '-synth', phase: 'Synthesize', schema: BRAINSTORM_SCHEMA }
  )
  return result
}
