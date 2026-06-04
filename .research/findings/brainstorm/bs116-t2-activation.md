# Brainstorm 116 — T2 Tier Activation

## Trigger: T1 cleared, T2 eligible
## Level: standard
## Date: 2026-06-05

---

## 1. T2 Readiness Assessment

### T1 Epic Status (all 5 completed)

| Epic | Status | Tier | Notes |
|------|--------|------|-------|
| native-rust-dx | completed | T0 | gpu-kernel ABI, MIR pass, gpu::run() |
| std-thread-gpu | completed | T0 | thread::spawn() maps to warp |
| cooperative-compute | completed | T1 | gpu::cooperative(), nn ops callable |
| structured-concurrency | completed | T1 | BlockScope/GridScope, channels, cancellation |
| library-api | completed | T1 | facade crate, setup.sh, getting-started guide |
| kernel-perf | completed | T1 | SGEMM 90% cuBLAS, FA >=35%, GPT-2 <40ms |

**All T1 epics completed. T2 tier is eligible for activation.**

### Dependency Check for T2 Epics

| T2 Epic | Depends On | Dependency Met? |
|---------|-----------|----------------|
| gpu-iterator (HIGH) | structured-concurrency | YES (completed) |
| auto-fusion (MEDIUM) | (none explicit) | YES |
| cuda-graph-scheduling (MEDIUM) | (none explicit) | YES |
| unified-runtime (MEDIUM) | (none explicit) | YES |
| conv-perf (MEDIUM) | kernel-perf | YES (completed) |
| cross-vendor (LOW) | (none explicit) | YES |
| std-completeness (LOW) | (none explicit) | YES |

All T2 dependencies satisfied.

---

## 2. Priority Analysis

### Tier Protocol

Per CLAUDE.md: "Never start T(N+1) tasks while T(N) has unmet criteria, unless T(N) is
explicitly blocked on external factors." T1 is cleared, so T2 activation is valid.

Within T2, priority ordering governs: start highest-priority T2 epics first.

### T2 Priority Ranking

1. **gpu-iterator** — HIGH priority (only T2 epic at HIGH)
2. **auto-fusion** — MEDIUM priority
3. **cuda-graph-scheduling** — MEDIUM priority
4. **unified-runtime** — MEDIUM priority
5. **conv-perf** — MEDIUM priority
6. **cross-vendor** — LOW priority
7. **std-completeness** — LOW priority

### Phase 2 Roadmap Alignment

The Phase 2 roadmap (brainstorm 115, user-approved) says:
> Phase 2: gpu-iterator + auto-fusion

Both epics are the designated next wave. gpu-iterator is HIGH, auto-fusion is MEDIUM.
Per tier protocol, gpu-iterator starts first. auto-fusion can begin once gpu-iterator
has no blocking work (i.e., investigation/design tasks completed and experiments running).

### Why gpu-iterator First

1. **Priority**: Only HIGH-priority T2 epic
2. **Roadmap**: Explicitly first in Phase 2
3. **North Star fit**: "no GPU concepts leak" — par_iter() eliminates kernel launch concepts
4. **Foundation for auto-fusion**: Iterator chain compilation shares MIR pass infrastructure
   with fusion analysis. Building the MIR pass for iterators creates reusable machinery.
5. **Demo impact**: `data.par_iter().map(|x| x * 2.0).collect()` is immediately compelling

### Why auto-fusion Second (not parallel)

1. **MIR pass overlap**: Both epics need MIR-level analysis and transformation. Building
   gpu-iterator's MIR pass first provides patterns and infrastructure for fusion.
2. **Max 2 subagents**: Resource constraint means we can't fully parallelize both epics.
3. **Dependency**: While not formal, auto-fusion's "identify chains of elementwise operations"
   builds on the same iterator-chain-to-kernel compilation that gpu-iterator develops.
4. **Activation timing**: Start fusion-analysis.1 (investigation) once iter-design.3
   (MIR strategy) completes — the investigation is independent and low-cost.

---

## 3. Existing Theme/Task Verification

### gpu-iterator: Themes and Tasks Review

**4 themes, 11 tasks** — all well-defined and properly scoped.

#### Theme: iter-design (3 tasks)
- iter-design.1: Investigation — Rust Iterator/Rayon trait mapping to GPU ✓ Good scope
- iter-design.2: Design — ParallelIterator trait API + closure capture rules ✓ Good scope
- iter-design.3: Design — MIR pass strategy for iterator chain → kernel ✓ Good scope
- **Assessment**: Sound design-first approach. Critical path starts here.

#### Theme: iter-compiler (3 tasks)
- iter-compiler.1: Experiment — MIR pass for map() ✓ Good scope, builds on iter-design.3
- iter-compiler.2: Experiment — filter() + fold() via ballot + shuffle ✓ Correctly identifies GPU primitives
- iter-compiler.3: Experiment — collect() with atomic output sizing ✓ Key challenge addressed
- **Assessment**: Well-scoped. Each operation has specific GPU mapping. Dependencies correct.

#### Theme: iter-runtime (2 tasks)
- iter-runtime.1: Experiment — warp-parallel partitioning + output buffers ✓ Good scope
- iter-runtime.2: Experiment — chained fusion, zero intermediate buffers ✓ This is where iterator and fusion overlap
- **Assessment**: Sound. iter-runtime.2 is the bridge to auto-fusion conceptually.

#### Theme: iter-demo (2 tasks)
- iter-demo.1: Experiment — par_iter().map().collect() on 1M+ elements ✓ Proves correctness
- iter-demo.2: Experiment — GPU par_iter vs CPU Rayon benchmark ✓ Proves value proposition
- **Assessment**: Good. Both demos directly verify epic success criteria.

**Verdict: gpu-iterator themes/tasks are still relevant, well-scoped, and ready to execute.**

### auto-fusion: Themes and Tasks Review

**3 themes, 8 tasks** — all well-defined.

#### Theme: fusion-analysis (3 tasks)
- fusion-analysis.1: Investigation — fusable MIR patterns ✓ Good entry point
- fusion-analysis.2: Design — pattern matching rules ✓ Depends on investigation
- fusion-analysis.3: Experiment — implement detection ✓ First code deliverable
- **Assessment**: Sound. Investigation can start early (before gpu-iterator completes).

#### Theme: fusion-codegen (3 tasks)
- fusion-codegen.1: Investigation — code gen strategies (PTX vs NVRTC vs dynamic) ✓ Important architectural decision
- fusion-codegen.2: Experiment — generate fused PTX from fusion graph ✓ Core deliverable
- fusion-codegen.3: Experiment — register-only intermediates + float4 ✓ Optimization layer
- **Assessment**: Good. The "compile-time" constraint (not runtime JIT) is correctly reflected.

#### Theme: fusion-integrate (2 tasks)
- fusion-integrate.1: Experiment — auto-fuse nn::Linear ✓ Proves end-to-end
- fusion-integrate.2: Experiment — GPT-2 benchmark with fusion ✓ Measures real-world impact
- **Assessment**: Good end-to-end validation.

**Verdict: auto-fusion themes/tasks are still relevant and well-scoped.**

### Concern: MIR Pass Feasibility

Both epics assume new MIR passes. The existing infrastructure is:
- `warp_cooperative.rs` (605 lines): transforms coroutine dispatch for warp-cooperative execution
- Integrated into `rustc_mir_transform` pipeline via patches

The existing MIR pass handles coroutine state machines, not iterator chains or elementwise
fusion. The gpu-iterator epic needs a DIFFERENT kind of MIR pass: one that recognizes
iterator trait method chains and compiles them to data-parallel GPU code. This is
substantially more complex than the warp-cooperative transform.

**Risk level**: MEDIUM. The iter-design.1 investigation should explicitly assess whether
MIR-level transformation is the right approach vs. a proc-macro or trait-based approach
that generates GPU code at the library level (like Rayon does for CPU threads).

---

## 4. Critical Path

### gpu-iterator Critical Path

```
iter-design.1 (investigation: Rayon/Iterator mapping)
    └→ iter-design.2 (design: ParallelIterator trait)
        └→ iter-design.3 (design: MIR pass strategy)
            ├→ iter-compiler.1 (experiment: map() MIR pass)
            │   ├→ iter-compiler.2 (experiment: filter() + fold())
            │   ├→ iter-compiler.3 (experiment: collect())
            │   └→ iter-runtime.1 (experiment: partitioning)
            │       └→ iter-runtime.2 (experiment: chain fusion)
            │           └→ iter-demo.1 (experiment: 1M+ elements)
            │               └→ iter-demo.2 (experiment: Rayon benchmark)
            └→ [gate: design review before compiler work]
```

**Shortest path to working demo**: 7 tasks on the critical path
(design.1 → design.2 → design.3 → compiler.1 → runtime.1 → demo.1 → demo.2)

**Estimated cycles**: 14-20 (2 cycles per task average, design tasks faster)

### auto-fusion Critical Path

```
fusion-analysis.1 (investigation: fusable patterns)
    └→ fusion-analysis.2 (design: pattern matching rules)
        ├→ fusion-analysis.3 (experiment: implement detection)
        └→ fusion-codegen.1 (investigation: codegen strategy)
            └→ fusion-codegen.2 (experiment: generate fused PTX)
                ├→ fusion-codegen.3 (experiment: register + float4)
                └→ fusion-integrate.1 (experiment: nn::Linear fusion)
                    └→ fusion-integrate.2 (experiment: GPT-2 benchmark)
```

**Shortest path to working demo**: 6 tasks
(analysis.1 → analysis.2 → codegen.1 → codegen.2 → integrate.1 → integrate.2)

**Estimated cycles**: 12-18

### Combined Phase 2 Critical Path (staggered start)

```
Week 1-2: iter-design.1, iter-design.2
Week 3:   iter-design.3 + fusion-analysis.1 (start fusion investigation)
Week 4-5: iter-compiler.1 + fusion-analysis.2
Week 6-7: iter-compiler.2/3 + iter-runtime.1 + fusion-codegen.1
Week 8-9: iter-runtime.2 + fusion-codegen.2
Week 10:  iter-demo.1/2 + fusion-codegen.3
Week 11:  fusion-integrate.1/2
```

Staggered start saves ~3-4 weeks vs sequential execution.

---

## 5. Activation Decision

### Activate Now: gpu-iterator

- **Status**: pending → **active**
- **Rationale**: Highest T2 priority, dependencies met, themes/tasks ready
- **First task**: iter-design.1 (Investigation: Rayon/Iterator trait mapping to GPU)
- **All 4 themes move to pending** (iter-design becomes active immediately)

### Activate Now (investigation only): auto-fusion

- **Status**: pending → **active** (but only fusion-analysis theme active)
- **Rationale**: Phase 2 roadmap pairs these two. The investigation task
  (fusion-analysis.1) has no dependencies and can start in parallel with
  iter-design work without conflicting.
- **First task**: fusion-analysis.1 (Investigation: fusable patterns in MIR)
- **Gate**: fusion-codegen and fusion-integrate themes stay pending until
  iter-design.3 completes (shared MIR pass infrastructure)

### Do NOT Activate Yet

| Epic | Why Not |
|------|---------|
| cuda-graph-scheduling | No themes/tasks defined yet. Lower priority than gpu-iterator. Brainstorm needed when ready. |
| unified-runtime | Phase 3 per roadmap. No themes/tasks defined. Wait for Phase 2 completion. |
| conv-perf | Has themes/tasks but is orthogonal to Phase 2 narrative. Activate only if Phase 2 stalls. |
| cross-vendor | LOW priority. Major architectural lift (AMD backend). Not aligned with current focus. |
| std-completeness | LOW priority. Has themes/tasks but no Phase 2 urgency. |

---

## 6. Recommended First Tasks

### Immediate (next cycle)

**1. iter-design.1** — Investigation: Rust Iterator/Rayon traits — which semantics map to GPU
- Theme: iter-design
- Priority: CRITICAL PATH
- Notes: Must answer key architectural question: MIR-level transformation vs
  library-level (trait-based) approach. Study Rayon's split/join model,
  Iterator::next() pull-based vs push-based GPU execution, closure capture
  constraints on nvptx64 (no heap allocation, no trait objects, POD types only).

**2. fusion-analysis.1** — Investigation: fusable patterns in MIR — elementwise chains, broadcast, reductions
- Theme: fusion-analysis
- Priority: Can start in parallel with iter-design.1
- Notes: Survey existing nn ops for fusion candidates. Look at GPT-2 forward pass
  for actual elementwise chains. Catalog patterns: mul+add+activation, residual+layernorm,
  bias+activation. Check what the existing perf-fusion theme already accomplished
  (manual fused LayerNorm+residual exists — can this be automated?).

### Next Wave (after design.1 completes)

**3. iter-design.2** — Design: ParallelIterator trait API surface + closure capture rules
- Depends: iter-design.1
- This is where the GPU-specific constraints (no heap, no dyn, POD closures) get baked
  into the trait design.

**4. fusion-analysis.2** — Design: MIR pass pattern matching rules
- Depends: fusion-analysis.1
- Can run in parallel with iter-design.2

### State Changes Required

```toml
# Epic status changes
gpu-iterator: pending → active
auto-fusion: pending → active

# Theme status changes
iter-design: pending → active
fusion-analysis: pending → active
# (all other themes in both epics remain pending)

# Brainstorm counter
brainstorm_seq = 116
tasks_since_brainstorm = 0
```

---

## Summary

T1 is fully cleared (all 5 epics completed, 724 tasks done). T2 activation is
straightforward: gpu-iterator is the only HIGH-priority T2 epic, and the Phase 2
roadmap explicitly pairs it with auto-fusion. Both epics already have well-scoped
themes and tasks from brainstorm 115.

The key architectural risk is the MIR pass approach — iter-design.1 must evaluate
whether a MIR-level transformation (like warp_cooperative.rs) is the right strategy
vs. a library-level approach (like Rayon's trait-based split/join). This investigation
gates all subsequent compiler work.

Staggered activation (gpu-iterator full, auto-fusion investigation-only) maximizes
throughput while respecting the 2-subagent concurrency limit and the shared MIR
infrastructure dependency.

## New tasks spawned: (none — existing tasks are sufficient)
## Themes activated: iter-design, fusion-analysis
## Epics activated: gpu-iterator, auto-fusion
