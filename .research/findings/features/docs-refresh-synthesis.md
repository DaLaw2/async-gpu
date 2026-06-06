# docs-refresh — Feature Synthesis

## State

All docs live in untracked `docs/` (gitignored). No docs at repo root.

## ARCHITECTURE.md Gaps (Critical)

- Crate map shows 6 crates + 4 examples; actual: 15 crates + 27 examples
- Missing: kernel split (4 crates), unified runtime, tiered memory,
  auto-fusion, GpuArray, AutoTuner, nn/onnx stack, GPU test framework
- Hostcall protocol section still accurate; crate map and build sections stale

## CHANGELOG.md Gaps (Minor)

- Covers cycles 309-638; current is 642 (4 cycles behind)
- Unreleased section exists but cycle range header is stale

## Stale Docs

- DESIGN-executor.md: unimplemented design (637 lines) — remove or archive
- VALIDATION.md: mostly current, example list incomplete
- getting-started.md: needs separate audit

## Key Decision

docs/ is gitignored — documentation is invisible to collaborators.
Un-gitignore and track, or accept local-only status?
