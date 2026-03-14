# hashmap-fix.1: Investigate HashMap seed on GPU
**Cycle**: 207 | **Theme**: hashmap-fix | **Kind**: investigation | **Status**: done

## Summary
HashMap does NOT panic on GPU. The `unsupported` random module's
`hashmap_random_keys()` returns address-based pseudo-random keys (stack addr
+ heap Box addr) WITHOUT calling `fill_bytes()`. Only direct `fill_bytes()`
calls panic.

## Findings
### Q: Does HashMap::new() work on GPU?
A: Yes. `hashmap_random_keys()` in the unsupported module uses stack pointer
and Box allocation addresses for seed derivation. It allocates a Box<u8> on
the bump allocator (now atomic-safe) and uses the resulting addresses. No
panic path is triggered.
**Confidence**: high

### Q: Is the seed quality acceptable?
A: The seed is deterministic per-launch (same heap layout → same addresses →
same seed). This means HashMap ordering is deterministic but functionally
correct. For GPU workloads, hash-DoS is not a concern. Could be improved
with `%clock64` + `%tid.x` for better entropy, but not required.
**Confidence**: high

## Impact on Downstream Tasks
- hashmap-fix.2 (HashMap test kernel) should still be done to verify e2e
- No std patch changes needed — the existing random module is functional
