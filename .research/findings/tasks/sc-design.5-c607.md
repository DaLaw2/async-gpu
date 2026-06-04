# sc-design.5 — Cancellation Chain-Walk Implementation

## Status: done

## Summary

Implemented parent-to-child cancellation propagation via chain-walk in both `BlockScope` and `GridScope`. When a parent scope cancels, children discover it on their next `is_cancelled()` poll without the parent needing to enumerate or notify children explicitly.

## Changes Made

### File: `crates/core/gpu-runtime/src/scope.rs`

#### 1. BlockScope: `parent_cancel_ptr` field

Added `parent_cancel_ptr: *const u32` to `BlockScope`. This pointer is:
- Null for top-level scopes (created via `block_scope()`)
- Points to shared memory for nested BlockScope parents
- Points to global memory for GridScope parents

#### 2. BlockScope: `is_cancelled()` chain-walk

Updated `is_cancelled()` to check local flag first (~2 cycles shared memory read), then parent flag if local is clear. The parent read costs ~2 cycles if parent is a BlockScope (shared memory) or ~100 cycles if parent is a GridScope (global memory). Short-circuits on first set flag.

Design note: only checks one level up (local + parent), not the full chain recursively. Grandparent propagation happens when the parent scope checks its own parent at its yield points. This keeps each check O(1).

#### 3. BlockScope: `cancel_ptr()` accessor

Returns a raw `*const u32` to this scope's cancel flag in shared memory. Callers pass this to `block_scope_with_parent()` when creating nested scopes.

#### 4. BlockScope: `propagate_cancel()` helper

Convenience method that reads the parent cancel flag and, if set, sets the local cancel flag. Useful at the top of a scope body for eager propagation.

#### 5. `block_scope_with_parent()` entry function

New public function that accepts a `parent_cancel: *const u32` and constructs a `BlockScope` with the chain link set. `block_scope()` now delegates to this with `null` parent.

#### 6. GridScope: `parent_cancel_ptr` field + chain-walk `is_cancelled()`

Added `parent_cancel_ptr: *const u32` to `GridScope` (currently always null since GridScope does not nest). Updated `is_cancelled()` to check local flag then parent flag, mirroring BlockScope's pattern. The `grid_scope()` entry function initializes `parent_cancel_ptr` to null.

## Verification

- `bash scripts/ci-lint.sh` passes (fmt, clippy, all PTX kernels, all host examples).
- Pre-existing clippy errors in `unified_channel.rs` (unrelated to this change) were confirmed as pre-existing by testing main branch.

## API Surface

### New public items
- `block_scope_with_parent(parent_cancel: *const u32, f: F) -> R` — nested scope entry
- `BlockScope::cancel_ptr(&self) -> *const u32` — expose cancel flag for children
- `BlockScope::propagate_cancel(&self)` — pull parent cancellation eagerly

### Modified public items
- `BlockScope::is_cancelled(&self) -> bool` — now chain-walks to parent
- `BlockScope::cancel(&self)` — doc updated to clarify local-only semantics
- `GridScope::is_cancelled(&self) -> bool` — now chain-walks to parent (currently no-op since parent is always null)

### Unchanged
- `block_scope()` — delegates to `block_scope_with_parent(null, f)`
- `GridScope::cancel_flag_ptr()` — already existed, used as `parent_cancel` for BlockScopes inside a GridScope

## Files Changed

- `crates/core/gpu-runtime/src/scope.rs`
