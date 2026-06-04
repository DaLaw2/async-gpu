# sc-design.4 — Compiler Requirements for Scope Enforcement

## Status: done
## Summary: The structured concurrency system (BlockScope, GridScope) does NOT need MIR pass changes for scope enforcement. The existing library-level mechanisms — `for<'scope>` HRTB, `PhantomData<&'scope mut &'scope ()>` invariance, `T: Copy` bounds, and runtime `debug_assert` for warp-0-only access — provide sound lifetime enforcement that the borrow checker validates identically on nvptx64 and x86_64. Known escape hatches (transmute, forget, raw pointers) all require `unsafe` and are acceptable under Rust's safety contract. The one MIR pass in the patched rustc (`WarpCooperativeTransform`) serves async/await warp convergence, not scope enforcement, and should remain separate. A future T3 epic (`ownership-memory`) envisions compile-time memory-tier enforcement, but this is exploratory and not needed for the structured concurrency system to ship safely.

## 1. Library-Level Enforcement Analysis

### 1.1 HRTB prevents scope escape (compile-time, sound)

The `for<'scope>` pattern on `block_scope` and `grid_scope`:

```rust
pub fn block_scope<F, R>(f: F) -> R
where
    F: for<'scope> FnOnce(&mut BlockScope<'scope>) -> R,
```

This is the same pattern used by `std::thread::scope`, `crossbeam::scope`, and `rayon::scope`. The universal quantification means the caller cannot name `'scope`, so any `&'scope mut [T]` obtained from `scope.alloc()` cannot be assigned to a variable with a longer lifetime. The borrow checker rejects:

```rust
let mut escaped: &mut [f32] = &mut [];
block_scope(|scope| {
    escaped = scope.alloc::<f32>(64); // COMPILE ERROR
});
```

This enforcement is target-independent — `rustc_borrowck` runs entirely at MIR level before any nvptx64 codegen. Confirmed in sc-resource.1: "The `for<'scope>` HRTB mechanism, `PhantomData<&'scope mut &'scope ()>` invariance, and all lifetime inference run in `rustc_borrowck`, which is target-independent."

### 1.2 PhantomData invariance prevents lifetime weakening (compile-time, sound)

`BlockScope` and `GridScope` both use `PhantomData<&'scope mut &'scope ()>` to make `'scope` invariant. Without this, Rust's covariance rules could allow `&'scope [T]` to be widened to `&'longer [T]`, potentially enabling escape. The `&'scope mut &'scope ()` pattern forces the compiler to treat `'scope` as neither covariant nor contravariant.

This is the standard library technique (used in `std::thread::Scope`).

### 1.3 T: Copy constraint prevents destructor issues (compile-time, sound)

All `scope.alloc::<T>()` methods require `T: Copy`. Since `Copy` types cannot implement `Drop`, the watermark-pop approach (bulk deallocation without per-element destruction) is sound. No MIR pass needed to verify this — it's a trait bound.

### 1.4 Warp-0-only allocation (runtime, acceptable)

`scope.alloc()` uses `debug_assert_eq!(warp_id(), 0)` to enforce single-writer access. This is a runtime check, not a compile-time one. However:

- **Cannot be a compile-time check**: Warp ID is a hardware register (`%tid.x / 32`) — it's a runtime value. There is no type-level representation of "which warp am I" in the current system.
- **Structurally enforced by API design**: Only warp 0 enters `block_scope()` (itself guarded by `debug_assert`). The `BlockScope` handle is passed to the closure and never shared across warps (it's `&mut`, not `&`). Spawned closures receive captured values, not the scope handle itself.
- **The `debug_assert` is defense-in-depth**, not the primary enforcement mechanism.

A compile-time `WarpZero` capability type is theoretically possible but would require fundamental changes to how GPU thread identity is modeled in the type system — far beyond a MIR pass. This is the domain of the T3 `ownership-memory` epic, not the structured concurrency system.

### 1.5 Send bound on spawned closures (compile-time, sound)

`scope.spawn` requires `F: FnOnce() -> T + Send + 'scope`. The `Send` bound ensures the closure can safely move to another warp. The `'scope` bound (replacing the `'static` requirement of `thread::spawn`) allows borrowing scope-allocated data. This is the same mechanism as `std::thread::scope`.

## 2. Potential MIR Pass Improvements

### 2.1 Shared memory tier enforcement (NOT needed)

**Question**: Should a MIR pass distinguish `&[f32]` in shared memory from `&[f32]` in global memory?

**Answer: No, for practical and theoretical reasons.**

1. **The borrow checker already prevents the dangerous case.** The concern is: "could a shared-memory reference escape the block scope and be accessed from another block?" The `for<'scope>` HRTB already prevents this — the reference cannot escape `block_scope()` at all, regardless of what memory backs it.

2. **Address space is invisible to the type system.** After `cvta.shared.u64` converts a shared-memory address to generic address space, Rust sees only `*mut u8` / `&mut [T]`. The LLVM address space (addrspace 3 vs 0) is below the type system. A MIR pass could inspect LLVM IR annotations, but this would require a post-LLVM-lowering pass, not a MIR pass — MIR does not carry address space information.

3. **The T3 `ownership-memory` epic explores this.** The `om-compiler` theme envisions "lifetime parameters in rustc map to address space annotations in LLVM IR." This is a research direction, not a requirement for structured concurrency. The success criteria are aspirational: "Cross-scope borrows rejected at compile time" — which the HRTB already achieves for the current scope model.

4. **Cost is extremely high.** Mapping lifetimes to address spaces would require changes to rustc's type system (not just a MIR pass), changes to the LLVM IR emission, and potentially changes to how references are represented. This is a multi-month effort with ongoing maintenance burden across rustc versions.

### 2.2 Warp-0-only allocation enforcement (NOT needed as MIR pass)

A MIR pass could theoretically reject `scope.alloc()` calls from basic blocks reachable from non-warp-0 code paths. But:

- The analysis would need to trace which closures run on which warps — this is a runtime property (warp scheduling), not a static property of the MIR.
- The API already prevents this structurally: `scope.alloc()` takes `&self` on `BlockScope`, and the scope is passed as `&mut BlockScope` to the closure. Spawned closures receive captured data, not the scope handle.
- The `debug_assert` catches any violations in development builds.

### 2.3 Custom GPU types (NOT needed)

All scope-allocated types must satisfy `T: Copy`. GPU-specific types like atomic wrappers (`AtomicU32`) are `!Copy` and cannot be scope-allocated — which is correct, since atomics need careful initialization and should not be bulk-deallocated. No MIR pass needed.

## 3. Known Escape Hatches

### 3.1 `unsafe` escape hatches (acceptable)

The following can bypass lifetime enforcement, but all require `unsafe`:

1. **`core::mem::transmute`**: Can transmute `&'scope mut [f32]` to `&'static mut [f32]`. Requires `unsafe`. This is inherent to Rust — no library or MIR pass can prevent `unsafe` transmute.

2. **`core::mem::forget`**: Can forget the `BlockScope`, preventing `Drop` from running (watermark not popped, spawned warps not joined). However, `block_scope()` explicitly calls `scope.join_all()` and `Drop` is only the safety net. Even with `forget`, the `for<'scope>` bound still prevents the reference from escaping the closure — `forget` cannot extend a lifetime.

3. **Raw pointers**: A user can `scope.alloc::<f32>(64).as_mut_ptr()` to get a raw pointer, then use it after the scope exits. This is an `unsafe` operation (dereferencing the pointer after the scope exits is UB). The borrow checker does not track raw pointer lifetimes — this is standard Rust behavior, not a GPU-specific issue.

4. **`core::ptr::read` / `core::ptr::write`**: Can read/write through raw pointers to scope memory. Same as above — requires `unsafe`.

**Assessment**: All escape hatches require `unsafe`. This matches Rust's safety model: safe code cannot violate scope boundaries. A MIR pass cannot prevent `unsafe` code from doing unsafe things — that is what `unsafe` means. The `unsafe` keyword itself is the enforcement mechanism.

### 3.2 Safe-code edge cases (none found)

No safe-code patterns were identified that could bypass the `for<'scope>` + invariance enforcement. The standard library's `std::thread::scope` uses the same mechanism and has been audited by the Rust team. The key properties:

- `for<'scope>` prevents naming the lifetime (cannot store in a struct with a named lifetime parameter).
- Invariance prevents subtyping (cannot weaken `'scope` to a longer lifetime).
- `R` (return value) must not borrow `'scope` — enforced by the borrow checker since `R` is returned from the closure to the caller, where `'scope` does not exist.

## 4. Existing MIR Passes

### 4.1 WarpCooperativeTransform (unrelated to scope enforcement)

The patched rustc has exactly one GPU-specific MIR pass: `WarpCooperativeTransform` in `rustc-patches/warp_cooperative.rs`. It:

1. Runs after `coroutine::StateTransform` in the `mir_drops_elaborated_and_const_checked` pipeline.
2. Is gated by `sess.target.arch == Arch::Nvptx64`.
3. Applies to all coroutine bodies (async fn state machines).
4. Inserts `activemask.b32` + `shfl.sync.idx.b32` to broadcast the coroutine dispatch discriminant across all 32 lanes in a warp.
5. Inserts `bar.warp.sync` before every `Return` terminator.

**Purpose**: Warp convergence for async/await. This pass ensures all 32 lanes in a warp stay in the same coroutine state, which is required for the hostcall protocol.

**Relationship to scopes**: None. This pass operates on coroutine state machines, not on scope lifetimes or memory allocation. Scope enforcement is purely a borrow-checker concern, handled before MIR optimization passes run.

### 4.2 Other rustc patches

The remaining patches are purely for the `#[warp_cooperative]` attribute:
- `rustc_feature_src_builtin_attrs.patch` — registers the attribute
- `rustc_passes_src_check_attr.patch` — allows the attribute on functions
- `rustc_span_src_symbol.patch` — adds the symbol
- `rustc_mir_transform_src_lib.patch` — registers the MIR pass in the pipeline

No other GPU-specific MIR passes exist.

### 4.3 Could WarpCooperativeTransform be extended for scope enforcement?

**No, and it should not be.** The pass operates at a fundamentally different level:

- WarpCooperativeTransform: post-StateTransform, operates on coroutine MIR, inserts PTX instructions.
- Scope enforcement: pre-codegen, operates on lifetime checking in `rustc_borrowck`.

Trying to extend a MIR optimization pass to do borrow checking would be architecturally wrong — it would bypass the established borrow-checker infrastructure and would not integrate with Rust's error reporting.

## 5. Cost-Benefit: Library vs Compiler

### 5.1 What library-level enforcement gives us

| Property | Mechanism | Enforcement | Sound? |
|---|---|---|---|
| Scope-allocated refs cannot escape | `for<'scope>` HRTB | Compile-time | Yes |
| Lifetime cannot be weakened | `PhantomData` invariance | Compile-time | Yes |
| No destructors needed | `T: Copy` bound | Compile-time | Yes |
| Closures safe to move across warps | `F: Send + 'scope` | Compile-time | Yes |
| Only warp 0 allocates | `debug_assert` | Runtime (dev) | Yes (API structural) |
| Spawned work cannot outlive scope | `Drop` + explicit join | Runtime | Yes |

### 5.2 What a MIR pass could add

| Property | Would require | Maintenance cost | Safety benefit |
|---|---|---|---|
| Shared-memory refs type-distinct from global-memory refs | Type system change + LLVM IR mapping | Very high (touches rustc core, breaks on version updates) | Low (HRTB already prevents cross-scope escape) |
| Compile-time warp-0-only enforcement | Thread-identity type system | Very high (GPU execution model in types) | Low (API already structural, debug_assert catches bugs) |
| Reject `unsafe` transmute of scope refs | Impossible (defeats purpose of `unsafe`) | N/A | N/A |

### 5.3 Maintenance cost of MIR passes

The existing `WarpCooperativeTransform` provides a concrete data point:

- 606 lines of code in `warp_cooperative.rs`
- 4 additional patch files to rustc
- Must be validated against every rustc nightly update (MIR representation can change)
- Uses internal compiler APIs (`rustc_middle::mir`, `rustc_span::sym`) that are unstable

Each additional MIR pass multiplies this maintenance burden. The warp-cooperative pass is justified because it enables a fundamentally new capability (async/await on GPU) that CANNOT be achieved with library-level techniques alone. Scope enforcement does NOT have this property — it IS achievable with library techniques.

### 5.4 Safety guarantees given up by staying library-only

1. **No compile-time memory-tier distinction**: `&'scope [f32]` from `BlockScope` (shared memory) looks identical to `&'scope [f32]` from `GridScope` (global memory) in the type system. A user could hypothetically pass a shared-memory reference to a cross-block API. However, the HRTB prevents the reference from escaping the scope, so the only dangerous pattern would be within the scope closure itself — and cross-block APIs take raw pointers (`*const u8`), not references, so the type system already distinguishes them.

2. **No compile-time warp-identity tracking**: Code executing on warp 3 could call `scope.alloc()` if it somehow obtained the scope handle. The API makes this very hard (the handle is `&mut`, not `&`; spawned closures receive captured values), but it's not provably impossible in all code paths.

3. **Runtime assertions in release builds are compiled out**: `debug_assert` is removed in release mode. If a user reaches `scope.alloc()` from a non-warp-0 path in release, it will silently corrupt the allocator. This is mitigated by the structural enforcement (see 1.4 above).

**Net assessment**: The safety gap is minimal and all edge cases require either `unsafe` code or pathological API misuse that the type system makes very difficult.

## 6. Recommendation

**Stay library-only for scope enforcement. Do NOT add a MIR pass.**

Rationale:

1. **The existing mechanisms are sound.** The `for<'scope>` + invariance pattern is proven in production (std, crossbeam, rayon) and works identically on nvptx64 (confirmed in sc-resource.1).

2. **The safety gap is negligible.** All known escape hatches require `unsafe`, which is Rust's intended contract. The one runtime check (`debug_assert` for warp-0-only) is defense-in-depth over structural API enforcement.

3. **The maintenance cost of a MIR pass is high.** The existing `WarpCooperativeTransform` is justified by enabling an otherwise-impossible capability. A scope-enforcement MIR pass would be a high-cost, low-benefit addition.

4. **The T3 `ownership-memory` epic is the right place for memory-tier enforcement.** If compile-time shared-vs-global distinction is ever needed, it should be part of a broader type-system effort (`om-compiler` theme), not a one-off MIR pass bolted onto the structured concurrency system.

5. **Practical priority**: The structured concurrency system should ship with library-level enforcement. Compiler-level memory-tier mapping is a T3 research direction that depends on the T1 structured concurrency system being stable first.

**One actionable improvement** (no compiler changes needed): Consider promoting the `debug_assert_eq!(warp_id(), 0)` in `scope.alloc()` to a full `assert!` rather than `debug_assert!`. The cost is one PTX instruction (`mov.u32 %r, %tid.x; shr.b32 %r, %r, 5; setp.ne.u32 ...`) — negligible compared to the shared memory allocation itself. This would provide runtime safety even in release builds, closing the last non-`unsafe` gap.

## Files Changed: none
