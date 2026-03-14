# warp-async-v2.3: Implement .await syntax recognition in proc macro
**Cycle**: 238 | **Theme**: warp-async-v2 | **Kind**: experiment | **Status**: done

## Summary
Implemented `.await` in `#[warp_async]` proc macro. The macro detects `Expr::Await`, infers the concrete future type from the base expression (e.g., `GpuPrintFuture::new(...)` → `GpuPrintFuture`), generates a `MaybeUninit<Type>` struct field, and creates INIT + POLL states. The POLL state calls `warp_poll_future()` for warp-cooperative polling (lane 0 polls, broadcasts result via shfl.sync). Tested with two sequential `.await`s on real GPU — both messages received and result correct.

## Findings
### Q: Can the proc macro recognize and transform `.await` into warp-cooperative states?
A: Yes. The implementation adds a `CfgNode::Await` variant that generates 2 states per `.await`:
1. **INIT**: Creates the inner future, stores it in `MaybeUninit<FutureType>` struct field
2. **POLL**: Calls `warp_poll_future(Pin::new_unchecked(future), &mut cx)` — lane 0 polls, broadcasts `Poll` result to all 32 lanes

**Confidence**: high (GPU-verified)

### Type Inference
The macro infers the future type from the base expression:
- `Type::method(args...)` → field type is `Type`
- `path::Type::method(args...)` → field type is `path::Type`
- `Type { fields... }` (struct literal) → `Type`
- Otherwise: compile error asking user to use constructor pattern

For the test: `gpu_runtime::std_future::GpuPrintFuture::new(buf, b"msg")` → type is `gpu_runtime::std_future::GpuPrintFuture`.

### Implementation Details

**New CfgNode variant**:
```rust
CfgNode::Await {
    base_expr: Expr,         // Expression being awaited
    result_var: Option<Ident>, // Variable to capture result
    future_type: Type,        // Concrete future type
    index: usize,            // Unique field index
}
```

**Generated struct fields**: `__await_0: core::mem::MaybeUninit<GpuPrintFuture>`

**No-op Waker**: Created once per `poll_warp()` call using `RawWakerVTable` with no-op functions, passed to all `warp_poll_future()` calls.

**Captures**: `.await` INIT state captures `buf` (first param) + extra params + known vars, since the await expression may reference any of them.

### Test Results
```
--- Warp .await test (warp-async-v2.3) ---
  Launching warp_await_test (32 threads)...
  Result: 1 (1=true, 0=false)
  Messages: ["[B0.T0] await: hello", "[B0.T0] await: done"]
  warp_await_test: PASSED!
    .await works in #[warp_async]: two sequential futures polled warp-cooperatively
```

### Supported patterns
- `expr.await;` — standalone await, no result capture
- `let ok = expr.await;` — await with result binding (stored as u64: 1=true, 0=false)
- Works inside if/else, loop, match (inherits from existing CfgNode handling)

## Open Questions
- `.await` result is stored as `u64` (1=true, 0=false). For `Future<Output=u64>` or other types, would need to extend. Currently only `Future<Output=bool>` is supported (matches `warp_poll_future` signature).

## Impact on Downstream Tasks
- **warp-async-v2.4**: Can now test combined `.await?`, branching, error handling (all building blocks available)
