# warp-async-v2.1: Design .await recognition in proc macro
**Cycle**: 236 | **Theme**: warp-async-v2 | **Kind**: design | **Status**: done

## Summary
Designed the transformation from `.await` expressions to warp-cooperative state machine states in `#[warp_async]`. Each `.await` becomes a single AWAIT state that calls `warp_poll_future()`, with the inner future stored as a struct field. Combined `.await?` is also supported via a follow-up DECISION state.

## Findings
### Q: How should the proc macro recognize and transform `.await` expressions?
A: syn parses `.await` as `Expr::Await(ExprAwait { base, .. })`. The macro walker already recursively processes expressions — adding a case for `Expr::Await` is straightforward.

**Confidence**: high

### Transformation Design

**Input:**
```rust
#[warp_async]
unsafe fn example(buf: *mut u8) -> bool {
    let ok = GpuPrintFuture::new(buf, b"Hello!").await;
    ok
}
```

**Generated struct fields:**
```rust
struct Example {
    buf: *mut u8,
    state: u32,
    // Inner future stored as MaybeUninit<ConcreteType>
    __await_0: core::mem::MaybeUninit<GpuPrintFuture>,
    // Captured result
    ok: bool,
}
```

**Generated state machine:**
```rust
// State 0 (AWAIT_INIT): Create inner future, store in struct
0 => {
    self.__await_0.write(GpuPrintFuture::new(self.buf, b"Hello!"));
    if wcx.is_leader() { self.state = 1; }
    return WarpPoll::Pending;
}
// State 1 (AWAIT_POLL): Warp-cooperative poll
1 => {
    let future = unsafe { self.__await_0.assume_init_mut() };
    let result = unsafe { warp_poll_future(
        Pin::new_unchecked(future), &mut __cx
    ) };
    match result {
        Poll::Ready(val) => {
            if wcx.is_leader() { self.ok = val; self.state = 2; }
            return WarpPoll::Pending; // advance state
        }
        Poll::Pending => return WarpPoll::Pending,
    }
}
// State 2 (DONE): Return captured value
2 => { return WarpPoll::Ready(self.ok); }
```

### CFG Node Extension

Add a new `CfgNode::Await` variant:
```rust
CfgNode::Await {
    base_expr: Expr,         // The expression being awaited
    result_var: Option<Ident>, // Variable to capture the result
    future_type: Type,        // Type of the inner future (for struct field)
    index: usize,            // Unique index for naming the struct field
}
```

This generates 2 states: INIT (create future) + POLL (warp-cooperative poll).

### .await? Support

Combined `.await?` means: await the future, then propagate error.

**Parsed as**: `Expr::Try(ExprTry { expr: Expr::Await(ExprAwait { base }) })`

**Generated states**: AWAIT_INIT → AWAIT_POLL → ERROR_DECISION
- AWAIT_POLL captures `Result<T, E>` value
- ERROR_DECISION broadcasts Ok/Err discriminant via `broadcast_u32`
- If Ok: continue with unwrapped value
- If Err: return `WarpPoll::Ready(Err(e))` — all lanes early-return

### Type Inference Challenge

The proc macro needs to know the concrete type of each inner future to create struct fields. Options:
1. **Explicit type annotation**: `let ok: bool = GpuPrintFuture::new(buf, b"Hello!").await;` — macro extracts type from the expression
2. **Named type inference**: If the awaited expression is a constructor call (`Type::new(...)`), extract the type name
3. **Turbofish**: `let ok = expr.await::<GpuPrintFuture>` — not standard Rust syntax
4. **MaybeUninit<[u8; N]>** with a fixed size — fragile

**Decision**: Option 2 with fallback to option 1. The macro analyzes the base expression:
- If it's `Type::method(...)` → field type is `Type`
- If it's `path::Type::method(...)` → field type is `path::Type`
- Otherwise: require explicit `#[warp_await_type(ConcreteType)]` annotation

### No-op Waker

The generated `poll_warp()` creates a no-op Waker at the top (same pattern as `warp_cooperative::warp_run_future()`):
```rust
const VTABLE: RawWakerVTable = RawWakerVTable::new(|_| ..., |_| {}, |_| {}, |_| {});
let raw = RawWaker::new(null(), &VTABLE);
let waker = Waker::from_raw(raw);
let mut __cx = Context::from_waker(&waker);
```

This is created once per `poll_warp()` call and passed to all inner future polls.

## Open Questions
- Should the macro accept `async fn` keyword and strip it, or require regular `fn`?
  - Proposal: Accept `async fn`, strip the keyword, generate WarpFuture impl
  - This gives users the visual cue that `.await` is available

## Impact on Downstream Tasks
- **warp-async-v2.3**: Implementation can follow this design directly
- **warp-async-v2.2**: ? operator implementation is independent (works with existing `warp_*!()` calls)
