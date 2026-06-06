# om-borrow-safety — Feature Synthesis

**Status**: DONE

The type system catches all tested cross-scope memory hierarchy violations at compile time. Three independent mechanisms enforce safety: (1) `for<'scope>` HRTB prevents lifetime escape from scope closures, (2) `!Send`/`!Sync` on `SharedRef` prevents cross-block sharing, (3) invariant lifetime (`PhantomData<&'scope mut &'scope ()>`) prevents covariant widening.

8 compile-fail tests (6 negative, 2 positive) verify these guarantees. All pass. Test files live in `crates/core/gpu-runtime/tests/compile_fail/`, runner script at `tests/compile_fail_runner.sh`.

Key finding: no additional runtime checks or new types are needed. The existing `GpuRef` design already provides complete compile-time safety for the shared/global memory hierarchy.
