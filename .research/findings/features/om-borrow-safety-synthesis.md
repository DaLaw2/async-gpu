# om-borrow-safety — Feature Synthesis

**Status**: DONE

Compile-time safety and runtime correctness of SharedRef/GlobalRef are both verified. Three type-system mechanisms enforce safety: (1) `for<'scope>` HRTB prevents lifetime escape, (2) `!Send`/`!Sync` on SharedRef prevents cross-block sharing, (3) invariant lifetime prevents covariant widening.

Task .1: 8 compile-fail tests (6 negative, 2 positive) prove invalid patterns are rejected at compile time. Files in `crates/core/gpu-runtime/tests/compile_fail/`.

Task .2: 2 GPU kernel tests prove valid patterns work on hardware. SharedRef: alloc_shared, read/write, sub_ref tiling, helper function passing, f32 type. GlobalRef: alloc_global, read/write, sub_ref, cross-warp via raw ptr + SendPtrMut, u64 type. PTX contains `ld.shared`/`st.shared` and `ld.global.u32`/`st.global.u64` as expected. Both tests pass on GPU. Files in `crates/kernel/gpu-kernel-test/src/lib.rs` and `crates/test/gpu-test-harness/tests/gpu_tests.rs`.

Key finding: GlobalRef's Send+Sync does not allow direct capture in spawn_all across scope boundaries (HRTB blocks it). The correct pattern is raw pointer extraction via as_global_mut_ptr(), matching the sc_grid_reduce demo.
