# dyn-probe synthesis

## Status: compilation confirmed, runtime test infrastructure ready

`&dyn Trait` compiles to valid PTX on nvptx64 and ptxas JIT-accepts the
indirect call instructions without error. The LLVM NVPTX backend emits
vtables in `.global` memory, indirect calls via register-based `call`
with `.callprototype`, and `cvta.global.u64` for address space conversion.

## Key findings
1. **PTX generation**: clean, no warnings (dyn-probe.1)
2. **ptxas JIT acceptance**: cuModuleLoadData succeeds on PTX with indirect
   calls — ptxas does not reject vtable lookups or indirect call patterns
3. **Host-side test**: added to gpu_tests.rs, compiles and passes lint
4. **Blocker**: no cubin containing test_gpu_dyn_trait exists; PTX JIT
   takes ~25 min for the 228K-line kernel_test.ptx

## Next steps
1. Rebuild cubin via `scripts/build-kernel-test.sh` to include dyn trait kernel
2. Re-run test with cubin (sub-second load) to get execution results
3. If execution passes: Box<dyn Trait>, &dyn Fn() closures, performance
4. If execution fails: document CUDA error, investigate workarounds

## Risk assessment
ptxas acceptance is a strong positive signal — most rejections happen at
JIT time. Remaining risk is GPU hardware execution of indirect calls
(unlikely to fail given ptxas success, but unverified until test completes).
