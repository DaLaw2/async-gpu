# dyn-compat — Third-party no_std crate with dyn Trait on GPU

## Status: COMPLETE

hashbrown v0.15.5 (`#![no_std]`, uses `&dyn FnMut` and `&dyn Fn` internally
in its raw hash table) compiles to valid PTX on nvptx64 completely unmodified.
Added as `hashbrown = { version = "0.15", default-features = false }` — zero
patches, zero workarounds. The crate's `extern crate alloc` is satisfied by
patched std, and hashbrown's internal dyn dispatch code paths compile cleanly.

## Evidence Chain
- dyn-probe: `&dyn Trait` compiles to valid PTX with indirect calls
- dyn-box: `Box<dyn Trait>` compiles with heap allocation
- dyn-perf: overhead ~1.0-1.15x vs static dispatch
- **dyn-compat**: real third-party no_std crate works unmodified

## Conclusion
The dynamic dispatch story is complete: from basic `&dyn Trait` through
`Box<dyn>` to unmodified third-party crates. No blockers remain for
no_std ecosystem compatibility on GPU.
