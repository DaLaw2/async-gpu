# Dynamic Dispatch

Box<dyn Trait> polymorphism running on GPU hardware.

## What It Demonstrates

- `&dyn Trait` references with vtable-based method dispatch on GPU
- `Box<dyn Trait>` heap-allocated trait objects with dynamic dispatch
- `Vec<Box<dyn Trait>>` heterogeneous collections iterated on GPU
- Runtime-selected trait objects via conditionals
- Drop semantics for `Box<dyn Trait>` on GPU

The GPU kernel defines traits and structs, then dispatches method calls through
trait object vtables — identical to how dynamic dispatch works on CPU Rust.

## Running

```bash
cargo run -p dyn-dispatch --release
```

## Key Results

Dynamic dispatch via vtables works transparently on GPU. Trait objects
(`&dyn Trait` and `Box<dyn Trait>`) use indirect calls through compiler-generated
vtables, with heap allocation backed by the GPU hostcall allocator.
