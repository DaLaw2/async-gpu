// compile_fail: SharedRef cannot be stored in a struct that holds GlobalRef.
//
// A cross-tier container (e.g., a struct containing both GlobalRef and
// SharedRef) should fail because SharedRef's lifetime and address-space
// constraints are incompatible with global-scope storage.
//
// Specifically, GlobalRef is Send+Sync but SharedRef is !Send+!Sync,
// so a container holding SharedRef cannot be shared across blocks.

use gpu_runtime::tiered_mem::{GpuRef, Global, Shared, SharedRef};

// A user might try to create a "cross-tier" container
struct CrossTierContainer<'a> {
    shared: SharedRef<'a, f32>,
}

fn assert_send<T: Send>() {}

fn main() {
    // Trying to send a container with SharedRef across blocks
    assert_send::<CrossTierContainer<'_>>(); //~ ERROR: `*mut f32` cannot be sent between threads safely
}
