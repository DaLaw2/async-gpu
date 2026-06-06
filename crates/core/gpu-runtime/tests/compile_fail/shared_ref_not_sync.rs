// compile_fail: SharedRef is !Sync — cannot be shared across threads/warps.
//
// SharedRef<'scope, T> does not implement Sync because shared memory is
// per-block. Attempting to share it across threads should fail.

use gpu_runtime::tiered_mem::SharedRef;

fn assert_sync<T: Sync>() {}

fn main() {
    assert_sync::<SharedRef<'_, f32>>(); //~ ERROR: `*mut f32` cannot be shared between threads safely
}
