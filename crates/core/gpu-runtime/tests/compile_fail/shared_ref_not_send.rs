// compile_fail: SharedRef is !Send — cannot be sent across threads/warps.
//
// SharedRef<'scope, T> does not implement Send because shared memory is
// per-block. Attempting to send it to another thread should produce a
// compile error.

use gpu_runtime::tiered_mem::SharedRef;

fn assert_send<T: Send>() {}

fn main() {
    assert_send::<SharedRef<'_, f32>>(); //~ ERROR: `*mut f32` cannot be sent between threads safely
}
