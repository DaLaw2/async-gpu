// should_compile: GlobalRef IS Send+Sync (unlike SharedRef).
//
// GlobalRef<'scope, T> implements Send and Sync because global memory
// is accessible from all blocks. This is a positive test confirming
// the deliberate asymmetry between SharedRef and GlobalRef.

use gpu_runtime::tiered_mem::GlobalRef;

fn assert_send<T: Send>() {}
fn assert_sync<T: Sync>() {}

fn main() {
    assert_send::<GlobalRef<'_, f32>>();
    assert_sync::<GlobalRef<'_, f32>>();
    assert_send::<GlobalRef<'_, u32>>();
    assert_sync::<GlobalRef<'_, u32>>();
}
