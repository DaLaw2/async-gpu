// compile_fail: SharedRef cannot be used after block_scope exits.
//
// The `for<'scope>` pattern ties SharedRef's lifetime to the scope closure.
// Storing a SharedRef and using it after the closure returns should fail
// because the lifetime 'scope does not live long enough.

use gpu_runtime::scope::block_scope;
use gpu_runtime::tiered_mem::SharedRef;

fn main() {
    let mut stashed: Option<SharedRef<'_, f32>> = None;
    block_scope(|scope| {
        let buf = scope.alloc_shared::<f32>(64);
        stashed = Some(buf); //~ ERROR: lifetime may not live long enough
    });
    // If the above compiled, this would be use-after-free:
    // if let Some(ref r) = stashed {
    //     r.read(0);
    // }
}
