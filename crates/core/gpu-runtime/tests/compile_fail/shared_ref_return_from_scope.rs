// compile_fail: SharedRef cannot be returned from block_scope closure.
//
// Attempting to return a SharedRef from the block_scope closure should fail
// because the lifetime 'scope does not outlive the closure return.

use gpu_runtime::scope::block_scope;
use gpu_runtime::tiered_mem::SharedRef;

fn main() {
    let _result = block_scope(|scope| {
        let buf = scope.alloc_shared::<f32>(64);
        buf //~ ERROR: lifetime may not live long enough
    });
}
