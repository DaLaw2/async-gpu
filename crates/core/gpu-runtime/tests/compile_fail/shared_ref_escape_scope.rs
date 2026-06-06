// compile_fail: SharedRef<'scope, T> cannot escape block_scope closure.
//
// The `for<'scope>` higher-ranked trait bound on `block_scope` ensures that
// the lifetime `'scope` is strictly shorter than any outer lifetime. Storing
// a SharedRef in an outer variable would require `'scope: 'outer`, which the
// HRTB prevents.

use gpu_runtime::scope::block_scope;
use gpu_runtime::tiered_mem::SharedRef;

fn main() {
    let escaped: SharedRef<'_, f32>;
    block_scope(|scope| {
        let buf = scope.alloc_shared::<f32>(64);
        escaped = buf; //~ ERROR: lifetime may not live long enough
    });
}
