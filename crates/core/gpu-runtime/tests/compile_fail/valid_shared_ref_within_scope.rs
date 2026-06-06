// should_compile: SharedRef used correctly within its scope.
//
// Valid patterns:
// 1. Allocating and using SharedRef within block_scope
// 2. Reading and writing elements
// 3. Creating sub_ref for tiling

use gpu_runtime::scope::block_scope;
use gpu_runtime::tiered_mem::SharedRef;

fn use_shared_ref(r: &SharedRef<'_, f32>) -> f32 {
    r.read(0)
}

fn main() {
    block_scope(|scope| {
        // Pattern 1: allocate and use within scope
        let buf: SharedRef<'_, f32> = scope.alloc_shared::<f32>(256);
        buf.write(0, 42.0);
        let val = buf.read(0);
        assert_eq!(val, 42.0);

        // Pattern 2: pass to a function within the scope
        let v = use_shared_ref(&buf);
        assert_eq!(v, 42.0);

        // Pattern 3: sub_ref for tiling
        let tile = buf.sub_ref(0, 64);
        let v2 = tile.read(0);
        assert_eq!(v2, 42.0);
    });
}
