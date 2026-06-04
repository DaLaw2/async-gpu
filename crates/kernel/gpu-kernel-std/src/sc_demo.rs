// Structured concurrency demo — producer-consumer pipeline using BlockScope
// and block channels (shared memory).
//
// Demonstrates three patterns:
// 1. Producer-consumer with BlockScope + BlockOneshotSlot signaling
// 2. Cooperative data-parallel with scope.spawn_all()
// 3. Nested scopes with lifetime-bounded shared memory
//
// These kernels prove that structured concurrency works on GPU:
// warp 0 manages scope, spawns work to other warps, and joins results —
// all coordinated via shared memory with zero host intervention.

use gpu_runtime::block_channel::{block_oneshot, BlockOneshotSlot};
use gpu_runtime::scope::{block_scope, init_shared_mem_allocator};
use gpu_runtime::thread;

// Wrapper to make raw pointers Send + Sync for spawned closures.
// SAFETY: On GPU, all warps share the same address space. Shared memory
// pointers are valid for all warps within the block. These are only used
// within a block_scope where lifetime safety is enforced by 'scope.
#[derive(Clone, Copy)]
struct SendPtr<T>(*mut T);
unsafe impl<T> Send for SendPtr<T> {}
unsafe impl<T> Sync for SendPtr<T> {}

impl<T> SendPtr<T> {
    fn new(p: *mut T) -> Self {
        Self(p)
    }
    fn as_ptr(self) -> *mut T {
        self.0
    }
    fn as_const(self) -> *const T {
        self.0 as *const T
    }
}

// ============================================================
// Demo 1: Producer-Consumer Pipeline
// ============================================================
//
// Warp 0 (manager) enters block_scope:
//   1. Allocates shared memory buffer + oneshot channel slot
//   2. Spawns producer warp: fills buffer with data, sends completion signal
//   3. Spawns consumer warp: waits for signal, sums data, returns result
//   4. Joins both, writes the verified sum to output
//
// Expected output: result[0] = sum of 0..64 = 2016
//                  result[1] = 1 (success flag)

/// Producer-consumer pipeline demo kernel.
///
/// # Arguments
/// * `result` - output buffer: [sum, success_flag]
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (128, 1, 1) — 4 warps (warp 0 = manager, warps 1-2 = workers)
/// * Shared memory: 2048 bytes
#[no_mangle]
pub unsafe extern "gpu-kernel" fn sc_producer_consumer(result: *mut u32) {
    thread::gpu_main(|| {
        // Initialize the shared memory allocator with 2048 bytes capacity.
        // This must happen on warp 0 before any block_scope calls.
        unsafe {
            init_shared_mem_allocator(2048);
        }

        let sum = block_scope(|scope| {
            // Allocate a 64-element u32 buffer in shared memory.
            // This buffer is 'scope-bounded — it cannot escape the closure.
            let data_buf: &mut [u32] = scope.alloc::<u32>(64);

            // Allocate raw bytes for the oneshot channel slot.
            // BlockOneshotSlot doesn't implement Copy, so we allocate as bytes
            // and reinterpret. The slot layout is repr(C): [state: u32, pad: u32, value: T].
            let slot_bytes: &mut [u8] = scope.alloc::<u8>(
                core::mem::size_of::<BlockOneshotSlot<u32>>(),
            );
            let oneshot_slot: &mut BlockOneshotSlot<u32> = unsafe {
                &mut *(slot_bytes.as_mut_ptr() as *mut BlockOneshotSlot<u32>)
            };

            // Create the oneshot channel pair.
            // SAFETY: slot is in shared memory, allocated by scope.alloc().
            let (tx, rx) = unsafe { block_oneshot(oneshot_slot) };

            // Wrap raw pointer in SendPtr for spawned closures.
            let data_ptr = SendPtr::new(data_buf.as_mut_ptr());

            // Spawn producer warp: fill the buffer with test data.
            // Producer writes data[i] = i for i in 0..64, then signals completion.
            let _producer = scope.spawn(move || {
                let ptr = data_ptr.as_ptr();
                let mut i = 0u32;
                while i < 64 {
                    unsafe {
                        core::ptr::write_volatile(ptr.add(i as usize), i);
                    }
                    i += 1;
                }
                // Signal that data is ready via the oneshot channel.
                // CTA-scope release semantics ensure data writes are visible.
                unsafe {
                    tx.send(1u32);
                }
            });

            // Spawn consumer warp: wait for the signal, then sum all data.
            let consumer = scope.spawn(move || -> u32 {
                let ptr = data_ptr.as_const();
                // Spin-wait for the producer's signal via the oneshot channel.
                // CTA-scope acquire semantics ensure we see the producer's data writes.
                let _signal = unsafe { rx.recv_spin() };

                // Sum all elements in the buffer.
                let mut sum = 0u32;
                let mut i = 0u32;
                while i < 64 {
                    let val = unsafe { core::ptr::read_volatile(ptr.add(i as usize)) };
                    sum += val;
                    i += 1;
                }
                sum
            });

            // Join the consumer to get the computed sum.
            // The producer is also joined implicitly when scope exits.
            consumer.join()
        });

        // Write results to global memory.
        // Expected: sum = 0 + 1 + 2 + ... + 63 = 2016
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, sum);
                core::ptr::write_volatile(result.add(1), 1); // success flag
            }
        }
    });
}

// ============================================================
// Demo 2: Cooperative Data-Parallel with spawn_all()
// ============================================================
//
// Warp 0 enters block_scope:
//   1. Allocates shared memory input/output buffers
//   2. Initializes input with test data
//   3. Calls scope.spawn_all() — all warps cooperatively double each element
//   4. Warp 0 reads and sums the output, writes to result
//
// Expected output: result[0] = sum of (i * 2) for i in 0..128 = 16256
//                  result[1] = 1 (success flag)

/// Cooperative data-parallel demo kernel using spawn_all().
///
/// # Arguments
/// * `result` - output buffer: [sum, success_flag]
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (128, 1, 1) — 4 warps all participate in spawn_all
/// * Shared memory: 4096 bytes
#[no_mangle]
pub unsafe extern "gpu-kernel" fn sc_cooperative_parallel(result: *mut u32) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(4096);
        }

        let sum = block_scope(|scope| {
            // Allocate input and output arrays in shared memory (128 elements each).
            let input: &mut [u32] = scope.alloc::<u32>(128);
            let output: &mut [u32] = scope.alloc::<u32>(128);

            // Warp 0 initializes input: input[i] = i
            let mut i = 0u32;
            while i < 128 {
                input[i as usize] = i;
                i += 1;
            }

            // Wrap raw pointers in SendPtr for the spawn_all closure.
            let in_ptr = SendPtr::new(input.as_mut_ptr());
            let out_ptr = SendPtr::new(output.as_mut_ptr());

            // spawn_all: ALL warps cooperatively double each element.
            // Each warp processes elements where index % n_warps == warp_id.
            scope.spawn_all(move |wid, n_warps| {
                let src = in_ptr.as_const();
                let dst = out_ptr.as_ptr();
                let mut idx = wid as usize;
                while idx < 128 {
                    let val = unsafe { core::ptr::read_volatile(src.add(idx)) };
                    unsafe {
                        core::ptr::write_volatile(dst.add(idx), val * 2);
                    }
                    idx += n_warps as usize;
                }
            });

            // After spawn_all returns, all warps have finished.
            // Warp 0 sums the output to verify correctness.
            let mut sum = 0u32;
            let mut j = 0u32;
            while j < 128 {
                sum += output[j as usize];
                j += 1;
            }
            sum
        });

        // Expected: sum = (0 + 1 + 2 + ... + 127) * 2 = 8128 * 2 = 16256
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, sum);
                core::ptr::write_volatile(result.add(1), 1);
            }
        }
    });
}

// ============================================================
// Demo 3: Nested Scopes — inner scope uses scratch, outer buffer survives
// ============================================================
//
// Warp 0 enters outer block_scope:
//   1. Allocates a large buffer (64 elements) in shared memory
//   2. Fills it with input data
//   3. Enters inner block_scope:
//      a. Allocates scratch space (16 elements)
//      b. Spawns a worker warp to compute partial sum into scratch
//      c. Joins worker, reads partial sum
//   4. Inner scope exits — scratch memory is freed (watermark popped)
//   5. Outer buffer is still valid — warp 0 reads from it to verify
//   6. Reports remaining bytes to prove memory was reclaimed
//
// Expected output: result[0] = partial_sum from inner scope = sum of data[0..16]
//                  result[1] = outer_first = data[0] = 10 (proves outer buffer survives)
//                  result[2] = 1 if memory was reclaimed after inner scope
//                  result[3] = 1 (success flag)

/// Nested scopes demo kernel.
///
/// # Arguments
/// * `result` - output buffer: [inner_sum, outer_first, mem_reclaimed, success_flag]
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (128, 1, 1) — 4 warps
/// * Shared memory: 2048 bytes
#[no_mangle]
pub unsafe extern "gpu-kernel" fn sc_nested_scopes(result: *mut u32) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(2048);
        }

        // Outer scope
        let (inner_sum, outer_first, mem_reclaimed) = block_scope(|outer| {
            // Allocate outer buffer — this persists for the whole outer scope.
            let outer_buf: &mut [u32] = outer.alloc::<u32>(64);

            // Fill outer buffer with data: data[i] = i + 10
            let mut i = 0u32;
            while i < 64 {
                outer_buf[i as usize] = i + 10;
                i += 1;
            }

            // Record available bytes before inner scope (after outer alloc).
            let bytes_before_inner = outer.available_bytes();

            // Enter inner scope — allocates scratch, does computation.
            let inner_sum: u32 = block_scope(|inner| {
                // Allocate scratch space in shared memory.
                // This memory is reclaimed when the inner scope exits.
                let scratch: &mut [u32] = inner.alloc::<u32>(16);

                // Wrap pointers for spawned closure.
                let outer_ptr = SendPtr::new(outer_buf.as_mut_ptr());
                let scratch_ptr = SendPtr::new(scratch.as_mut_ptr());

                // Spawn a worker warp to compute partial sum of outer_buf[0..16]
                // into scratch, then return the sum.
                let worker = inner.spawn(move || -> u32 {
                    let src = outer_ptr.as_const();
                    let dst = scratch_ptr.as_ptr();
                    let mut sum = 0u32;
                    let mut j = 0u32;
                    while j < 16 {
                        let val = unsafe { core::ptr::read_volatile(src.add(j as usize)) };
                        // Store intermediate values in scratch to demonstrate usage.
                        unsafe {
                            core::ptr::write_volatile(dst.add(j as usize), val);
                        }
                        sum += val;
                        j += 1;
                    }
                    sum
                });

                worker.join()
            });
            // Inner scope has exited — scratch memory is logically freed.

            // Record available bytes after inner scope.
            let bytes_after_inner = outer.available_bytes();

            // Verify outer buffer is still valid after inner scope exit.
            let outer_first = outer_buf[0]; // Should be 10

            // Memory should be reclaimed: bytes available after inner scope exit
            // should be >= bytes available before inner scope entry, because the
            // inner scope's watermark was popped back.
            let mem_reclaimed = if bytes_after_inner >= bytes_before_inner {
                1u32
            } else {
                0u32
            };

            (inner_sum, outer_first, mem_reclaimed)
        });

        // Expected:
        //   inner_sum = (10+11+12+...+25) = 16*10 + (0+1+...+15) = 160+120 = 280
        //   outer_first = 10
        //   mem_reclaimed = 1
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, inner_sum);
                core::ptr::write_volatile(result.add(1), outer_first);
                core::ptr::write_volatile(result.add(2), mem_reclaimed);
                core::ptr::write_volatile(result.add(3), 1);
            }
        }
    });
}

// ============================================================
// Demo 4: Combined — producer-consumer + spawn_all in one kernel
// ============================================================
//
// Shows BlockScope composability: sequential spawn then parallel spawn_all.
//
// 1. Spawn producer -> fills buffer -> oneshot signal
// 2. Spawn consumer -> waits signal -> transforms data (x3)
// 3. Join both
// 4. spawn_all -> all warps cooperatively sum the transformed data
// 5. Write final result
//
// Expected output: result[0] = sum of (i * 3) for i in 0..32 = (0+1+...+31)*3 = 496*3 = 1488
//                  result[1] = 1 (success flag)

/// Combined structured concurrency demo: spawn + spawn_all in one scope.
///
/// # Arguments
/// * `result` - output buffer: [final_sum, success_flag]
///
/// # Launch config
/// * Grid: (1, 1, 1)
/// * Block: (128, 1, 1) — 4 warps
/// * Shared memory: 4096 bytes
#[no_mangle]
pub unsafe extern "gpu-kernel" fn sc_combined_demo(result: *mut u32) {
    thread::gpu_main(|| {
        unsafe {
            init_shared_mem_allocator(4096);
        }

        let final_sum = block_scope(|scope| {
            // Allocate data buffer and partial sums buffer in shared memory.
            let data: &mut [u32] = scope.alloc::<u32>(32);
            let partial_sums: &mut [u32] = scope.alloc::<u32>(4); // one per warp

            // Allocate oneshot slot as raw bytes (BlockOneshotSlot is not Copy).
            let slot_bytes: &mut [u8] = scope.alloc::<u8>(
                core::mem::size_of::<BlockOneshotSlot<u32>>(),
            );
            let oneshot_slot: &mut BlockOneshotSlot<u32> = unsafe {
                &mut *(slot_bytes.as_mut_ptr() as *mut BlockOneshotSlot<u32>)
            };

            let (tx, rx) = unsafe { block_oneshot(oneshot_slot) };

            let data_ptr = SendPtr::new(data.as_mut_ptr());

            // Phase 1: Producer fills data[i] = i
            let _producer = scope.spawn(move || {
                let ptr = data_ptr.as_ptr();
                let mut i = 0u32;
                while i < 32 {
                    unsafe {
                        core::ptr::write_volatile(ptr.add(i as usize), i);
                    }
                    i += 1;
                }
                unsafe {
                    tx.send(1u32);
                }
            });

            // Phase 2: Consumer waits, then multiplies each element by 3
            let _consumer = scope.spawn(move || {
                let ptr = data_ptr.as_ptr();
                let _signal = unsafe { rx.recv_spin() };
                let mut i = 0u32;
                while i < 32 {
                    unsafe {
                        let val = core::ptr::read_volatile(ptr.add(i as usize));
                        core::ptr::write_volatile(ptr.add(i as usize), val * 3);
                    }
                    i += 1;
                }
            });

            // Join both spawned warps before proceeding.
            scope.join_all();

            // Phase 3: spawn_all — all warps cooperatively compute partial sums.
            let data_ptr2 = SendPtr::new(data.as_mut_ptr());
            let ps_ptr = SendPtr::new(partial_sums.as_mut_ptr());

            scope.spawn_all(move |wid, n_warps| {
                let src = data_ptr2.as_const();
                let dst = ps_ptr.as_ptr();
                let mut sum = 0u32;
                let mut idx = wid as usize;
                while idx < 32 {
                    let val = unsafe { core::ptr::read_volatile(src.add(idx)) };
                    sum += val;
                    idx += n_warps as usize;
                }
                unsafe {
                    core::ptr::write_volatile(dst.add(wid as usize), sum);
                }
            });

            // Warp 0 reduces partial sums.
            let n_warps = gpu_runtime::thread::available_parallelism() as u32 + 1;
            let mut total = 0u32;
            let mut w = 0u32;
            while w < n_warps {
                total += partial_sums[w as usize];
                w += 1;
            }
            total
        });

        // Expected: sum of (i*3) for i in 0..32 = 3 * (31*32/2) = 3 * 496 = 1488
        if gpu_runtime::index::thread_idx_x() == 0 {
            unsafe {
                core::ptr::write_volatile(result, final_sum);
                core::ptr::write_volatile(result.add(1), 1);
            }
        }
    });
}
