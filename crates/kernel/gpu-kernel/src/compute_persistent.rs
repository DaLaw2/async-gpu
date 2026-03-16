// Persistent kernel with mapped memory work queue.
//
// Work queue protocol:
// - Host writes work items to mapped memory with status READY
// - GPU block 0 polls for READY items, executes them, writes result, sets DONE
// - Host reads result when DONE, resets to FREE for reuse
//
// Work item layout (64 bytes):
//   [0..4]   status: u32 (0=FREE, 1=READY, 2=DONE, 3=SHUTDOWN)
//   [4..8]   fn_id: u32 (work function identifier)
//   [8..12]  n_args: u32
//   [12..44] args: [f32; 8]
//   [44..48] result: f32
//   [48..52] padding
//   [52..56] sequence: u32 (monotonic counter for ordering)
//   [56..64] reserved

use core::arch::nvptx;

const STATUS_FREE: u32 = 0;
const STATUS_READY: u32 = 1;
const STATUS_DONE: u32 = 2;
const STATUS_SHUTDOWN: u32 = 3;

const ITEM_SIZE: usize = 64; // bytes per work item
const STATUS_OFFSET: usize = 0;
const FN_ID_OFFSET: usize = 4;
const N_ARGS_OFFSET: usize = 8;
const ARGS_OFFSET: usize = 12;
const RESULT_OFFSET: usize = 44;
const SEQ_OFFSET: usize = 52;

/// Persistent kernel: polls work queue and executes work items.
///
/// The kernel runs until it sees a SHUTDOWN status in any slot.
/// Each block processes one work item at a time (block 0 only for simplicity).
///
/// Args: queue_ptr (mapped memory), n_slots (work queue depth), status_out (completion flag).
///
/// Grid: (1, 1, 1), Block: (1, 1, 1) — single thread for simplicity.
/// For production: use multiple blocks with per-block slot assignment.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn persistent_worker(
    queue: *mut u8,
    n_slots: u32,
    items_processed: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        if tid != 0 {
            return;
        }

        let mut count: u32 = 0;
        let mut slot: u32 = 0;
        let max_idle_spins: u32 = 100_000; // ~6.4ms with 64ns nanosleep

        'outer: loop {
            // Scan all slots for READY items
            let mut found = false;
            for _ in 0..n_slots {
                let item = queue.add((slot as usize) * ITEM_SIZE);
                let item_status: u32;
                core::arch::asm!(
                    "ld.acquire.sys.global.u32 {out}, [{addr}];",
                    out = out(reg32) item_status,
                    addr = in(reg64) item.add(STATUS_OFFSET),
                );

                if item_status == STATUS_SHUTDOWN {
                    break 'outer;
                }

                if item_status == STATUS_READY {
                    found = true;
                    // Read work item
                    let fn_id = *(item.add(FN_ID_OFFSET) as *const u32);
                    let n_args = *(item.add(N_ARGS_OFFSET) as *const u32);
                    let args = item.add(ARGS_OFFSET) as *const f32;

                    // Execute work function
                    let result = match fn_id {
                        0 => {
                            // NOP: return 0.0
                            0.0f32
                        }
                        1 => {
                            // ADD: args[0] + args[1]
                            *args.add(0) + *args.add(1)
                        }
                        2 => {
                            // MUL: args[0] * args[1]
                            *args.add(0) * *args.add(1)
                        }
                        3 => {
                            // DOT4: dot product of args[0..4] and args[4..8]
                            let mut sum = 0.0f32;
                            for i in 0..4 {
                                let a = *args.add(i);
                                let b = *args.add(4 + i);
                                core::arch::asm!(
                                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                                    d = out(reg32) sum,
                                    a = in(reg32) a,
                                    b = in(reg32) b,
                                    c = in(reg32) sum,
                                );
                            }
                            sum
                        }
                        4 => {
                            // SQRT: sqrt(args[0])
                            let val = *args.add(0);
                            let result: f32;
                            core::arch::asm!(
                                "sqrt.approx.f32 {out}, {in_};",
                                out = out(reg32) result,
                                in_ = in(reg32) val,
                            );
                            result
                        }
                        _ => {
                            // Unknown function — return NaN
                            f32::NAN
                        }
                    };

                    // Write result
                    *(item.add(RESULT_OFFSET) as *mut f32) = result;

                    // Signal completion (release-store)
                    core::arch::asm!(
                        "st.release.sys.global.u32 [{addr}], {val};",
                        addr = in(reg64) item.add(STATUS_OFFSET),
                        val = in(reg32) STATUS_DONE,
                    );

                    count += 1;
                }

                slot = (slot + 1) % n_slots;
            }

            // If no work found, nanosleep briefly then try again
            if !found {
                let mut idle_spins: u32 = 0;
                loop {
                    // Nanosleep 64ns
                    core::arch::asm!("nanosleep.u32 64;");
                    idle_spins += 1;

                    // Check first slot for any activity
                    let check: u32;
                    core::arch::asm!(
                        "ld.acquire.sys.global.u32 {out}, [{addr}];",
                        out = out(reg32) check,
                        addr = in(reg64) queue.add(STATUS_OFFSET),
                    );
                    if check == STATUS_READY || check == STATUS_SHUTDOWN {
                        break;
                    }
                    if idle_spins >= max_idle_spins {
                        // Check all slots one more time before giving up
                        break;
                    }
                }
            }
        }

        *items_processed = count;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (queue, n_slots, items_processed);
    }

    if tid == 0 {
        *status = 0;
    }
}
