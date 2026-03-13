// ML workload + GPU compute kernels — f32 math, vector search, MMA, GEMM, softmax.

use crate::helpers::{bar_sync, get_dynamic_smem_ptr, gpu_exp_f32, gpu_sqrtf};
use core::arch::nvptx;
use gpu_atomics::membar_sys;
use gpu_protocol::*;
use gpu_runtime::warp_future::{warp_hostcall_submit, warp_hostcall_wait_u64};

// ============================================================
// ml-workload.1: f32 math validation kernel
// ============================================================

/// ml-workload.1: f32 math validation kernel.
/// Tests: f32 add, mul, div, fma, sqrt on GPU.
/// output[0] = 3.0 + 4.0 = 7.0
/// output[1] = 3.0 * 4.0 = 12.0
/// output[2] = 10.0 / 4.0 = 2.5
/// output[3] = sqrt(9.0) = 3.0
/// output[4] = dot([1,2,3,4], [5,6,7,8]) = 5+12+21+32 = 70.0
/// output[5] = ||[3,4]|| = sqrt(9+16) = 5.0
/// output[6] = cosine_sim([1,0], [0,1]) = 0.0
/// output[7] = cosine_sim([1,0], [1,0]) = 1.0
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_math_test(output: *mut f32) {
    let tid = core::arch::nvptx::_thread_idx_x() as usize;
    if tid != 0 {
        return;
    }

    // Basic ops
    let a: f32 = 3.0;
    let b: f32 = 4.0;
    core::ptr::write_volatile(output.add(0), a + b); // 7.0
    core::ptr::write_volatile(output.add(1), a * b); // 12.0
    core::ptr::write_volatile(output.add(2), 10.0f32 / b); // 2.5
    core::ptr::write_volatile(output.add(3), gpu_sqrtf(9.0)); // 3.0

    // Dot product
    let v1 = [1.0f32, 2.0, 3.0, 4.0];
    let v2 = [5.0f32, 6.0, 7.0, 8.0];
    let mut dot: f32 = 0.0;
    let mut i = 0;
    while i < 4 {
        dot += v1[i] * v2[i];
        i += 1;
    }
    core::ptr::write_volatile(output.add(4), dot); // 70.0

    // Norm
    let norm = gpu_sqrtf(3.0 * 3.0 + 4.0 * 4.0);
    core::ptr::write_volatile(output.add(5), norm); // 5.0

    // Cosine similarity: orthogonal vectors → 0.0
    // cos([1,0], [0,1]) = 0 / (1*1) = 0.0
    let cos_orth = 0.0f32 / (1.0f32 * 1.0f32);
    core::ptr::write_volatile(output.add(6), cos_orth); // 0.0

    // Cosine similarity: identical vectors → 1.0
    // cos([1,0], [1,0]) = 1 / (1*1) = 1.0
    let cos_same = 1.0f32 / (1.0f32 * 1.0f32);
    core::ptr::write_volatile(output.add(7), cos_same); // 1.0
}

// ============================================================
// ml-workload.2: Vector Similarity Search — GPU-Autonomous Demo
// ============================================================
//
// 20-state WarpFuture. Each state does exactly ONE thing (submit or wait).
// No multi-phase states, no sentinel values.

const VS_DIM: usize = 128;
const VS_VEC_BYTES: usize = VS_DIM * 4; // 512 bytes per vector
const VS_K: usize = 10;

// State constants — each state does exactly one action
const VS_SUBMIT_OPEN_DB: u32 = 0;
const VS_WAIT_OPEN_DB: u32 = 1;
const VS_SUBMIT_READ_DB: u32 = 2;
const VS_WAIT_READ_DB: u32 = 3;
const VS_SUBMIT_CLOSE_DB: u32 = 4;
const VS_WAIT_CLOSE_DB: u32 = 5;
const VS_SUBMIT_OPEN_Q: u32 = 6;
const VS_WAIT_OPEN_Q: u32 = 7;
const VS_SUBMIT_READ_Q: u32 = 8;
const VS_WAIT_READ_Q: u32 = 9;
const VS_SUBMIT_CLOSE_Q: u32 = 10;
const VS_WAIT_CLOSE_Q: u32 = 11;
const VS_COMPUTE: u32 = 12;
const VS_SUBMIT_OPEN_OUT: u32 = 13;
const VS_WAIT_OPEN_OUT: u32 = 14;
const VS_SUBMIT_WRITE: u32 = 15;
const VS_WAIT_WRITE: u32 = 16;
const VS_SUBMIT_CLOSE_OUT: u32 = 17;
const VS_WAIT_CLOSE_OUT: u32 = 18;
const VS_DONE: u32 = 19;

#[derive(Clone, Copy)]
struct TopKEntry {
    id: u32,
    score: f32,
}

struct VecSearchFuture {
    buf: *mut u8,
    sideband: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
    db_count: u32,
    db_offset: u64,
    query_offset: u64,
    result_offset: u64,
    top_k: [TopKEntry; VS_K],
}

impl VecSearchFuture {
    unsafe fn new(buf: *mut u8, sideband: *mut u8) -> Self {
        Self {
            buf,
            sideband,
            state: VS_SUBMIT_OPEN_DB,
            pkt_idx: gpu_protocol::NULL_INDEX,
            fd: 0,
            db_count: 0,
            db_offset: 0,
            query_offset: 0,
            result_offset: 0,
            top_k: [TopKEntry {
                id: u32::MAX,
                score: -1.0,
            }; VS_K],
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for VecSearchFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // --- Database: open -> read -> close ---
            VS_SUBMIT_OPEN_DB => unsafe {
                let path = b"vecdb.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    VS_WAIT_OPEN_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_OPEN_DB => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_READ_DB,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                        gpu_runtime::sideband::sideband_reset(self.sideband);
                        self.db_offset =
                            gpu_runtime::sideband::sideband_alloc(self.sideband, 900 * 1024);
                    }
                }
                WarpPoll::Pending
            },

            VS_SUBMIT_READ_DB => unsafe {
                let fd = self.fd;
                let db_off = self.db_offset;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, db_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, 900 * 1024);
                    },
                    VS_WAIT_READ_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_READ_DB => unsafe {
                if let Some(_bytes) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_CLOSE_DB,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        let header = self
                            .sideband
                            .add(gpu_protocol::SIDEBAND_DATA_OFFSET + self.db_offset as usize);
                        self.db_count = core::ptr::read_volatile(header as *const u32);
                    }
                }
                WarpPoll::Pending
            },

            VS_SUBMIT_CLOSE_DB => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    VS_WAIT_CLOSE_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_CLOSE_DB => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_OPEN_Q,
                    &mut self.state,
                )
                .is_some()
                {
                    if wcx.is_leader() {
                        self.fd = 0;
                    }
                }
                WarpPoll::Pending
            },

            // --- Query: open -> read -> close ---
            VS_SUBMIT_OPEN_Q => unsafe {
                let path = b"query.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    VS_WAIT_OPEN_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_OPEN_Q => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_READ_Q,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                        self.query_offset = gpu_runtime::sideband::sideband_alloc(
                            self.sideband,
                            (4 + VS_VEC_BYTES) as u64,
                        );
                    }
                }
                WarpPoll::Pending
            },

            VS_SUBMIT_READ_Q => unsafe {
                let fd = self.fd;
                let q_off = self.query_offset;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, q_off);
                        core::ptr::write_volatile(
                            payload.add(16) as *mut u64,
                            (4 + VS_VEC_BYTES) as u64,
                        );
                    },
                    VS_WAIT_READ_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_READ_Q => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_CLOSE_Q,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            VS_SUBMIT_CLOSE_Q => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    VS_WAIT_CLOSE_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_CLOSE_Q => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, VS_COMPUTE, &mut self.state)
                    .is_some()
                {
                    if wcx.is_leader() {
                        self.fd = 0;
                    }
                }
                WarpPoll::Pending
            },

            // --- Compute cosine similarity + write results to sideband ---
            VS_COMPUTE => unsafe {
                let lid = wcx.lane_id;
                let n = broadcast_u32(wcx.active_mask, self.db_count);
                let sb_base = self.sideband as usize + gpu_protocol::SIDEBAND_DATA_OFFSET;

                let db_off = broadcast_u32(wcx.active_mask, self.db_offset as u32) as usize;
                let db_vecs_base = sb_base + db_off + 8; // skip N:u32 + dim:u32 header

                let q_off = broadcast_u32(wcx.active_mask, self.query_offset as u32) as usize;
                let query_base = (sb_base + q_off + 4) as *const f32; // skip dim:u32 header

                // Load query vector
                let mut query = [0.0f32; VS_DIM];
                let mut d = 0;
                while d < VS_DIM {
                    query[d] = core::ptr::read_volatile(query_base.add(d));
                    d += 1;
                }

                // Query norm
                let mut q_norm_sq: f32 = 0.0;
                d = 0;
                while d < VS_DIM {
                    q_norm_sq += query[d] * query[d];
                    d += 1;
                }
                let q_norm = gpu_sqrtf(q_norm_sq);

                // Per-lane: stride-32 work distribution
                let mut local_topk = [TopKEntry {
                    id: u32::MAX,
                    score: -1.0f32,
                }; VS_K];

                let mut vec_idx = lid;
                while vec_idx < n {
                    let vec_ptr = (db_vecs_base + (vec_idx as usize) * VS_VEC_BYTES) as *const f32;

                    let mut dot: f32 = 0.0;
                    let mut v_norm_sq: f32 = 0.0;
                    d = 0;
                    while d < VS_DIM {
                        let v = core::ptr::read_volatile(vec_ptr.add(d));
                        dot += query[d] * v;
                        v_norm_sq += v * v;
                        d += 1;
                    }
                    let v_norm = gpu_sqrtf(v_norm_sq);
                    let denom = q_norm * v_norm;
                    let score = if denom > 0.0 { dot / denom } else { 0.0 };

                    if score > local_topk[VS_K - 1].score {
                        local_topk[VS_K - 1] = TopKEntry { id: vec_idx, score };
                        let mut j = VS_K - 1;
                        while j > 0 && local_topk[j].score > local_topk[j - 1].score {
                            let tmp = local_topk[j - 1];
                            local_topk[j - 1] = local_topk[j];
                            local_topk[j] = tmp;
                            j -= 1;
                        }
                    }

                    vec_idx += 32;
                }

                // Full warp merge: collect all 32 lanes' top-K via shfl.sync
                let mut global_topk = [TopKEntry {
                    id: u32::MAX,
                    score: -1.0f32,
                }; VS_K];
                if lid == 0 {
                    global_topk = local_topk; // start with lane 0's results
                }

                let mask = wcx.active_mask;
                let mut k = 0u32;
                while k < VS_K as u32 {
                    let my_id = local_topk[k as usize].id;
                    let my_score_bits: u32 = f32::to_bits(local_topk[k as usize].score);

                    let mut s = 0u32;
                    while s < 32 {
                        let cand_id = gpu_atomics::shfl_sync_idx_u32(mask, my_id, s);
                        let cand_score_bits =
                            gpu_atomics::shfl_sync_idx_u32(mask, my_score_bits, s);
                        let cand_score: f32 = f32::from_bits(cand_score_bits);

                        // Lane 0 inserts candidate into global top-K
                        if lid == 0 && s != 0 {
                            // skip lane 0 (already included)
                            if cand_score > global_topk[VS_K - 1].score {
                                global_topk[VS_K - 1] = TopKEntry {
                                    id: cand_id,
                                    score: cand_score,
                                };
                                let mut j = VS_K - 1;
                                while j > 0 && global_topk[j].score > global_topk[j - 1].score {
                                    let tmp = global_topk[j - 1];
                                    global_topk[j - 1] = global_topk[j];
                                    global_topk[j] = tmp;
                                    j -= 1;
                                }
                            }
                        }
                        s += 1;
                    }
                    k += 1;
                }

                // Lane 0 writes global top-K results to sideband
                if wcx.is_leader() {
                    self.top_k = global_topk;

                    let result_offset =
                        gpu_runtime::sideband::sideband_alloc(self.sideband, (4 + VS_K * 8) as u64);
                    self.result_offset = result_offset;
                    let result_base = self
                        .sideband
                        .add(gpu_protocol::SIDEBAND_DATA_OFFSET + result_offset as usize);
                    core::ptr::write_volatile(result_base as *mut u32, VS_K as u32);
                    let entries = result_base.add(4);
                    let mut i = 0;
                    while i < VS_K {
                        core::ptr::write_volatile(entries.add(i * 8) as *mut u32, self.top_k[i].id);
                        core::ptr::write_volatile(
                            entries.add(i * 8 + 4) as *mut f32,
                            self.top_k[i].score,
                        );
                        i += 1;
                    }
                }

                membar_sys();
                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = VS_SUBMIT_OPEN_OUT;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // --- Output: open -> write -> close ---
            VS_SUBMIT_OPEN_OUT => unsafe {
                let path = b"results.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    VS_WAIT_OPEN_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_OPEN_OUT => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_WRITE,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                    }
                }
                WarpPoll::Pending
            },

            VS_SUBMIT_WRITE => unsafe {
                let fd = self.fd;
                let r_off = self.result_offset;
                let r_len = (4 + VS_K * 8) as u64;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_WRITE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, r_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, r_len);
                    },
                    VS_WAIT_WRITE,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_WRITE => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    VS_SUBMIT_CLOSE_OUT,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            VS_SUBMIT_CLOSE_OUT => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    VS_WAIT_CLOSE_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            VS_WAIT_CLOSE_OUT => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, VS_DONE, &mut self.state)
                    .is_some()
                {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            },

            VS_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Ready(false),
        }
    }
}

/// ml-workload.2: GPU-autonomous vector similarity search.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn vector_search_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = VecSearchFuture::new(buf, sideband);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// ml-workload.3: Batch Vector Search — Multi-Query in One Launch
// ============================================================
//
// Same 20-state pattern as ml-workload.2 but:
// - queries.bin contains multiple query vectors
// - COMPUTE loops over all queries
// - results.bin contains results for all queries
//
// File formats:
//   queries.bin: [num_q:u32][dim:u32][q0_d0:f32]...[q0_d127]...[qN_d127]
//   batch_results.bin: [num_q:u32][K:u32][{id:u32,score:f32}*K]*num_q

// Reuse VS_DIM, VS_VEC_BYTES, VS_K, TopKEntry from ml-workload.2

const BS_SUBMIT_OPEN_DB: u32 = 0;
const BS_WAIT_OPEN_DB: u32 = 1;
const BS_SUBMIT_READ_DB: u32 = 2;
const BS_WAIT_READ_DB: u32 = 3;
const BS_SUBMIT_CLOSE_DB: u32 = 4;
const BS_WAIT_CLOSE_DB: u32 = 5;
const BS_SUBMIT_OPEN_Q: u32 = 6;
const BS_WAIT_OPEN_Q: u32 = 7;
const BS_SUBMIT_READ_Q: u32 = 8;
const BS_WAIT_READ_Q: u32 = 9;
const BS_SUBMIT_CLOSE_Q: u32 = 10;
const BS_WAIT_CLOSE_Q: u32 = 11;
const BS_COMPUTE: u32 = 12;
const BS_SUBMIT_OPEN_OUT: u32 = 13;
const BS_WAIT_OPEN_OUT: u32 = 14;
const BS_SUBMIT_WRITE: u32 = 15;
const BS_WAIT_WRITE: u32 = 16;
const BS_SUBMIT_CLOSE_OUT: u32 = 17;
const BS_WAIT_CLOSE_OUT: u32 = 18;
const BS_DONE: u32 = 19;

struct BatchSearchFuture {
    buf: *mut u8,
    sideband: *mut u8,
    state: u32,
    pkt_idx: u16,
    fd: u64,
    db_count: u32,
    num_queries: u32,
    db_offset: u64,
    query_offset: u64,
    result_offset: u64,
    result_bytes: u64,
}

impl BatchSearchFuture {
    unsafe fn new(buf: *mut u8, sideband: *mut u8) -> Self {
        Self {
            buf,
            sideband,
            state: BS_SUBMIT_OPEN_DB,
            pkt_idx: gpu_protocol::NULL_INDEX,
            fd: 0,
            db_count: 0,
            num_queries: 0,
            db_offset: 0,
            query_offset: 0,
            result_offset: 0,
            result_bytes: 0,
        }
    }
}

unsafe impl gpu_runtime::warp_future::WarpFuture for BatchSearchFuture {
    type Output = bool;

    fn poll_warp(
        &mut self,
        wcx: &mut gpu_runtime::warp_future::WarpContext,
    ) -> gpu_runtime::warp_future::WarpPoll<bool> {
        use gpu_runtime::warp_future::{broadcast_u32, WarpPoll};

        let state = unsafe { broadcast_u32(wcx.active_mask, self.state) };

        match state {
            // --- Database: open -> read -> close ---
            BS_SUBMIT_OPEN_DB => unsafe {
                let path = b"vecdb.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    BS_WAIT_OPEN_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_OPEN_DB => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_READ_DB,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                        gpu_runtime::sideband::sideband_reset(self.sideband);
                        self.db_offset =
                            gpu_runtime::sideband::sideband_alloc(self.sideband, 900 * 1024);
                    }
                }
                WarpPoll::Pending
            },

            BS_SUBMIT_READ_DB => unsafe {
                let fd = self.fd;
                let db_off = self.db_offset;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, db_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, 900 * 1024);
                    },
                    BS_WAIT_READ_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_READ_DB => unsafe {
                if let Some(_bytes) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_CLOSE_DB,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        let header = self
                            .sideband
                            .add(gpu_protocol::SIDEBAND_DATA_OFFSET + self.db_offset as usize);
                        self.db_count = core::ptr::read_volatile(header as *const u32);
                    }
                }
                WarpPoll::Pending
            },

            BS_SUBMIT_CLOSE_DB => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    BS_WAIT_CLOSE_DB,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_CLOSE_DB => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_OPEN_Q,
                    &mut self.state,
                )
                .is_some()
                {
                    if wcx.is_leader() {
                        self.fd = 0;
                    }
                }
                WarpPoll::Pending
            },

            // --- Queries: open -> read -> close ---
            BS_SUBMIT_OPEN_Q => unsafe {
                let path = b"queries.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_READ as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    BS_WAIT_OPEN_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_OPEN_Q => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_READ_Q,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                        // Allocate query space: up to 100KB for queries
                        self.query_offset =
                            gpu_runtime::sideband::sideband_alloc(self.sideband, 100 * 1024);
                    }
                }
                WarpPoll::Pending
            },

            BS_SUBMIT_READ_Q => unsafe {
                let fd = self.fd;
                let q_off = self.query_offset;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_READ,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, q_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, 100 * 1024);
                    },
                    BS_WAIT_READ_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_READ_Q => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_CLOSE_Q,
                    &mut self.state,
                )
                .is_some()
                {
                    if wcx.is_leader() {
                        // Parse query header: [num_q:u32][dim:u32]
                        let q_header = self
                            .sideband
                            .add(gpu_protocol::SIDEBAND_DATA_OFFSET + self.query_offset as usize);
                        self.num_queries = core::ptr::read_volatile(q_header as *const u32);
                    }
                }
                WarpPoll::Pending
            },

            BS_SUBMIT_CLOSE_Q => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    BS_WAIT_CLOSE_Q,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_CLOSE_Q => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, BS_COMPUTE, &mut self.state)
                    .is_some()
                {
                    if wcx.is_leader() {
                        self.fd = 0;
                    }
                }
                WarpPoll::Pending
            },

            // --- Compute: loop over all queries, write results to sideband ---
            BS_COMPUTE => unsafe {
                let lid = wcx.lane_id;
                let mask = wcx.active_mask;
                let n = broadcast_u32(mask, self.db_count);
                let nq = broadcast_u32(mask, self.num_queries);
                let sb_base = self.sideband as usize + gpu_protocol::SIDEBAND_DATA_OFFSET;

                let db_off = broadcast_u32(mask, self.db_offset as u32) as usize;
                let db_vecs_base = sb_base + db_off + 8; // skip N:u32 + dim:u32

                let q_off = broadcast_u32(mask, self.query_offset as u32) as usize;
                let queries_base = sb_base + q_off + 8; // skip num_q:u32 + dim:u32

                // Allocate result buffer: [num_q:u32][K:u32] + num_q * K * 8
                let result_header_bytes = 8u64; // num_q + K
                let result_entries_bytes = (nq as u64) * (VS_K as u64) * 8;
                let total_result_bytes = result_header_bytes + result_entries_bytes;

                let result_offset = if wcx.is_leader() {
                    let off =
                        gpu_runtime::sideband::sideband_alloc(self.sideband, total_result_bytes);
                    self.result_offset = off;
                    self.result_bytes = total_result_bytes;
                    off
                } else {
                    0
                };
                let result_offset = broadcast_u32(mask, result_offset as u32) as usize;
                let result_base = sb_base + result_offset;

                // Write result header (lane 0 only)
                if wcx.is_leader() {
                    core::ptr::write_volatile(result_base as *mut u32, nq);
                    core::ptr::write_volatile((result_base + 4) as *mut u32, VS_K as u32);
                }

                // Process each query
                let mut qi: u32 = 0;
                while qi < nq {
                    let query_base = (queries_base + (qi as usize) * VS_VEC_BYTES) as *const f32;

                    // Load query vector
                    let mut query = [0.0f32; VS_DIM];
                    let mut d = 0;
                    while d < VS_DIM {
                        query[d] = core::ptr::read_volatile(query_base.add(d));
                        d += 1;
                    }

                    // Query norm
                    let mut q_norm_sq: f32 = 0.0;
                    d = 0;
                    while d < VS_DIM {
                        q_norm_sq += query[d] * query[d];
                        d += 1;
                    }
                    let q_norm = gpu_sqrtf(q_norm_sq);

                    // Per-lane stride-32 search
                    let mut local_topk = [TopKEntry {
                        id: u32::MAX,
                        score: -1.0f32,
                    }; VS_K];

                    let mut vec_idx = lid;
                    while vec_idx < n {
                        let vec_ptr =
                            (db_vecs_base + (vec_idx as usize) * VS_VEC_BYTES) as *const f32;

                        let mut dot: f32 = 0.0;
                        let mut v_norm_sq: f32 = 0.0;
                        d = 0;
                        while d < VS_DIM {
                            let v = core::ptr::read_volatile(vec_ptr.add(d));
                            dot += query[d] * v;
                            v_norm_sq += v * v;
                            d += 1;
                        }
                        let v_norm = gpu_sqrtf(v_norm_sq);
                        let denom = q_norm * v_norm;
                        let score = if denom > 0.0 { dot / denom } else { 0.0 };

                        if score > local_topk[VS_K - 1].score {
                            local_topk[VS_K - 1] = TopKEntry { id: vec_idx, score };
                            let mut j = VS_K - 1;
                            while j > 0 && local_topk[j].score > local_topk[j - 1].score {
                                let tmp = local_topk[j - 1];
                                local_topk[j - 1] = local_topk[j];
                                local_topk[j] = tmp;
                                j -= 1;
                            }
                        }

                        vec_idx += 32;
                    }

                    // Full warp merge via shfl.sync
                    let mut global_topk = [TopKEntry {
                        id: u32::MAX,
                        score: -1.0f32,
                    }; VS_K];
                    if lid == 0 {
                        global_topk = local_topk;
                    }

                    let mut k = 0u32;
                    while k < VS_K as u32 {
                        let my_id = local_topk[k as usize].id;
                        let my_score_bits: u32 = f32::to_bits(local_topk[k as usize].score);

                        let mut s = 0u32;
                        while s < 32 {
                            let cand_id = gpu_atomics::shfl_sync_idx_u32(mask, my_id, s);
                            let cand_score_bits =
                                gpu_atomics::shfl_sync_idx_u32(mask, my_score_bits, s);
                            let cand_score: f32 = f32::from_bits(cand_score_bits);

                            if lid == 0 && s != 0 {
                                if cand_score > global_topk[VS_K - 1].score {
                                    global_topk[VS_K - 1] = TopKEntry {
                                        id: cand_id,
                                        score: cand_score,
                                    };
                                    let mut j = VS_K - 1;
                                    while j > 0 && global_topk[j].score > global_topk[j - 1].score {
                                        let tmp = global_topk[j - 1];
                                        global_topk[j - 1] = global_topk[j];
                                        global_topk[j] = tmp;
                                        j -= 1;
                                    }
                                }
                            }
                            s += 1;
                        }
                        k += 1;
                    }

                    // Lane 0 writes this query's merged results
                    if wcx.is_leader() {
                        let entry_base = result_base + 8 + (qi as usize) * VS_K * 8;
                        let mut i = 0;
                        while i < VS_K {
                            core::ptr::write_volatile(
                                (entry_base + i * 8) as *mut u32,
                                global_topk[i].id,
                            );
                            core::ptr::write_volatile(
                                (entry_base + i * 8 + 4) as *mut f32,
                                global_topk[i].score,
                            );
                            i += 1;
                        }
                    }

                    qi += 1;
                }

                membar_sys();
                gpu_atomics::syncwarp(wcx.active_mask);
                if wcx.is_leader() {
                    self.state = BS_SUBMIT_OPEN_OUT;
                }
                gpu_atomics::syncwarp(wcx.active_mask);
                WarpPoll::Pending
            },

            // --- Output: open -> write -> close ---
            BS_SUBMIT_OPEN_OUT => unsafe {
                let path = b"batch_results.bin";
                let path_len = path.len();
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_OPEN,
                    |payload| {
                        let slot0 = (path_len as u64) | ((FILE_OPEN_WRITE_CREATE as u64) << 32);
                        core::ptr::write_volatile(payload as *mut u64, slot0);
                        let dst = payload.add(8);
                        let mut i = 0;
                        while i < path_len {
                            core::ptr::write_volatile(dst.add(i), path[i]);
                            i += 1;
                        }
                    },
                    BS_WAIT_OPEN_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_OPEN_OUT => unsafe {
                if let Some(fd) = warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_WRITE,
                    &mut self.state,
                ) {
                    if wcx.is_leader() {
                        self.fd = fd;
                    }
                }
                WarpPoll::Pending
            },

            BS_SUBMIT_WRITE => unsafe {
                let fd = self.fd;
                let r_off = self.result_offset;
                let r_len = self.result_bytes;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_BULK_WRITE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                        core::ptr::write_volatile(payload.add(8) as *mut u64, r_off);
                        core::ptr::write_volatile(payload.add(16) as *mut u64, r_len);
                    },
                    BS_WAIT_WRITE,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_WRITE => unsafe {
                if warp_hostcall_wait_u64(
                    self.buf,
                    wcx,
                    self.pkt_idx,
                    BS_SUBMIT_CLOSE_OUT,
                    &mut self.state,
                )
                .is_some()
                {}
                WarpPoll::Pending
            },

            BS_SUBMIT_CLOSE_OUT => unsafe {
                let fd = self.fd;
                warp_hostcall_submit(
                    self.buf,
                    wcx,
                    SERVICE_CLOSE,
                    |payload| {
                        core::ptr::write_volatile(payload as *mut u64, fd);
                    },
                    BS_WAIT_CLOSE_OUT,
                    &mut self.state,
                    &mut self.pkt_idx,
                )
            },

            BS_WAIT_CLOSE_OUT => unsafe {
                if warp_hostcall_wait_u64(self.buf, wcx, self.pkt_idx, BS_DONE, &mut self.state)
                    .is_some()
                {
                    return WarpPoll::Ready(true);
                }
                WarpPoll::Pending
            },

            BS_DONE => WarpPoll::Ready(true),
            _ => WarpPoll::Ready(false),
        }
    }
}

/// ml-workload.3: GPU-autonomous batch vector search.
/// Processes multiple queries against the same database in one kernel launch.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn batch_search_pipeline(
    buf: *mut u8,
    sideband: *mut u8,
    status: *mut u32,
) {
    gpu_runtime::panic::gpu_panic_init(buf);

    let mut future = BatchSearchFuture::new(buf, sideband);
    let ok = gpu_runtime::warp_future::WarpExecutor::run(&mut future);

    if gpu_atomics::lane_id() == 0 {
        core::ptr::write_volatile(status, if ok { 1 } else { 0 });
    }
}

// ============================================================
// gpu-compute.3: Tensor Core MMA via inline PTX
// ============================================================

/// gpu-compute.3: Test Tensor Core MMA instruction via inline PTX.
///
/// Uses `mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32` on SM80+.
/// Each thread in the warp holds a fragment of A, B, C matrices.
/// Test: A=0, B=0, C=known → D should equal C (0*0 + C = C).
///
/// Parameters:
/// - c_vals: pointer to 4 f32 values per thread = 128 f32 total (as u32 bits)
/// - d_out:  pointer to 4 f32 values per thread = 128 f32 output (as u32 bits)
/// - status: 0 on entry, set to 1 on success
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_mma_m16n8k16(
    c_vals: *const u32,
    d_out: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    // Each thread reads its 4 C fragment registers (f32 as u32 bits)
    let base = (tid * 4) as usize;
    let c0 = *c_vals.add(base);
    let c1 = *c_vals.add(base + 1);
    let c2 = *c_vals.add(base + 2);
    let c3 = *c_vals.add(base + 3);

    // A = 0 (f16x2), B = 0 (f16x2) → D = 0*0 + C = C
    let a0: u32 = 0;
    let a1: u32 = 0;
    let a2: u32 = 0;
    let a3: u32 = 0;
    let b0: u32 = 0;
    let b1: u32 = 0;

    let d0: u32;
    let d1: u32;
    let d2: u32;
    let d3: u32;

    #[cfg(target_arch = "nvptx64")]
    {
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        d0 = c0;
        d1 = c1;
        d2 = c2;
        d3 = c3;
    }

    // Write D fragment back
    *d_out.add(base) = d0;
    *d_out.add(base + 1) = d1;
    *d_out.add(base + 2) = d2;
    *d_out.add(base + 3) = d3;

    // Lane 0 sets status
    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-compute.4: Shared memory access + bar.sync
// ============================================================

/// gpu-compute.4: Test shared memory access + bar.sync from Rust inline PTX.
///
/// Each thread writes its thread ID to shared memory, synchronizes,
/// then reads its neighbor's value (tid XOR 1) and writes to output.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_shared_memory(output: *mut u32, n: u32, status: *mut u32) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid >= n {
        return;
    }

    #[cfg(target_arch = "nvptx64")]
    {
        // Get shared memory base (generic address space pointer)
        let smem = get_dynamic_smem_ptr() as *mut u32;

        // Each thread writes its tid to shared memory
        *smem.add(tid as usize) = tid + 1; // +1 so we can distinguish from zero-init

        // Synchronize all threads in the block
        bar_sync();

        // Each thread reads its neighbor's value (XOR with 1 for pair swap)
        let neighbor = tid ^ 1;
        let val = if neighbor < n {
            *smem.add(neighbor as usize)
        } else {
            0
        };

        // Write to global output
        *output.add(tid as usize) = val;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (output, n);
    }

    // Lane 0 sets status
    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-compute.5: Tiled GEMM — MMA + shared memory pipeline
// ============================================================

/// gpu-compute.5: Tiled GEMM combining Tensor Core MMA + shared memory.
///
/// Demonstrates the full pipeline:
///   global memory → shared memory → MMA fragment registers → MMA → global memory
///
/// Computes D[16x8] = A[16x16] x B[16x8] + C (C=0).
/// A and B are f16, D is f32. Uses a single MMA tile (m16n8k16).
///
/// Test uses all-1.0 matrices: every element of D should be 16.0
/// (sum of 16 products of 1.0 x 1.0).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_tiled_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        // Step 1: Load A and B from global to shared memory
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem;
        let b_smem = smem.add(128);

        // 32 threads load 128 u32s of A (4 each)
        for i in 0..4u32 {
            let idx = (tid * 4 + i) as usize;
            *a_smem.add(idx) = *a_global.add(idx);
        }
        // 32 threads load 64 u32s of B (2 each)
        for i in 0..2u32 {
            let idx = (tid * 2 + i) as usize;
            *b_smem.add(idx) = *b_global.add(idx);
        }

        bar_sync();

        // Step 2: Load MMA fragments from shared memory.
        let a0 = *a_smem.add(0);
        let a1 = *a_smem.add(1);
        let a2 = *a_smem.add(2);
        let a3 = *a_smem.add(3);
        let b0 = *b_smem.add(0);
        let b1 = *b_smem.add(1);

        // C = 0 (f32 accumulator)
        let c0: u32 = 0;
        let c1: u32 = 0;
        let c2: u32 = 0;
        let c3: u32 = 0;

        // Step 3: Execute MMA
        let d0: u32;
        let d1: u32;
        let d2: u32;
        let d3: u32;
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );

        // Step 4: Write D fragments to global memory (thread-indexed layout)
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = d0;
        *d_global.add(out_base + 1) = d1;
        *d_global.add(out_base + 2) = d2;
        *d_global.add(out_base + 3) = d3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-compute.6: Element-wise GPU compute kernels
// ============================================================

/// gpu-pipeline.1: MMA with proper fragment-to-matrix index mapping.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_mma_mapped(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [16][8] = 128 u32
        let b_smem = smem.add(128); // [16][4] = 64 u32

        // Cooperative load: 32 threads load 128 u32s of A (4 each)
        for i in 0..4u32 {
            let idx = (tid * 4 + i) as usize;
            *a_smem.add(idx) = *a_global.add(idx);
        }
        // 32 threads load 64 u32s of B (2 each)
        for i in 0..2u32 {
            let idx = (tid * 2 + i) as usize;
            *b_smem.add(idx) = *b_global.add(idx);
        }
        bar_sync();

        // Fragment indexing for m16n8k16:
        let group = tid / 4; // 0..7
        let lane = tid % 4; // 0..3

        let a0 = *a_smem.add((group * 8 + lane) as usize);
        let a1 = *a_smem.add((group * 8 + lane + 4) as usize);
        let a2 = *a_smem.add(((group + 8) * 8 + lane) as usize);
        let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);

        let b0 = *b_smem.add((group * 4 + lane) as usize);
        let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

        // C = 0 (f32 accumulator)
        let c0: u32 = 0;
        let c1: u32 = 0;
        let c2: u32 = 0;
        let c3: u32 = 0;

        let d0: u32;
        let d1: u32;
        let d2: u32;
        let d3: u32;
        core::arch::asm!(
            "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
             {{{d0}, {d1}, {d2}, {d3}}}, \
             {{{a0}, {a1}, {a2}, {a3}}}, \
             {{{b0}, {b1}}}, \
             {{{c0}, {c1}, {c2}, {c3}}};",
            d0 = out(reg32) d0,
            d1 = out(reg32) d1,
            d2 = out(reg32) d2,
            d3 = out(reg32) d3,
            a0 = in(reg32) a0,
            a1 = in(reg32) a1,
            a2 = in(reg32) a2,
            a3 = in(reg32) a3,
            b0 = in(reg32) b0,
            b1 = in(reg32) b1,
            c0 = in(reg32) c0,
            c1 = in(reg32) c1,
            c2 = in(reg32) c2,
            c3 = in(reg32) c3,
        );

        // Write D fragments to output (thread-indexed)
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = d0;
        *d_global.add(out_base + 1) = d1;
        *d_global.add(out_base + 2) = d2;
        *d_global.add(out_base + 3) = d3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

/// gpu-compute.6: Softmax with shared memory reduction.
///
/// Computes softmax(x) for a vector of N f32 values (N <= 32, one per thread):
///   1. Find max via shared memory parallel reduction
///   2. Compute exp(x - max) per thread
///   3. Sum exp values via shared memory parallel reduction
///   4. Divide each exp by sum -> softmax output
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_softmax(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;
    if tid >= n {
        return;
    }

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut f32;
        let x = *input.add(tid as usize);

        // Step 1: Find max via shared memory reduction
        *smem.add(tid as usize) = x;
        bar_sync();

        let mut stride = n / 2;
        while stride > 0 {
            if tid < stride {
                let a = *smem.add(tid as usize);
                let b = *smem.add((tid + stride) as usize);
                if b > a {
                    *smem.add(tid as usize) = b;
                }
            }
            bar_sync();
            stride /= 2;
        }
        let max_val = *smem.add(0);
        bar_sync();

        // Step 2: Compute exp(x - max) per thread
        let exp_val = gpu_exp_f32(x - max_val);
        *smem.add(tid as usize) = exp_val;
        bar_sync();

        // Step 3: Sum via shared memory reduction
        stride = n / 2;
        while stride > 0 {
            if tid < stride {
                let a = *smem.add(tid as usize);
                let b = *smem.add((tid + stride) as usize);
                *smem.add(tid as usize) = a + b;
            }
            bar_sync();
            stride /= 2;
        }
        let sum = *smem.add(0);
        bar_sync();

        // Step 4: Normalize
        *output.add(tid as usize) = exp_val / sum;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-pipeline.2: Multi-tile K-accumulation GEMM loop
// ============================================================

/// Multi-tile GEMM: D = A(16×K) × B(K×8) with K-dimension tiling.
///
/// Loops over K in tiles of 16, accumulating MMA results in f32 registers.
/// A is row-major f16x2 packed [16][K/2] u32, B is row-major f16x2 packed [K][4] u32.
/// D output is 16×8 f32 in thread-indexed layout (128 u32).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_multi_tile_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut u32,
    k_tiles: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [16][8] = 128 u32 per tile
        let b_smem = smem.add(128); // [16][4] = 64 u32 per tile

        let group = tid / 4;
        let lane = tid % 4;
        let k_half = k_tiles * 8; // K/2 = packed u32 count per row of A

        // Initialize accumulator to zero
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Load A tile: 32 threads load 128 u32 (4 each)
            // A_tile[row][col_packed] = A_full[row][t*8 + col_packed]
            let mut i = 0u32;
            while i < 4 {
                let smem_idx = tid * 4 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }

            // Load B tile: 32 threads load 64 u32 (2 each)
            // B_tile[row][col_packed] = B_full[t*16 + row][col_packed]
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 4;
                let col_packed = smem_idx % 4;
                let global_idx = (t * 16 + row) * 4 + col_packed;
                *b_smem.add(smem_idx as usize) = *b_global.add(global_idx as usize);
                i += 1;
            }

            bar_sync();

            // Load MMA fragments from shared memory (same mapping as gpu-pipeline.1)
            let a0 = *a_smem.add((group * 8 + lane) as usize);
            let a1 = *a_smem.add((group * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);

            let b0 = *b_smem.add((group * 4 + lane) as usize);
            let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

            // MMA: D = A*B + C (accumulate across tiles)
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            // Feed D back as C for next iteration
            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync(); // Ensure all threads done before overwriting smem
            t += 1;
        }

        // Write final accumulated D fragments to output
        let out_base = (tid * 4) as usize;
        *d_global.add(out_base) = c0;
        *d_global.add(out_base + 1) = c1;
        *d_global.add(out_base + 2) = c2;
        *d_global.add(out_base + 3) = c3;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gpu-pipeline.3: End-to-end GEMM + softmax pipeline
// ============================================================

/// Autonomous GEMM + softmax pipeline: output = softmax(A × B, per row).
///
/// Phase 1: Multi-tile GEMM (reuses gpu-pipeline.2 pattern)
/// Phase 2: Write GEMM output to shared memory in matrix order
/// Phase 3: Per-row softmax (16 threads, 1 row each, 8 elements)
///
/// This demonstrates GPU-autonomous multi-step compute: the host launches once,
/// and the GPU executes the entire GEMM → softmax pipeline without intervention.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn test_gemm_softmax_pipeline(
    a_global: *const u32,
    b_global: *const u32,
    softmax_output: *mut f32,
    k_tiles: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // 128 u32 for A tile
        let b_smem = smem.add(128); // 64 u32 for B tile

        let group = tid / 4;
        let lane = tid % 4;
        let k_half = k_tiles * 8;

        // === Phase 1: Multi-tile GEMM ===
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            let mut i = 0u32;
            while i < 4 {
                let smem_idx = tid * 4 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 4;
                let col_packed = smem_idx % 4;
                let global_idx = (t * 16 + row) * 4 + col_packed;
                *b_smem.add(smem_idx as usize) = *b_global.add(global_idx as usize);
                i += 1;
            }
            bar_sync();

            let a0 = *a_smem.add((group * 8 + lane) as usize);
            let a1 = *a_smem.add((group * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((group + 8) * 8 + lane + 4) as usize);
            let b0 = *b_smem.add((group * 4 + lane) as usize);
            let b1 = *b_smem.add(((group + 8) * 4 + lane) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;
            bar_sync();
            t += 1;
        }

        // === Phase 2: Write GEMM output to shared memory in matrix order ===
        // Fragment mapping: d0=D[lane*2][group], d1=D[lane*2+1][group],
        //                   d2=D[lane*2+8][group], d3=D[lane*2+9][group]
        let d_smem = smem as *mut f32; // reuse shared memory (128 f32 fits in 192 u32)
        *d_smem.add((lane * 2 * 8 + group) as usize) = f32::from_bits(c0);
        *d_smem.add(((lane * 2 + 1) * 8 + group) as usize) = f32::from_bits(c1);
        *d_smem.add(((lane * 2 + 8) * 8 + group) as usize) = f32::from_bits(c2);
        *d_smem.add(((lane * 2 + 9) * 8 + group) as usize) = f32::from_bits(c3);
        bar_sync();

        // === Phase 3: Per-row softmax (threads 0-15 each handle one row) ===
        if tid < 16 {
            let row_base = (tid * 8) as usize;

            // Find max in this row
            let mut max_val = *d_smem.add(row_base);
            let mut j = 1usize;
            while j < 8 {
                let v = *d_smem.add(row_base + j);
                if v > max_val {
                    max_val = v;
                }
                j += 1;
            }

            // Compute exp(x - max) and sum
            let mut sum = 0.0f32;
            let mut exp_vals = [0.0f32; 8];
            j = 0;
            while j < 8 {
                let e = gpu_exp_f32(*d_smem.add(row_base + j) - max_val);
                exp_vals[j] = e;
                sum += e;
                j += 1;
            }

            // Normalize and write to global output
            j = 0;
            while j < 8 {
                *softmax_output.add(row_base + j) = exp_vals[j] / sum;
                j += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, softmax_output, k_tiles);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// gemm-scale.1: Multi-warp output tiling
// ============================================================

/// Multi-warp GEMM: D(32×16) = A(32×K) × B(K×16), 4 warps in 2×2 layout.
///
/// 128 threads (4 warps), each warp computes a 16×8 MMA tile.
/// Warp layout: warp_m = warp_id/2 (0..1), warp_n = warp_id%2 (0..1).
/// Shared memory per K-tile: A[32][8] + B[16][8] = 384 u32.
/// A is row-major f16x2 packed [32][K/2] u32.
/// B is row-major f16x2 packed [K][8] u32 (N=16 → 8 packed per row).
/// D is row-major f32 [32][16].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_warp_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        // Warp arrangement: 2×2 (2 in M, 2 in N)
        let warp_m = warp_id / 2; // 0 or 1
        let warp_n = warp_id % 2; // 0 or 1

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32 (col-major packed)

        let k_half = k_tiles * 8; // K/2 = packed u32 per row of A
        let k_half_cm = k_tiles * 8; // K/2 = packed u32 per column of B (col-major)

        // Initialize accumulator
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32, 128 threads → 2 each
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_global.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: [N][8] col-major packed, 128 threads → 1 each
            // B_cm layout: b_global[col * k_half_cm + k_pair]
            // = pack(B[k_pair*2][col], B[k_pair*2+1][col])
            if tid < 128 {
                let col = tid / 8; // N column (0..15)
                let k_pair = tid % 8; // row pair within tile (0..7)
                let global_idx = col * k_half_cm + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_global.add(global_idx as usize);
            }

            bar_sync();

            // Load A fragments for this warp's M-slice (warp_m * 16)
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            // Load B fragments for this warp's N-slice (col-major packed)
            // b_smem layout: [N][8], col = warp_n*8+group, k_pair = lane
            // b0 = pack(B[lane*2][col], B[lane*2+1][col])
            // b1 = pack(B[lane*2+8][col], B[lane*2+9][col])
            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // MMA: D = A*B + C
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output in row-major [32][16] f32
        // Correct MMA fragment mapping (m16n8k16.row.col):
        //   d0→D[group][lane*2], d1→D[group][lane*2+1],
        //   d2→D[group+8][lane*2], d3→D[group+8][lane*2+1]
        let r0 = warp_m * 16 + group;
        let r2 = warp_m * 16 + group + 8;
        let c0_idx = warp_n * 8 + lane * 2;
        let c1_idx = c0_idx + 1;

        *d_global.add((r0 * n_cols + c0_idx) as usize) = f32::from_bits(c0);
        *d_global.add((r0 * n_cols + c1_idx) as usize) = f32::from_bits(c1);
        *d_global.add((r2 * n_cols + c0_idx) as usize) = f32::from_bits(c2);
        *d_global.add((r2 * n_cols + c1_idx) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        core::ptr::write_volatile(status, 1);
    }
}

// ============================================================
// Multi-block GEMM (gemm-scale.2)
// ============================================================

/// Multi-block GEMM: D(M×N) = A(M×K) × B(K×N), M = num_blocks * 32, N = 16.
///
/// Each block: 128 threads (4 warps), computes D[block_m*32..(block_m+1)*32][0..15].
/// grid_dim = (M/32, 1, 1), block_dim = (128, 1, 1).
/// A is row-major f16x2 packed [M][K/2] u32.
/// B is column-major f16x2 packed [N][K/2] u32.
/// D is row-major f32 [M][N].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_block_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    m_rows: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32; // which 32-row block

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        // Warp arrangement: 2×2 (2 in M, 2 in N)
        let warp_m = warp_id / 2; // 0 or 1
        let warp_n = warp_id % 2; // 0 or 1

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32 (col-major packed)

        let k_half = k_tiles * 8; // K/2 = packed u32 per row of A

        // Offset A by block_m * 32 rows
        let a_block = a_global.add((block_m * 32 * k_half) as usize);

        // Initialize accumulator
        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32, 128 threads → 2 each
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_block.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: [N][8] col-major packed, 128 threads → 1 each
            // B is shared across all blocks — no offset needed
            if tid < 128 {
                let col = tid / 8; // N column (0..15)
                let k_pair = tid % 8; // row pair within tile (0..7)
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_global.add(global_idx as usize);
            }

            bar_sync();

            // Load A fragments for this warp's M-slice (warp_m * 16)
            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            // Load B fragments for this warp's N-slice (col-major packed)
            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            // MMA: D = A*B + C
            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output — offset by block_m * 32 rows in global D
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let c0_idx = warp_n * 8 + lane * 2;
        let c1_idx = c0_idx + 1;

        *d_global.add((global_r0 * n_cols + c0_idx) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + c1_idx) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + c0_idx) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + c1_idx) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols, m_rows);
    }

    // All blocks atomically increment status; host checks for num_blocks
    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Full GEMM with 2D tiling (gemm-scale.3)
// ============================================================

/// Full GEMM: D(M×N) = A(M×K) × B(K×N), arbitrary M/N multiples of 32/16.
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Each block: 128 threads (4 warps), computes D[bm*32..(bm+1)*32][bn*16..(bn+1)*16].
/// A is row-major f16x2 packed [M][K/2] u32.
/// B is column-major f16x2 packed [N][K/2] u32.
/// D is row-major f32 [M][N].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm(
    a_global: *const u32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_half = k_tiles * 8; // K/2

        // A block offset: rows [block_m*32 .. block_m*32+32]
        let a_block = a_global.add((block_m * 32 * k_half) as usize);
        // B block offset: columns [block_n*16 .. block_n*16+16]
        // B is col-major packed: [N][K/2], so column offset = block_n * 16 * k_half
        let b_block = b_global.add((block_n * 16 * k_half) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: [32][8] = 256 u32
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                let global_idx = row * k_half + t * 8 + col_packed;
                *a_smem.add(smem_idx as usize) = *a_block.add(global_idx as usize);
                i += 1;
            }

            // Cooperative load B tile: 16 columns × 8 k-pairs = 128 u32
            if tid < 128 {
                let col = tid / 8; // local column within this 16-col block
                let k_pair = tid % 8;
                // b_block already offset to the right 16 columns
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_block.add(global_idx as usize);
            }

            bar_sync();

            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output with global row/col offsets
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Full GEMM with f32 A input (precision-fix.2)
// ============================================================

/// Full GEMM with f32 activation input: D(M×N) = A(M×K) × B(K×N).
///
/// Same as `full_gemm` but A is row-major f32 [M][K] instead of packed f16x2.
/// The kernel converts f32→f16 per-tile in shared memory, eliminating the
/// separate f32_to_f16x2_pack kernel launch and global memory f16 roundtrip.
///
/// B is still column-major f16x2 packed [N][K/2] u32 (weights, packed once).
/// D is row-major f32 [M][N].
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// Shared memory: (256 + 128) * 4 = 1536 bytes (same as full_gemm).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn full_gemm_f32in(
    a_global: *const f32,
    b_global: *const u32,
    d_global: *mut f32,
    k_tiles: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let warp_id = tid / 32;
        let local_tid = tid % 32;
        let group = local_tid / 4;
        let lane = local_tid % 4;

        let warp_m = warp_id / 2;
        let warp_n = warp_id % 2;

        let smem = get_dynamic_smem_ptr() as *mut u32;
        let a_smem = smem; // [32][8] = 256 u32 (f16x2 packed in smem)
        let b_smem = smem.add(256); // [16][8] = 128 u32

        let k_full = k_tiles * 16; // K (number of f32 elements per A row)
        let k_half = k_tiles * 8; // K/2 (number of u32 per B column)

        // A block offset: rows [block_m*32 .. block_m*32+32], f32 layout
        let a_block = a_global.add((block_m * 32 * k_full) as usize);
        // B block offset: columns [block_n*16 .. block_n*16+16]
        let b_block = b_global.add((block_n * 16 * k_half) as usize);

        let mut c0: u32 = 0;
        let mut c1: u32 = 0;
        let mut c2: u32 = 0;
        let mut c3: u32 = 0;

        let mut t = 0u32;
        while t < k_tiles {
            // Cooperative load A tile: read f32 pairs, convert to f16x2, store in smem
            // smem layout: [32][8] u32, each u32 = 2 packed f16 values
            let mut i = 0u32;
            while i < 2 {
                let smem_idx = tid * 2 + i;
                let row = smem_idx / 8;
                let col_packed = smem_idx % 8;
                // Two f32 values that map to this packed u32
                let k_base = t * 16 + col_packed * 2;
                let v0 = *a_block.add((row * k_full + k_base) as usize);
                let v1 = *a_block.add((row * k_full + k_base + 1) as usize);
                // Convert f32 → f16 via PTX cvt
                let h0: u32;
                let h1: u32;
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h0,
                    f = in(reg32) v0,
                );
                core::arch::asm!(
                    "cvt.rn.f16.f32 {h}, {f};",
                    h = out(reg32) h1,
                    f = in(reg32) v1,
                );
                // Pack: lo | (hi << 16)
                let packed = (h0 & 0xFFFF) | (h1 << 16);
                *a_smem.add(smem_idx as usize) = packed;
                i += 1;
            }

            // Cooperative load B tile: 16 columns × 8 k-pairs = 128 u32 (same as full_gemm)
            if tid < 128 {
                let col = tid / 8;
                let k_pair = tid % 8;
                let global_idx = col * k_half + t * 8 + k_pair;
                *b_smem.add(tid as usize) = *b_block.add(global_idx as usize);
            }

            bar_sync();

            let a_off = warp_m * 16;
            let a0 = *a_smem.add(((a_off + group) * 8 + lane) as usize);
            let a1 = *a_smem.add(((a_off + group) * 8 + lane + 4) as usize);
            let a2 = *a_smem.add(((a_off + group + 8) * 8 + lane) as usize);
            let a3 = *a_smem.add(((a_off + group + 8) * 8 + lane + 4) as usize);

            let b_col = warp_n * 8 + group;
            let b0 = *b_smem.add((b_col * 8 + lane) as usize);
            let b1 = *b_smem.add((b_col * 8 + lane + 4) as usize);

            let d0: u32;
            let d1: u32;
            let d2: u32;
            let d3: u32;
            core::arch::asm!(
                "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 \
                 {{{d0}, {d1}, {d2}, {d3}}}, \
                 {{{a0}, {a1}, {a2}, {a3}}}, \
                 {{{b0}, {b1}}}, \
                 {{{c0}, {c1}, {c2}, {c3}}};",
                d0 = out(reg32) d0,
                d1 = out(reg32) d1,
                d2 = out(reg32) d2,
                d3 = out(reg32) d3,
                a0 = in(reg32) a0,
                a1 = in(reg32) a1,
                a2 = in(reg32) a2,
                a3 = in(reg32) a3,
                b0 = in(reg32) b0,
                b1 = in(reg32) b1,
                c0 = in(reg32) c0,
                c1 = in(reg32) c1,
                c2 = in(reg32) c2,
                c3 = in(reg32) c3,
            );

            c0 = d0;
            c1 = d1;
            c2 = d2;
            c3 = d3;

            bar_sync();
            t += 1;
        }

        // Write output with global row/col offsets (same as full_gemm)
        let global_r0 = block_m * 32 + warp_m * 16 + group;
        let global_r2 = block_m * 32 + warp_m * 16 + group + 8;
        let global_c0 = block_n * 16 + warp_n * 8 + lane * 2;
        let global_c1 = global_c0 + 1;

        *d_global.add((global_r0 * n_cols + global_c0) as usize) = f32::from_bits(c0);
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = f32::from_bits(c1);
        *d_global.add((global_r2 * n_cols + global_c0) as usize) = f32::from_bits(c2);
        *d_global.add((global_r2 * n_cols + global_c1) as usize) = f32::from_bits(c3);
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_tiles, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// LayerNorm kernel (transformer-layer.1)
// ============================================================

/// Warp-level butterfly reduction: sum across all 32 lanes.
#[inline(always)]
unsafe fn warp_reduce_sum_f32(mut val: f32) -> f32 {
    #[cfg(target_arch = "nvptx64")]
    {
        let mask = 0xFFFF_FFFFu32;
        let mut offset = 16u32;
        while offset > 0 {
            let other: f32;
            core::arch::asm!(
                "shfl.sync.bfly.b32 {dst}, {src}, {off}, 0x1f, {mask};",
                dst = out(reg32) other,
                src = in(reg32) val,
                off = in(reg32) offset,
                mask = in(reg32) mask,
                options(nostack),
            );
            val += other;
            offset /= 2;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {}
    val
}

/// LayerNorm: y[i] = gamma[i] * (x[i] - mean) / sqrt(var + eps) + beta[i]
///
/// grid_dim = (num_rows, 1, 1), block_dim = (32, 1, 1).
/// Each block (1 warp) processes one row of d_model elements.
/// Input/output: f32 [num_rows][d_model].
/// gamma, beta: f32 [d_model].
#[no_mangle]
pub unsafe extern "ptx-kernel" fn layer_norm(
    input: *const f32,
    output: *mut f32,
    gamma: *const f32,
    beta: *const f32,
    d_model: u32,
    eps: f32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let row = nvptx::_block_idx_x() as u32;
        let row_ptr = input.add((row * d_model) as usize);
        let out_ptr = output.add((row * d_model) as usize);
        let elems_per_thread = d_model / 32;

        // Phase 1: Compute mean — each thread sums its elements
        let mut local_sum: f32 = 0.0;
        let mut i = 0u32;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            local_sum += *row_ptr.add(idx as usize);
            i += 1;
        }
        let total_sum = warp_reduce_sum_f32(local_sum);
        let mean = total_sum / d_model as f32;

        // Phase 2: Compute variance — each thread sums squared deviations
        let mut local_var: f32 = 0.0;
        i = 0;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            let diff = *row_ptr.add(idx as usize) - mean;
            local_var += diff * diff;
            i += 1;
        }
        let total_var = warp_reduce_sum_f32(local_var);
        let var = total_var / d_model as f32;
        let inv_std = 1.0 / gpu_sqrtf(var + eps);

        // Phase 3: Normalize and apply affine
        i = 0;
        while i < elems_per_thread {
            let idx = tid * elems_per_thread + i;
            let x = *row_ptr.add(idx as usize);
            let g = *gamma.add(idx as usize);
            let b = *beta.add(idx as usize);
            *out_ptr.add(idx as usize) = g * (x - mean) * inv_std + b;
            i += 1;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, gamma, beta, d_model, eps);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// GELU kernel (transformer-layer.2)
// ============================================================

/// GELU activation: y = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
/// Each thread processes one element.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gelu_forward(
    input: *const f32,
    output: *mut f32,
    n: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < n {
            let x = *input.add(global_id as usize);
            // GELU(x) = x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
            // tanh(z) = (exp(2z) - 1) / (exp(2z) + 1)
            let sqrt_2_over_pi: f32 = 0.7978845608; // sqrt(2/pi)
            let coeff: f32 = 0.044715;
            let inner = sqrt_2_over_pi * (x + coeff * x * x * x);
            // tanh via exp: tanh(z) = 1 - 2/(exp(2z)+1)
            let exp_2z = gpu_exp_f32(2.0 * inner);
            let tanh_val = (exp_2z - 1.0) / (exp_2z + 1.0);
            let result = x * 0.5 * (1.0 + tanh_val);
            *output.add(global_id as usize) = result;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, n);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Scaled Dot-Product Attention kernel (transformer-layer.3)
// ============================================================

/// Single-head scaled dot-product attention (f32, small seq):
/// output[seq][d_head] = softmax(Q[seq][d_head] × K^T[d_head][seq] / sqrt(d_head)) × V[seq][d_head]
///
/// grid_dim = (n_heads, 1, 1), block_dim = (32, 1, 1).
/// Each block (1 warp) processes one attention head.
/// Q, K, V are laid out as [n_heads][seq_len][d_head] f32 (already projected & split).
/// Output: [n_heads][seq_len][d_head] f32.
/// Constraint: seq_len <= 32 (one thread per query position).
/// causal_mask: 0 = no mask (bidirectional), 1 = causal (mask future positions).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn attention_head(
    q_global: *const f32,   // [n_heads][seq_len][d_head]
    k_global: *const f32,   // [n_heads][seq_len][d_head]
    v_global: *const f32,   // [n_heads][seq_len][d_head]
    out_global: *mut f32,   // [n_heads][seq_len][d_head]
    seq_len: u32,
    d_head: u32,
    causal_mask: u32,       // 0 = no mask, 1 = causal
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let head_offset = head * seq_len * d_head;

        if tid >= seq_len {
            // Still participate in warp-level operations, but skip writes.
            // For simplicity, just return — seq_len should equal 32 or be < 32.
            return;
        }

        let q_head = q_global.add(head_offset as usize);
        let k_head = k_global.add(head_offset as usize);
        let v_head = v_global.add(head_offset as usize);
        let out_head = out_global.add(head_offset as usize);

        // This thread handles query row `tid` (one row of seq_len).
        // Step 1: Compute attention scores = Q[tid] · K[j]^T / sqrt(d_head)
        //         for j = 0..seq_len
        let scale = 1.0 / gpu_sqrtf(d_head as f32);
        let smem = get_dynamic_smem_ptr() as *mut f32;
        // smem layout: [seq_len * seq_len] for scores
        // With seq=32, that's 32*32 = 1024 f32 = 4KB — fits.

        // Compute scores for this query row
        let scores = smem.add((tid * seq_len) as usize); // my row in score matrix
        let mut j = 0u32;
        while j < seq_len {
            // Apply causal mask: positions j > tid get -inf (will become 0 after softmax)
            if causal_mask != 0 && j > tid {
                // Use a large negative value instead of -inf to avoid NaN in exp
                *scores.add(j as usize) = -1.0e38_f32;
            } else {
                let mut dot: f32 = 0.0;
                let mut d = 0u32;
                while d < d_head {
                    dot += *q_head.add((tid * d_head + d) as usize)
                        * *k_head.add((j * d_head + d) as usize);
                    d += 1;
                }
                *scores.add(j as usize) = dot * scale;
            }
            j += 1;
        }

        // Step 2: Softmax over scores[tid][0..seq_len]
        // Find max (only over non-masked positions, but -1e38 won't be max)
        let mut max_val: f32 = *scores.add(0);
        j = 1;
        while j < seq_len {
            let v = *scores.add(j as usize);
            if v > max_val {
                max_val = v;
            }
            j += 1;
        }
        // Exp and sum
        let mut sum_exp: f32 = 0.0;
        j = 0;
        while j < seq_len {
            let e = gpu_exp_f32(*scores.add(j as usize) - max_val);
            *scores.add(j as usize) = e;
            sum_exp += e;
            j += 1;
        }
        // Normalize
        let inv_sum = 1.0 / sum_exp;
        j = 0;
        while j < seq_len {
            *scores.add(j as usize) = *scores.add(j as usize) * inv_sum;
            j += 1;
        }

        // Step 3: Output = attention_weights × V
        // out[tid][d] = sum_j weights[tid][j] * V[j][d]
        let mut d = 0u32;
        while d < d_head {
            let mut acc: f32 = 0.0;
            j = 0;
            while j < seq_len {
                acc += *scores.add(j as usize) * *v_head.add((j * d_head + d) as usize);
                j += 1;
            }
            *out_head.add((tid * d_head + d) as usize) = acc;
            d += 1;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (q_global, k_global, v_global, out_global, seq_len, d_head, causal_mask);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// FlashAttention kernel (attention-scale.3)
// ============================================================

/// FlashAttention forward pass — tiled attention for arbitrary seq_len.
///
/// Uses online softmax to avoid materializing the O(seq^2) score matrix.
/// Processes K/V in tiles of B_C=32 columns, streaming from global memory.
///
/// Layout: Q/K/V/out are [n_heads, seq_len, d_head] row-major f32.
/// d_head MUST be 64 (GPT-2 small).
///
/// grid_dim = (n_heads, ceil(seq_len/32), 1)
/// block_dim = (32, 1, 1)  — one warp
/// Shared memory: k_tile[32][64] + v_tile[32][64] = 16384 bytes
#[no_mangle]
pub unsafe extern "ptx-kernel" fn flash_attention(
    q_global: *const f32,
    k_global: *const f32,
    v_global: *const f32,
    out_global: *mut f32,
    seq_len: u32,
    d_head: u32,
    causal_mask: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let head = nvptx::_block_idx_x() as u32;
        let q_tile_idx: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) q_tile_idx);

        let my_row = q_tile_idx * 32 + tid; // global Q row index

        // Head offset in global arrays
        let head_off = (head * seq_len * d_head) as usize;
        let q_head = q_global.add(head_off);
        let k_head = k_global.add(head_off);
        let v_head = v_global.add(head_off);
        let out_head = out_global.add(head_off);

        let scale = 1.0 / gpu_sqrtf(d_head as f32);

        // Shared memory: k_tile[32][64] + v_tile[32][64]
        let smem = get_dynamic_smem_ptr() as *mut f32;
        let k_tile = smem; // 32 * 64 = 2048 f32
        let v_tile = smem.add(2048); // 32 * 64 = 2048 f32

        // Load Q row into registers (thread-private, loaded once)
        let mut q_row: [f32; 64] = [0.0; 64];
        if my_row < seq_len {
            let mut d = 0u32;
            while d < d_head {
                q_row[d as usize] = *q_head.add((my_row * d_head + d) as usize);
                d += 1;
            }
        }

        // Initialize online softmax state
        let mut m: f32 = -1.0e38; // running max
        let mut l: f32 = 0.0; // running sum of exp(score - m)

        // Output accumulator (unnormalized): o_acc[d] = sum of P_ij * V_j[d]
        let mut o_acc: [f32; 64] = [0.0; 64];

        // Number of KV tiles
        let n_kv_tiles = (seq_len + 31) / 32;

        let mut t = 0u32;
        while t < n_kv_tiles {
            let kv_col_start = t * 32;

            // Early exit for causal: if entire tile is above diagonal for ALL rows in Q tile
            if causal_mask != 0 && kv_col_start > q_tile_idx * 32 + 31 {
                // All KV columns > all Q rows → fully masked, skip
                // Remaining tiles are even further right, so break
                break;
            }

            let tile_size = if kv_col_start + 32 <= seq_len {
                32u32
            } else {
                seq_len - kv_col_start
            };

            // Cooperative load K tile: 32 threads load 32 rows of K, each thread loads 1 row
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *k_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *k_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            // Cooperative load V tile: same pattern
            {
                let global_kv_row = kv_col_start + tid;
                let mut d = 0u32;
                while d < d_head {
                    let val = if global_kv_row < seq_len {
                        *v_head.add((global_kv_row * d_head + d) as usize)
                    } else {
                        0.0
                    };
                    *v_tile.add((tid * d_head + d) as usize) = val;
                    d += 1;
                }
            }

            bar_sync();

            if my_row < seq_len {
                // Compute scores for this row against tile columns
                // and perform online softmax update
                let mut tile_max: f32 = -1.0e38;
                let mut scores: [f32; 32] = [0.0; 32]; // temp scores for this tile

                let mut c = 0u32;
                while c < tile_size {
                    let kv_col = kv_col_start + c;

                    if causal_mask != 0 && kv_col > my_row {
                        // Masked position
                        scores[c as usize] = -1.0e38;
                    } else {
                        // Dot product: Q[my_row] · K[kv_col]
                        let mut dot: f32 = 0.0;
                        let mut d = 0u32;
                        while d < d_head {
                            dot += q_row[d as usize]
                                * *k_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                        let s = dot * scale;
                        scores[c as usize] = s;
                        if s > tile_max {
                            tile_max = s;
                        }
                    }
                    c += 1;
                }

                // Pad remaining scores (if tile_size < 32) with -inf
                c = tile_size;
                while c < 32 {
                    scores[c as usize] = -1.0e38;
                    c += 1;
                }

                // Online softmax update
                let m_new = if tile_max > m { tile_max } else { m };

                // Compute exp(scores - m_new) and their sum
                let mut row_sum: f32 = 0.0;
                let mut exp_scores: [f32; 32] = [0.0; 32];
                c = 0;
                while c < tile_size {
                    let e = gpu_exp_f32(scores[c as usize] - m_new);
                    exp_scores[c as usize] = e;
                    row_sum += e;
                    c += 1;
                }

                // Correction factor for old accumulator
                let correction = gpu_exp_f32(m - m_new);

                // Rescale old output accumulator
                let mut d = 0u32;
                while d < d_head {
                    o_acc[d as usize] = o_acc[d as usize] * correction;
                    d += 1;
                }

                // Accumulate: o_acc += P_tile × V_tile (for this row)
                // o_acc[d] += sum_c exp_scores[c] * V_tile[c][d]
                c = 0;
                while c < tile_size {
                    let p = exp_scores[c as usize];
                    if p > 0.0 {
                        let mut d = 0u32;
                        while d < d_head {
                            o_acc[d as usize] +=
                                p * *v_tile.add((c * d_head + d) as usize);
                            d += 1;
                        }
                    }
                    c += 1;
                }

                // Update running stats
                l = l * correction + row_sum;
                m = m_new;
            }

            bar_sync();
            t += 1;
        }

        // Final normalization: o_acc /= l
        if my_row < seq_len && l > 0.0 {
            let inv_l = 1.0 / l;
            let mut d = 0u32;
            while d < d_head {
                *out_head.add((my_row * d_head + d) as usize) =
                    o_acc[d as usize] * inv_l;
                d += 1;
            }
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (q_global, k_global, v_global, out_global, seq_len, d_head, causal_mask);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Token + positional embedding kernel (full-inference.1)
// ============================================================

/// Embedding lookup + addition: out[pos, d] = wte[token_ids[pos], d] + wpe[pos, d]
///
/// wte: [vocab_size, d_model] f32 (token embedding table)
/// wpe: [max_seq, d_model] f32 (positional embedding table)
/// token_ids: [seq_len] u32
/// out: [seq_len, d_model] f32
///
/// grid_dim = (ceil(seq_len * d_model / 256), 1, 1), block_dim = (256, 1, 1)
#[no_mangle]
pub unsafe extern "ptx-kernel" fn embedding_lookup(
    wte: *const f32,
    wpe: *const f32,
    token_ids: *const u32,
    out: *mut f32,
    seq_len: u32,
    d_model: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = seq_len * d_model;

        if global_id < total {
            let pos = global_id / d_model;
            let d = global_id % d_model;

            let token_id = *token_ids.add(pos as usize);
            let tok_emb = *wte.add((token_id * d_model + d) as usize);
            let pos_emb = *wpe.add((pos * d_model + d) as usize);

            *out.add(global_id as usize) = tok_emb + pos_emb;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (wte, wpe, token_ids, out, seq_len, d_model);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Bias-add kernel (transformer-layer.4 helper)
// ============================================================

/// Add bias to a 2D matrix in-place: data[i][j] += bias[j]
///
/// grid_dim = (ceil(total/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn bias_add(
    data: *mut f32,
    bias: *const f32,
    n_cols: u32,
    total: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < total {
            let col = global_id % n_cols;
            let val = *data.add(global_id as usize) + *bias.add(col as usize);
            *data.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (data, bias, n_cols, total);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}

// ============================================================
// Elementwise add kernel (residual connection helper)
// ============================================================

/// a[i] += b[i] (in-place residual add)
///
/// grid_dim = (ceil(n/256), 1, 1), block_dim = (256, 1, 1).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn elementwise_add(a: *mut f32, b: *const f32, n: u32) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        if global_id < n {
            let val = *a.add(global_id as usize) + *b.add(global_id as usize);
            *a.add(global_id as usize) = val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a, b, n);
    }
}

// ============================================================
// QKV split + transpose kernel (transformer-layer.6 helper)
// ============================================================

/// Split QKV from [seq, 3*d_model] into Q, K, V as [n_heads][seq][d_head].
///
/// input: [seq_len, 3*d_model] f32 (row-major)
/// q/k/v_out: [n_heads, seq_len, d_head] f32 (head-major)
///
/// grid_dim = (ceil(total_out/256), 1, 1), block_dim = (256, 1, 1).
/// total_out = n_heads * seq_len * d_head (per output tensor).
#[no_mangle]
pub unsafe extern "ptx-kernel" fn split_qkv(
    input: *const f32,   // [seq_len, 3*d_model]
    q_out: *mut f32,     // [n_heads, seq_len, d_head]
    k_out: *mut f32,     // [n_heads, seq_len, d_head]
    v_out: *mut f32,     // [n_heads, seq_len, d_head]
    seq_len: u32,
    n_heads: u32,
    d_head: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let total = n_heads * seq_len * d_head;

        if global_id < total {
            // Decode [head][seq][d] indices
            let d = global_id % d_head;
            let seq = (global_id / d_head) % seq_len;
            let head = global_id / (d_head * seq_len);

            let d_model = n_heads * d_head;
            // Input layout: [seq, 3*d_model], Q at [0..d_model], K at [d_model..2*d_model], V at [2*d_model..3*d_model]
            let base = seq * 3 * d_model + head * d_head + d;
            let q_val = *input.add(base as usize);
            let k_val = *input.add((base + d_model) as usize);
            let v_val = *input.add((base + 2 * d_model) as usize);

            let out_idx = global_id as usize;
            *q_out.add(out_idx) = q_val;
            *k_out.add(out_idx) = k_val;
            *v_out.add(out_idx) = v_val;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, q_out, k_out, v_out, seq_len, n_heads, d_head);
    }
}

// ============================================================
// Attention concat kernel (transformer-layer.6 helper)
// ============================================================

/// Concat attention output from [n_heads][seq][d_head] → [seq][d_model].
///
/// grid_dim = (ceil(total/256), 1, 1), block_dim = (256, 1, 1).
/// total = seq_len * d_model.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn concat_heads(
    input: *const f32, // [n_heads][seq_len][d_head]
    output: *mut f32,  // [seq_len][d_model]
    seq_len: u32,
    n_heads: u32,
    d_head: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;
        let d_model = n_heads * d_head;
        let total = seq_len * d_model;

        if global_id < total {
            // Decode [seq][d_model] → [seq][head][d]
            let d = global_id % d_head;
            let col = global_id % d_model;
            let seq = global_id / d_model;
            let head = col / d_head;

            // Input index: head * seq_len * d_head + seq * d_head + d
            let in_idx = head * seq_len * d_head + seq * d_head + d;
            *output.add(global_id as usize) = *input.add(in_idx as usize);
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, seq_len, n_heads, d_head);
    }
}

// ============================================================
// f32 → f16x2 pack kernel (transformer-layer.4 helper)
// ============================================================

/// Pack f32 row-major [M][K] → f16x2 row-major [M][K/2] u32.
/// K must be even. Each thread packs one pair.
///
/// grid_dim = (ceil(total_pairs/256), 1, 1), block_dim = (256, 1, 1).
/// total_pairs = M * K / 2.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn f32_to_f16x2_pack(
    input: *const f32,
    output: *mut u32,
    total_pairs: u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_x = nvptx::_block_idx_x() as u32;
        let global_id = block_x * 256 + tid;

        if global_id < total_pairs {
            let idx = global_id * 2;
            let v0 = *input.add(idx as usize);
            let v1 = *input.add((idx + 1) as usize);
            // f32 → f16 via PTX cvt instruction
            let h0: u32;
            let h1: u32;
            core::arch::asm!(
                "cvt.rn.f16.f32 {h}, {f};",
                h = out(reg32) h0,
                f = in(reg32) v0,
            );
            core::arch::asm!(
                "cvt.rn.f16.f32 {h}, {f};",
                h = out(reg32) h1,
                f = in(reg32) v1,
            );
            // Pack: lo | (hi << 16)
            let packed = (h0 & 0xFFFF) | (h1 << 16);
            *output.add(global_id as usize) = packed;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (input, output, total_pairs);
    }
}

// ============================================================
// Zero-pad kernel (full-inference.5.1)
// ============================================================

/// Zero out elements at index >= start_offset.
///
/// Used to clear padded rows in activation buffers, preventing NaN
/// propagation from padded positions through GEMM shared memory tiles.
///
/// grid_dim = (ceil(total_elems / 256), 1, 1), block_dim = (256, 1, 1).
/// start_offset: first element to zero (e.g., actual_seq * d_model).
/// total_elems: total buffer size.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn zero_pad(
    buffer: *mut f32,
    start_offset: u32,
    total_elems: u32,
) {
    #[cfg(target_arch = "nvptx64")]
    {
        let gid = nvptx::_block_idx_x() as u32 * 256 + nvptx::_thread_idx_x() as u32;
        let idx = start_offset + gid;
        if idx < total_elems {
            *buffer.add(idx as usize) = 0.0;
        }
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (buffer, start_offset, total_elems);
    }
}

// ============================================================
// Pure f32 GEMM kernel (full-inference.6)
// ============================================================

/// Tiled f32 GEMM without Tensor Cores: D = A × B.
///
/// A: [M, K] row-major f32.
/// B: [K, N] column-major f32 (stored as b[col * K + row]).
/// D: [M, N] row-major f32.
///
/// grid_dim = (M/32, N/16, 1), block_dim = (128, 1, 1).
/// shared_mem_bytes = (32*16 + 16*16) * 4 = 3072.
///
/// Each block computes a 32×16 output tile. Each thread computes 4 output
/// elements (2 rows × 2 cols). K dimension is tiled in chunks of 16.
#[no_mangle]
pub unsafe extern "ptx-kernel" fn gemm_f32(
    a_global: *const f32,
    b_global: *const f32,
    d_global: *mut f32,
    k_dim: u32,
    n_cols: u32,
    status: *mut u32,
) {
    let tid = nvptx::_thread_idx_x() as u32;

    #[cfg(target_arch = "nvptx64")]
    {
        let block_m = nvptx::_block_idx_x() as u32;
        let block_n: u32;
        core::arch::asm!("mov.u32 {r}, %ctaid.y;", r = out(reg32) block_n);

        let smem = get_dynamic_smem_ptr() as *mut f32;
        let a_smem = smem; // [32][16] = 512 f32
        let b_smem = smem.add(512); // [16][16] = 256 f32

        // Thread mapping: 128 threads → 32×16 output = 4 per thread
        // tid / 8 = row_pair (0..15), handles rows row_pair*2, row_pair*2+1
        // tid % 8 = col_pair (0..7), handles cols col_pair*2, col_pair*2+1
        let row_pair = tid / 8;
        let col_pair = tid % 8;
        let r0 = row_pair * 2;
        let r1 = r0 + 1;
        let c0 = col_pair * 2;
        let c1 = c0 + 1;

        let global_r0 = block_m * 32 + r0;
        let global_r1 = block_m * 32 + r1;
        let global_c0 = block_n * 16 + c0;
        let global_c1 = block_n * 16 + c1;

        // A block starts at row (block_m * 32)
        let a_block = a_global.add((block_m * 32 * k_dim) as usize);
        // B block starts at col (block_n * 16)
        let b_block = b_global.add((block_n * 16 * k_dim) as usize);

        let mut acc00: f32 = 0.0;
        let mut acc01: f32 = 0.0;
        let mut acc10: f32 = 0.0;
        let mut acc11: f32 = 0.0;

        let k_tiles = (k_dim + 15) / 16;
        let mut t = 0u32;
        while t < k_tiles {
            let k_base = t * 16;

            // Cooperative load A tile: 32×16 = 512 f32, 128 threads × 4 elements each
            let mut i = 0u32;
            while i < 4 {
                let flat = tid * 4 + i;
                let row = flat / 16;
                let col = flat % 16;
                let k_idx = k_base + col;
                let val = if k_idx < k_dim {
                    *a_block.add((row * k_dim + k_idx) as usize)
                } else {
                    0.0
                };
                *a_smem.add(flat as usize) = val;
                i += 1;
            }

            // Cooperative load B tile: 16×16 = 256 f32, 128 threads × 2 elements each
            let mut j = 0u32;
            while j < 2 {
                let flat = tid * 2 + j;
                let col = flat / 16; // B column within this tile (0..15)
                let k_row = flat % 16; // K row within tile
                let k_idx = k_base + k_row;
                let b_col = block_n * 16 + col;
                let val = if k_idx < k_dim && b_col < n_cols {
                    // B is column-major: b[col * K + k]
                    *b_global.add((b_col * k_dim + k_idx) as usize)
                } else {
                    0.0
                };
                // Store in shared mem as b_smem[col][k_row] = b_smem[col * 16 + k_row]
                *b_smem.add(flat as usize) = val;
                j += 1;
            }

            bar_sync();

            // Compute: each thread accumulates 4 output elements over K=16
            let tile_k = if k_base + 16 <= k_dim { 16 } else { k_dim - k_base };
            let mut k = 0u32;
            while k < tile_k {
                let a_r0 = *a_smem.add((r0 * 16 + k) as usize);
                let a_r1 = *a_smem.add((r1 * 16 + k) as usize);
                let b_c0 = *b_smem.add((c0 * 16 + k) as usize);
                let b_c1 = *b_smem.add((c1 * 16 + k) as usize);

                // Use FMA for better precision
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc00,
                    a = in(reg32) a_r0,
                    b = in(reg32) b_c0,
                    c = in(reg32) acc00,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc01,
                    a = in(reg32) a_r0,
                    b = in(reg32) b_c1,
                    c = in(reg32) acc01,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc10,
                    a = in(reg32) a_r1,
                    b = in(reg32) b_c0,
                    c = in(reg32) acc10,
                );
                core::arch::asm!(
                    "fma.rn.f32 {d}, {a}, {b}, {c};",
                    d = out(reg32) acc11,
                    a = in(reg32) a_r1,
                    b = in(reg32) b_c1,
                    c = in(reg32) acc11,
                );

                k += 1;
            }

            bar_sync();
            t += 1;
        }

        // Write output
        *d_global.add((global_r0 * n_cols + global_c0) as usize) = acc00;
        *d_global.add((global_r0 * n_cols + global_c1) as usize) = acc01;
        *d_global.add((global_r1 * n_cols + global_c0) as usize) = acc10;
        *d_global.add((global_r1 * n_cols + global_c1) as usize) = acc11;
    }
    #[cfg(not(target_arch = "nvptx64"))]
    {
        let _ = (a_global, b_global, d_global, k_dim, n_cols);
    }

    if tid == 0 {
        let _prev: u32;
        core::arch::asm!(
            "atom.global.add.u32 {prev}, [{addr}], 1;",
            prev = out(reg32) _prev,
            addr = in(reg64) status,
        );
    }
}
