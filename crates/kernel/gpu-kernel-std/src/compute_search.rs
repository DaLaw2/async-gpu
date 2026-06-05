// ml-workload.2 & .3: Vector Similarity Search — GPU-Autonomous Demo
//
// 20-state WarpFuture. Each state does exactly ONE thing (submit or wait).
// No multi-phase states, no sentinel values.

use gpu_atomics::membar_sys;
use gpu_kernel_core::helpers::gpu_sqrtf;
use gpu_protocol::*;
use gpu_runtime::warp_future::{warp_hostcall_submit, warp_hostcall_wait_u64};

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
pub unsafe extern "gpu-kernel" fn vector_search_pipeline(
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
pub unsafe extern "gpu-kernel" fn batch_search_pipeline(
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
