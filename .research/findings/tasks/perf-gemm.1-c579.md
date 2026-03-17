# High-Performance SGEMM Kernel Design for SM 86 (Ampere)

## Task: perf-gemm.1 — Research SGEMM Optimization Techniques

**Status**: Complete
**Goal**: Understand how to build an SGEMM kernel achieving 70%+ of cuBLAS (~2000 GFLOPS on SM 86)

---

## Current Kernel Analysis

Our `gemm_f32` kernel at `crates/kernel/gpu-kernel/src/compute_gemm.rs:1649`:
- **Tile**: 32x16 (CTA tile), K-tile = 16
- **Threads**: 128 per block
- **Thread output**: 2x2 = 4 elements per thread (128 threads * 4 = 512 = 32*16)
- **Shared memory**: 3072 bytes = (512 + 256) * 4 = A[32][16] + B[16][16]
- **No register blocking**: Each thread loads 2 A values and 2 B values from smem per K iteration
- **No vectorized loads**: Scalar `ld.global.f32` only
- **No double buffering**: Load, sync, compute, sync in lockstep
- **Result**: ~157 GFLOPS = 5.6% of cuBLAS (2780 GFLOPS)

### Why It's Slow
1. **Tiny tile** — 32x16 = 512 output elements per block. High-perf kernels use 128x128 = 16384.
2. **No register blocking** — Only 4 FMAs per K iteration per thread. Should be 64 (8x8).
3. **Terrible arithmetic intensity** — 4 FMAs need 4 smem loads (ratio 1:1). Target is 8:1 or better.
4. **Scalar memory access** — No float4 vectorization = 4x fewer bytes per transaction.
5. **Grid is too large** — Many tiny blocks means scheduling overhead dominates.

---

## 1. CUTLASS SGEMM Tile Configuration (SM 80-86)

### Recommended Configuration for SM 86
| Level | Tile Size | Notes |
|-------|-----------|-------|
| **ThreadBlock (CTA)** | 128 x 128 x 8 | Primary tile. 128x128 output, K-step=8 |
| **Warp** | 32 x 64 | 4 warps: 2 along M, 2 along N |
| **Thread** | 8 x 8 | Each thread computes 64 output elements |

### Alternative Configurations
| CTA Tile | Warps | Threads | Thread Tile | Use Case |
|----------|-------|---------|-------------|----------|
| 128x128x8 | 4 (2x2) | 128 | 8x8 | Default SGEMM, best for 1K-4K |
| 128x128x8 | 8 (4x2) | 256 | 8x4 | Higher occupancy variant |
| 128x256x8 | 8 (2x4) | 256 | 8x8 | Large matrices (M,N > 2500) |
| 64x64x16 | 4 (2x2) | 128 | 4x4 | A100-specific (more K per step) |

### Warp Arrangement (128x128, 4 warps, 128 threads)
```
CTA tile 128x128:
  Warp 0: rows [0..31],  cols [0..63]   → 32x64 sub-tile
  Warp 1: rows [0..31],  cols [64..127] → 32x64 sub-tile
  Warp 2: rows [32..63], cols [0..63]   → 32x64 sub-tile
  Warp 3: rows [32..63], cols [64..127] → 32x64 sub-tile

Wait — that's only 64 rows. Alternative (verified from CUTLASS):
  4 warps arranged as 2x2 grid over 128x128:
  Each warp owns 64x64 output sub-tile.

With 256 threads (8 warps):
  8 warps in 4x2 arrangement over 128x128:
  Each warp owns 32x64 output sub-tile.
```

### Why 128x128x8?
- **Shared memory**: A[128][8] + B[8][128] = (1024 + 1024) * 4 = 8 KB per stage
- With 2 stages (double buffer): 16 KB — fits easily in 100 KB SM shared memory
- With 3 stages: 24 KB — still fine
- **Registers**: 8x8 accumulators = 64 regs + fragments ≈ 90 regs/thread → fits in 255 max
- **Occupancy**: 128 threads/block → 2 blocks/SM possible (256 threads, 48 warps available, using 8)

---

## 2. Register Blocking (Thread-Level Tiling)

### How Each Thread Computes Its 8x8 Output Sub-Tile

Each thread "owns" an 8x8 patch of the output matrix C. Within the K-loop:

```
// Per K-step (k = 0..7 for BK=8):
// 1. Load 8 values from A column into regM[8]
// 2. Load 8 values from B row into regN[8]
// 3. Outer product: 8x8 = 64 FMAs

for k in 0..BK {
    regM[0..8] = A_smem[thread_row*8 + 0..8][k]  // 8 loads from smem
    regN[0..8] = B_smem[k][thread_col*8 + 0..8]  // 8 loads from smem

    // 64 FMAs (outer product)
    for i in 0..8 {
        for j in 0..8 {
            acc[i][j] = fma(regM[i], regN[j], acc[i][j]);
        }
    }
}
```

### Register Budget Per Thread

| Register Type | Count | Purpose |
|---------------|-------|---------|
| **Accumulators** | 64 (8x8) | Running sum of C[i][j] |
| **A fragments (regM)** | 8 | Current A column slice |
| **B fragments (regN)** | 8 | Current B row slice |
| **A fragments (prefetch)** | 8 | Next A slice (double-buffered regs) |
| **B fragments (prefetch)** | 8 | Next B slice (double-buffered regs) |
| **Loop vars, addresses** | ~16 | k, t, pointers, temp |
| **TOTAL** | ~112 | Well within 255 limit |

### FMA-to-Load Ratio (Arithmetic Intensity)

**Per K iteration (one k value within BK):**
- Shared memory loads: 8 (regM) + 8 (regN) = 16 loads
- FMA operations: 8 * 8 = 64 FMAs
- **Ratio: 64 FMAs / 16 loads = 4:1**

**Per full BK=8 step:**
- Total smem loads: 16 * 8 = 128
- Total FMAs: 64 * 8 = 512
- **Ratio: 4:1 (same, amortized)**

**Compare to our current kernel:**
- 4 smem loads → 4 FMAs per K step
- **Ratio: 1:1** — This is why we're slow!

**Why this matters:** SM 86 has 128 FP32 CUDA cores per SM partition but shared memory bandwidth is limited. The 4:1 ratio means the FP32 pipes stay fed while smem loads overlap via instruction-level parallelism.

### How Register Blocking Hides Memory Latency

The key insight: with 64 FMAs per K-step, the warp scheduler can interleave FMA instructions with memory load instructions. While one FMA is executing (4 cycles), the scheduler can issue smem loads for the next iteration. The 4:1 ratio means there are enough FMAs "in flight" to cover the ~20-30 cycle smem read latency.

---

## 3. Shared Memory Layout

### A and B Tile Storage

For CTA tile 128x128 with BK=8:

```
Shared Memory Layout:
  A_smem: [BK][BM] = [8][128] f32, TRANSPOSED from original [BM][BK]
  B_smem: [BK][BN] = [8][128] f32, row-major

  Size per buffer:
    A: 8 * 128 * 4 = 4096 bytes
    B: 8 * 128 * 4 = 4096 bytes
    Total: 8192 bytes per stage

  Double-buffered (2 stages): 16384 bytes = 16 KB
  Triple-buffered (3 stages): 24576 bytes = 24 KB
```

### Why Transpose A?

Original A is [M][K] row-major. During computation, each thread needs A[row][k] for 8 consecutive rows and a single k. If stored as A_smem[row][k], accessing 8 rows at the same k means stride-128 access in smem — guaranteed bank conflicts!

**Transposed layout A_smem[k][row]:**
- Accessing A_smem[k][row..row+8] is contiguous
- Enables `ld.shared.v4.f32` (vectorized 128-bit load) for 4 consecutive elements
- Two `ld.shared.v4.f32` loads fetch all 8 A values per thread

### Bank Conflict Avoidance

NVIDIA shared memory has **32 banks**, each 4 bytes wide. Bank = (byte_offset / 4) % 32.

**Problem with A_smem[BK][BM] = [8][128]:**
- Columns 0, 32, 64, 96 all map to bank 0
- If 32 threads in a warp access rows 0..31 at the same k, elements at offsets 0,1,2,...31 → banks 0,1,...31 → **NO conflict** (perfect)

**Problem with B_smem[BK][BN] = [8][128]:**
- Same analysis — accessing B_smem[k][col..col+8] is contiguous → no conflict

**When conflicts DO occur — padding solution:**
- If threads access elements with stride that's a multiple of 32, add 4 bytes of padding:
  - `A_smem[8][132]` instead of `A_smem[8][128]` — leading dimension = 132
  - This shifts every row by 4 floats, breaking the 32-bank pattern
  - Cost: 8 * 4 * 4 = 128 extra bytes — negligible

**Swizzling (advanced):**
- CUTLASS uses XOR-based swizzling: `smem_addr = row * BN + (col ^ (row * some_factor))`
- Avoids padding waste but harder to implement
- For a first high-perf kernel, padding is sufficient

### Double-Buffering Shared Memory

Two buffers for A and two for B. While computing with buffer 0, load the next K-tile into buffer 1.

```
Total SMEM with double-buffering:
  2 * (A[8][132] + B[8][128]) * 4 bytes
  = 2 * (1056 + 1024) * 4
  = 2 * 8320
  = 16,640 bytes ≈ 16.3 KB

Buffer swap via XOR trick:
  buf_offset ^= 8320;  // Toggle between buffer 0 and buffer 1
```

---

## 4. Global Memory Access Pattern

### Vectorized Loads (float4 = 128-bit)

Each `ld.global.v4.f32` loads 4 consecutive floats (16 bytes) in a single transaction. This is critical because:
- GPU memory transactions are 32 bytes minimum (L2 sector)
- A single `ld.global.f32` wastes 28 bytes of the 32-byte transaction
- `ld.global.v4.f32` uses 16 of 32 bytes — 2 threads per transaction → fully coalesced

### Thread-to-Tile Loading Mapping

**Loading A_smem (128x8 = 1024 elements, 128 threads):**
```
Each thread loads 1024 / 128 = 8 elements.
Using float4: each thread issues 2 float4 loads.

Thread mapping for A (original A is row-major [M][K]):
  // A is [128][K], we need A[0..128][k_base..k_base+8]
  // 128 threads, each loads 8 elements
  // Thread tid loads:
  //   Row group: tid / (8/4) = tid / 2  → rows are assigned 2 threads each
  //   But wait — K-tile is only 8 wide. float4 loads 4, so 2 float4 per row.
  //   Better: 128 threads load 128 rows, each thread loads 1 row of 8 elements
  //   = 2 float4 loads per thread

  row = tid;               // thread 0..127 → row 0..127
  float4 load1 = *(float4*)(A + (block_m*128 + row) * K + k_base + 0);
  float4 load2 = *(float4*)(A + (block_m*128 + row) * K + k_base + 4);

  // Store transposed into smem:
  A_smem[0][row] = load1.x;   // k=0
  A_smem[1][row] = load1.y;   // k=1
  A_smem[2][row] = load1.z;   // k=2
  A_smem[3][row] = load1.w;   // k=3
  A_smem[4][row] = load2.x;   // k=4
  A_smem[5][row] = load2.y;   // k=5
  A_smem[6][row] = load2.z;   // k=6
  A_smem[7][row] = load2.w;   // k=7
```

Note: The global loads ARE vectorized (coalesced), but the smem stores are scattered (transpose). This is acceptable because:
- Global loads happen once, smem reads happen 8x (once per K)
- Store-to-smem is cheaper than load-from-global

**Loading B_smem (8x128 = 1024 elements, 128 threads):**
```
Each thread loads 1024 / 128 = 8 elements = 2 float4 loads.

  // B is column-major or pre-transposed. If B is [K][N]:
  col_group = tid / 4;       // 32 column groups
  k_offset = (tid % 4) * 4;  // 0, 4 within BK=8... doesn't divide evenly

  // Simpler: flatten
  flat_start = tid * 8;
  row = flat_start / 128;  // K dimension (0..7)
  col = flat_start % 128;  // N dimension

  float4 load1 = *(float4*)(B + (k_base + row) * N + block_n*128 + col);
  float4 load2 = *(float4*)(B + (k_base + row) * N + block_n*128 + col + 4);

  B_smem[row][col+0..3] = load1;
  B_smem[row][col+4..7] = load2;
```

### Edge Handling (M/N/K not divisible by tile size)

Two approaches:
1. **Pad to tile boundaries** (our current approach) — simple but wastes memory and compute
2. **Predicated loads** — check bounds per element, load 0.0 if out of bounds:
   ```
   float val = (global_row < M && global_col < N) ? *ptr : 0.0f;
   ```
   This is what high-perf kernels do. Predication has minimal cost when most threads are in-bounds.

---

## 5. Software Pipelining

### cp.async (SM 80+)

`cp.async.ca.shared.global` copies data directly from global memory to shared memory, bypassing registers. Available on SM 80+ (Ampere).

**PTX syntax:**
```
cp.async.ca.shared.global [smem_ptr], [gmem_ptr], 16;  // 16 bytes = float4
cp.async.commit_group;
cp.async.wait_group N;  // Wait until at most N groups pending
```

**Benefits over register-mediated loads:**
1. Frees up registers — no temp registers needed for global→smem transfer
2. Can overlap with computation without occupying FP32 pipes
3. Enables multi-stage pipelining (3+ stages)

### Pipeline Stages

| Stages | SMEM per stage | Total SMEM | Benefit |
|--------|---------------|------------|---------|
| 2 (double-buffer) | 8 KB | 16 KB | Overlap load[k+1] with compute[k] |
| 3 | 8 KB | 24 KB | Overlap load[k+2] with compute[k], load[k+1] in flight |
| 4 | 8 KB | 32 KB | Deeper pipeline, more latency hiding |

**CUTLASS default for SM 80 SGEMM: 3 stages** (experimentally optimal).

### Multi-Stage Pipeline Structure

```
// Prologue: Fill pipeline
cp.async(smem[0], gmem[0]);  // Load stage 0
cp.async(smem[1], gmem[1]);  // Load stage 1
cp.async(smem[2], gmem[2]);  // Load stage 2

// Main loop
for tile_k in 3..K/BK {
    wait_group(2);  // Ensure stage (tile_k-3) is ready
    compute(smem[(tile_k-3) % 3]);  // Compute on oldest ready stage
    cp.async(smem[tile_k % 3], gmem[tile_k]);  // Load next stage
}

// Epilogue: Drain pipeline
wait_group(1); compute(smem[...]);
wait_group(0); compute(smem[...]);
compute(smem[...]);
```

### Benefit vs Simple Double-Buffering

- Double-buffering: latency of one global load must be hidden by one compute phase
- 3-stage: latency of one global load can be hidden by TWO compute phases
- On SM 86: global load latency ~200-400 cycles, compute per BK=8 step ~100-200 cycles
- With 2 stages, we might stall; with 3, we almost certainly don't

### For Our Rust/PTX Codebase

We can use `cp.async` via inline PTX assembly:
```rust
core::arch::asm!(
    "cp.async.ca.shared.global [{smem}], [{gmem}], 16;",
    smem = in(reg64) smem_ptr,
    gmem = in(reg64) gmem_ptr,
);
core::arch::asm!("cp.async.commit_group;");
// ... later ...
core::arch::asm!("cp.async.wait_group 0;");
```

**Recommendation for v1: Start with simple double-buffering (2 stages, no cp.async).** It's simpler and gets us 80%+ of the benefit. Add cp.async as a v2 optimization.

---

## 6. Concrete Implementation Plan for Our Codebase

### Recommended Configuration for SM 86

```
CTA Tile:    BM=128, BN=128, BK=8
Threads:     256 (8 warps)
Warp Tile:   32x64  (8 warps in 4x2 arrangement over 128x128)
Thread Tile: TM=8, TN=8 (each thread computes 8x8 = 64 output elements)
SMEM:        2 * (128*8 + 128*8) * 4 = 16,384 bytes (double-buffered)
Stages:      2 (double-buffer, no cp.async for v1)
```

**Verification:**
- 256 threads * 64 outputs/thread = 16,384 = 128*128 ✓
- 8 warps in 4x2 grid: each warp = 32 threads, owns 32x64 output
- 32 threads * 64 outputs = 2048 = 32*64 ✓
- Within warp: 32 threads, each with 8x8... actually 32*64=2048, 32x64/32=64 per thread ✓

### Thread-to-Output Mapping

```
// Block-level: 256 threads = 8 warps
warp_id = tid / 32;        // 0..7
lane_id = tid % 32;        // 0..31

// Warp grid: 4 rows x 2 cols over 128x128
warp_row = warp_id / 2;    // 0..3, each covers 32 rows
warp_col = warp_id % 2;    // 0..1, each covers 64 cols

// Within warp: 32 threads over 32x64 = 2048 elements
// Each thread owns 8x8 = 64 elements
// Threads arranged as 4x8 grid within warp tile:
thread_row = lane_id / 8;  // 0..3, each covers 8 rows
thread_col = lane_id % 8;  // 0..7, each covers 8 cols

// Global output position:
out_row_base = block_m * 128 + warp_row * 32 + thread_row * 8;
out_col_base = block_n * 128 + warp_col * 64 + thread_col * 8;

// This thread computes C[out_row_base..+8][out_col_base..+8]
```

### Shared Memory Layout with Bank Conflict Avoidance

```rust
// Shared memory layout (transposed A, padded):
const BM: usize = 128;
const BN: usize = 128;
const BK: usize = 8;
const A_PAD: usize = 4;  // Padding to avoid bank conflicts

// Buffer 0:
//   A_smem[BK][BM + A_PAD] at offset 0
//   B_smem[BK][BN]         at offset BK * (BM + A_PAD)
// Buffer 1:
//   Same layout, offset by STAGE_SIZE

const A_STRIDE: usize = BM + A_PAD;  // 132
const STAGE_SIZE: usize = (BK * A_STRIDE + BK * BN);  // 8*132 + 8*128 = 1056+1024 = 2080 floats
// Total SMEM: 2 * 2080 * 4 = 16,640 bytes
```

### Main Loop Structure (Pseudocode)

```rust
pub unsafe extern "ptx-kernel" fn gemm_f32_v2(
    a_global: *const f32,  // [M, K] row-major
    b_global: *const f32,  // [K, N] row-major (NOT col-major)
    c_global: *mut f32,    // [M, N] row-major output
    M: u32, N: u32, K: u32,
) {
    let tid = thread_idx_x();
    let warp_id = tid / 32;
    let lane_id = tid % 32;
    let block_m = block_idx_x();
    let block_n = block_idx_y();

    // Compute thread's output position
    let warp_row = warp_id / 2;  // 0..3
    let warp_col = warp_id % 2;  // 0..1
    let thread_row = lane_id / 8;  // 0..3
    let thread_col = lane_id % 8;  // 0..7

    // Shared memory pointers
    let smem = get_dynamic_smem_ptr() as *mut f32;
    let mut buf_idx: usize = 0;

    // Accumulator registers: 8x8 = 64 f32
    let mut acc: [f32; 64] = [0.0; 64];

    // === Prologue: Load first tile into buffer 0 ===
    load_a_tile(smem, buf_idx, a_global, block_m, 0, K);
    load_b_tile(smem, buf_idx, b_global, block_n, 0, K, N);
    bar_sync();

    let k_tiles = (K + BK - 1) / BK;

    for t in 0..k_tiles {
        // If not last tile, start loading next tile into other buffer
        if t + 1 < k_tiles {
            let next_buf = 1 - buf_idx;
            load_a_tile(smem, next_buf, a_global, block_m, (t+1)*BK, K);
            load_b_tile(smem, next_buf, b_global, block_n, (t+1)*BK, K, N);
        }

        // Compute on current buffer
        let a_base = buf_idx * STAGE_SIZE;
        let b_base = buf_idx * STAGE_SIZE + BK * A_STRIDE;

        // Register-level double buffering
        let mut regM: [f32; 8];
        let mut regN: [f32; 8];

        for k in 0..BK {
            // Load A fragment: 8 values from A_smem[k][warp_row*32 + thread_row*8 .. +8]
            let a_smem_row = warp_row * 32 + thread_row * 8;
            for i in 0..8 {
                regM[i] = smem[a_base + k * A_STRIDE + a_smem_row + i];
            }

            // Load B fragment: 8 values from B_smem[k][warp_col*64 + thread_col*8 .. +8]
            let b_smem_col = warp_col * 64 + thread_col * 8;
            for j in 0..8 {
                regN[j] = smem[b_base + k * BN + b_smem_col + j];
            }

            // Outer product: 64 FMAs
            for i in 0..8 {
                for j in 0..8 {
                    acc[i*8 + j] = fma(regM[i], regN[j], acc[i*8 + j]);
                }
            }
        }

        bar_sync();
        buf_idx = 1 - buf_idx;  // Swap buffers
    }

    // === Epilogue: Write 8x8 output ===
    let out_row = block_m * 128 + warp_row * 32 + thread_row * 8;
    let out_col = block_n * 128 + warp_col * 64 + thread_col * 8;
    for i in 0..8 {
        for j in 0..8 {
            if out_row + i < M && out_col + j < N {
                *c_global.add(((out_row + i) * N + out_col + j) as usize) = acc[i*8 + j];
            }
        }
    }
}
```

### Register Budget Analysis

| Item | Registers | Notes |
|------|-----------|-------|
| Accumulators `acc[64]` | 64 | 8x8 f32 |
| A fragment `regM[8]` | 8 | Current K slice |
| B fragment `regN[8]` | 8 | Current K slice |
| Loop variables | 8 | k, t, buf_idx, k_tiles, etc. |
| Address computation | 16 | Pointers, offsets, strides |
| Temp / spill | 8 | Misc |
| **TOTAL** | **~112** | **Well within 255 limit** |

At 112 registers/thread, 256 threads/block = 28,672 registers per block.
SM 86 has 65,536 registers per SM → can run 2 blocks per SM (57,344 regs) → **good occupancy**.

### Expected GFLOPS Estimate

**SM 86 (RTX 3060 / A2) theoretical peak:**
- 128 FP32 cores/SM * 28 SMs * 2 (FMA) * 1.777 GHz = ~12,800 GFLOPS (RTX 3060)
- Or for A2: 40 SMs, 1.77 GHz → ~18,150 GFLOPS
- cuBLAS achieves ~2780 GFLOPS on our GPU → GPU has ~10 SMs or is A2 (10 SMs)

**Expected progression:**
| Optimization | Expected GFLOPS | % of cuBLAS |
|-------------|----------------|-------------|
| Current (32x16, no reg blocking) | 157 | 5.6% |
| 128x128 + register blocking (8x8) | 1200-1600 | 43-57% |
| + Vectorized loads (float4) | 1800-2100 | 65-75% |
| + Double-buffered smem | 2000-2300 | 72-83% |
| + Bank conflict avoidance (padding) | 2100-2400 | 75-86% |
| + Warp-level tiling + tuning | 2300-2500 | 83-90% |
| + cp.async 3-stage pipeline | 2400-2600 | 86-94% |

**Target: 2000+ GFLOPS (70%+) is achievable with steps 1-4.**

---

## 7. Key PTX Instructions Needed

### Core Compute
```ptx
fma.rn.f32 %d, %a, %b, %c;    // d = a*b + c, round-to-nearest
```

### Global Memory (Vectorized)
```ptx
// Load 4 floats (128-bit) from global memory
ld.global.v4.f32 {%f0, %f1, %f2, %f3}, [%addr];

// Store 4 floats (128-bit) to global memory
st.global.v4.f32 [%addr], {%f0, %f1, %f2, %f3};
```

### Shared Memory
```ptx
// Load single float from shared memory
ld.shared.f32 %f0, [%smem_addr];

// Load 4 floats from shared memory (vectorized)
ld.shared.v4.f32 {%f0, %f1, %f2, %f3}, [%smem_addr];

// Store single float to shared memory
st.shared.f32 [%smem_addr], %f0;

// Store 4 floats to shared memory (for B tile loading)
st.shared.v4.f32 [%smem_addr], {%f0, %f1, %f2, %f3};
```

### Synchronization
```ptx
bar.sync 0;                     // __syncthreads()
```

### Asynchronous Copy (SM 80+, for v2)
```ptx
// Copy 16 bytes directly from global to shared memory
cp.async.ca.shared.global [%smem], [%gmem], 16;

// Commit current group of async copies
cp.async.commit_group;

// Wait until at most N groups are pending
cp.async.wait_group N;
```

### SM 86 Specific Notes

1. **No SM 86 exclusive instructions** for SGEMM — SM 80 instructions work.
2. SM 86 has **128 FP32 cores per SM** (same as SM 80) but fewer SMs in consumer GPUs.
3. SM 86 supports `cp.async` same as SM 80.
4. SM 86 has **100 KB configurable shared memory** per SM (vs 164 KB on SM 80 A100).
5. Max registers per thread: **255** (same as SM 80).
6. Max warps per SM: **48** (vs 64 on SM 80).
7. Max blocks per SM: **16** (vs 32 on SM 80).

### Inline PTX in Rust (Examples)

```rust
// Vectorized global load (float4)
let f0: f32; let f1: f32; let f2: f32; let f3: f32;
core::arch::asm!(
    "ld.global.v4.f32 {{{f0}, {f1}, {f2}, {f3}}}, [{addr}];",
    f0 = out(reg32) f0,
    f1 = out(reg32) f1,
    f2 = out(reg32) f2,
    f3 = out(reg32) f3,
    addr = in(reg64) global_ptr,
);

// FMA
core::arch::asm!(
    "fma.rn.f32 {d}, {a}, {b}, {c};",
    d = out(reg32) acc,
    a = in(reg32) reg_m,
    b = in(reg32) reg_n,
    c = in(reg32) acc,
);

// cp.async (SM 80+)
core::arch::asm!(
    "cp.async.ca.shared.global [{smem}], [{gmem}], 16;",
    smem = in(reg64) smem_ptr,
    gmem = in(reg64) gmem_ptr,
);
```

---

## Implementation Roadmap

### Phase 1: Basic 128x128 with Register Blocking (Target: 1200-1600 GFLOPS)
1. Change tile from 32x16 to 128x128
2. Change thread count from 128 to 256
3. Add 8x8 register blocking (64 accumulators per thread)
4. Keep scalar loads for now
5. Keep single-buffered smem

### Phase 2: Vectorized Loads + Double Buffering (Target: 2000-2300 GFLOPS)
1. Add float4 global loads (`ld.global.v4.f32`)
2. Transpose A during smem store
3. Add smem padding (A_stride = 132 instead of 128)
4. Double-buffer shared memory
5. Add bounds checking for edge tiles

### Phase 3: Advanced Optimizations (Target: 2400+ GFLOPS)
1. cp.async 3-stage pipeline
2. Warp-level load scheduling
3. Autotuning tile sizes for different matrix shapes
4. Shared memory swizzling (replacing padding)

---

## Sources

- [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog (siboehm)](https://siboehm.com/articles/22/CUDA-MMM)
- [Advanced Matrix Multiplication Optimization on NVIDIA GPUs (salykova)](https://salykova.github.io/sgemm-gpu)
- [Inside NVIDIA GPUs: Anatomy of high performance matmul kernels (Aleksa Gordic)](https://www.aleksagordic.com/blog/matmul)
- [NVIDIA CUTLASS Efficient GEMM documentation](https://github.com/NVIDIA/cutlass/blob/main/media/docs/cpp/efficient_gemm.md)
- [CUTLASS SGEMM SM80 example](https://github.com/NVIDIA/cutlass/blob/main/examples/cute/tutorial/sgemm_sm80.cu)
- [CUTLASS Ampere SGEMM Python DSL example](https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/sgemm.py)
- [CUTLASS Tutorial: Efficient GEMM kernel designs with Pipelining (Colfax)](https://research.colfax-intl.com/cutlass-tutorial-design-of-a-gemm-kernel/)
- [NVIDIA Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/index.html)
- [CUDA Shared Memory Swizzling (Lei Mao)](https://leimao.github.io/blog/CUDA-Shared-Memory-Swizzling/)
- [CUDA Shared Memory Bank Conflicts (Axel Feldmann)](https://feldmann.nyc/blog/smem-microbenchmarks)
