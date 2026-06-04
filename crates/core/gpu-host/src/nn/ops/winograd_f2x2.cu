// Winograd F(2×2, 3×3) convolution kernel for NVRTC compilation.
//
// Algorithm overview (F(2,3) means 2×2 output from 3×3 filter on 4×4 input):
//   - Input tile d[4×4] → transformed U = BᵀdB  (16 elements)
//   - Filter g[3×3] → transformed V = GgGᵀ  (16 elements, precomputed once)
//   - Element-wise M = U ⊙ V  (16 multiplies)
//   - Output Y[2×2] = AᵀMA  (inverse transform)
//
// Transform matrices for F(2,3):
//   B^T = [ 1  0 -1  0 ]    G = [ 1    0    0  ]    A^T = [ 1  1   1   0 ]
//         [ 0  1  1  0 ]        [ 1/2  1/2  1/2]          [ 0  1  -1  -1 ]
//         [ 0 -1  1  0 ]        [ 1/2 -1/2  1/2]
//         [ 0  1  0 -1 ]        [ 0    0    1  ]
//
// Kernel design:
//   - One thread block per (output_tile_row, output_tile_col) position
//   - Block dim = (C_in_chunk, C_out_chunk) threads for parallelism
//   - Uses shared memory for input/filter transforms
//
// Supports both single-sample and batched modes:
//   - Single: input [C_in, H, W], output [C_out, H_out, W_out], batch_size=1
//   - Batched: input [N, C_in, H, W], output [N, C_out, H_out, W_out], batch_size=N
// In batched mode, grid.z encodes the batch index.

extern "C" __global__ void winograd_filter_transform(
    const float* __restrict__ filter,   // [C_out, C_in, 3, 3]
    float* __restrict__ filter_wino,    // [16, C_out, C_in]
    unsigned int C_out,
    unsigned int C_in
) {
    // G matrix (4×3) for F(2,3):
    // [ 1     0     0   ]
    // [ 1/2   1/2   1/2 ]
    // [ 1/2  -1/2   1/2 ]
    // [ 0     0     1   ]

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C_out * C_in;
    if (idx >= total) return;

    unsigned int co = idx / C_in;
    unsigned int ci = idx % C_in;

    // Load 3×3 filter
    const float* g = filter + (co * C_in + ci) * 9;
    float f00 = g[0], f01 = g[1], f02 = g[2];
    float f10 = g[3], f11 = g[4], f12 = g[5];
    float f20 = g[6], f21 = g[7], f22 = g[8];

    // Compute temp = G * g  (4×3 = G[4×3] * g[3×3])
    // Row 0: g[0]
    float t00 = f00, t01 = f01, t02 = f02;
    // Row 1: (g[0]+g[1]+g[2])/2
    float t10 = (f00 + f10 + f20) * 0.5f;
    float t11 = (f01 + f11 + f21) * 0.5f;
    float t12 = (f02 + f12 + f22) * 0.5f;
    // Row 2: (g[0]-g[1]+g[2])/2
    float t20 = (f00 - f10 + f20) * 0.5f;
    float t21 = (f01 - f11 + f21) * 0.5f;
    float t22 = (f02 - f12 + f22) * 0.5f;
    // Row 3: g[2]
    float t30 = f20, t31 = f21, t32 = f22;

    // Compute V = temp * G^T  (4×4 = temp[4×3] * G^T[3×4])
    // G^T col 0: [1, 1/2, 1/2, 0]
    // G^T col 1: [0, 1/2, -1/2, 0]
    // G^T col 2: [0, 1/2, 1/2, 1]
    // Wait, G^T is transpose of G, so:
    // G^T = [ 1    1/2   1/2   0 ]
    //       [ 0    1/2  -1/2   0 ]
    //       [ 0    1/2   1/2   1 ]
    // So temp * G^T means each row of temp multiplied by cols of G^T:
    // V[i][0] = t[i][0]*1   + t[i][1]*0   + t[i][2]*0   = t[i][0]
    // V[i][1] = t[i][0]*1/2 + t[i][1]*1/2 + t[i][2]*1/2 = (t[i][0]+t[i][1]+t[i][2])/2
    // V[i][2] = t[i][0]*1/2 - t[i][1]*1/2 + t[i][2]*1/2 = (t[i][0]-t[i][1]+t[i][2])/2
    // V[i][3] = t[i][0]*0   + t[i][1]*0   + t[i][2]*1   = t[i][2]

    float v[16];
    // Row 0
    v[0]  = t00;
    v[1]  = (t00 + t01 + t02) * 0.5f;
    v[2]  = (t00 - t01 + t02) * 0.5f;
    v[3]  = t02;
    // Row 1
    v[4]  = t10;
    v[5]  = (t10 + t11 + t12) * 0.5f;
    v[6]  = (t10 - t11 + t12) * 0.5f;
    v[7]  = t12;
    // Row 2
    v[8]  = t20;
    v[9]  = (t20 + t21 + t22) * 0.5f;
    v[10] = (t20 - t21 + t22) * 0.5f;
    v[11] = t22;
    // Row 3
    v[12] = t30;
    v[13] = (t30 + t31 + t32) * 0.5f;
    v[14] = (t30 - t31 + t32) * 0.5f;
    v[15] = t32;

    // Store as [16, C_out, C_in] layout for coalesced access during GEMM
    unsigned int oc_idx = co * C_in + ci;
    for (int k = 0; k < 16; k++) {
        filter_wino[k * total + oc_idx] = v[k];
    }
}

extern "C" __global__ void winograd_conv2d_f2x2(
    const float* __restrict__ input,         // [C_in, H, W] or [N, C_in, H, W]
    const float* __restrict__ filter_wino,   // [16, C_out, C_in] (pre-transformed)
    float* __restrict__ output,              // [C_out, H_out, W_out] or [N, C_out, H_out, W_out]
    unsigned int C_in,
    unsigned int C_out,
    unsigned int H,
    unsigned int W,
    unsigned int H_out,
    unsigned int W_out,
    unsigned int n_tile_x,   // number of output tiles in x (width) direction
    unsigned int n_tile_y,   // number of output tiles in y (height) direction
    unsigned int padding
) {
    // Grid: (n_tiles, C_out_blocks, batch_size)
    // Block: (TILE_C_OUT, 1, 1) where TILE_C_OUT threads each handle one c_out
    //
    // Each thread computes one 2×2 output tile for one output channel and one batch sample.
    // The thread loops over all C_in channels, accumulating in Winograd domain.
    // For single-sample mode, grid.z = 1. For batched mode, grid.z = N.

    const unsigned int TILE_C_OUT = 32;

    unsigned int tile_idx = blockIdx.x;  // which spatial tile
    unsigned int co_block = blockIdx.y;  // which block of output channels
    unsigned int batch_idx = blockIdx.z; // which batch sample
    unsigned int co_local = threadIdx.x; // thread within block

    unsigned int co = co_block * TILE_C_OUT + co_local;
    if (co >= C_out) return;

    unsigned int total_tiles = n_tile_x * n_tile_y;
    if (tile_idx >= total_tiles) return;

    unsigned int tile_y = tile_idx / n_tile_x;
    unsigned int tile_x = tile_idx % n_tile_x;

    // Output tile origin in output space
    unsigned int out_y = tile_y * 2;
    unsigned int out_x = tile_x * 2;

    // Input tile origin (4×4 patch covering this 2×2 output)
    int in_y = (int)(out_y) - (int)padding;
    int in_x = (int)(out_x) - (int)padding;

    // Per-sample offsets into input/output arrays
    unsigned int in_sample_offset = batch_idx * C_in * H * W;
    unsigned int out_sample_offset = batch_idx * C_out * H_out * W_out;

    // Each thread accumulates 16 Winograd-domain values
    float m[16];
    for (int k = 0; k < 16; k++) m[k] = 0.0f;

    unsigned int filter_plane = C_out * C_in;

    for (unsigned int ci = 0; ci < C_in; ci++) {
        // Load 4×4 input tile with boundary checking
        float d[4][4];
        for (int r = 0; r < 4; r++) {
            for (int c = 0; c < 4; c++) {
                int iy = in_y + r;
                int ix = in_x + c;
                if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {
                    d[r][c] = input[in_sample_offset + ci * H * W + iy * W + ix];
                } else {
                    d[r][c] = 0.0f;
                }
            }
        }

        // Input transform: U = B^T * d * B
        // B^T = [ 1  0 -1  0 ]
        //       [ 0  1  1  0 ]
        //       [ 0 -1  1  0 ]
        //       [ 0  1  0 -1 ]
        //
        // First compute temp = B^T * d (4×4)
        float t[4][4];
        for (int c = 0; c < 4; c++) {
            t[0][c] = d[0][c] - d[2][c];
            t[1][c] = d[1][c] + d[2][c];
            t[2][c] = -d[1][c] + d[2][c];
            t[3][c] = d[1][c] - d[3][c];
        }

        // Then U = temp * B (B is transpose of B^T columns → rows)
        // B = [ 1   0   0   0 ]
        //     [ 0   1  -1   1 ]
        //     [-1   1   1   0 ]
        //     [ 0   0   0  -1 ]
        float u[16];
        for (int r = 0; r < 4; r++) {
            u[r*4 + 0] = t[r][0] - t[r][2];
            u[r*4 + 1] = t[r][1] + t[r][2];
            u[r*4 + 2] = -t[r][1] + t[r][2];
            u[r*4 + 3] = t[r][1] - t[r][3];
        }

        // Element-wise multiply with pre-transformed filter and accumulate
        // filter_wino layout: [16, C_out, C_in]
        unsigned int filt_base = co * C_in + ci;
        for (int k = 0; k < 16; k++) {
            m[k] += u[k] * filter_wino[k * filter_plane + filt_base];
        }
    }

    // Output transform: Y = A^T * M * A
    // A^T = [ 1  1  1   0 ]
    //       [ 0  1 -1  -1 ]
    //
    // First compute temp = A^T * M (2×4)
    // M is stored as m[r*4+c] for r in 0..4, c in 0..4
    float at_m[2][4];
    for (int c = 0; c < 4; c++) {
        at_m[0][c] = m[0*4+c] + m[1*4+c] + m[2*4+c];
        at_m[1][c] = m[1*4+c] - m[2*4+c] - m[3*4+c];
    }

    // Then Y = temp * A (2×2)
    // A = [ 1   0 ]
    //     [ 1   1 ]
    //     [ 1  -1 ]
    //     [ 0  -1 ]
    float y00 = at_m[0][0] + at_m[0][1] + at_m[0][2];
    float y01 = at_m[0][1] - at_m[0][2] - at_m[0][3];
    float y10 = at_m[1][0] + at_m[1][1] + at_m[1][2];
    float y11 = at_m[1][1] - at_m[1][2] - at_m[1][3];

    // Write 2×2 output tile (offset by batch sample)
    unsigned int out_base = out_sample_offset + co * H_out * W_out;
    if (out_y < H_out && out_x < W_out)
        output[out_base + out_y * W_out + out_x] = y00;
    if (out_y < H_out && out_x + 1 < W_out)
        output[out_base + out_y * W_out + out_x + 1] = y01;
    if (out_y + 1 < H_out && out_x < W_out)
        output[out_base + (out_y + 1) * W_out + out_x] = y10;
    if (out_y + 1 < H_out && out_x + 1 < W_out)
        output[out_base + (out_y + 1) * W_out + out_x + 1] = y11;
}
