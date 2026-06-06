// Winograd F(4x4, 3x3) batched-GEMM pipeline: filter, input, and output transform kernels.
//
// F(4x4,3x3) produces 4x4 output tiles from 6x6 input tiles with 3x3 filters.
// This reduces tile count by 4x compared to F(2x2,3x3), making the GEMM matrices
// wider and more efficient for cuBLAS.
//
// Phase 0: winograd_filter_transform_f4x4
//   filter[C_out, C_in, 3, 3] -> U[36, C_out, C_in]
//   Transform: G * g * G^T where G is the 6x3 filter transform matrix
//
// Phase 1: winograd_input_transform_f4x4
//   Extracts 6x6 tiles from input, applies B^T * d * B,
//   scatters into V[36, C_in, n_tiles] for cuBLAS strided batched GEMM.
//
// Phase 3: winograd_output_transform_f4x4
//   Reads from M[36, C_out, n_tiles], applies A^T * M * A,
//   writes 4x4 output tiles to output tensor.
//   Optionally adds bias.
//
// Transform matrices for F(4,3):
//
//   B^T (6x6):
//     [  4   0  -5   0   1   0 ]
//     [  0  -4  -4   1   1   0 ]
//     [  0   4  -4  -1   1   0 ]
//     [  0  -2  -1   2   1   0 ]
//     [  0   2  -1  -2   1   0 ]
//     [  0   4   0  -5   0   1 ]
//
//   G (6x3):
//     [  1/4    0      0    ]
//     [ -1/6  -1/6   -1/6   ]
//     [ -1/6   1/6   -1/6   ]
//     [  1/24  1/12   1/6   ]
//     [  1/24 -1/12   1/6   ]
//     [  0      0      1    ]
//
//   A^T (4x6):
//     [  1   1   1   1   1   0 ]
//     [  0   1  -1   2  -2   0 ]
//     [  0   1   1   4   4   0 ]
//     [  0   1  -1   8  -8   1 ]

// Filter transform: filter[C_out, C_in, 3, 3] -> U[36, C_out, C_in]
// Transform: U = G * g * G^T
//
// Grid:  (ceil(C_out * C_in / 256), 1, 1)
// Block: (256, 1, 1)
extern "C" __global__ void winograd_filter_transform_f4x4(
    const float* __restrict__ filter,       // [C_out, C_in, 3, 3]
    float* __restrict__ filter_wino,        // [36, C_out, C_in]
    unsigned int C_out,
    unsigned int C_in
) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total = C_out * C_in;
    if (idx >= total) return;

    unsigned int co = idx / C_in;
    unsigned int ci = idx % C_in;

    // Load 3x3 filter
    const float* g = filter + (co * C_in + ci) * 9;
    float g0 = g[0], g1 = g[1], g2 = g[2];
    float g3 = g[3], g4 = g[4], g5 = g[5];
    float g6 = g[6], g7 = g[7], g8 = g[8];

    // Compute temp = G * g  (6x3)
    // G = [1/4, 0, 0; -1/6, -1/6, -1/6; -1/6, 1/6, -1/6; 1/24, 1/12, 1/6; 1/24, -1/12, 1/6; 0, 0, 1]
    float t[6][3];
    // Row 0: (1/4) * g_row0
    t[0][0] = g0 * 0.25f;    t[0][1] = g1 * 0.25f;    t[0][2] = g2 * 0.25f;
    // Row 1: (-1/6) * (g_row0 + g_row1 + g_row2)
    float s10 = g0 + g3 + g6, s11 = g1 + g4 + g7, s12 = g2 + g5 + g8;
    float inv6 = 1.0f / 6.0f;
    t[1][0] = -s10 * inv6; t[1][1] = -s11 * inv6; t[1][2] = -s12 * inv6;
    // Row 2: (-1/6) * (g_row0 - g_row1 + g_row2)
    float d10 = g0 - g3 + g6, d11 = g1 - g4 + g7, d12 = g2 - g5 + g8;
    t[2][0] = -d10 * inv6; t[2][1] = -d11 * inv6; t[2][2] = -d12 * inv6;
    // Row 3: (1/24)*g_row0 + (1/12)*g_row1 + (1/6)*g_row2
    float inv24 = 1.0f / 24.0f;
    float inv12 = 1.0f / 12.0f;
    t[3][0] = g0*inv24 + g3*inv12 + g6*inv6;
    t[3][1] = g1*inv24 + g4*inv12 + g7*inv6;
    t[3][2] = g2*inv24 + g5*inv12 + g8*inv6;
    // Row 4: (1/24)*g_row0 - (1/12)*g_row1 + (1/6)*g_row2
    t[4][0] = g0*inv24 - g3*inv12 + g6*inv6;
    t[4][1] = g1*inv24 - g4*inv12 + g7*inv6;
    t[4][2] = g2*inv24 - g5*inv12 + g8*inv6;
    // Row 5: g_row2
    t[5][0] = g6; t[5][1] = g7; t[5][2] = g8;

    // Compute U = temp * G^T  (6x6)
    // G^T columns are rows of G transposed
    float u[36];
    for (int r = 0; r < 6; r++) {
        float a = t[r][0], b = t[r][1], c = t[r][2];
        // Col 0: [1/4, -1/6, -1/6, 1/24, 1/24, 0]^T -> t[r][0]*1/4
        // Actually G^T[j][k] = G[k][j], so:
        // U[r][j] = sum_k t[r][k] * G^T[k][j] = sum_k t[r][k] * G[j][k]
        u[r*6 + 0] = a * 0.25f;
        u[r*6 + 1] = -(a + b + c) * inv6;
        u[r*6 + 2] = -(a - b + c) * inv6;
        u[r*6 + 3] = a*inv24 + b*inv12 + c*inv6;
        u[r*6 + 4] = a*inv24 - b*inv12 + c*inv6;
        u[r*6 + 5] = c;
    }

    // Store as [36, C_out, C_in]
    unsigned int oc_idx = co * C_in + ci;
    #pragma unroll
    for (int k = 0; k < 36; k++) {
        filter_wino[k * total + oc_idx] = u[k];
    }
}


// Input transform: extract 6x6 tiles, apply B^T * d * B, write to V[36, C_in, n_tiles].
//
// Grid:  (ceil(total_tiles / 256), C_in, 1)
// Block: (256, 1, 1)
//
// Each thread processes one (tile, batch_sample) pair for one input channel.
extern "C" __global__ void winograd_input_transform_f4x4(
    const float* __restrict__ input,  // [N, C_in, H, W] or [C_in, H, W]
    float* __restrict__ V,            // [36, C_in, total_tiles]
    unsigned int C_in,
    unsigned int H, unsigned int W,
    unsigned int n_tile_x,
    unsigned int n_tile_y,
    unsigned int n_tiles_per_sample,
    unsigned int batch_size,
    unsigned int padding
) {
    unsigned int ci = blockIdx.y;
    if (ci >= C_in) return;

    unsigned int global_tile = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_tiles = n_tiles_per_sample * batch_size;
    if (global_tile >= total_tiles) return;

    unsigned int batch_idx = global_tile / n_tiles_per_sample;
    unsigned int tile_idx = global_tile % n_tiles_per_sample;
    unsigned int tile_y = tile_idx / n_tile_x;
    unsigned int tile_x = tile_idx % n_tile_x;

    // Input tile origin (6x6 patch covering this 4x4 output tile)
    int in_y = (int)(tile_y * 4) - (int)padding;
    int in_x = (int)(tile_x * 4) - (int)padding;

    unsigned int in_offset = batch_idx * C_in * H * W + ci * H * W;

    // Load 6x6 input tile with boundary checking
    float d[6][6];
    #pragma unroll
    for (int r = 0; r < 6; r++) {
        #pragma unroll
        for (int c = 0; c < 6; c++) {
            int iy = in_y + r;
            int ix = in_x + c;
            if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {
                d[r][c] = input[in_offset + iy * W + ix];
            } else {
                d[r][c] = 0.0f;
            }
        }
    }

    // B^T * d (6x6 result)
    // B^T:
    //   [  4   0  -5   0   1   0 ]
    //   [  0  -4  -4   1   1   0 ]
    //   [  0   4  -4  -1   1   0 ]
    //   [  0  -2  -1   2   1   0 ]
    //   [  0   2  -1  -2   1   0 ]
    //   [  0   4   0  -5   0   1 ]
    float t[6][6];
    #pragma unroll
    for (int c = 0; c < 6; c++) {
        float d0 = d[0][c], d1 = d[1][c], d2 = d[2][c];
        float d3 = d[3][c], d4 = d[4][c], d5 = d[5][c];
        t[0][c] =  4.0f*d0 - 5.0f*d2 + d4;
        t[1][c] = -4.0f*d1 - 4.0f*d2 + d3 + d4;
        t[2][c] =  4.0f*d1 - 4.0f*d2 - d3 + d4;
        t[3][c] = -2.0f*d1 -      d2 + 2.0f*d3 + d4;
        t[4][c] =  2.0f*d1 -      d2 - 2.0f*d3 + d4;
        t[5][c] =  4.0f*d1 - 5.0f*d3 + d5;
    }

    // u = t * B (6x6, B is columns of B^T transposed)
    // B:
    //   [  4   0   0   0   0   0 ]
    //   [  0  -4   4  -2   2   4 ]
    //   [ -5  -4  -4  -1  -1   0 ]
    //   [  0   1  -1   2  -2  -5 ]
    //   [  1   1   1   1   1   0 ]
    //   [  0   0   0   0   0   1 ]
    float u[36];
    #pragma unroll
    for (int r = 0; r < 6; r++) {
        float t0 = t[r][0], t1 = t[r][1], t2 = t[r][2];
        float t3 = t[r][3], t4 = t[r][4], t5 = t[r][5];
        u[r*6 + 0] =  4.0f*t0 - 5.0f*t2 + t4;
        u[r*6 + 1] = -4.0f*t1 - 4.0f*t2 + t3 + t4;
        u[r*6 + 2] =  4.0f*t1 - 4.0f*t2 - t3 + t4;
        u[r*6 + 3] = -2.0f*t1 -      t2 + 2.0f*t3 + t4;
        u[r*6 + 4] =  2.0f*t1 -      t2 - 2.0f*t3 + t4;
        u[r*6 + 5] =  4.0f*t1 - 5.0f*t3 + t5;
    }

    // Scatter to V[36, C_in, total_tiles]
    unsigned int plane = C_in * total_tiles;
    unsigned int col = ci * total_tiles + global_tile;
    #pragma unroll
    for (int k = 0; k < 36; k++) {
        V[k * plane + col] = u[k];
    }
}


// Output transform: read M[36, C_out, total_tiles], apply A^T * M * A,
// write 4x4 output tiles. Optionally adds bias.
//
// Grid:  (ceil(total_tiles / 256), C_out, 1)
// Block: (256, 1, 1)
//
// A^T (4x6):
//   [  1   1   1   1   1   0 ]
//   [  0   1  -1   2  -2   0 ]
//   [  0   1   1   4   4   0 ]
//   [  0   1  -1   8  -8   1 ]
extern "C" __global__ void winograd_output_transform_f4x4(
    const float* __restrict__ M,       // [36, C_out, total_tiles]
    const float* __restrict__ bias,    // [C_out] or nullptr
    float* __restrict__ output,        // [N, C_out, H_out, W_out]
    unsigned int C_out,
    unsigned int H_out, unsigned int W_out,
    unsigned int n_tile_x,
    unsigned int n_tile_y,
    unsigned int n_tiles_per_sample,
    unsigned int batch_size,
    unsigned int has_bias
) {
    unsigned int co = blockIdx.y;
    if (co >= C_out) return;

    unsigned int global_tile = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_tiles = n_tiles_per_sample * batch_size;
    if (global_tile >= total_tiles) return;

    unsigned int batch_idx = global_tile / n_tiles_per_sample;
    unsigned int tile_idx = global_tile % n_tiles_per_sample;
    unsigned int tile_y = tile_idx / n_tile_x;
    unsigned int tile_x = tile_idx % n_tile_x;

    // Read 36 values from M[k][co][global_tile]
    unsigned int plane = C_out * total_tiles;
    unsigned int col = co * total_tiles + global_tile;

    float m[36];
    #pragma unroll
    for (int k = 0; k < 36; k++) {
        m[k] = M[k * plane + col];
    }

    // m as 6x6 matrix: m[r*6+c]
    // First: temp = A^T * m  (4x6)
    // A^T:
    //   [  1   1   1   1   1   0 ]
    //   [  0   1  -1   2  -2   0 ]
    //   [  0   1   1   4   4   0 ]
    //   [  0   1  -1   8  -8   1 ]
    float at_m[4][6];
    #pragma unroll
    for (int c = 0; c < 6; c++) {
        float m0 = m[0*6+c], m1 = m[1*6+c], m2 = m[2*6+c];
        float m3 = m[3*6+c], m4 = m[4*6+c], m5 = m[5*6+c];
        float s12 = m1 + m2;
        float d12 = m1 - m2;
        float s34 = m3 + m4;
        float d34 = m3 - m4;
        at_m[0][c] = m0 + s12 + s34;
        at_m[1][c] = d12 + 2.0f * d34;
        at_m[2][c] = s12 + 4.0f * s34;
        at_m[3][c] = d12 + 8.0f * d34 + m5;
    }

    // Then Y = temp * A  (4x4)
    // A (6x4):
    //   [ 1   0   0   0 ]
    //   [ 1   1   1   1 ]
    //   [ 1  -1   1  -1 ]
    //   [ 1   2   4   8 ]
    //   [ 1  -2   4  -8 ]
    //   [ 0   0   0   1 ]
    float y[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        float t0 = at_m[r][0], t1 = at_m[r][1], t2 = at_m[r][2];
        float t3 = at_m[r][3], t4 = at_m[r][4], t5 = at_m[r][5];
        float s12 = t1 + t2;
        float d12 = t1 - t2;
        float s34 = t3 + t4;
        float d34 = t3 - t4;
        y[r][0] = t0 + s12 + s34;
        y[r][1] = d12 + 2.0f * d34;
        y[r][2] = s12 + 4.0f * s34;
        y[r][3] = d12 + 8.0f * d34 + t5;
    }

    // Add bias
    float b = 0.0f;
    if (has_bias) {
        b = bias[co];
    }

    // Write 4x4 output tile
    unsigned int out_y = tile_y * 4;
    unsigned int out_x = tile_x * 4;
    unsigned int out_base = batch_idx * C_out * H_out * W_out + co * H_out * W_out;

    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            unsigned int oy = out_y + r;
            unsigned int ox = out_x + c;
            if (oy < H_out && ox < W_out) {
                output[out_base + oy * W_out + ox] = y[r][c] + b;
            }
        }
    }
}
