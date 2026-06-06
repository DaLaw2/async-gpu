// Winograd F(2x2, 3x3) batched-GEMM pipeline: input/output transform kernels.
//
// Phase 1: winograd_input_transform
//   Extracts 4x4 tiles from input, applies B^T * d * B,
//   scatters into V[16, C_in, n_tiles] for cuBLAS strided batched GEMM.
//
// Phase 3: winograd_output_transform
//   Reads from M[16, C_out, n_tiles], applies A^T * M * A,
//   writes 2x2 output tiles to output tensor.
//   Optionally adds bias.
//
// Transform matrices for F(2,3):
//   B^T = [ 1  0 -1  0 ]    A^T = [ 1  1   1   0 ]
//         [ 0  1  1  0 ]          [ 0  1  -1  -1 ]
//         [ 0 -1  1  0 ]
//         [ 0  1  0 -1 ]

// Input transform: extract 4x4 tiles, apply B^T * d * B, write to V[16, C_in, n_tiles].
//
// Grid:  (ceil(n_tiles_per_sample * batch_size / 256), C_in, 1)
// Block: (256, 1, 1)
//
// Each thread processes one (tile, batch_sample) pair for one input channel.
// Total tiles across batch = n_tiles_per_sample * batch_size.
extern "C" __global__ void winograd_input_transform(
    const float* __restrict__ input,  // [N, C_in, H, W] or [C_in, H, W]
    float* __restrict__ V,            // [16, C_in, total_tiles]
    unsigned int C_in,
    unsigned int H, unsigned int W,
    unsigned int n_tile_x,   // tiles per row in output
    unsigned int n_tile_y,   // tiles per column in output
    unsigned int n_tiles_per_sample,  // n_tile_x * n_tile_y
    unsigned int batch_size,
    unsigned int padding
) {
    unsigned int ci = blockIdx.y;
    if (ci >= C_in) return;

    unsigned int global_tile = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int total_tiles = n_tiles_per_sample * batch_size;
    if (global_tile >= total_tiles) return;

    // Decode batch index and per-sample tile index
    unsigned int batch_idx = global_tile / n_tiles_per_sample;
    unsigned int tile_idx = global_tile % n_tiles_per_sample;
    unsigned int tile_y = tile_idx / n_tile_x;
    unsigned int tile_x = tile_idx % n_tile_x;

    // Input tile origin (4x4 patch)
    int in_y = (int)(tile_y * 2) - (int)padding;
    int in_x = (int)(tile_x * 2) - (int)padding;

    unsigned int in_offset = batch_idx * C_in * H * W + ci * H * W;

    // Load 4x4 input tile with boundary checking
    float d[4][4];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        #pragma unroll
        for (int c = 0; c < 4; c++) {
            int iy = in_y + r;
            int ix = in_x + c;
            if (iy >= 0 && iy < (int)H && ix >= 0 && ix < (int)W) {
                d[r][c] = input[in_offset + iy * W + ix];
            } else {
                d[r][c] = 0.0f;
            }
        }
    }

    // Input transform: U = B^T * d * B
    // First compute temp = B^T * d (4x4)
    float t[4][4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        t[0][c] = d[0][c] - d[2][c];
        t[1][c] = d[1][c] + d[2][c];
        t[2][c] = -d[1][c] + d[2][c];
        t[3][c] = d[1][c] - d[3][c];
    }

    // Then U = temp * B
    float u[16];
    #pragma unroll
    for (int r = 0; r < 4; r++) {
        u[r*4 + 0] = t[r][0] - t[r][2];
        u[r*4 + 1] = t[r][1] + t[r][2];
        u[r*4 + 2] = -t[r][1] + t[r][2];
        u[r*4 + 3] = t[r][1] - t[r][3];
    }

    // Scatter to V[16, C_in, total_tiles]
    // V[k][ci][global_tile] = u[k]
    unsigned int plane = C_in * total_tiles;
    unsigned int col = ci * total_tiles + global_tile;
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        V[k * plane + col] = u[k];
    }
}


// Output transform: read M[16, C_out, total_tiles], apply A^T * M * A,
// write 2x2 output tiles. Optionally adds bias.
//
// Grid:  (ceil(n_tiles_per_sample * batch_size / 256), C_out, 1)
// Block: (256, 1, 1)
//
// Each thread processes one (tile, batch_sample) pair for one output channel.
extern "C" __global__ void winograd_output_transform(
    const float* __restrict__ M,       // [16, C_out, total_tiles]
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

    // Decode batch index and per-sample tile index
    unsigned int batch_idx = global_tile / n_tiles_per_sample;
    unsigned int tile_idx = global_tile % n_tiles_per_sample;
    unsigned int tile_y = tile_idx / n_tile_x;
    unsigned int tile_x = tile_idx % n_tile_x;

    // Read 16 values from M[k][co][global_tile]
    unsigned int plane = C_out * total_tiles;
    unsigned int col = co * total_tiles + global_tile;

    float m[16];
    #pragma unroll
    for (int k = 0; k < 16; k++) {
        m[k] = M[k * plane + col];
    }

    // Output transform: Y = A^T * m_4x4 * A
    // m_4x4[r][c] = m[r*4+c]
    //
    // A^T = [ 1  1  1   0 ]
    //       [ 0  1 -1  -1 ]
    //
    // First compute temp = A^T * m_4x4 (2x4)
    float at_m[2][4];
    #pragma unroll
    for (int c = 0; c < 4; c++) {
        at_m[0][c] = m[0*4+c] + m[1*4+c] + m[2*4+c];
        at_m[1][c] = m[1*4+c] - m[2*4+c] - m[3*4+c];
    }

    // Then Y = temp * A (2x2)
    // A = [ 1   0 ]
    //     [ 1   1 ]
    //     [ 1  -1 ]
    //     [ 0  -1 ]
    float y00 = at_m[0][0] + at_m[0][1] + at_m[0][2];
    float y01 = at_m[0][1] - at_m[0][2] - at_m[0][3];
    float y10 = at_m[1][0] + at_m[1][1] + at_m[1][2];
    float y11 = at_m[1][1] - at_m[1][2] - at_m[1][3];

    // Add bias
    if (has_bias) {
        float b = bias[co];
        y00 += b;
        y01 += b;
        y10 += b;
        y11 += b;
    }

    // Write 2x2 output tile
    unsigned int out_y = tile_y * 2;
    unsigned int out_x = tile_x * 2;
    unsigned int out_base = batch_idx * C_out * H_out * W_out + co * H_out * W_out;

    if (out_y < H_out && out_x < W_out)
        output[out_base + out_y * W_out + out_x] = y00;
    if (out_y < H_out && out_x + 1 < W_out)
        output[out_base + out_y * W_out + out_x + 1] = y01;
    if (out_y + 1 < H_out && out_x < W_out)
        output[out_base + (out_y + 1) * W_out + out_x] = y10;
    if (out_y + 1 < H_out && out_x + 1 < W_out)
        output[out_base + (out_y + 1) * W_out + out_x + 1] = y11;
}
