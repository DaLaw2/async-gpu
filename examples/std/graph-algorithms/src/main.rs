// Graph algorithms on GPU — BFS, PageRank with async scheduling
//
// Task graph-bfs.1: CSR graph representation, RMAT generator, CPU BFS reference
// Task graph-bfs.2: GPU BFS (level-synchronous) + CPU reference comparison

use std::collections::VecDeque;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::{compile_ptx, Ptx};

// ---------------------------------------------------------------------------
// Compressed Sparse Row (CSR) graph
// ---------------------------------------------------------------------------

/// CSR (Compressed Sparse Row) graph representation.
///
/// For vertex `v`, its neighbors are stored in
/// `col_idx[row_ptr[v] .. row_ptr[v+1]]`.
struct CsrGraph {
    num_vertices: u32,
    /// Length = num_vertices + 1. row_ptr[v] is the start offset in col_idx
    /// for vertex v; row_ptr[num_vertices] == col_idx.len().
    row_ptr: Vec<u32>,
    /// Destination vertex for each edge.
    col_idx: Vec<u32>,
}

impl CsrGraph {
    /// Build a CSR graph from a directed edge list.
    ///
    /// Edges are (src, dst) pairs. Self-loops are removed. Duplicate edges
    /// are removed. The resulting adjacency lists are sorted.
    fn from_edges(num_vertices: u32, edges: &[(u32, u32)]) -> Self {
        // Count degree per vertex (after filtering self-loops).
        let mut adj: Vec<Vec<u32>> = vec![Vec::new(); num_vertices as usize];
        for &(src, dst) in edges {
            if src == dst {
                continue; // skip self-loops
            }
            if src < num_vertices && dst < num_vertices {
                adj[src as usize].push(dst);
            }
        }

        // Sort + deduplicate each adjacency list.
        for list in &mut adj {
            list.sort_unstable();
            list.dedup();
        }

        // Build row_ptr and col_idx.
        let mut row_ptr = Vec::with_capacity(num_vertices as usize + 1);
        let mut col_idx = Vec::new();
        let mut offset: u32 = 0;
        for list in &adj {
            row_ptr.push(offset);
            col_idx.extend_from_slice(list);
            offset += list.len() as u32;
        }
        row_ptr.push(offset);

        Self {
            num_vertices,
            row_ptr,
            col_idx,
        }
    }

    /// Number of directed edges in the graph.
    fn num_edges(&self) -> u32 {
        *self.row_ptr.last().unwrap()
    }

    /// Degree of vertex `v`.
    fn degree(&self, v: u32) -> u32 {
        self.row_ptr[v as usize + 1] - self.row_ptr[v as usize]
    }

    /// Neighbors of vertex `v`.
    fn neighbors(&self, v: u32) -> &[u32] {
        let start = self.row_ptr[v as usize] as usize;
        let end = self.row_ptr[v as usize + 1] as usize;
        &self.col_idx[start..end]
    }

    /// Average degree (edges / vertices).
    fn avg_degree(&self) -> f64 {
        self.num_edges() as f64 / self.num_vertices as f64
    }
}

// ---------------------------------------------------------------------------
// Xorshift64 RNG — simple, no external deps
// ---------------------------------------------------------------------------

struct Xorshift64 {
    state: u64,
}

impl Xorshift64 {
    fn new(seed: u64) -> Self {
        // Ensure non-zero state.
        Self {
            state: if seed == 0 {
                0xDEAD_BEEF_CAFE_1234
            } else {
                seed
            },
        }
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Returns a float in [0, 1).
    fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / ((1u64 << 53) as f64)
    }
}

// ---------------------------------------------------------------------------
// RMAT synthetic graph generator
// ---------------------------------------------------------------------------

/// Generate an RMAT (Recursive MATrix) graph.
///
/// - `scale`: number of vertices = 2^scale
/// - `edge_factor`: target edges = edge_factor * num_vertices
/// - Standard Kronecker probabilities: a=0.57, b=0.19, c=0.19, d=0.05
///
/// Returns a list of directed (src, dst) edges before dedup/self-loop removal.
fn rmat_generate(scale: u32, edge_factor: u32, seed: u64) -> Vec<(u32, u32)> {
    let num_vertices: u32 = 1 << scale;
    let num_edges = (edge_factor as u64) * (num_vertices as u64);

    let a: f64 = 0.57;
    let b: f64 = 0.19;
    let c: f64 = 0.19;
    // d = 0.05 (implicit: 1 - a - b - c)

    let ab = a + b;
    let abc = a + b + c;

    let mut rng = Xorshift64::new(seed);
    let mut edges = Vec::with_capacity(num_edges as usize);

    for _ in 0..num_edges {
        let mut src: u32 = 0;
        let mut dst: u32 = 0;

        for depth in 0..scale {
            let r = rng.next_f64();
            let bit = 1u32 << (scale - 1 - depth);

            if r < a {
                // quadrant (0,0) — no change
            } else if r < ab {
                // quadrant (0,1)
                dst |= bit;
            } else if r < abc {
                // quadrant (1,0)
                src |= bit;
            } else {
                // quadrant (1,1)
                src |= bit;
                dst |= bit;
            }
        }

        edges.push((src, dst));
    }

    edges
}

// ---------------------------------------------------------------------------
// CPU BFS reference — level-synchronous
// ---------------------------------------------------------------------------

/// Level-synchronous BFS from `source`.
///
/// Returns a distance array where `dist[v]` is the shortest hop-count from
/// `source` to `v`, or `u32::MAX` if `v` is unreachable.
fn cpu_bfs(graph: &CsrGraph, source: u32) -> Vec<u32> {
    let n = graph.num_vertices as usize;
    let mut dist = vec![u32::MAX; n];
    dist[source as usize] = 0;

    let mut queue = VecDeque::new();
    queue.push_back(source);

    while let Some(v) = queue.pop_front() {
        let d = dist[v as usize];
        for &w in graph.neighbors(v) {
            if dist[w as usize] == u32::MAX {
                dist[w as usize] = d + 1;
                queue.push_back(w);
            }
        }
    }

    dist
}

// ---------------------------------------------------------------------------
// GPU BFS — level-synchronous with CUDA kernel
// ---------------------------------------------------------------------------

/// CUDA C kernel for level-synchronous BFS.
///
/// Each thread processes one vertex. If the vertex's distance equals the
/// current level, it explores all neighbors and atomically sets their
/// distance to (level + 1) if they are unvisited (distance == 0xFFFFFFFF).
/// A global counter tracks how many new vertices were discovered.
const BFS_KERNEL_SRC: &str = r#"
extern "C" __global__ void bfs_level_expand(
    const unsigned int* __restrict__ row_ptr,
    const unsigned int* __restrict__ col_idx,
    unsigned int* __restrict__ dist,
    unsigned int num_vertices,
    unsigned int current_level,
    unsigned int* __restrict__ frontier_count
) {
    unsigned int v = blockIdx.x * blockDim.x + threadIdx.x;
    if (v >= num_vertices) return;

    // Only process vertices at the current frontier level.
    if (dist[v] != current_level) return;

    unsigned int start = row_ptr[v];
    unsigned int end   = row_ptr[v + 1];
    unsigned int next_level = current_level + 1;

    for (unsigned int e = start; e < end; e++) {
        unsigned int neighbor = col_idx[e];
        // atomicCAS: if dist[neighbor] == 0xFFFFFFFF, set to next_level.
        unsigned int old = atomicCAS(&dist[neighbor], 0xFFFFFFFFu, next_level);
        if (old == 0xFFFFFFFFu) {
            atomicAdd(frontier_count, 1u);
        }
    }
}
"#;

/// Run level-synchronous BFS on GPU using cudarc.
///
/// Uploads CSR graph to device, launches one kernel per BFS level,
/// reads back results. Returns the distance array.
fn gpu_bfs(
    graph: &CsrGraph,
    source: u32,
    dev: &Arc<CudaDevice>,
    ptx: &Ptx,
) -> Result<Vec<u32>, Box<dyn std::error::Error>> {
    let n = graph.num_vertices as usize;

    // Load the BFS kernel.
    dev.load_ptx(ptx.clone(), "bfs", &["bfs_level_expand"])?;
    let func_name = "bfs_level_expand";

    // Upload CSR arrays.
    let d_row_ptr = dev.htod_sync_copy(&graph.row_ptr)?;
    let d_col_idx = dev.htod_sync_copy(&graph.col_idx)?;

    // Initialize distance array: all u32::MAX, then set source = 0.
    let mut host_dist = vec![u32::MAX; n];
    host_dist[source as usize] = 0;
    let mut d_dist = dev.htod_sync_copy(&host_dist)?;

    // Frontier counter (single u32 on device).
    let mut d_frontier_count = dev.htod_sync_copy(&[0u32])?;

    // Launch config: 256 threads per block, enough blocks to cover all vertices.
    let block_size = 256u32;
    let grid_size = (graph.num_vertices + block_size - 1) / block_size;

    let mut current_level: u32 = 0;
    let mut total_discovered: u64 = 1; // source vertex

    loop {
        // Reset frontier counter to 0.
        dev.htod_sync_copy_into(&[0u32], &mut d_frontier_count)?;

        // Launch kernel for this level.
        let func = dev
            .get_func("bfs", func_name)
            .ok_or("BFS kernel function not found")?;
        let cfg = LaunchConfig {
            grid_dim: (grid_size, 1, 1),
            block_dim: (block_size, 1, 1),
            shared_mem_bytes: 0,
        };
        unsafe {
            func.launch(
                cfg,
                (
                    &d_row_ptr,
                    &d_col_idx,
                    &mut d_dist,
                    graph.num_vertices,
                    current_level,
                    &mut d_frontier_count,
                ),
            )?;
        }
        dev.synchronize()?;

        // Read back frontier count.
        let frontier_count = dev.dtoh_sync_copy(&d_frontier_count)?;
        let discovered = frontier_count[0] as u64;

        if discovered == 0 {
            break; // No new vertices discovered — BFS complete.
        }

        total_discovered += discovered;
        current_level += 1;
    }

    // Read back final distance array.
    let result = dev.dtoh_sync_copy(&d_dist)?;
    let _ = total_discovered; // suppress unused warning in non-verbose mode
    Ok(result)
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    let scale: u32 = 17; // 2^17 = 131072 vertices
    let edge_factor: u32 = 16;
    let num_vertices: u32 = 1 << scale;
    let seed: u64 = 42;

    // -- Generate RMAT graph --------------------------------------------------
    println!("Generating RMAT graph: scale={scale}, edge_factor={edge_factor}");
    let t0 = Instant::now();
    let edges = rmat_generate(scale, edge_factor, seed);
    let gen_time = t0.elapsed();
    println!(
        "  Raw edges generated: {} ({:.2} ms)",
        edges.len(),
        gen_time.as_secs_f64() * 1000.0
    );

    // -- Build CSR ------------------------------------------------------------
    let t0 = Instant::now();
    let graph = CsrGraph::from_edges(num_vertices, &edges);
    let build_time = t0.elapsed();

    println!("\nGraph statistics:");
    println!("  Vertices:   {}", graph.num_vertices);
    println!(
        "  Edges:      {} (after dedup + self-loop removal)",
        graph.num_edges()
    );
    println!("  Avg degree: {:.2}", graph.avg_degree());

    // Compute max degree for interest.
    let max_deg = (0..graph.num_vertices)
        .map(|v| graph.degree(v))
        .max()
        .unwrap_or(0);
    println!("  Max degree: {max_deg}");
    println!("  CSR build:  {:.2} ms", build_time.as_secs_f64() * 1000.0);

    // -- CPU BFS --------------------------------------------------------------
    let source: u32 = 0;
    println!("\nRunning CPU BFS from vertex {source}...");
    let t0 = Instant::now();
    let cpu_dist = cpu_bfs(&graph, source);
    let cpu_bfs_time = t0.elapsed();

    let reachable = cpu_dist.iter().filter(|&&d| d != u32::MAX).count();
    let max_depth = cpu_dist
        .iter()
        .filter(|&&d| d != u32::MAX)
        .copied()
        .max()
        .unwrap_or(0);

    println!("CPU BFS results:");
    println!("  Reachable vertices: {reachable} / {num_vertices}");
    println!("  Max depth (diameter lower bound): {max_depth}");
    println!("  BFS time:  {:.2} ms", cpu_bfs_time.as_secs_f64() * 1000.0);

    // Print level histogram (first few levels).
    println!("\n  Level histogram (first 15 levels):");
    for level in 0..15.min(max_depth + 1) {
        let count = cpu_dist.iter().filter(|&&d| d == level).count();
        println!("    level {level:3}: {count:>8} vertices");
    }

    // -- GPU BFS --------------------------------------------------------------
    println!("\n--- GPU BFS (level-synchronous, CUDA kernel) ---");

    // Initialize CUDA device and compile BFS kernel.
    let dev = match CudaDevice::new(0) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Failed to initialize CUDA device: {e}");
            eprintln!("Skipping GPU BFS.");
            return;
        }
    };

    println!("Compiling BFS CUDA kernel via NVRTC...");
    let t0 = Instant::now();
    let ptx = match compile_ptx(BFS_KERNEL_SRC) {
        Ok(p) => p,
        Err(e) => {
            eprintln!("Failed to compile BFS kernel: {e}");
            eprintln!("Skipping GPU BFS.");
            return;
        }
    };
    let compile_time = t0.elapsed();
    println!(
        "  Kernel compiled in {:.2} ms",
        compile_time.as_secs_f64() * 1000.0
    );

    // Warmup run (first launch has JIT overhead).
    println!("Running GPU BFS warmup...");
    let _ = gpu_bfs(&graph, source, &dev, &ptx);

    // Timed run.
    println!("Running GPU BFS from vertex {source}...");
    let t0 = Instant::now();
    let gpu_dist = match gpu_bfs(&graph, source, &dev, &ptx) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("GPU BFS failed: {e}");
            return;
        }
    };
    let gpu_bfs_time = t0.elapsed();

    let gpu_reachable = gpu_dist.iter().filter(|&&d| d != u32::MAX).count();
    let gpu_max_depth = gpu_dist
        .iter()
        .filter(|&&d| d != u32::MAX)
        .copied()
        .max()
        .unwrap_or(0);

    println!("GPU BFS results:");
    println!("  Reachable vertices: {gpu_reachable} / {num_vertices}");
    println!("  Max depth: {gpu_max_depth}");
    println!("  BFS time:  {:.2} ms", gpu_bfs_time.as_secs_f64() * 1000.0);

    // -- Verify GPU vs CPU ----------------------------------------------------
    println!("\n--- Verification: GPU vs CPU ---");
    let mut mismatches = 0u64;
    let mut first_mismatch: Option<(u32, u32, u32)> = None;
    for v in 0..num_vertices as usize {
        if cpu_dist[v] != gpu_dist[v] {
            mismatches += 1;
            if first_mismatch.is_none() {
                first_mismatch = Some((v as u32, cpu_dist[v], gpu_dist[v]));
            }
        }
    }

    if mismatches == 0 {
        println!("  PASS: GPU BFS matches CPU BFS for all {num_vertices} vertices");
    } else {
        println!("  FAIL: {mismatches} mismatches out of {num_vertices} vertices");
        if let Some((v, cpu_d, gpu_d)) = first_mismatch {
            println!("  First mismatch: vertex {v}, CPU dist={cpu_d}, GPU dist={gpu_d}");
        }
    }

    // -- Timing comparison ----------------------------------------------------
    let cpu_ms = cpu_bfs_time.as_secs_f64() * 1000.0;
    let gpu_ms = gpu_bfs_time.as_secs_f64() * 1000.0;
    let speedup = cpu_ms / gpu_ms;

    println!("\n--- Timing Summary ---");
    println!("  CPU BFS: {cpu_ms:.2} ms");
    println!("  GPU BFS: {gpu_ms:.2} ms (includes {max_depth} kernel launches + sync)");
    println!("  Speedup: {speedup:.2}x");

    if speedup < 1.0 {
        println!(
            "  Note: GPU is slower due to kernel launch overhead ({} levels).",
            max_depth
        );
        println!("  GPU BFS shines on larger graphs (scale >= 20) with fewer levels.");
    }

    // -- CSR data round-trip verification -------------------------------------
    println!("\n--- CSR GPU Round-Trip Verification ---");
    let d_row_ptr = dev.htod_sync_copy(&graph.row_ptr).unwrap();
    let d_col_idx = dev.htod_sync_copy(&graph.col_idx).unwrap();
    let rt_row_ptr: Vec<u32> = dev.dtoh_sync_copy(&d_row_ptr).unwrap();
    let rt_col_idx: Vec<u32> = dev.dtoh_sync_copy(&d_col_idx).unwrap();
    let row_ptr_ok = rt_row_ptr == graph.row_ptr;
    let col_idx_ok = rt_col_idx == graph.col_idx;
    println!(
        "  row_ptr round-trip: {} ({} elements)",
        if row_ptr_ok { "PASS" } else { "FAIL" },
        graph.row_ptr.len()
    );
    println!(
        "  col_idx round-trip: {} ({} elements)",
        if col_idx_ok { "PASS" } else { "FAIL" },
        graph.col_idx.len()
    );

    println!("\nDone.");
}
