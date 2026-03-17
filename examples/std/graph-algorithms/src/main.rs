// Graph algorithms on GPU — BFS, PageRank with async scheduling
//
// Task graph-bfs.1: CSR graph representation, RMAT generator, CPU BFS reference

use std::collections::VecDeque;
use std::time::Instant;

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
    let dist = cpu_bfs(&graph, source);
    let bfs_time = t0.elapsed();

    let reachable = dist.iter().filter(|&&d| d != u32::MAX).count();
    let max_depth = dist
        .iter()
        .filter(|&&d| d != u32::MAX)
        .copied()
        .max()
        .unwrap_or(0);

    println!("BFS results:");
    println!("  Reachable vertices: {reachable} / {num_vertices}");
    println!("  Max depth (diameter lower bound): {max_depth}");
    println!("  BFS time:  {:.2} ms", bfs_time.as_secs_f64() * 1000.0);

    // Print level histogram (first few levels).
    println!("\n  Level histogram (first 15 levels):");
    for level in 0..15.min(max_depth + 1) {
        let count = dist.iter().filter(|&&d| d == level).count();
        println!("    level {level:3}: {count:>8} vertices");
    }

    println!("\nDone. CPU BFS reference is ready for GPU verification.");
}
