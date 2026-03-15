//! Benchmark harness: result types, statistical helpers, and JSON output.

use std::fmt;

/// A single benchmark scenario result with all collected metrics.
#[derive(Clone)]
pub struct BenchmarkResult {
    /// Scenario name (e.g., "nop_latency_1thread").
    pub name: String,
    /// Number of GPU threads used.
    pub num_threads: u32,
    /// Number of iterations per thread.
    pub num_iters: u32,
    /// Number of hostcall packets in the pool.
    pub num_packets: u16,
    /// Grid dimension (blocks).
    pub grid_dim: u32,
    /// Block dimension (threads per block).
    pub block_dim: u32,
    /// Wall-clock time in milliseconds (host-side).
    pub wall_ms: f64,
    /// Total completed hostcalls across all threads.
    pub total_completed: u64,
    /// Total CAS retries across all threads.
    pub total_retries: u64,
    /// Per-iteration latencies in nanoseconds (if available).
    pub per_iter_latencies_ns: Vec<f64>,
    /// Per-thread average latencies in nanoseconds (fallback).
    pub per_thread_latencies_ns: Vec<f64>,
}

/// Summary statistics for a latency distribution.
#[derive(Clone)]
pub struct LatencyStats {
    pub count: usize,
    pub mean_ns: f64,
    pub stddev_ns: f64,
    pub min_ns: f64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    pub p99_ns: f64,
    pub p999_ns: f64,
    pub max_ns: f64,
}

impl BenchmarkResult {
    /// Compute aggregate throughput (hostcalls per second).
    pub fn throughput(&self) -> f64 {
        if self.wall_ms > 0.0 {
            self.total_completed as f64 / (self.wall_ms / 1000.0)
        } else {
            0.0
        }
    }

    /// Compute CAS retry rate (retries per completed call).
    pub fn cas_retry_rate(&self) -> f64 {
        if self.total_completed > 0 {
            self.total_retries as f64 / self.total_completed as f64
        } else {
            0.0
        }
    }

    /// Compute latency statistics from the best available data.
    /// Prefers per-iteration latencies; falls back to per-thread averages.
    pub fn latency_stats(&self) -> LatencyStats {
        let data = if !self.per_iter_latencies_ns.is_empty() {
            &self.per_iter_latencies_ns
        } else {
            &self.per_thread_latencies_ns
        };
        compute_stats(data)
    }

    /// Format as a concise one-line summary.
    pub fn summary_line(&self) -> String {
        let stats = self.latency_stats();
        format!(
            "{:<30} threads={:<5} iters={:<5} | \
             p50={:>8.0}ns p95={:>8.0}ns p99={:>8.0}ns mean={:>8.0}ns stddev={:>8.0}ns | \
             CAS/call={:.2} throughput={:.0}/s completed={}/{} wall={:.1}ms",
            self.name,
            self.num_threads,
            self.num_iters,
            stats.p50_ns,
            stats.p95_ns,
            stats.p99_ns,
            stats.mean_ns,
            stats.stddev_ns,
            self.cas_retry_rate(),
            self.throughput(),
            self.total_completed,
            self.num_threads as u64 * self.num_iters as u64,
            self.wall_ms,
        )
    }

    /// Serialize to JSON string for regression tracking.
    pub fn to_json(&self) -> String {
        let stats = self.latency_stats();
        format!(
            r#"{{"name":"{}","num_threads":{},"num_iters":{},"num_packets":{},"grid_dim":{},"block_dim":{},"wall_ms":{:.3},"total_completed":{},"total_retries":{},"throughput":{:.1},"cas_retry_rate":{:.4},"latency_ns":{{"count":{},"mean":{:.1},"stddev":{:.1},"min":{:.1},"p50":{:.1},"p95":{:.1},"p99":{:.1},"p999":{:.1},"max":{:.1}}}}}"#,
            self.name,
            self.num_threads,
            self.num_iters,
            self.num_packets,
            self.grid_dim,
            self.block_dim,
            self.wall_ms,
            self.total_completed,
            self.total_retries,
            self.throughput(),
            self.cas_retry_rate(),
            stats.count,
            stats.mean_ns,
            stats.stddev_ns,
            stats.min_ns,
            stats.p50_ns,
            stats.p95_ns,
            stats.p99_ns,
            stats.p999_ns,
            stats.max_ns,
        )
    }
}

impl fmt::Display for BenchmarkResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}", self.summary_line())
    }
}

/// Compute statistical summary from a slice of values.
pub fn compute_stats(data: &[f64]) -> LatencyStats {
    if data.is_empty() {
        return LatencyStats {
            count: 0,
            mean_ns: 0.0,
            stddev_ns: 0.0,
            min_ns: 0.0,
            p50_ns: 0.0,
            p95_ns: 0.0,
            p99_ns: 0.0,
            p999_ns: 0.0,
            max_ns: 0.0,
        };
    }

    let mut sorted = data.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let n = sorted.len();
    let mean = sorted.iter().sum::<f64>() / n as f64;
    let variance = if n > 1 {
        sorted.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / (n - 1) as f64
    } else {
        0.0
    };

    LatencyStats {
        count: n,
        mean_ns: mean,
        stddev_ns: variance.sqrt(),
        min_ns: sorted[0],
        p50_ns: percentile_sorted(&sorted, 50.0),
        p95_ns: percentile_sorted(&sorted, 95.0),
        p99_ns: percentile_sorted(&sorted, 99.0),
        p999_ns: percentile_sorted(&sorted, 99.9),
        max_ns: sorted[n - 1],
    }
}

/// Compute percentile from an already-sorted slice (nearest-rank method).
pub fn percentile_sorted(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let rank = (p / 100.0 * sorted.len() as f64).ceil() as usize;
    let idx = rank.saturating_sub(1).min(sorted.len() - 1);
    sorted[idx]
}

/// Write benchmark results to a JSON file.
pub fn write_results_json(
    path: &std::path::Path,
    results: &[BenchmarkResult],
) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "[")?;
    for (i, r) in results.iter().enumerate() {
        let comma = if i + 1 < results.len() { "," } else { "" };
        writeln!(f, "  {}{}", r.to_json(), comma)?;
    }
    writeln!(f, "]")?;
    Ok(())
}

/// Print a formatted benchmark report to stdout.
#[allow(dead_code)]
pub fn print_report(title: &str, results: &[BenchmarkResult]) {
    println!("\n--- {} ---", title);
    for r in results {
        println!("  {}", r.summary_line());
    }
}
