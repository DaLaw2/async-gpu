//! Auto-tuning framework for GPU kernel launch parameter selection.
//!
//! The auto-tuner finds optimal launch configurations by running warmup-based
//! parameter search across candidate block sizes, measuring wall-clock timing,
//! and caching the best result per (kernel, problem-size bucket, device).
//!
//! # Architecture
//!
//! - [`TuningKey`] — composite key: (kernel name, problem-size bucket, device ID)
//! - [`TuningCache`] — thread-safe cache mapping keys to best [`LaunchConfig`]
//! - [`AutoTuner`] — orchestrates candidate generation, benchmark, and selection
//!
//! # Example
//!
//! ```no_run
//! use gpu_host::auto_tune::{AutoTuner, TuningCache};
//!
//! let cache = TuningCache::new();
//! let tuner = AutoTuner::new();
//!
//! // Tune block size for a kernel processing 65536 elements
//! let best = tuner.tune_block_size(
//!     65536,            // problem size (elements)
//!     None,             // optional kernel resource info for occupancy filtering
//!     &|block_size| {   // benchmark closure: launch kernel with given block size, return elapsed
//!         // ... launch kernel, synchronize, measure time ...
//!         Some(std::time::Duration::from_micros(100))
//!     },
//! );
//!
//! // Cache the result for future calls
//! if let Some(result) = best {
//!     cache.insert_config("vector_add", 65536, 0, result);
//! }
//! ```

use std::collections::HashMap;
use std::fmt;
use std::sync::Mutex;
use std::time::Duration;

use cudarc::driver::LaunchConfig;

use crate::resource_report::{KernelResources, SmConfig};

// ============================================================================
// TuningKey — composite cache key
// ============================================================================

/// Composite key for the tuning cache.
///
/// Encodes kernel identity, problem-size bucket, and device ordinal so that
/// different problem sizes on different GPUs get independent tuning results.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct TuningKey {
    /// Kernel function name (e.g., "vector_add").
    pub kernel_name: String,
    /// Problem-size bucket (power-of-2 rounding of actual size).
    pub size_bucket: u64,
    /// CUDA device ordinal.
    pub device_id: u32,
}

impl TuningKey {
    /// Create a new tuning key, automatically bucketing the problem size.
    pub fn new(kernel_name: &str, problem_size: u64, device_id: u32) -> Self {
        Self {
            kernel_name: kernel_name.to_string(),
            size_bucket: bucket_size(problem_size),
            device_id,
        }
    }
}

impl fmt::Display for TuningKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "({}, bucket={}, dev={})",
            self.kernel_name, self.size_bucket, self.device_id
        )
    }
}

/// Bucket a problem size to the nearest power of 2.
///
/// This groups similar problem sizes together so that a kernel tuned for
/// N=60000 also covers N=65536 without re-tuning.
///
/// Returns the smallest power of 2 >= `size`, clamped to at least 32.
pub fn bucket_size(size: u64) -> u64 {
    if size <= 32 {
        return 32;
    }
    size.next_power_of_two()
}

// ============================================================================
// TuningResult — what we cache per key
// ============================================================================

/// Result of an auto-tuning search for a single key.
#[derive(Debug, Clone)]
pub struct TuningResult {
    /// The best launch configuration found.
    pub config: LaunchConfig,
    /// Wall-clock median duration of the best configuration.
    pub best_time: Duration,
    /// Number of candidates evaluated.
    pub candidates_tested: usize,
    /// All candidate results (block_size -> median time).
    pub all_results: Vec<(u32, Duration)>,
}

impl fmt::Display for TuningResult {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "best block_dim=({},{},{}), grid=({},{},{}), time={:?} ({} candidates)",
            self.config.block_dim.0,
            self.config.block_dim.1,
            self.config.block_dim.2,
            self.config.grid_dim.0,
            self.config.grid_dim.1,
            self.config.grid_dim.2,
            self.best_time,
            self.candidates_tested,
        )
    }
}

// ============================================================================
// TuningCache — thread-safe result cache
// ============================================================================

/// Thread-safe cache of tuning results.
///
/// Stores the best [`LaunchConfig`] per [`TuningKey`]. Can be shared across
/// threads via `Arc<TuningCache>`.
pub struct TuningCache {
    entries: Mutex<HashMap<TuningKey, TuningResult>>,
}

impl TuningCache {
    /// Create an empty cache.
    pub fn new() -> Self {
        Self {
            entries: Mutex::new(HashMap::new()),
        }
    }

    /// Look up a cached tuning result.
    pub fn get(&self, key: &TuningKey) -> Option<TuningResult> {
        self.entries.lock().unwrap().get(key).cloned()
    }

    /// Look up by components (convenience wrapper).
    pub fn get_config(
        &self,
        kernel_name: &str,
        problem_size: u64,
        device_id: u32,
    ) -> Option<TuningResult> {
        let key = TuningKey::new(kernel_name, problem_size, device_id);
        self.get(&key)
    }

    /// Insert a tuning result.
    pub fn insert(&self, key: TuningKey, result: TuningResult) {
        self.entries.lock().unwrap().insert(key, result);
    }

    /// Insert by components (convenience wrapper).
    pub fn insert_config(
        &self,
        kernel_name: &str,
        problem_size: u64,
        device_id: u32,
        result: TuningResult,
    ) {
        let key = TuningKey::new(kernel_name, problem_size, device_id);
        self.insert(key, result);
    }

    /// Number of cached entries.
    pub fn len(&self) -> usize {
        self.entries.lock().unwrap().len()
    }

    /// Whether the cache is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all cached entries.
    pub fn clear(&self) {
        self.entries.lock().unwrap().clear();
    }

    /// List all cached keys (for diagnostics).
    pub fn keys(&self) -> Vec<TuningKey> {
        self.entries.lock().unwrap().keys().cloned().collect()
    }
}

impl Default for TuningCache {
    fn default() -> Self {
        Self::new()
    }
}

impl fmt::Debug for TuningCache {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let entries = self.entries.lock().unwrap();
        f.debug_struct("TuningCache")
            .field("entries", &entries.len())
            .finish()
    }
}

// ============================================================================
// AutoTuner — the tuning engine
// ============================================================================

/// Configuration for the auto-tuning search.
#[derive(Debug, Clone)]
pub struct AutoTunerConfig {
    /// Candidate block sizes to evaluate.
    /// Default: [32, 64, 128, 256, 512, 1024].
    pub candidate_block_sizes: Vec<u32>,
    /// Number of warmup iterations before measurement.
    /// Default: 3.
    pub warmup_iterations: u32,
    /// Number of measurement iterations (median is taken).
    /// Default: 7.
    pub measure_iterations: u32,
    /// Minimum occupancy percentage to consider a candidate viable.
    /// Default: 12 (12.5% rounded down).
    pub min_occupancy_pct: u32,
    /// SM configuration for occupancy filtering.
    /// Default: sm_75 (GTX 1660 / RTX 20xx).
    pub sm_config: SmConfig,
}

impl Default for AutoTunerConfig {
    fn default() -> Self {
        Self {
            candidate_block_sizes: vec![32, 64, 128, 256, 512, 1024],
            warmup_iterations: 3,
            measure_iterations: 7,
            min_occupancy_pct: 12,
            sm_config: SmConfig::sm_75(),
        }
    }
}

/// Auto-tuner engine for GPU kernel launch parameters.
///
/// Given a kernel and problem size, evaluates multiple block-size candidates
/// using warmup + timed runs, and returns the configuration with the lowest
/// median execution time.
pub struct AutoTuner {
    config: AutoTunerConfig,
}

impl AutoTuner {
    /// Create an auto-tuner with default configuration.
    pub fn new() -> Self {
        Self {
            config: AutoTunerConfig::default(),
        }
    }

    /// Create an auto-tuner with custom configuration.
    pub fn with_config(config: AutoTunerConfig) -> Self {
        Self { config }
    }

    /// Get the tuner configuration.
    pub fn config(&self) -> &AutoTunerConfig {
        &self.config
    }

    /// Generate candidate block sizes, filtered by occupancy constraints.
    ///
    /// If `kernel_resources` is provided, candidates that would yield occupancy
    /// below `min_occupancy_pct` are rejected.
    pub fn generate_candidates(&self, kernel_resources: Option<&KernelResources>) -> Vec<u32> {
        self.config
            .candidate_block_sizes
            .iter()
            .copied()
            .filter(|&bs| {
                if let Some(kr) = kernel_resources {
                    let occ = kr.occupancy(&self.config.sm_config, bs);
                    occ >= self.config.min_occupancy_pct
                } else {
                    true // No resource info — keep all candidates
                }
            })
            .collect()
    }

    /// Run the auto-tuning search for block size.
    ///
    /// The `benchmark_fn` closure receives a block size and must:
    /// 1. Launch the kernel with that block size (and appropriate grid)
    /// 2. Synchronize the GPU
    /// 3. Return the elapsed wall-clock time
    ///
    /// Returns `None` if no candidates are viable (all filtered out or all fail).
    pub fn tune_block_size<F>(
        &self,
        problem_size: u64,
        kernel_resources: Option<&KernelResources>,
        benchmark_fn: &F,
    ) -> Option<TuningResult>
    where
        F: Fn(u32) -> Option<Duration>,
    {
        let candidates = self.generate_candidates(kernel_resources);
        if candidates.is_empty() {
            return None;
        }

        let mut all_results: Vec<(u32, Duration)> = Vec::new();

        for &block_size in &candidates {
            // Warmup runs
            for _ in 0..self.config.warmup_iterations {
                let _ = benchmark_fn(block_size);
            }

            // Measurement runs
            let mut times: Vec<Duration> = Vec::new();
            for _ in 0..self.config.measure_iterations {
                if let Some(elapsed) = benchmark_fn(block_size) {
                    times.push(elapsed);
                }
            }

            if times.is_empty() {
                continue; // This candidate failed — skip
            }

            // Take the median
            times.sort();
            let median = times[times.len() / 2];
            all_results.push((block_size, median));
        }

        if all_results.is_empty() {
            return None;
        }

        // Pick the best (lowest median time)
        let (best_block_size, best_time) =
            all_results.iter().min_by_key(|(_, t)| *t).copied().unwrap();

        // Compute grid for the best block size
        let grid_x = (problem_size as u32).div_ceil(best_block_size);
        let config = LaunchConfig {
            grid_dim: (grid_x, 1, 1),
            block_dim: (best_block_size, 1, 1),
            shared_mem_bytes: 0,
        };

        let candidates_tested = all_results.len();

        Some(TuningResult {
            config,
            best_time,
            candidates_tested,
            all_results,
        })
    }

    /// Convenience: tune and insert into cache.
    ///
    /// If a cached result already exists for this key, returns it without
    /// re-tuning (lazy caching).
    pub fn tune_or_cached<F>(
        &self,
        cache: &TuningCache,
        kernel_name: &str,
        problem_size: u64,
        device_id: u32,
        kernel_resources: Option<&KernelResources>,
        benchmark_fn: &F,
    ) -> Option<TuningResult>
    where
        F: Fn(u32) -> Option<Duration>,
    {
        // Check cache first
        if let Some(cached) = cache.get_config(kernel_name, problem_size, device_id) {
            return Some(cached);
        }

        // Tune
        let result = self.tune_block_size(problem_size, kernel_resources, benchmark_fn)?;

        // Cache and return
        cache.insert_config(kernel_name, problem_size, device_id, result.clone());
        Some(result)
    }

    /// Format a tuning report showing all candidate results.
    pub fn format_report(kernel_name: &str, problem_size: u64, result: &TuningResult) -> String {
        use std::fmt::Write;

        let mut out = String::new();
        let _ = writeln!(
            out,
            "Auto-tune report: {} (N={})",
            kernel_name, problem_size
        );
        let _ = writeln!(out, "{:-<60}", "");
        let _ = writeln!(
            out,
            "  {:>12}  {:>12}  {:>10}",
            "Block Size", "Median Time", "Speedup"
        );
        let _ = writeln!(out, "  {:-<40}", "");

        // Find the default (256) time for speedup calculation
        let default_time = result
            .all_results
            .iter()
            .find(|(bs, _)| *bs == 256)
            .map(|(_, t)| *t);

        for (bs, time) in &result.all_results {
            let speedup_str = if let Some(dt) = default_time {
                if *time > Duration::ZERO {
                    format!("{:.2}x", dt.as_secs_f64() / time.as_secs_f64())
                } else {
                    "-".to_string()
                }
            } else {
                "-".to_string()
            };

            let marker = if *bs == result.config.block_dim.0 {
                " <-- best"
            } else {
                ""
            };

            let _ = writeln!(
                out,
                "  {:>12}  {:>12.2?}  {:>10}{}",
                bs, time, speedup_str, marker
            );
        }

        let _ = writeln!(out);
        let _ = writeln!(
            out,
            "Best: block_size={}, time={:?}",
            result.config.block_dim.0, result.best_time
        );

        out
    }
}

impl Default for AutoTuner {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_bucket_size_small() {
        assert_eq!(bucket_size(1), 32);
        assert_eq!(bucket_size(16), 32);
        assert_eq!(bucket_size(32), 32);
    }

    #[test]
    fn test_bucket_size_powers_of_two() {
        assert_eq!(bucket_size(64), 64);
        assert_eq!(bucket_size(1024), 1024);
        assert_eq!(bucket_size(65536), 65536);
    }

    #[test]
    fn test_bucket_size_non_powers() {
        assert_eq!(bucket_size(33), 64);
        assert_eq!(bucket_size(100), 128);
        assert_eq!(bucket_size(1000), 1024);
        assert_eq!(bucket_size(60000), 65536);
    }

    #[test]
    fn test_tuning_key_equality() {
        let k1 = TuningKey::new("vector_add", 60000, 0);
        let k2 = TuningKey::new("vector_add", 65536, 0);
        // Both bucket to 65536
        assert_eq!(k1, k2);
    }

    #[test]
    fn test_tuning_key_inequality() {
        let k1 = TuningKey::new("vector_add", 1024, 0);
        let k2 = TuningKey::new("vector_add", 2048, 0);
        assert_ne!(k1, k2); // Different buckets
    }

    #[test]
    fn test_cache_basic() {
        let cache = TuningCache::new();
        assert!(cache.is_empty());

        let result = TuningResult {
            config: LaunchConfig {
                grid_dim: (256, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            best_time: Duration::from_micros(100),
            candidates_tested: 6,
            all_results: vec![(256, Duration::from_micros(100))],
        };

        cache.insert_config("test_kernel", 65536, 0, result);
        assert_eq!(cache.len(), 1);

        let retrieved = cache.get_config("test_kernel", 65536, 0);
        assert!(retrieved.is_some());
        assert_eq!(retrieved.unwrap().config.block_dim.0, 256);

        // Same bucket
        let retrieved2 = cache.get_config("test_kernel", 60000, 0);
        assert!(retrieved2.is_some());
    }

    #[test]
    fn test_cache_miss() {
        let cache = TuningCache::new();
        assert!(cache.get_config("nonexistent", 1024, 0).is_none());
    }

    #[test]
    fn test_generate_candidates_no_filter() {
        let tuner = AutoTuner::new();
        let candidates = tuner.generate_candidates(None);
        assert_eq!(candidates, vec![32, 64, 128, 256, 512, 1024]);
    }

    #[test]
    fn test_generate_candidates_with_filter() {
        let tuner = AutoTuner::new();
        // A kernel with 255 registers — very high pressure
        let kr = KernelResources {
            name: "heavy_kernel".to_string(),
            registers: 255,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        };
        let candidates = tuner.generate_candidates(Some(&kr));
        // With 255 regs on sm_75, occupancy is very low for larger block sizes
        // Only small block sizes should pass the 12% occupancy filter
        assert!(!candidates.is_empty());
        // Large blocks (512, 1024) should be filtered out
        assert!(!candidates.contains(&1024));
    }

    #[test]
    fn test_tune_block_size_synthetic() {
        let tuner = AutoTuner::with_config(AutoTunerConfig {
            candidate_block_sizes: vec![64, 128, 256],
            warmup_iterations: 1,
            measure_iterations: 3,
            min_occupancy_pct: 0,
            sm_config: SmConfig::sm_75(),
        });

        // Synthetic benchmark: 128 is "fastest"
        let result = tuner.tune_block_size(1024, None, &|block_size| {
            let time_us = match block_size {
                64 => 200,
                128 => 50,
                256 => 100,
                _ => 500,
            };
            Some(Duration::from_micros(time_us))
        });

        let result = result.expect("tuning should succeed");
        assert_eq!(result.config.block_dim.0, 128);
        assert_eq!(result.candidates_tested, 3);
    }

    #[test]
    fn test_tune_or_cached() {
        let tuner = AutoTuner::with_config(AutoTunerConfig {
            candidate_block_sizes: vec![128, 256],
            warmup_iterations: 1,
            measure_iterations: 3,
            min_occupancy_pct: 0,
            sm_config: SmConfig::sm_75(),
        });
        let cache = TuningCache::new();

        let call_count = std::sync::atomic::AtomicU32::new(0);

        let bench = |_bs: u32| -> Option<Duration> {
            call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            Some(Duration::from_micros(100))
        };

        // First call — should tune
        let r1 = tuner.tune_or_cached(&cache, "k", 1024, 0, None, &bench);
        assert!(r1.is_some());
        let first_calls = call_count.load(std::sync::atomic::Ordering::Relaxed);
        assert!(first_calls > 0);

        // Second call — should use cache (no additional benchmark calls)
        call_count.store(0, std::sync::atomic::Ordering::Relaxed);
        let r2 = tuner.tune_or_cached(&cache, "k", 1024, 0, None, &bench);
        assert!(r2.is_some());
        let second_calls = call_count.load(std::sync::atomic::Ordering::Relaxed);
        assert_eq!(second_calls, 0);
    }

    #[test]
    fn test_format_report() {
        let result = TuningResult {
            config: LaunchConfig {
                grid_dim: (512, 1, 1),
                block_dim: (128, 1, 1),
                shared_mem_bytes: 0,
            },
            best_time: Duration::from_micros(50),
            candidates_tested: 3,
            all_results: vec![
                (64, Duration::from_micros(200)),
                (128, Duration::from_micros(50)),
                (256, Duration::from_micros(100)),
            ],
        };

        let report = AutoTuner::format_report("vector_add", 65536, &result);
        assert!(report.contains("vector_add"));
        assert!(report.contains("128"));
        assert!(report.contains("best"));
    }
}
