//! Per-kernel GPU resource estimation and occupancy analysis.
//!
//! Parses `ptxas -v` output to extract physical register counts, spill
//! information, and constant memory usage per kernel entry point.  Calculates
//! theoretical occupancy for a given SM architecture and emits structured
//! diagnostics (warnings for low occupancy, register spills, etc.).
//!
//! # Intended usage
//!
//! ```no_run
//! use gpu_host::resource_report::{parse_ptxas_output, SmConfig, OccupancyLevel};
//!
//! let ptxas_stderr = "ptxas info : Compiling entry function 'my_kern' for 'sm_75'\n\
//!     ptxas info : Function properties for my_kern\n\
//!         0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n\
//!     ptxas info : Used 32 registers, used 0 barriers, 400 bytes cmem[0]\n";
//!
//! let sm = SmConfig::sm_75();
//! let kernels = parse_ptxas_output(ptxas_stderr);
//! for k in &kernels {
//!     let occ = k.occupancy(&sm, 256);
//!     let level = OccupancyLevel::from_percentage(occ);
//!     println!("{}: {} regs, {}% occ → {:?}", k.name, k.registers, occ, level);
//! }
//! ```

use std::fmt;

/// SM architecture configuration for occupancy calculation.
#[derive(Debug, Clone)]
pub struct SmConfig {
    /// SM name (e.g., "sm_75").
    pub name: &'static str,
    /// Total 32-bit registers per SM.
    pub regs_per_sm: u32,
    /// Maximum resident threads per SM.
    pub max_threads_per_sm: u32,
    /// Maximum thread blocks per SM.
    pub max_blocks_per_sm: u32,
    /// Total shared memory per SM (bytes).
    pub smem_per_sm: u32,
    /// Warp size (always 32 on NVIDIA hardware).
    pub warp_size: u32,
    /// Register allocation granularity (registers per warp, rounded up to this).
    pub reg_alloc_granularity: u32,
}

impl SmConfig {
    /// GTX 1660 / RTX 2060–2080 (Turing, sm_75).
    pub fn sm_75() -> Self {
        Self {
            name: "sm_75",
            regs_per_sm: 65536,
            max_threads_per_sm: 1024,
            max_blocks_per_sm: 16,
            smem_per_sm: 49152, // 48 KB (configurable vs L1, this is max smem)
            warp_size: 32,
            reg_alloc_granularity: 256, // Turing allocates in units of 256 regs/warp
        }
    }
}

/// Occupancy severity level.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OccupancyLevel {
    /// Occupancy >= 50% — generally healthy.
    Ok,
    /// Occupancy 25..49% — may limit performance.
    Warn,
    /// Occupancy < 25% — severe performance limitation.
    Critical,
}

impl OccupancyLevel {
    /// Classify an occupancy percentage into a severity level.
    pub fn from_percentage(pct: u32) -> Self {
        if pct < 25 {
            Self::Critical
        } else if pct < 50 {
            Self::Warn
        } else {
            Self::Ok
        }
    }
}

impl fmt::Display for OccupancyLevel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Ok => write!(f, "OK"),
            Self::Warn => write!(f, "WARN"),
            Self::Critical => write!(f, "CRITICAL"),
        }
    }
}

/// Per-kernel resource usage extracted from `ptxas -v` output.
#[derive(Debug, Clone)]
pub struct KernelResources {
    /// Kernel entry-point name.
    pub name: String,
    /// Physical registers per thread (from ptxas allocation).
    pub registers: u32,
    /// Spill store bytes (register pressure exceeded available regs).
    pub spill_stores: u32,
    /// Spill load bytes.
    pub spill_loads: u32,
    /// Per-thread stack frame size in bytes.
    pub stack_frame: u32,
    /// Cumulative stack size (including called device functions).
    pub cumulative_stack: u32,
    /// Constant memory bank 0 usage in bytes.
    pub cmem0: u32,
}

impl KernelResources {
    /// Calculate theoretical occupancy percentage for a given SM config and block size.
    ///
    /// Returns a value 0..=100. Only considers register pressure; does NOT
    /// account for shared memory (which is dynamic in this codebase).
    pub fn occupancy(&self, sm: &SmConfig, block_size: u32) -> u32 {
        if self.registers == 0 || block_size == 0 {
            return 100;
        }

        let regs_per_warp = self.registers * sm.warp_size;
        // Round up to allocation granularity
        let regs_per_warp_alloc =
            regs_per_warp.div_ceil(sm.reg_alloc_granularity) * sm.reg_alloc_granularity;

        let warps_per_block = block_size.div_ceil(sm.warp_size);
        let max_warps_by_regs = sm.regs_per_sm / regs_per_warp_alloc;
        let max_blocks = (max_warps_by_regs / warps_per_block).min(sm.max_blocks_per_sm);

        let active_threads = (max_blocks * block_size).min(sm.max_threads_per_sm);
        active_threads * 100 / sm.max_threads_per_sm
    }

    /// Returns true if this kernel has register spills.
    pub fn has_spills(&self) -> bool {
        self.spill_stores > 0 || self.spill_loads > 0
    }
}

/// Parse ptxas -v stderr output into per-kernel resource records.
///
/// Only extracts entry function data (not device function internals).
/// Each entry function in the ptxas output produces one `KernelResources`.
pub fn parse_ptxas_output(output: &str) -> Vec<KernelResources> {
    let mut results = Vec::new();
    let mut current_kernel: Option<String> = None;
    let mut found_entry_props = false;
    let mut spill_stores = 0u32;
    let mut spill_loads = 0u32;
    let mut stack_frame = 0u32;

    for line in output.lines() {
        // Match: ptxas info    : Compiling entry function 'NAME' for 'sm_XX'
        if line.contains("Compiling entry function") {
            if let Some(name) = extract_quoted(line, "entry function '") {
                current_kernel = Some(name);
                found_entry_props = false;
                spill_stores = 0;
                spill_loads = 0;
                stack_frame = 0;
            }
            continue;
        }

        let Some(ref kernel_name) = current_kernel else {
            continue;
        };

        // Match: ptxas info    : Function properties for KERNEL_NAME
        if !found_entry_props
            && line.contains("Function properties for ")
            && line.contains(kernel_name.as_str())
        {
            found_entry_props = true;
            continue;
        }

        // Match spill line (indented, right after Function properties):
        // "    N bytes stack frame, N bytes spill stores, N bytes spill loads"
        if found_entry_props && line.contains("bytes stack frame") {
            let nums = extract_numbers(line);
            if nums.len() >= 3 {
                stack_frame = nums[0];
                spill_stores = nums[1];
                spill_loads = nums[2];
            }
            continue;
        }

        // Match: ptxas info    : Used N registers, ...
        if line.contains("Used ") && line.contains(" registers") {
            let regs = extract_after(line, "Used ", " registers").unwrap_or(0);
            let cmem0 = extract_after(line, "bytes cmem[0]", "")
                .or_else(|| {
                    // cmem[0] value appears BEFORE the text "bytes cmem[0]"
                    line.find("bytes cmem[0]").and_then(|pos| {
                        let before = &line[..pos].trim_end();
                        before
                            .rsplit_once(|c: char| !c.is_ascii_digit())
                            .and_then(|(_, n)| n.parse().ok())
                    })
                })
                .unwrap_or(0);

            let cumulative_stack = line
                .find("bytes cumulative stack")
                .and_then(|pos| {
                    let before = &line[..pos].trim_end();
                    before
                        .rsplit_once(|c: char| !c.is_ascii_digit())
                        .and_then(|(_, n)| n.parse().ok())
                })
                .unwrap_or(0);

            results.push(KernelResources {
                name: kernel_name.clone(),
                registers: regs,
                spill_stores,
                spill_loads,
                stack_frame,
                cumulative_stack,
                cmem0,
            });
            current_kernel = None;
            found_entry_props = false;
        }
    }

    results
}

/// Format a resource report as a human-readable string.
///
/// Produces a table with per-kernel register count, occupancy percentage,
/// spill info, and warning level.
pub fn format_report(kernels: &[KernelResources], sm: &SmConfig, block_size: u32) -> String {
    use std::fmt::Write;

    let mut out = String::new();
    let _ = writeln!(
        out,
        "GPU Kernel Resource Report  (target: {}, block: {} threads)",
        sm.name, block_size
    );
    let _ = writeln!(out, "{:─<75}", "");
    let _ = writeln!(
        out,
        "  {:<40} {:>5} {:>5} {:>8} {:>6} {:>4}",
        "KERNEL", "REGS", "OCC%", "SPILL", "STACK", "WARN"
    );
    let _ = writeln!(out, "  {:-<75}", "");

    let mut total = 0u32;
    let mut crit_count = 0u32;
    let mut warn_count = 0u32;
    let mut spill_count = 0u32;

    for k in kernels {
        total += 1;
        let occ = k.occupancy(sm, block_size);
        let level = OccupancyLevel::from_percentage(occ);

        let spill_str = if k.has_spills() {
            spill_count += 1;
            format!("{}/{}B", k.spill_stores, k.spill_loads)
        } else {
            "-".to_string()
        };

        let stack_str = if k.cumulative_stack > 0 {
            format!("{}B", k.cumulative_stack)
        } else if k.stack_frame > 0 {
            format!("{}B", k.stack_frame)
        } else {
            "-".to_string()
        };

        let level_str = match level {
            OccupancyLevel::Critical => {
                crit_count += 1;
                "CRIT"
            }
            OccupancyLevel::Warn => {
                warn_count += 1;
                "WARN"
            }
            OccupancyLevel::Ok => "",
        };

        let _ = writeln!(
            out,
            "  {:<40} {:>5} {:>4}% {:>8} {:>6} {:>4}",
            k.name, k.registers, occ, spill_str, stack_str, level_str
        );
    }

    let _ = writeln!(
        out,
        "\n  Total: {} kernels | {} critical | {} warn | {} spills | {} healthy",
        total,
        crit_count,
        warn_count,
        spill_count,
        total - crit_count - warn_count
    );

    out
}

// ── Internal helpers ──────────────────────────────────────────────

/// Extract the string between `prefix'VALUE'` from a line.
fn extract_quoted(line: &str, prefix: &str) -> Option<String> {
    let start = line.find(prefix)? + prefix.len();
    let end = line[start..].find('\'')?;
    Some(line[start..start + end].to_string())
}

/// Extract all decimal numbers from a string.
fn extract_numbers(s: &str) -> Vec<u32> {
    let mut nums = Vec::new();
    let mut in_num = false;
    let mut current = 0u32;
    for ch in s.chars() {
        if ch.is_ascii_digit() {
            current = current * 10 + (ch as u32 - '0' as u32);
            in_num = true;
        } else if in_num {
            nums.push(current);
            current = 0;
            in_num = false;
        }
    }
    if in_num {
        nums.push(current);
    }
    nums
}

/// Extract the number immediately before `suffix` in the line, or after `prefix`.
fn extract_after(line: &str, prefix: &str, suffix: &str) -> Option<u32> {
    if !suffix.is_empty() {
        // Find "N <suffix>" pattern
        let pos = line.find(suffix)?;
        let before = line[..pos].trim_end();
        let num_str: String = before
            .chars()
            .rev()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        let num_str: String = num_str.chars().rev().collect();
        num_str.parse().ok()
    } else {
        // Find "<prefix> N" pattern
        let pos = line.find(prefix)?;
        let after = &line[pos + prefix.len()..];
        let num_str: String = after
            .trim_start()
            .chars()
            .take_while(|c| c.is_ascii_digit())
            .collect();
        num_str.parse().ok()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_basic_ptxas_output() {
        let output = "\
ptxas info    : Compiling entry function 'vector_add' for 'sm_75'
ptxas info    : Function properties for vector_add
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
";
        let kernels = parse_ptxas_output(output);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "vector_add");
        assert_eq!(kernels[0].registers, 12);
        assert_eq!(kernels[0].spill_stores, 0);
        assert_eq!(kernels[0].spill_loads, 0);
        assert_eq!(kernels[0].cmem0, 380);
    }

    #[test]
    fn parse_high_register_kernel() {
        let output = "\
ptxas info    : Compiling entry function 'gemm_f32_v3' for 'sm_75'
ptxas info    : Function properties for gemm_f32_v3
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 111 registers, used 1 barriers, 400 bytes cmem[0]
";
        let kernels = parse_ptxas_output(output);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].registers, 111);

        let sm = SmConfig::sm_75();
        let occ = kernels[0].occupancy(&sm, 256);
        assert_eq!(occ, 50);
    }

    #[test]
    fn parse_kernel_with_cumulative_stack() {
        let output = "\
ptxas info    : Compiling entry function 'flash_attention_v2' for 'sm_75'
ptxas info    : Function properties for flash_attention_v2
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 112 registers, used 1 barriers, 768 bytes cumulative stack size, 408 bytes cmem[0], 512 bytes cmem[2]
";
        let kernels = parse_ptxas_output(output);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].cumulative_stack, 768);
        assert_eq!(kernels[0].cmem0, 408);
    }

    #[test]
    fn parse_multiple_kernels() {
        let output = "\
ptxas info    : Compiling entry function 'vector_add' for 'sm_75'
ptxas info    : Function properties for vector_add
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 12 registers, used 0 barriers, 380 bytes cmem[0]
ptxas info    : Compiling entry function 'gemm_f32_v3' for 'sm_75'
ptxas info    : Function properties for gemm_f32_v3
    0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 111 registers, used 1 barriers, 400 bytes cmem[0]
";
        let kernels = parse_ptxas_output(output);
        assert_eq!(kernels.len(), 2);
        assert_eq!(kernels[0].name, "vector_add");
        assert_eq!(kernels[1].name, "gemm_f32_v3");
    }

    #[test]
    fn parse_kernel_with_device_functions() {
        // Entry function followed by device function properties
        let output = "\
ptxas info    : Compiling entry function 'executor_demo' for 'sm_75'
ptxas info    : Function properties for executor_demo
    48 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
ptxas info    : Used 112 registers, used 0 barriers, 48 bytes cumulative stack size, 368 bytes cmem[0], 608 bytes cmem[2]
ptxas info    : Function properties for _RINvNtCs4UKSxfezG2Q_4core3mem4drop
    120 bytes stack frame, 116 bytes spill stores, 116 bytes spill loads
";
        let kernels = parse_ptxas_output(output);
        assert_eq!(kernels.len(), 1);
        assert_eq!(kernels[0].name, "executor_demo");
        assert_eq!(kernels[0].registers, 112);
        assert_eq!(kernels[0].stack_frame, 48);
        assert_eq!(kernels[0].spill_stores, 0); // Entry function has no spills
    }

    #[test]
    fn occupancy_low_registers() {
        let sm = SmConfig::sm_75();
        let k = KernelResources {
            name: "simple".into(),
            registers: 12,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        };
        assert_eq!(k.occupancy(&sm, 256), 100);
    }

    #[test]
    fn occupancy_255_registers() {
        let sm = SmConfig::sm_75();
        let k = KernelResources {
            name: "heavy".into(),
            registers: 255,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        };
        // 255 * 32 = 8160, rounded to 8192 (next mult of 256)
        // 65536 / 8192 = 8 warps max
        // 8 warps / 8 warps_per_block = 1 block
        // 1 * 256 = 256 threads / 1024 max = 25%
        assert_eq!(k.occupancy(&sm, 256), 25);
        assert_eq!(
            OccupancyLevel::from_percentage(k.occupancy(&sm, 256)),
            OccupancyLevel::Warn
        );
    }

    #[test]
    fn format_report_smoke() {
        let kernels = vec![
            KernelResources {
                name: "vector_add".into(),
                registers: 12,
                spill_stores: 0,
                spill_loads: 0,
                stack_frame: 0,
                cumulative_stack: 0,
                cmem0: 380,
            },
            KernelResources {
                name: "gemm_heavy".into(),
                registers: 200,
                spill_stores: 64,
                spill_loads: 64,
                stack_frame: 0,
                cumulative_stack: 0,
                cmem0: 400,
            },
        ];
        let sm = SmConfig::sm_75();
        let report = format_report(&kernels, &sm, 256);
        assert!(report.contains("vector_add"));
        assert!(report.contains("gemm_heavy"));
        assert!(report.contains("WARN") || report.contains("CRIT"));
    }
}
