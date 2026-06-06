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

/// Configurable warning thresholds for kernel resource analysis.
///
/// Controls at what point occupancy and register usage trigger warnings.
/// Defaults: CRITICAL below 25% occupancy, WARN below 50%.
#[derive(Debug, Clone)]
pub struct WarningConfig {
    /// Occupancy percentage below which a warning is emitted (default: 50).
    pub warn_occupancy_pct: u32,
    /// Occupancy percentage below which a critical warning is emitted (default: 25).
    pub critical_occupancy_pct: u32,
    /// Block size assumption for occupancy calculation (default: 256).
    pub block_size: u32,
}

impl Default for WarningConfig {
    fn default() -> Self {
        Self {
            warn_occupancy_pct: 50,
            critical_occupancy_pct: 25,
            block_size: 256,
        }
    }
}

/// A specific, actionable performance warning for a kernel.
#[derive(Debug, Clone)]
pub struct KernelWarning {
    /// Kernel entry-point name.
    pub kernel: String,
    /// Warning severity.
    pub severity: WarningSeverity,
    /// Machine-readable warning kind.
    pub kind: WarningKind,
    /// Human-readable actionable message.
    pub message: String,
}

/// Warning severity (ordered: Info < Warn < Critical).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum WarningSeverity {
    /// Informational — not a problem, but worth noting.
    Info,
    /// Warning — may limit performance.
    Warn,
    /// Critical — severe performance limitation.
    Critical,
}

impl fmt::Display for WarningSeverity {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Info => write!(f, "INFO"),
            Self::Warn => write!(f, "WARN"),
            Self::Critical => write!(f, "CRIT"),
        }
    }
}

/// Machine-readable warning classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WarningKind {
    /// Occupancy below warning threshold.
    LowOccupancy,
    /// Register spills to local memory.
    RegisterSpill,
    /// Potential shared memory bank conflict stride.
    BankConflict,
}

impl fmt::Display for WarningKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::LowOccupancy => write!(f, "low-occupancy"),
            Self::RegisterSpill => write!(f, "register-spill"),
            Self::BankConflict => write!(f, "bank-conflict"),
        }
    }
}

/// Analyze kernel resources and emit actionable warnings.
///
/// Returns a list of warnings with specific remediation advice.
/// Warnings include:
/// - Low occupancy with register reduction targets
/// - Register spills with byte counts
/// - Bank conflict patterns (if PTX source is provided)
pub fn analyze_warnings(
    kernels: &[KernelResources],
    sm: &SmConfig,
    config: &WarningConfig,
    ptx_source: Option<&str>,
) -> Vec<KernelWarning> {
    let mut warnings = Vec::new();

    for k in kernels {
        let occ = k.occupancy(sm, config.block_size);

        // Low occupancy warnings with actionable register targets
        if occ < config.critical_occupancy_pct {
            let target_regs = next_occupancy_target_regs(sm, config.block_size, occ);
            let msg = if let Some((target, target_occ)) = target_regs {
                format!(
                    "Kernel '{}' uses {} registers → {}% occupancy (CRITICAL). \
                     Reduce to ≤{} registers for {}% occupancy. \
                     Consider: #[launch_bounds], loop tiling, or reducing live variables.",
                    k.name, k.registers, occ, target, target_occ
                )
            } else {
                format!(
                    "Kernel '{}' uses {} registers → {}% occupancy (CRITICAL). \
                     At hardware maximum — consider algorithmic restructuring.",
                    k.name, k.registers, occ
                )
            };
            warnings.push(KernelWarning {
                kernel: k.name.clone(),
                severity: WarningSeverity::Critical,
                kind: WarningKind::LowOccupancy,
                message: msg,
            });
        } else if occ < config.warn_occupancy_pct {
            let target_regs = next_occupancy_target_regs(sm, config.block_size, occ);
            let msg = if let Some((target, target_occ)) = target_regs {
                format!(
                    "Kernel '{}' uses {} registers → {}% occupancy. \
                     Reduce to ≤{} registers for {}% occupancy.",
                    k.name, k.registers, occ, target, target_occ
                )
            } else {
                format!(
                    "Kernel '{}' uses {} registers → {}% occupancy.",
                    k.name, k.registers, occ
                )
            };
            warnings.push(KernelWarning {
                kernel: k.name.clone(),
                severity: WarningSeverity::Warn,
                kind: WarningKind::LowOccupancy,
                message: msg,
            });
        }

        // Spill warnings
        if k.has_spills() {
            let total_spill = k.spill_stores + k.spill_loads;
            let msg = format!(
                "Kernel '{}' spills {} bytes to local memory \
                 ({} store + {} load). Each spill adds ~100 cycles latency. \
                 Reduce live variables or use shared memory for temporaries.",
                k.name, total_spill, k.spill_stores, k.spill_loads
            );
            warnings.push(KernelWarning {
                kernel: k.name.clone(),
                severity: WarningSeverity::Warn,
                kind: WarningKind::RegisterSpill,
                message: msg,
            });
        }
    }

    // Bank conflict detection from PTX source
    if let Some(ptx) = ptx_source {
        let bank_warnings = detect_bank_conflicts(ptx, kernels);
        warnings.extend(bank_warnings);
    }

    warnings
}

/// Calculate the register target needed for the next occupancy level.
///
/// Returns `Some((max_regs, occupancy_pct))` for the next higher occupancy tier,
/// or `None` if already at 100%.
fn next_occupancy_target_regs(
    sm: &SmConfig,
    block_size: u32,
    current_occ: u32,
) -> Option<(u32, u32)> {
    // Try occupancy levels from current+25 up to 100 in steps of 25
    let target_occ = ((current_occ / 25) + 1) * 25;
    if target_occ > 100 {
        return None;
    }

    // Binary search for max registers that achieve target occupancy
    // Start from 255 (max) down to 1
    let mut best_regs = None;
    for regs in (1..=255).rev() {
        let k = KernelResources {
            name: String::new(),
            registers: regs,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        };
        let occ = k.occupancy(sm, block_size);
        if occ >= target_occ {
            best_regs = Some((regs, occ));
            break;
        }
    }

    best_regs
}

/// Detect potential bank conflict patterns in PTX source.
///
/// Analyzes shared memory access patterns for known-bad strides:
/// - Stride that is a multiple of 128 bytes (32 banks * 4 bytes) causes all
///   threads to hit the same bank
/// - Absence of padding in tiled shared memory arrays
///
/// This is a heuristic analysis — it catches common patterns but cannot detect
/// all bank conflicts (data-dependent or indirect accesses are invisible).
fn detect_bank_conflicts(ptx: &str, kernels: &[KernelResources]) -> Vec<KernelWarning> {
    let mut warnings = Vec::new();

    // Build a set of kernel names that use shared memory
    let smem_kernels = find_smem_kernels(ptx);

    for kernel_name in &smem_kernels {
        // Check if this kernel is in our resource list
        let in_resources = kernels.iter().any(|k| &k.name == kernel_name);
        if !in_resources {
            continue;
        }

        // Extract the PTX for this kernel entry
        let kernel_ptx = extract_kernel_ptx(ptx, kernel_name);
        if kernel_ptx.is_empty() {
            continue;
        }

        // Look for stride patterns that cause bank conflicts
        let bad_strides = detect_bad_strides(&kernel_ptx);
        for stride in bad_strides {
            let msg = format!(
                "Kernel '{}' accesses shared memory with stride {} \
                 (multiple of 128 bytes → all threads hit same bank). \
                 Add padding: use stride {} instead of {} to eliminate bank conflicts.",
                kernel_name,
                stride,
                stride + 4,
                stride
            );
            warnings.push(KernelWarning {
                kernel: kernel_name.clone(),
                severity: WarningSeverity::Warn,
                kind: WarningKind::BankConflict,
                message: msg,
            });
        }
    }

    warnings
}

/// Find kernel names that access dynamic shared memory.
fn find_smem_kernels(ptx: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut current_entry: Option<String> = None;

    for line in ptx.lines() {
        // Match .visible .entry KERNEL_NAME(
        if line.contains(".entry ") {
            if let Some(name) = extract_entry_name(line) {
                current_entry = Some(name);
            }
            continue;
        }

        // If we're inside a kernel and see shared memory access, record it
        if let Some(ref name) = current_entry {
            if (line.contains("cvta.shared")
                || line.contains("ld.shared")
                || line.contains("st.shared"))
                && !result.contains(name)
            {
                result.push(name.clone());
            }
        }

        // New entry function resets tracking
        // (we don't need to track function boundaries precisely;
        //  the .entry marker is enough for our heuristic)
    }

    result
}

/// Extract the PTX instructions for a single kernel entry point.
fn extract_kernel_ptx(ptx: &str, kernel_name: &str) -> String {
    let mut result = String::new();
    let mut in_kernel = false;
    let mut brace_depth = 0i32;

    for line in ptx.lines() {
        if !in_kernel && line.contains(".entry ") && line.contains(kernel_name) {
            in_kernel = true;
            continue;
        }

        if in_kernel {
            for ch in line.chars() {
                if ch == '{' {
                    brace_depth += 1;
                } else if ch == '}' {
                    brace_depth -= 1;
                }
            }
            result.push_str(line);
            result.push('\n');

            if brace_depth <= 0 && result.contains('{') {
                break;
            }
        }
    }

    result
}

/// Detect shared memory access strides that cause bank conflicts.
///
/// Looks for `mad.lo.s32` or `mul.lo.s32` instructions with an immediate
/// operand that is a multiple of 128 (32 banks * 4 bytes). These are the
/// stride computations for shared memory tile indexing.
///
/// Returns the list of bad stride values found.
fn detect_bad_strides(kernel_ptx: &str) -> Vec<u32> {
    let mut bad_strides = Vec::new();

    // Check for shared memory usage first
    let uses_smem = kernel_ptx.contains("cvta.shared")
        || kernel_ptx.contains("ld.shared")
        || kernel_ptx.contains("st.shared");

    if !uses_smem {
        return bad_strides;
    }

    for line in kernel_ptx.lines() {
        let trimmed = line.trim();

        // Look for mul/mad with immediate stride operands near shared memory accesses
        // Pattern: mad.lo.s32 %rN, %rM, STRIDE, %rK  (where STRIDE is a constant)
        // Pattern: mul.lo.s32 %rN, %rM, STRIDE
        if trimmed.starts_with("mad.lo.s32") || trimmed.starts_with("mul.lo.s32") {
            if let Some(stride) = extract_stride_immediate(trimmed) {
                // A stride that is a multiple of 128 bytes (32 banks * 4B each)
                // causes all threads to hit the same bank
                if stride > 0 && stride % 128 == 0 && !bad_strides.contains(&stride) {
                    bad_strides.push(stride);
                }
            }
        }

        // Also check shl (shift left) which is an implicit multiply:
        // shl.b32 %rN, %rM, 7  → multiply by 128, bad stride
        if trimmed.starts_with("shl.b32") || trimmed.starts_with("shl.b64") {
            if let Some(shift) = extract_shift_immediate(trimmed) {
                // shift of 7+ means stride of 128+ bytes
                // But only flag if it's exactly a power-of-2 multiple of 128
                if shift >= 7 {
                    let stride = 1u32 << shift;
                    // Only flag if this is used for shared memory indexing
                    // (heuristic: the stride is a reasonable array dimension)
                    if stride <= 8192 && !bad_strides.contains(&stride) {
                        // Check: this is suspicious but common (e.g., block-level offsets).
                        // Only flag strides that are exactly 128 or 256 as these are
                        // the most common bank-conflict-causing patterns in tiled code.
                        if stride == 128 || stride == 256 {
                            bad_strides.push(stride);
                        }
                    }
                }
            }
        }
    }

    bad_strides
}

/// Extract an immediate numeric operand from a mad/mul instruction.
///
/// Example: `mad.lo.s32 %r15, %r5, 132, %r4` → Some(132)
fn extract_stride_immediate(instr: &str) -> Option<u32> {
    // Split by comma, look for numeric operands
    let parts: Vec<&str> = instr.split(',').collect();
    if parts.len() < 3 {
        return None;
    }

    // The stride is typically the 3rd operand (index 2) in mad,
    // or the 3rd operand (index 2) in mul
    for part in &parts[1..] {
        let trimmed = part.trim().trim_end_matches(';');
        if let Ok(n) = trimmed.parse::<u32>() {
            return Some(n);
        }
    }

    None
}

/// Extract the shift amount from a shl instruction.
///
/// Example: `shl.b32 %r2, %r59, 7` → Some(7)
fn extract_shift_immediate(instr: &str) -> Option<u32> {
    let parts: Vec<&str> = instr.split(',').collect();
    if let Some(last) = parts.last() {
        let trimmed = last.trim().trim_end_matches(';');
        return trimmed.parse::<u32>().ok();
    }
    None
}

/// Extract kernel entry-point name from a PTX `.entry` line.
fn extract_entry_name(line: &str) -> Option<String> {
    let entry_idx = line.find(".entry ")?;
    let after = &line[entry_idx + 7..];
    let name: String = after
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() {
        None
    } else {
        Some(name)
    }
}

/// Format actionable warnings as a human-readable report.
pub fn format_warnings(warnings: &[KernelWarning]) -> String {
    use std::fmt::Write;

    if warnings.is_empty() {
        return String::from("  No performance warnings.\n");
    }

    let mut out = String::new();

    let crit_count = warnings
        .iter()
        .filter(|w| w.severity == WarningSeverity::Critical)
        .count();
    let warn_count = warnings
        .iter()
        .filter(|w| w.severity == WarningSeverity::Warn)
        .count();

    let _ = writeln!(
        out,
        "  {} warning(s): {} critical, {} warning",
        warnings.len(),
        crit_count,
        warn_count
    );
    let _ = writeln!(out);

    for w in warnings {
        let _ = writeln!(out, "  [{}] {}: {}", w.severity, w.kind, w.message);
    }

    out
}

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

    // Add actionable warnings section if any issues found
    if crit_count > 0 || warn_count > 0 || spill_count > 0 {
        let config = WarningConfig::default();
        let warnings = analyze_warnings(kernels, sm, &config, None);
        if !warnings.is_empty() {
            let _ = writeln!(out, "\n  Actionable Warnings:");
            let _ = writeln!(out, "  {:-<75}", "");
            let _ = write!(out, "{}", format_warnings(&warnings));
        }
    }

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
    fn warning_config_defaults() {
        let config = WarningConfig::default();
        assert_eq!(config.warn_occupancy_pct, 50);
        assert_eq!(config.critical_occupancy_pct, 25);
        assert_eq!(config.block_size, 256);
    }

    #[test]
    fn analyze_warnings_critical_occupancy() {
        let sm = SmConfig::sm_75();
        let config = WarningConfig::default();
        // 255 regs → 25% occ (exactly at critical threshold: < 25 is false)
        // Use a kernel that truly gets < 25% — not possible on sm_75 with 256
        // block size (25% is the minimum), so use custom thresholds
        let config_strict = WarningConfig {
            warn_occupancy_pct: 50,
            critical_occupancy_pct: 30,
            block_size: 256,
        };
        let kernels = vec![KernelResources {
            name: "heavy_kernel".into(),
            registers: 255,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        // 25% < 30% critical threshold → CRITICAL
        let warnings = analyze_warnings(&kernels, &sm, &config_strict, None);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, WarningSeverity::Critical);
        assert_eq!(warnings[0].kind, WarningKind::LowOccupancy);
        assert!(warnings[0].message.contains("255 registers"));
        assert!(warnings[0].message.contains("25%"));
        assert!(warnings[0].message.contains("Reduce to"));

        // With default thresholds: 25% is exactly at the boundary (not < 25)
        // so it gets classified as WARN (25..49%)
        let warnings_default = analyze_warnings(&kernels, &sm, &config, None);
        assert_eq!(warnings_default.len(), 1);
        assert_eq!(warnings_default[0].severity, WarningSeverity::Warn);
    }

    #[test]
    fn analyze_warnings_warn_occupancy() {
        let sm = SmConfig::sm_75();
        let config = WarningConfig::default();
        // 111 regs → 50% occ (exactly at warn threshold: < 50 is false)
        // Use 129 regs → 25% occ, which triggers WARN (25..49%)
        let kernels = vec![KernelResources {
            name: "medium_kernel".into(),
            registers: 129,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = analyze_warnings(&kernels, &sm, &config, None);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].severity, WarningSeverity::Warn);
        assert_eq!(warnings[0].kind, WarningKind::LowOccupancy);
        assert!(warnings[0].message.contains("129 registers"));
    }

    #[test]
    fn analyze_warnings_healthy_no_warnings() {
        let sm = SmConfig::sm_75();
        let config = WarningConfig::default();
        let kernels = vec![KernelResources {
            name: "simple_kernel".into(),
            registers: 16,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = analyze_warnings(&kernels, &sm, &config, None);
        assert!(warnings.is_empty());
    }

    #[test]
    fn analyze_warnings_spills() {
        let sm = SmConfig::sm_75();
        let config = WarningConfig::default();
        let kernels = vec![KernelResources {
            name: "spilly_kernel".into(),
            registers: 16,
            spill_stores: 128,
            spill_loads: 64,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = analyze_warnings(&kernels, &sm, &config, None);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::RegisterSpill);
        assert!(warnings[0].message.contains("192 bytes"));
        assert!(warnings[0].message.contains("128 store"));
    }

    #[test]
    fn analyze_warnings_custom_thresholds() {
        let sm = SmConfig::sm_75();
        let config = WarningConfig {
            warn_occupancy_pct: 75,
            critical_occupancy_pct: 50,
            block_size: 256,
        };
        // 69 regs → 75% occupancy; with threshold at 75% this is not warned
        let kernels = vec![KernelResources {
            name: "tuned_kernel".into(),
            registers: 69,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = analyze_warnings(&kernels, &sm, &config, None);
        assert!(warnings.is_empty());
    }

    #[test]
    fn bank_conflict_detection_bad_stride() {
        // PTX with a kernel using stride 128 in shared memory
        let ptx = "\
.visible .entry bad_stride_kernel(
    .param .u64 .ptr .align 1 bad_stride_kernel_param_0
)
{
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    cvta.shared.u64 %rd1, dynamic_smem;
    mad.lo.s32 %r5, %r1, 128, %r2;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.b32 [%rd3], %r6;
}
";
        let kernels = vec![KernelResources {
            name: "bad_stride_kernel".into(),
            registers: 16,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = detect_bank_conflicts(ptx, &kernels);
        assert_eq!(warnings.len(), 1);
        assert_eq!(warnings[0].kind, WarningKind::BankConflict);
        assert!(warnings[0].message.contains("stride 128"));
        assert!(warnings[0].message.contains("stride 132"));
    }

    #[test]
    fn bank_conflict_detection_good_stride() {
        // PTX with padded stride (132 = 128 + 4) — no bank conflict
        let ptx = "\
.visible .entry good_stride_kernel(
    .param .u64 .ptr .align 1 good_stride_kernel_param_0
)
{
    .reg .b32 %r<10>;
    .reg .b64 %rd<10>;
    cvta.shared.u64 %rd1, dynamic_smem;
    mad.lo.s32 %r5, %r1, 132, %r2;
    mul.wide.u32 %rd2, %r5, 4;
    add.s64 %rd3, %rd1, %rd2;
    st.b32 [%rd3], %r6;
}
";
        let kernels = vec![KernelResources {
            name: "good_stride_kernel".into(),
            registers: 16,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        }];

        let warnings = detect_bank_conflicts(ptx, &kernels);
        assert!(
            warnings.is_empty(),
            "Padded stride 132 should NOT trigger bank conflict warning"
        );
    }

    #[test]
    fn next_occupancy_target_for_25_pct() {
        let sm = SmConfig::sm_75();
        // 255 regs → 25% occ; next target should be 50% occ
        let result = next_occupancy_target_regs(&sm, 256, 25);
        assert!(result.is_some());
        let (target_regs, target_occ) = result.unwrap();
        assert!(target_occ >= 50);
        // Verify: a kernel with target_regs actually achieves target_occ
        let k = KernelResources {
            name: String::new(),
            registers: target_regs,
            spill_stores: 0,
            spill_loads: 0,
            stack_frame: 0,
            cumulative_stack: 0,
            cmem0: 0,
        };
        assert!(k.occupancy(&sm, 256) >= 50);
    }

    #[test]
    fn format_warnings_output() {
        let warnings = vec![KernelWarning {
            kernel: "test_kern".into(),
            severity: WarningSeverity::Critical,
            kind: WarningKind::LowOccupancy,
            message: "test warning message".into(),
        }];
        let out = format_warnings(&warnings);
        assert!(out.contains("1 warning(s)"));
        assert!(out.contains("1 critical"));
        assert!(out.contains("[CRIT]"));
        assert!(out.contains("test warning message"));
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
