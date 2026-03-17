// Monte Carlo simulations: xoshiro256++ PRNG, Pi estimation, Black-Scholes pricing
//
// CPU reference + GPU (cudarc/NVRTC) implementations for mc-finance theme.
// GPU kernels use per-thread xoshiro256++ seeded via SplitMix64.

use std::f64::consts::PI;
use std::sync::Arc;
use std::time::Instant;

use cudarc::driver::{CudaDevice, LaunchAsync, LaunchConfig};
use cudarc::nvrtc::compile_ptx;

// ---------------------------------------------------------------------------
// SplitMix64 — used to seed xoshiro256++ from a single u64
// ---------------------------------------------------------------------------

struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    fn next(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
}

// ---------------------------------------------------------------------------
// Xoshiro256PlusPlus — high-quality, fast PRNG
// ---------------------------------------------------------------------------

struct Xoshiro256PlusPlus {
    s: [u64; 4],
}

impl Xoshiro256PlusPlus {
    /// Create a new generator seeded via SplitMix64 from `seed`.
    fn from_seed(seed: u64) -> Self {
        let mut sm = SplitMix64::new(seed);
        Self {
            s: [sm.next(), sm.next(), sm.next(), sm.next()],
        }
    }

    /// Return the next pseudorandom u64.
    fn next_u64(&mut self) -> u64 {
        let result = (self.s[0].wrapping_add(self.s[3]))
            .rotate_left(23)
            .wrapping_add(self.s[0]);

        let t = self.s[1] << 17;

        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];

        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);

        result
    }

    /// Return a uniform f64 in [0, 1).
    fn next_f64(&mut self) -> f64 {
        // Use the upper 53 bits for a double in [0, 1).
        (self.next_u64() >> 11) as f64 * (1.0_f64 / (1u64 << 53) as f64)
    }

    /// Return a standard-normal sample via the Box-Muller transform.
    /// Generates two normals but discards one for simplicity.
    fn next_normal(&mut self) -> f64 {
        loop {
            let u1 = self.next_f64();
            let u2 = self.next_f64();
            if u1 > 0.0 {
                return (-2.0 * u1.ln()).sqrt() * (2.0 * PI * u2).cos();
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Monte Carlo Pi estimation (CPU)
// ---------------------------------------------------------------------------

fn estimate_pi(n: u64, base_seed: u64) -> (f64, u64) {
    let mut rng = Xoshiro256PlusPlus::from_seed(base_seed);
    let mut inside = 0u64;

    for _ in 0..n {
        let x = rng.next_f64();
        let y = rng.next_f64();
        if x * x + y * y < 1.0 {
            inside += 1;
        }
    }

    let pi_est = 4.0 * inside as f64 / n as f64;
    (pi_est, inside)
}

// ---------------------------------------------------------------------------
// Black-Scholes analytical price (for comparison)
// ---------------------------------------------------------------------------

/// Complementary error function (Horner-form rational approximation).
/// Maximum relative error < 1.5e-7.
fn erfc_approx(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.5 * x.abs());
    #[rustfmt::skip]
    let tau = t * (-x * x - 1.265_512_23
        + t * (1.000_023_68
        + t * (0.374_091_96
        + t * (0.096_784_18
        + t * (-0.186_288_06
        + t * (0.278_868_07
        + t * (-1.135_203_98
        + t * (1.488_515_87
        + t * (-0.822_152_23
        + t * 0.170_872_77)))))))))
    .exp();
    if x >= 0.0 {
        tau
    } else {
        2.0 - tau
    }
}

/// Standard normal CDF via erfc approximation (error < 1.5e-7).
fn norm_cdf(x: f64) -> f64 {
    0.5 * erfc_approx(-x / std::f64::consts::SQRT_2)
}

/// Analytical European call price under Black-Scholes.
fn black_scholes_call(s0: f64, k: f64, r: f64, sigma: f64, t: f64) -> f64 {
    let d1 = ((s0 / k).ln() + (r + 0.5 * sigma * sigma) * t) / (sigma * t.sqrt());
    let d2 = d1 - sigma * t.sqrt();
    s0 * norm_cdf(d1) - k * (-r * t).exp() * norm_cdf(d2)
}

// ---------------------------------------------------------------------------
// Monte Carlo Black-Scholes pricing (CPU)
// ---------------------------------------------------------------------------

fn mc_black_scholes(
    n: u64,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    base_seed: u64,
) -> (f64, f64) {
    let drift = (r - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let discount = (-r * t).exp();

    let mut rng = Xoshiro256PlusPlus::from_seed(base_seed);
    let mut payoff_sum = 0.0_f64;
    let mut payoff_sq_sum = 0.0_f64;

    for _ in 0..n {
        let z = rng.next_normal();
        let s_t = s0 * (drift + vol * z).exp();
        let payoff = (s_t - k).max(0.0);
        payoff_sum += payoff;
        payoff_sq_sum += payoff * payoff;
    }

    let mean_payoff = payoff_sum / n as f64;
    let price = discount * mean_payoff;

    // Standard error of the discounted payoff
    let variance = payoff_sq_sum / n as f64 - mean_payoff * mean_payoff;
    let std_err = discount * (variance / n as f64).sqrt();

    (price, std_err)
}

// ---------------------------------------------------------------------------
// CUDA kernel source — shared PRNG + Pi estimation + Black-Scholes
// ---------------------------------------------------------------------------

const CUDA_KERNEL_SRC: &str = r#"
// xoshiro256++ and SplitMix64 in CUDA C (identical algorithm to CPU Rust code)

// SplitMix64: deterministic seeding from a single u64
__device__ unsigned long long splitmix64(unsigned long long *state) {
    *state += 0x9e3779b97f4a7c15ULL;
    unsigned long long z = *state;
    z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
    z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
    return z ^ (z >> 31);
}

// Rotate left for unsigned long long
__device__ unsigned long long rotl64(unsigned long long x, int k) {
    return (x << k) | (x >> (64 - k));
}

// xoshiro256++ state
struct Xoshiro256PP {
    unsigned long long s[4];
};

__device__ void xoshiro_seed(Xoshiro256PP *rng, unsigned long long seed) {
    unsigned long long sm = seed;
    rng->s[0] = splitmix64(&sm);
    rng->s[1] = splitmix64(&sm);
    rng->s[2] = splitmix64(&sm);
    rng->s[3] = splitmix64(&sm);
}

__device__ unsigned long long xoshiro_next(Xoshiro256PP *rng) {
    unsigned long long result = rotl64(rng->s[0] + rng->s[3], 23) + rng->s[0];
    unsigned long long t = rng->s[1] << 17;

    rng->s[2] ^= rng->s[0];
    rng->s[3] ^= rng->s[1];
    rng->s[1] ^= rng->s[2];
    rng->s[0] ^= rng->s[3];

    rng->s[2] ^= t;
    rng->s[3] = rotl64(rng->s[3], 45);

    return result;
}

// Uniform double in [0, 1) — same method as the Rust code (upper 53 bits)
__device__ double xoshiro_next_f64(Xoshiro256PP *rng) {
    return (double)(xoshiro_next(rng) >> 11) * (1.0 / (double)(1ULL << 53));
}

// ---------------------------------------------------------------------------
// Pi estimation kernel: each thread tests N_PER_THREAD points, atomicAdd count
// ---------------------------------------------------------------------------
extern "C" __global__ void mc_pi_kernel(
    unsigned long long base_seed,
    int n_per_thread,
    unsigned long long *global_count,  // single element, atomicAdd target
    int total_threads
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    // Each thread gets a unique seed derived from (base_seed + tid)
    Xoshiro256PP rng;
    xoshiro_seed(&rng, base_seed + (unsigned long long)tid);

    unsigned long long local_count = 0;
    for (int i = 0; i < n_per_thread; i++) {
        double x = xoshiro_next_f64(&rng);
        double y = xoshiro_next_f64(&rng);
        if (x * x + y * y < 1.0) {
            local_count++;
        }
    }

    // Accumulate into global counter
    atomicAdd(global_count, local_count);
}

// ---------------------------------------------------------------------------
// Black-Scholes Monte Carlo kernel: each thread computes N_PER_THREAD payoffs
// ---------------------------------------------------------------------------
extern "C" __global__ void mc_bs_kernel(
    float s0,
    float k,
    float drift,     // (r - 0.5*sigma^2)*T
    float vol,       // sigma * sqrt(T)
    unsigned long long base_seed,
    int n_per_thread,
    float *payoffs,  // output: one sum per thread
    int total_threads
) {
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= total_threads) return;

    Xoshiro256PP rng;
    xoshiro_seed(&rng, base_seed + (unsigned long long)tid);

    float payoff_sum = 0.0f;
    for (int i = 0; i < n_per_thread; i++) {
        // Box-Muller transform for standard normal (f32 for GPU throughput)
        float u1, u2;
        do {
            u1 = (float)xoshiro_next_f64(&rng);
        } while (u1 == 0.0f);
        u2 = (float)xoshiro_next_f64(&rng);

        float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);

        float s_t = s0 * expf(drift + vol * z);
        float payoff = s_t - k;
        if (payoff < 0.0f) payoff = 0.0f;
        payoff_sum += payoff;
    }

    payoffs[tid] = payoff_sum;
}
"#;

// ---------------------------------------------------------------------------
// GPU Pi estimation via CUDA
// ---------------------------------------------------------------------------

fn gpu_estimate_pi(
    dev: &Arc<CudaDevice>,
    n: u64,
    base_seed: u64,
) -> Result<(f64, u64, std::time::Duration), Box<dyn std::error::Error>> {
    let block_size: u32 = 256;
    let total_threads: u32 = 65536; // 65536 threads for better GPU occupancy
    let n_per_thread = (n as u32 + total_threads - 1) / total_threads;
    let actual_n = n_per_thread as u64 * total_threads as u64;
    let grid_size = (total_threads + block_size - 1) / block_size;

    // Allocate global counter (single u64) on device, initialized to 0
    let d_count = dev.htod_sync_copy(&[0u64])?;

    let func = dev
        .get_func("mc_kernels", "mc_pi_kernel")
        .ok_or("mc_pi_kernel not found")?;

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let t0 = Instant::now();
    unsafe {
        func.launch(
            cfg,
            (
                base_seed,
                n_per_thread as i32,
                &d_count,
                total_threads as i32,
            ),
        )?;
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed();

    let result: Vec<u64> = dev.dtoh_sync_copy(&d_count)?;
    let inside = result[0];
    let pi_est = 4.0 * inside as f64 / actual_n as f64;

    Ok((pi_est, inside, elapsed))
}

// ---------------------------------------------------------------------------
// GPU Black-Scholes Monte Carlo via CUDA
// ---------------------------------------------------------------------------

fn gpu_mc_black_scholes(
    dev: &Arc<CudaDevice>,
    n: u64,
    s0: f64,
    k: f64,
    r: f64,
    sigma: f64,
    t: f64,
    base_seed: u64,
) -> Result<(f64, f64, std::time::Duration), Box<dyn std::error::Error>> {
    let block_size: u32 = 256;
    let total_threads: u32 = 65536; // 65536 threads for better GPU occupancy
    let n_per_thread = (n as u32 + total_threads - 1) / total_threads;
    let actual_n = n_per_thread as u64 * total_threads as u64;
    let grid_size = (total_threads + block_size - 1) / block_size;

    let drift = (r - 0.5 * sigma * sigma) * t;
    let vol = sigma * t.sqrt();
    let discount = (-r * t).exp();

    // Each thread writes one f32 partial sum
    let d_payoffs = dev.htod_sync_copy(&vec![0.0f32; total_threads as usize])?;

    let func = dev
        .get_func("mc_kernels", "mc_bs_kernel")
        .ok_or("mc_bs_kernel not found")?;

    let cfg = LaunchConfig {
        grid_dim: (grid_size, 1, 1),
        block_dim: (block_size, 1, 1),
        shared_mem_bytes: 0,
    };

    let t0 = Instant::now();
    unsafe {
        func.launch(
            cfg,
            (
                s0 as f32,
                k as f32,
                drift as f32,
                vol as f32,
                base_seed,
                n_per_thread as i32,
                &d_payoffs,
                total_threads as i32,
            ),
        )?;
    }
    dev.synchronize()?;
    let elapsed = t0.elapsed();

    // Download f32 partial sums and reduce on host in f64
    let payoff_sums: Vec<f32> = dev.dtoh_sync_copy(&d_payoffs)?;
    let total_payoff: f64 = payoff_sums.iter().map(|x| *x as f64).sum();
    let mean_payoff = total_payoff / actual_n as f64;
    let price = discount * mean_payoff;

    // Compute standard error from partial sums (each is a sum of n_per_thread payoffs)
    // We approximate: variance ~ E[X^2] - E[X]^2, use per-thread means
    let per_thread_means: Vec<f64> = payoff_sums
        .iter()
        .map(|s| *s as f64 / n_per_thread as f64)
        .collect();
    let mean_of_means = per_thread_means.iter().sum::<f64>() / total_threads as f64;
    let var_of_means: f64 = per_thread_means
        .iter()
        .map(|m| (m - mean_of_means).powi(2))
        .sum::<f64>()
        / total_threads as f64;
    // Var(mean of n_per_thread samples) = Var(X)/n_per_thread
    // So Var(X) ~ var_of_means * n_per_thread
    // StdErr(grand mean) = sqrt(Var(X) / actual_n)
    let std_err = discount * (var_of_means * n_per_thread as f64 / actual_n as f64).sqrt();

    Ok((price, std_err, elapsed))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Monte Carlo Simulations ===\n");

    // -----------------------------------------------------------------------
    // CPU Pi estimation
    // -----------------------------------------------------------------------
    let n_pi: u64 = 10_000_000;
    let seed_pi: u64 = 12345;

    println!("--- Pi Estimation (CPU) ---");
    println!("Points: {n_pi}");

    let t0 = Instant::now();
    let (pi_est, inside) = estimate_pi(n_pi, seed_pi);
    let elapsed_cpu_pi = t0.elapsed();

    let pi_err = (pi_est - PI).abs();
    println!("Inside circle: {inside}");
    println!("Pi estimate:   {pi_est:.6}");
    println!("True Pi:       {PI:.6}");
    println!("Abs error:     {pi_err:.6e}");
    println!("Time:          {:.2?}", elapsed_cpu_pi);

    assert!(
        pi_err < 0.01,
        "Pi estimate too inaccurate: error = {pi_err}"
    );
    println!("PASS: Pi accurate to ~{} decimal places\n", {
        let digits = -pi_err.log10();
        digits.floor() as u32
    });

    // -----------------------------------------------------------------------
    // CPU Black-Scholes
    // -----------------------------------------------------------------------
    let n_bs: u64 = 1_000_000;
    let s0 = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let seed_bs: u64 = 67890;

    println!("--- Black-Scholes European Call (CPU Monte Carlo) ---");
    println!("Paths: {n_bs}");
    println!("S0={s0}, K={k}, r={r}, sigma={sigma}, T={t}");

    let analytical = black_scholes_call(s0, k, r, sigma, t);
    println!("Analytical BS price: {analytical:.4}");

    let t0 = Instant::now();
    let (mc_price_cpu, std_err_cpu) = mc_black_scholes(n_bs, s0, k, r, sigma, t, seed_bs);
    let elapsed_cpu_bs = t0.elapsed();

    let rel_err_cpu = ((mc_price_cpu - analytical) / analytical).abs();
    println!("MC price:            {mc_price_cpu:.4} +/- {std_err_cpu:.4}");
    println!("Relative error:      {:.4}%", rel_err_cpu * 100.0);
    println!("Time:                {:.2?}", elapsed_cpu_bs);

    assert!(
        rel_err_cpu < 0.01,
        "CPU MC price too far from analytical: {:.4}%",
        rel_err_cpu * 100.0
    );
    println!("PASS: CPU MC price within 1% of analytical\n");

    // -----------------------------------------------------------------------
    // GPU setup: compile CUDA kernels via NVRTC
    // -----------------------------------------------------------------------
    println!("--- GPU Setup ---");
    let dev = CudaDevice::new(0).expect("Failed to initialize CUDA device");
    println!("CUDA device initialized");

    println!("Compiling Monte Carlo CUDA kernels via NVRTC...");
    let compile_t0 = Instant::now();
    let ptx = compile_ptx(CUDA_KERNEL_SRC).expect("Failed to compile CUDA kernel");
    println!("Kernel compilation: {:.2?}", compile_t0.elapsed());

    dev.load_ptx(ptx, "mc_kernels", &["mc_pi_kernel", "mc_bs_kernel"])
        .expect("Failed to load PTX module");
    println!();

    // -----------------------------------------------------------------------
    // GPU Pi estimation
    // -----------------------------------------------------------------------
    let n_gpu_pi: u64 = 100_000_000;

    println!("--- Pi Estimation (GPU) ---");
    println!("Points: {n_gpu_pi}");

    let (gpu_pi_est, gpu_inside, elapsed_gpu_pi) =
        gpu_estimate_pi(&dev, n_gpu_pi, seed_pi).expect("GPU Pi estimation failed");

    let gpu_pi_err = (gpu_pi_est - PI).abs();
    println!("Inside circle: {gpu_inside}");
    println!("Pi estimate:   {gpu_pi_est:.6}");
    println!("True Pi:       {PI:.6}");
    println!("Abs error:     {gpu_pi_err:.6e}");
    println!("Time:          {:.2?}", elapsed_gpu_pi);
    println!(
        "Speedup vs CPU: {:.1}x",
        elapsed_cpu_pi.as_secs_f64() / elapsed_gpu_pi.as_secs_f64()
    );

    assert!(
        gpu_pi_err < 0.01,
        "GPU Pi estimate too inaccurate: error = {gpu_pi_err}"
    );
    println!("PASS: GPU Pi accurate to ~{} decimal places\n", {
        let digits = -gpu_pi_err.log10();
        digits.floor() as u32
    });

    // -----------------------------------------------------------------------
    // GPU Black-Scholes (10M paths for better speedup demonstration)
    // -----------------------------------------------------------------------
    let n_gpu_bs: u64 = 10_000_000;
    println!("--- Black-Scholes European Call (GPU Monte Carlo) ---");
    println!("Paths: {n_gpu_bs}");
    println!("S0={s0}, K={k}, r={r}, sigma={sigma}, T={t}");
    println!("Analytical BS price: {analytical:.4}");

    let (mc_price_gpu, std_err_gpu, elapsed_gpu_bs) =
        gpu_mc_black_scholes(&dev, n_gpu_bs, s0, k, r, sigma, t, seed_bs)
            .expect("GPU BS Monte Carlo failed");

    let rel_err_gpu = ((mc_price_gpu - analytical) / analytical).abs();
    println!("MC price:            {mc_price_gpu:.4} +/- {std_err_gpu:.4}");
    println!("Relative error:      {:.4}%", rel_err_gpu * 100.0);
    println!("Time:                {:.2?}", elapsed_gpu_bs);
    println!(
        "Speedup vs CPU: {:.1}x",
        elapsed_cpu_bs.as_secs_f64() / elapsed_gpu_bs.as_secs_f64()
    );

    assert!(
        rel_err_gpu < 0.01,
        "GPU MC price too far from analytical: {:.4}%",
        rel_err_gpu * 100.0
    );
    println!("PASS: GPU MC price within 1% of analytical\n");

    // -----------------------------------------------------------------------
    // Summary
    // -----------------------------------------------------------------------
    println!("=== Summary ===");
    println!(
        "Pi estimation:    CPU {:.2?} ({n_pi} pts) vs GPU {:.2?} ({n_gpu_pi} pts)",
        elapsed_cpu_pi, elapsed_gpu_pi,
    );
    // Normalized speedup: (cpu_time/cpu_N) / (gpu_time/gpu_N)
    let pi_norm_speedup = (elapsed_cpu_pi.as_secs_f64() / n_pi as f64)
        / (elapsed_gpu_pi.as_secs_f64() / n_gpu_pi as f64);
    println!("  Normalized throughput speedup: {pi_norm_speedup:.1}x");
    println!(
        "Black-Scholes:    CPU {:.2?} ({n_bs} paths) vs GPU {:.2?} ({n_gpu_bs} paths)",
        elapsed_cpu_bs, elapsed_gpu_bs,
    );
    let bs_norm_speedup = (elapsed_cpu_bs.as_secs_f64() / n_bs as f64)
        / (elapsed_gpu_bs.as_secs_f64() / n_gpu_bs as f64);
    println!("  Normalized throughput speedup: {bs_norm_speedup:.1}x");
    println!("\n=== All tests passed ===");
}
