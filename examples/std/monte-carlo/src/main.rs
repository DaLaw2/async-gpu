// Monte Carlo simulations: xoshiro256++ PRNG, Pi estimation, Black-Scholes pricing
//
// CPU reference implementation for mc-finance theme.
// GPU kernels will be added in subsequent tasks.

use std::f64::consts::PI;
use std::time::Instant;

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
// Monte Carlo Pi estimation
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
// Monte Carlo Black-Scholes pricing
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
// Main
// ---------------------------------------------------------------------------

fn main() {
    println!("=== Monte Carlo Simulations (CPU Reference) ===\n");

    // --- Pi estimation ---
    let n_pi: u64 = 10_000_000;
    let seed_pi: u64 = 12345;

    println!("--- Pi Estimation ---");
    println!("Points: {n_pi}");

    let t0 = Instant::now();
    let (pi_est, inside) = estimate_pi(n_pi, seed_pi);
    let elapsed_pi = t0.elapsed();

    let pi_err = (pi_est - PI).abs();
    println!("Inside circle: {inside}");
    println!("Pi estimate:   {pi_est:.6}");
    println!("True Pi:       {PI:.6}");
    println!("Abs error:     {pi_err:.6e}");
    println!("Time:          {:.2?}", elapsed_pi);

    assert!(
        pi_err < 0.01,
        "Pi estimate too inaccurate: error = {pi_err}"
    );
    println!("PASS: Pi accurate to ~{} decimal places\n", {
        let digits = -pi_err.log10();
        digits.floor() as u32
    });

    // --- Black-Scholes option pricing ---
    let n_bs: u64 = 1_000_000;
    let s0 = 100.0;
    let k = 100.0;
    let r = 0.05;
    let sigma = 0.2;
    let t = 1.0;
    let seed_bs: u64 = 67890;

    println!("--- Black-Scholes European Call (Monte Carlo) ---");
    println!("Paths: {n_bs}");
    println!("S0={s0}, K={k}, r={r}, sigma={sigma}, T={t}");

    let analytical = black_scholes_call(s0, k, r, sigma, t);
    println!("Analytical BS price: {analytical:.4}");

    let t0 = Instant::now();
    let (mc_price, std_err) = mc_black_scholes(n_bs, s0, k, r, sigma, t, seed_bs);
    let elapsed_bs = t0.elapsed();

    let rel_err = ((mc_price - analytical) / analytical).abs();
    println!("MC price:            {mc_price:.4} +/- {std_err:.4}");
    println!("Relative error:      {:.4}%", rel_err * 100.0);
    println!("Time:                {:.2?}", elapsed_bs);

    assert!(
        rel_err < 0.01,
        "MC price too far from analytical: {rel_err:.4}%"
    );
    println!("PASS: MC price within 1% of analytical\n");

    println!("=== All tests passed ===");
}
