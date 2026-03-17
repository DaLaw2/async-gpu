# mc-finance.1: GPU xoshiro256++ PRNG kernel with per-thread seeding

## Task
Implement CPU reference for Monte Carlo simulations: xoshiro256++ PRNG, Pi estimation, and Black-Scholes option pricing.

## Implementation

Created `examples/std/monte-carlo/` with:

### PRNG: xoshiro256++
- State: 4x u64 (`s0..s3`)
- SplitMix64 seeder: derives 4 state values from a single u64 seed
- `next_u64()`: standard xoshiro256++ with rotate_left(23) and rotate_left(45)
- `next_f64()`: upper 53 bits mapped to [0, 1)
- `next_normal()`: Box-Muller transform for standard normal samples
- Per-thread seeding: thread `i` gets `base_seed + i`

### Monte Carlo Pi Estimation
- 10M random (x, y) pairs in [0, 1)
- Count where x^2 + y^2 < 1.0, multiply by 4/N
- Result: Pi = 3.141585, abs error = 7.85e-6 (5 decimal places)
- Time: ~25ms on CPU (single-threaded, release mode)

### Black-Scholes European Call Pricing
- Parameters: S0=100, K=100 (ATM), r=5%, sigma=20%, T=1yr
- 1M GBM paths: S_T = S0 * exp((r - sigma^2/2)*T + sigma*sqrt(T)*Z)
- Discounted payoff: exp(-rT) * max(S_T - K, 0)
- Analytical BS price: 10.4506
- MC price: 10.4359 +/- 0.0147 (0.14% relative error)
- Time: ~24ms on CPU

### Analytical Black-Scholes
- `norm_cdf()` via erfc Horner-form rational approximation (error < 1.5e-7)
- Initial implementation used Abramowitz & Stegun formula 26.2.17 with erfc polynomial which had significant errors (N(1) = 0.8703 instead of 0.8413). Replaced with a 10-term Chebyshev rational approximation of erfc which achieves < 1.5e-7 accuracy.

## Key Finding
The commonly-cited "Abramowitz & Stegun" norm_cdf using `1/(1 + p*x)` polynomial is actually an approximation of erfc, not directly of the normal CDF. The version commonly copy-pasted online often has incorrect coefficient application. The erfc-based approach (Horner form with 10 Chebyshev coefficients) is more robust and equally fast.

## Files
- `examples/std/monte-carlo/Cargo.toml`
- `examples/std/monte-carlo/src/main.rs`

## Status
DONE — CPU reference works correctly. Ready for GPU kernel implementation in mc-finance.2.
