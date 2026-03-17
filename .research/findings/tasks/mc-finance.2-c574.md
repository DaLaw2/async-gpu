# mc-finance.2: Black-Scholes Monte Carlo on GPU

## Task
Implement GPU-accelerated Black-Scholes European call option pricing using
cudarc + NVRTC-compiled CUDA C kernels with per-thread xoshiro256++ PRNG.

## Implementation

### CUDA kernel (`mc_bs_kernel`)
- Each of 4096 threads seeds its own xoshiro256++ via SplitMix64 (unique seed per thread)
- Each thread computes ~244 paths (1M / 4096), accumulating payoff sums locally
- Box-Muller transform for standard normal: `Z = sqrt(-2*log(u1)) * cos(2*pi*u2)`
- Terminal price: `S_T = S0 * exp(drift + vol * Z)` where `drift = (r - 0.5*sigma^2)*T`
- Payoff: `max(S_T - K, 0)`
- Thread writes partial sum to `payoffs[tid]`; host reduces

### Host side
- Compiles kernel via NVRTC, uploads parameters, launches 16 blocks x 256 threads
- Downloads 4096 partial sums, computes grand mean, discounts: `price = exp(-rT) * mean_payoff`
- Standard error estimated from variance of per-thread means

## Results

| Metric          | CPU          | GPU          |
|-----------------|-------------|-------------|
| Paths           | 1,000,000   | 1,000,000   |
| Price           | 10.4359     | 10.4412     |
| Analytical      | 10.4506     | 10.4506     |
| Rel error       | 0.14%       | 0.09%       |
| Std error       | 0.0147      | 0.0148      |
| Time            | 23.74ms     | 3.44ms      |
| **Speedup**     | —           | **6.9x**    |

Both CPU and GPU prices are within 1% of the analytical Black-Scholes price (~10.45).

## Key Design Decisions
- **Per-thread PRNG** avoids synchronization; each thread has independent random stream
- **Partial sums per thread** avoid atomicAdd on doubles (not natively supported pre-sm_60)
- **4096 threads** balances occupancy vs per-thread work (~244 paths each)

## Files Modified
- `examples/std/monte-carlo/Cargo.toml` — added cudarc dependency
- `examples/std/monte-carlo/src/main.rs` — added CUDA kernel + GPU host code
