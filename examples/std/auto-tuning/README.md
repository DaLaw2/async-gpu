# Auto-Tuning

Warmup-based parameter search to find the optimal GPU kernel block size.

## What It Demonstrates

- `AutoTuner::new()` with default candidate block sizes [32, 64, 128, 256, 512, 1024]
- `tune_block_size()` runs warmup + timed iterations per candidate
- `TuningCache` stores results keyed by (kernel, problem-size bucket, device)
- `tune_or_cached()` skips re-tuning if a cached result exists
- `format_report()` produces a human-readable comparison table
- Speedup comparison: auto-tuned vs default block size

## Running

```bash
cargo run -p auto-tuning --release
```

## Key Results

The auto-tuner evaluates 6 block-size candidates and selects the one with the lowest median execution time. Typical speedups of 1.1-2x over a naive default block size.
