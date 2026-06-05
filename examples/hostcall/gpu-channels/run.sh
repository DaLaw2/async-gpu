#!/usr/bin/env bash
# Run the gpu-channels example — uses pre-built kernels from gpu-kernel-std.
set -e
cd "$(dirname "$0")"
cargo run --release "$@"
