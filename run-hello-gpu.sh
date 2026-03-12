#!/usr/bin/env bash
# Run the hello-gpu example — builds kernel + host automatically.
# Prerequisites: Rust nightly (auto-installed via rust-toolchain.toml), NVIDIA GPU with CUDA driver.
set -e
cd "$(dirname "$0")/examples/hello-gpu/host"
cargo run --release "$@"
