#!/usr/bin/env bash
# Run the vector-math example — builds kernel + host automatically.
set -e
cd "$(dirname "$0")/host"
cargo run --release "$@"
