#!/usr/bin/env bash
# Run the async-pipeline example — builds kernel + host automatically.
set -e
cd "$(dirname "$0")/host"
cargo run --release "$@"
