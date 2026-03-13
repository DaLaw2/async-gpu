#!/usr/bin/env bash
# Run the async-io example — builds kernel + host automatically.
set -e
cd "$(dirname "$0")/host"
cargo run --release "$@"
