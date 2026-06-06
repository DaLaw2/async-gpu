#!/usr/bin/env bash
# Run the par-iter example — launches pre-compiled par_iter kernels.
set -e
cd "$(dirname "$0")/host"
cargo run --release "$@"
