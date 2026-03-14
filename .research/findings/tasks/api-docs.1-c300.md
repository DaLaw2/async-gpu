# api-docs.1 — Module-level documentation for gpu-host

## Summary

Added comprehensive `//!` module-level documentation to two files in the
`gpu-host` crate:

### `crates/core/gpu-host/src/lib.rs`

- Expanded the existing overview to explain what the crate does (host-side SDK
  for kernel launch + hostcall protocol)
- Added a numbered usage pattern showing the typical workflow
  (GpuRuntime → PTX → HostcallBuffer → HostcallSession → launch → sync → shutdown)
- Added a `no_run` code example demonstrating the pattern end-to-end
- Added a "Key types" section listing all major public types with one-line descriptions
- Preserved existing "Core modules" and "Optional modules" sections

### `crates/core/gpu-host/src/hostcall.rs`

- Expanded from 3-line summary to full protocol documentation
- Documented the 6-step hostcall protocol flow (GPU acquires slot → writes
  doorbell → host polls → dispatches → writes response → GPU reads)
- Explained the sideband buffer for bulk data exceeding 56 bytes
- Documented sharding model (per-block shard assignment)
- Documented the FdResource model (unified fd table for File + TcpStream +
  TcpListener)
- Added "Key types" section listing HostcallBuffer, HostcallSession, Pipeline,
  CommandBuffer, FlightRecorder, HostcallError

## Verification

- `cargo +stable fmt` — clean
- `cargo +stable clippy -- -D warnings` — clean, no warnings
- No code logic modified, docs only
