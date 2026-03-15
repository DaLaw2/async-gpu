# Tokio GPU Offload

Async kernel launch + event streaming from a tokio runtime via `GpuTask`.

## What it demonstrates

- `AsyncGpuRuntime` — non-blocking device init and PTX loading
- `GpuTask::launch().await` — kernel launch without blocking the tokio executor
- `GpuTask::next_event().await` — async GPU event streaming (print messages)
- Concurrent host + GPU work via tokio task scheduling

## Running

```bash
cargo run --release
```

## How it works

The `GpuTask` struct orchestrates:
1. An `AsyncHostcallSession` — listener thread with tokio mpsc event channel
2. Kernel launch via `tokio::task::spawn_blocking` (offloads blocking CUDA calls)
3. Device synchronization via `spawn_blocking`

The hostcall listener stays as a dedicated `std::thread` for low-latency
doorbell polling, while events flow to tokio tasks via `tokio::sync::mpsc`.
