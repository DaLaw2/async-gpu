# channel-oneshot.3: Oneshot receiver as Future — demo with executor
**Cycle**: 327 | **Theme**: channel-oneshot | **Kind**: experiment | **Status**: done

## Summary
Added `channel_oneshot_demo` kernel that spawns 4 producer-consumer pairs communicating
via OneshotSlot<u32> channels. Each producer sends a value, each consumer polls until
the value arrives. Added host-side test `run_channel_oneshot_demo_test`. Compiles and
passes clippy on both GPU (nvptx64) and host targets.

## Implementation

### Kernel: `channel_oneshot_demo` (hostcall_kernels.rs)
- **OneshotProducer**: Future that writes value to OneshotSlot's value field, then
  release-stores SENT state. Completes in 1 poll.
- **OneshotConsumer**: Future that acquire-loads slot state. Returns Pending until
  SENT, then reads value and writes to result slot.
- Thread 0 creates 4 OneshotSlots after the executor in mapped memory, spawns
  4 consumer + 4 producer tasks (8 total), runs executor.
- Result layout: spawned, completed, tasks_executed, polls_total, values[4],
  channels_count, success_flag, phase_marker.

### Host test: `run_channel_oneshot_demo_test` (tests_scaling.rs)
- Same pattern as executor_demo: thread-based launch + phase polling + timeout
- Allocates 256KB + 256 bytes for executor + oneshot slots
- Asserts all 4 values received correctly (42, 100, 255, 1337)

### Key additions to gpu-runtime
- `OneshotSlot::state_ptr()` — public accessor for atomic state field
- `OneshotSlot::value_ptr()` — public accessor for value storage

## Verification
- `cargo +nightly-2026-03-11 build --release` (gpu-kernel): 0 errors, 18 pre-existing warnings
- `cargo +stable clippy -p gpu-host -- -D warnings`: 0 errors, 0 warnings

## Impact on Downstream Tasks
- **channel-oneshot theme**: All 3 tasks complete — theme success criteria met
- **channel-waker.2**: Can extend OneshotReceiver to store waker for wake-on-send
- **channel-demo.1**: Can use oneshot channels in the full pipeline demo
