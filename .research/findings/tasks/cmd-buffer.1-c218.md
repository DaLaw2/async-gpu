# cmd-buffer.1: Command Buffer Protocol Design
**Cycle**: 218 | **Theme**: cmd-buffer | **Kind**: design | **Status**: done

## Summary
Design a host→GPU command buffer protocol using mapped memory. The host writes commands to a ring buffer; the GPU kernel polls and processes them. This enables a "multi-command kernel" that handles heterogeneous commands in a single launch.

## Design

### Memory Layout

```
Command Buffer (mapped memory, 4KB default):

Header (64 bytes, cache-line aligned):
  Offset  0: write_idx   (u64, atomic) — host increments after writing command
  Offset  8: read_idx    (u64, atomic) — GPU increments after processing command
  Offset 16: capacity    (u32)         — number of command slots
  Offset 20: flags       (u32)         — reserved
  Offset 24: reserved    [40 bytes]

Command Slot (64 bytes each):
  Offset  0: cmd_type    (u32)         — command type enum
  Offset  4: cmd_flags   (u32)         — per-command flags (reserved)
  Offset  8: payload     [56 bytes]    — command-specific data
```

### Command Types

```rust
// In gpu-protocol/src/lib.rs
pub const CMD_NOP: u32 = 0;       // No-op (for testing)
pub const CMD_COMPUTE: u32 = 1;   // Execute computation on device buffer
pub const CMD_PRINT: u32 = 2;     // Print a message via hostcall
pub const CMD_EXIT: u32 = 3;      // Kernel should exit its command loop

// Payload layouts:
// CMD_COMPUTE: slot[0..8] = input_ptr (u64), slot[8..16] = output_ptr (u64),
//              slot[16..20] = count (u32), slot[20..24] = op_code (u32)
// CMD_PRINT:   slot[0..4] = msg_len (u32), slot[4..56] = message bytes (52 max)
// CMD_EXIT:    no payload
```

### Protocol

**Host side (submit command):**
```
1. Read write_idx (Relaxed)
2. Compute slot_offset = header_size + (write_idx % capacity) * slot_size
3. Write cmd_type + payload to slot (volatile writes)
4. Increment write_idx with Release store
   → GPU sees new command after acquire-loading write_idx
```

**GPU side (poll and process):**
```
1. Load read_idx (local — only GPU writes this)
2. Load write_idx with Acquire (sys-scope — host may have written)
3. If read_idx == write_idx → no commands, nanosleep, loop
4. Read command at slot (read_idx % capacity)
5. Dispatch based on cmd_type:
   - CMD_COMPUTE: run computation
   - CMD_PRINT: call gpu_hostcall_print()
   - CMD_EXIT: break loop
6. Increment read_idx with Release store
7. Loop to step 1
```

**Synchronization model:**
- write_idx: host writes (Release), GPU reads (Acquire) — sys-scope atomics
- read_idx: GPU writes (Release), host reads (Acquire) — for backpressure
- Ring buffer bounded: host must check `write_idx - read_idx < capacity` before submit
- Ordering: host Release on write_idx → GPU Acquire on write_idx ensures payload visibility

### Host-Side API

```rust
/// A mapped-memory command buffer for host→GPU command submission.
pub struct CommandBuffer {
    host_ptr: *mut u8,
    dev_ptr: CUdeviceptr,
    size: usize,
    capacity: u32,
}

impl CommandBuffer {
    /// Allocate a command buffer with the given slot capacity.
    pub fn new(capacity: u32) -> Result<Self>;

    /// Get device pointer for kernel arg.
    pub fn dev_ptr(&self) -> CUdeviceptr;

    /// Submit a command. Blocks if buffer is full (waits for GPU to drain).
    pub fn submit(&self, cmd: Command) -> Result<()>;

    /// Submit CMD_EXIT.
    pub fn submit_exit(&self);

    /// Check how many commands are pending (not yet processed by GPU).
    pub fn pending_count(&self) -> u32;

    /// Reset indices to 0 (between kernel launches).
    pub fn reset(&self);
}

pub enum Command {
    Nop,
    Compute { input_ptr: u64, output_ptr: u64, count: u32, op_code: u32 },
    Print { msg: String },  // truncated to 52 bytes
    Exit,
}
```

### GPU-Side API

```rust
/// In gpu-runtime or gpu-kernel:

/// Read the next command from the command buffer.
/// Returns None if no commands available (caller should retry).
pub unsafe fn cmd_poll(cmd_buf: *const u8) -> Option<(u32, *const u8)> {
    let read_idx = read_local_u64(cmd_buf.add(8));  // GPU's own read_idx
    let write_idx = sys_load_acquire_u64(cmd_buf as *const u64);  // host's write_idx
    if read_idx >= write_idx {
        return None;
    }
    let capacity = read_u32(cmd_buf.add(16));
    let slot_idx = (read_idx % capacity as u64) as u32;
    let slot_ptr = cmd_buf.add(64 + slot_idx as usize * 64);
    let cmd_type = read_volatile_u32(slot_ptr);
    Some((cmd_type, slot_ptr.add(8)))  // (type, payload_ptr)
}

/// Acknowledge that the current command has been processed.
pub unsafe fn cmd_ack(cmd_buf: *mut u8) {
    let read_idx = read_local_u64(cmd_buf.add(8));
    sys_store_release_u64(cmd_buf.add(8) as *mut u64, read_idx + 1);
}
```

### Multi-Command Kernel Pattern

```rust
#[no_mangle]
pub unsafe extern "ptx-kernel" fn multi_cmd_kernel(
    hc_buf: *mut u8,
    cmd_buf: *const u8,
) {
    gpu_runtime::panic::gpu_panic_init(hc_buf);

    loop {
        match cmd_poll(cmd_buf) {
            Some((CMD_COMPUTE, payload)) => {
                // Execute compute operation
            }
            Some((CMD_PRINT, payload)) => {
                // Use hostcall to print
                let msg_len = read_u32(payload);
                gpu_hostcall_print(hc_buf, payload.add(4), msg_len);
            }
            Some((CMD_EXIT, _)) => {
                cmd_ack(cmd_buf);
                break;
            }
            Some((_, _)) => { /* unknown command, skip */ }
            None => {
                // No commands — nanosleep and retry
                nanosleep(1000);  // 1µs
                continue;
            }
        }
        cmd_ack(cmd_buf);
    }
}
```

### TDR Safety

The kernel processes a batch of commands and exits when CMD_EXIT is received. The host is responsible for submitting CMD_EXIT before TDR timeout (~2 seconds on Windows).

For a "soft persistent" pattern:
1. Host submits N commands + CMD_EXIT
2. Kernel processes all N + exits
3. Host calls `cmd_buf.reset()` and relaunches if more work pending
4. Combined with HostcallSession, the hostcall listener stays alive across relaunches

### Constants (gpu-protocol)

```rust
pub const CMD_BUF_HEADER_SIZE: usize = 64;
pub const CMD_SLOT_SIZE: usize = 64;
pub const CMD_OFF_WRITE_IDX: usize = 0;
pub const CMD_OFF_READ_IDX: usize = 8;
pub const CMD_OFF_CAPACITY: usize = 16;
pub const CMD_OFF_FLAGS: usize = 20;

pub const CMD_SLOT_OFF_TYPE: usize = 0;
pub const CMD_SLOT_OFF_FLAGS: usize = 4;
pub const CMD_SLOT_OFF_PAYLOAD: usize = 8;
```

## Findings

### Q: Ring buffer vs lock-free stack for host→GPU commands?
A: Ring buffer is simpler and sufficient. Commands are FIFO (ordering matters for COMPUTE → PRINT → EXIT sequence). A lock-free stack would require reversal for ordering. The hostcall protocol uses stacks because multiple GPU threads push concurrently — here only the host pushes, so no contention.
**Confidence**: high

### Q: Should the command buffer be part of HostcallSession?
A: No — keep them separate allocations. The hostcall buffer is GPU→host; the command buffer is host→GPU. They serve different purposes and have different lifecycles. A HostcallSession could optionally own a CommandBuffer, but the protocol should be independent.
**Confidence**: high

## Impact on Downstream Tasks
- cmd-buffer.2: Implement CommandBuffer in gpu-host + GPU-side polling in gpu-runtime
- cmd-buffer.3: Integration test with multi-command kernel
