# TCP Echo

Demonstrates GPU-initiated TCP networking via the async_gpu hostcall protocol. A GPU kernel connects to a local TCP echo server, sends a message, reads back the echo response, and reports the result to the host.

## What It Demonstrates

- GPU-initiated TCP connection via SERVICE_TCP_CONNECT hostcall
- Sending data over TCP from a GPU kernel via SERVICE_TCP_WRITE
- Reading TCP data back into GPU memory via SERVICE_TCP_READ
- Closing a TCP socket from GPU via SERVICE_TCP_CLOSE
- Using the async `block_on()` executor with `GpuTcpConnectFuture`, `GpuTcpWriteFuture`, `GpuTcpReadFuture`, and `GpuTcpCloseFuture`
- Host-side TCP echo server cooperating with GPU kernel execution

## How It Works

1. The **host** binds a TCP echo server to `127.0.0.1:0` (random port) and spawns it in a background thread.
2. The **host** launches the `tcp_echo_kernel` with the server's port number as an argument.
3. The **GPU kernel** (thread 0 only):
   - Connects to `127.0.0.1:{port}` using `GpuTcpConnectFuture`
   - Sends "Hello from GPU!" using `GpuTcpWriteFuture`
   - Reads the echo response using `GpuTcpReadFuture`
   - Prints the echoed message via hostcall PRINT
   - Closes the socket using `GpuTcpCloseFuture`
   - Writes the response length to the output buffer (or `0xDEAD` on error)
4. The **host** verifies that the response length matches "Hello from GPU!" (15 bytes).

## Running

```bash
# Linux/macOS
bash run.sh

# Windows
run.bat
```

## Expected Output

```
=== TCP Echo Example ===

[host] TCP echo server listening on 127.0.0.1:XXXXX
[host] CUDA device initialized.
[host] PTX module loaded.

--- TCP Echo: GPU connects, sends, reads echo ---
[echo] Accepted connection from 127.0.0.1:YYYYY
[echo] Received 15 bytes: "Hello from GPU!"
[echo] Echoed 15 bytes back
  [HOST] TCP CONNECT: "127.0.0.1:XXXXX" -> fd=0
  [HOST] TCP WRITE: fd=0 15 bytes written
  [HOST] TCP READ: fd=0 15 bytes read
[GPU] Hello from GPU!
  [HOST] TCP CLOSE: fd=0 (TCP STREAM) closed
[host] tcp_echo_kernel: PASSED (response length: 15, expected: 15)

=== TCP Echo example complete! ===
```

## Key Concepts

- **Hostcall TCP protocol**: The GPU kernel uses async Future types that internally submit SERVICE_TCP_CONNECT/WRITE/READ/CLOSE packets through the hostcall buffer. The host-side `HostcallBuffer::listen()` loop dispatches these to actual `std::net::TcpStream` operations.
- **Thread gating**: Only thread 0 performs the TCP I/O to avoid concurrent hostcall conflicts.
- **Async execution**: The `block_on()` executor polls each Future in a spin loop until the host completes the operation and writes the response packet.
- **Error handling**: If any step fails, the kernel writes `0xDEAD` to the output buffer and cleans up the socket.
