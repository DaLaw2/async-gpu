# Brainstorm 54 — Skeptic Challenge: `gpu-autonomous` v2

**Role**: Skeptic
**Epic**: GPU as Autonomous Compute Environment

---

## Challenges to Each Success Criterion

### Criterion 1: Persistent Kernel (Command Buffer Loop)

**Untested assumptions:**

- The kernel can spin-wait on a shared memory command buffer without being killed by the OS. This is the most dangerous assumption in the entire epic and is directly contradicted by the Windows TDR constraint (see next section).
- A GPU kernel can usefully process heterogeneous commands (compute, print, exit) in a loop. But what is the value proposition vs. simply launching three short kernels? The persistent kernel pattern exists in CUDA (e.g., CUDA Persistent Threads) and is well-understood — this is not novel. The only novel aspect would be doing it in Rust, which is a thin veneer over an existing CUDA pattern.
- The command buffer protocol needs to be designed from scratch. The existing hostcall protocol goes GPU→host. A command buffer goes host→GPU. These are fundamentally different directions. The hostcall infrastructure does NOT directly help here — we need a new host→GPU signaling mechanism.

**What could fail in practice:**

- On Windows (the development environment), TDR will kill the kernel before it processes more than 1-2 commands unless the user modifies registry settings. This makes the demo unreliable and non-portable.
- If the kernel spin-waits on a command flag, it wastes GPU power and generates heat for zero useful work. A sleeping/yield mechanism does not exist on GPU — there is no `__sleep()` equivalent that actually deschedules a warp.
- Single-SM workload constraint means the persistent kernel occupies one SM indefinitely, blocking other GPU work. On a modern GPU with 80+ SMs this is tolerable, but it establishes a bad pattern.

**Is this just reimplementing CUDA?**

Yes. CUDA persistent threads/kernels are a 10+ year old pattern. The Rust angle adds type safety but the fundamental approach is identical. The question is: does wrapping it in Rust justify the engineering effort?

### Criterion 2: Cross-Launch Persistent State

**This criterion is trivially satisfied by existing CUDA semantics.**

- `cuMemAlloc` / `cuMemHostAlloc` memory persists until `cuMemFree` is called. This is not a feature to build — it is how CUDA already works.
- The existing `alloc_mapped_result_array` and `alloc_mapped_u64_array` functions already demonstrate this: they allocate mapped memory, pass it to kernels, and the memory survives until explicitly freed.
- The tests already demonstrate multi-launch patterns: `run_branching_pipeline_test` launches TWO kernels sequentially (run 1 and run 2) with file state persisting between them on disk.
- Memory ordering between kernel launches: CUDA guarantees that if kernels are launched in the same stream (default stream), all writes from kernel A are visible to kernel B. This is specified in the CUDA programming guide, section 3.2.6.5. There is nothing to implement.

**The real question:** Is there anything genuinely new here beyond writing a helper function that wraps `cuMemHostAlloc` with a nicer API? If the criterion is "prove that device memory persists across launches" — that is a CUDA hello-world, not research.

**Counter-proposal:** If cross-launch state is worth pursuing, the interesting challenge is **typed persistent buffers with lifetime safety** — ensuring at the Rust type level that Kernel B cannot be launched until Kernel A has completed, and that the buffer type seen by Kernel B matches what Kernel A wrote. This would be genuinely novel. Raw "don't free the buffer" is not.

### Criterion 3: GPU-Driven Autonomous Workflow

**This criterion is ALREADY MET by existing code.**

Evidence from the codebase:

1. **`file_transform_pipeline`** (pipeline.rs, lines 11-369): The GPU autonomously sequences 8 I/O steps + 1 compute step: open(in) → read → transform → open(out) → write → close(in) → close(out) → print. Zero CPU intervention between steps. The GPU decides the entire workflow.

2. **`branching_pipeline`** (pipeline.rs, lines 371-752): The GPU autonomously decides next operations based on intermediate results. It tries to open a file, and depending on whether the open succeeds or fails, takes a different branch (close+print vs. create+write+close+print). This is exactly "kernel autonomously decides next operations based on intermediate results."

3. **`autonomous_pipeline`** (tests_pipeline.rs, lines 862-982): This test is literally called "Autonomous Pipeline" and runs 3 modes:
   - Mode 0: File write pipeline (create → write → close) — GPU-driven
   - Mode 1: File read + classify pipeline (open → read → close → **branch based on data size**) — GPU-decided branching
   - Mode 2: Roundtrip pipeline (create → write → close → reopen → read → verify) — 6 hostcall steps, GPU-driven

   The test summary explicitly states: "GPU autonomously chose processing paths via match" and "GPU branched on hostcall results via if/else" and "13 total hostcall steps across 3 pipelines, zero host orchestration."

**The criterion says** "open file → read → process → write" — this is EXACTLY what `file_transform_pipeline` already does.

**What would be genuinely new?** The existing demos use hardcoded file paths. A truly autonomous workflow would need the GPU to compute derived values (e.g., construct a file path from data, make decisions based on numeric thresholds computed from file contents). But this is incremental, not epic-level work.

### Criterion 4: Hostcall Session Persistence

**Untested assumptions:**

- The hostcall buffer can be safely reused across kernel launches without reinitialization. Currently, every test creates a new `HostcallBuffer::new()` and a new listener thread per kernel launch. No test reuses a buffer.
- The listener thread (which runs `listen_unified`) exits when `signal_shutdown()` is called. To persist across launches, we need a "session mode" where the listener stays alive. This requires refactoring the listen loop to not exit after shutdown.

**What could fail:**

- **In-flight packets**: If Kernel A terminates while a packet is in FILLED state (submitted but not yet processed by host), the listener will process it after Kernel A exits. When Kernel B starts and the listener is still running, the free stack state must be consistent. If the listener hasn't finished releasing the packet, Kernel B may see a corrupted free stack.
- **Free stack / ready stack state**: When a kernel exits, some packets may be in various states:
  - FREE (on free stack) — safe for Kernel B to use
  - FILLED (submitted, not yet processed) — host will process these and release them, but Kernel B doesn't know when
  - READY (processed, waiting for GPU to release) — Kernel A died before releasing, so these slots are leaked until someone cleans them up

  A new kernel starting with stale READY packets in the buffer will find fewer free slots than expected. Over multiple launches, this leaks all packet slots.

- **Shutdown signaling**: Currently `signal_shutdown()` sets a flag. A "session mode" would need a different mechanism — maybe a per-kernel launch barrier or epoch counter, rather than a simple shutdown flag.

**This is the one criterion with genuine engineering complexity.** The hostcall protocol was designed for single-launch use. Making it multi-launch-safe requires careful consideration of packet lifecycle across kernel boundaries.

---

## Windows TDR Problem — The #1 Blocker

**Facts:**
- Windows TDR (Timeout Detection and Recovery) default timeout: **2 seconds**.
- Any GPU kernel exceeding this timeout causes the Windows display driver to reset the GPU, killing the kernel and potentially blue-screening the system.
- The TDR timeout is controlled by `HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\GraphicsDrivers\TdrDelay` (DWORD, in seconds).
- Changing this requires administrator privileges AND a system reboot.

**Impact on Criterion 1 (Persistent Kernel):**

A persistent kernel, by definition, runs indefinitely. On Windows, it will be killed after 2 seconds unless TDR is modified. This makes the "persistent kernel" demo:

1. **Unreliable by default** — it will crash on any unmodified Windows system
2. **Require system modification** — violates the project's own "Host Environment Policy" (CLAUDE.md: "No modifying PATH, environment variables, or system config")
3. **Non-portable** — Linux does not have TDR (no display driver timeout), so the demo would only work on Linux or modified Windows

**Possible workarounds (all flawed):**

- **Short command loops**: Process commands quickly and exit the kernel before TDR triggers. But this defeats the purpose of a "persistent" kernel — it becomes a regular kernel that processes one batch.
- **TDR reset trick**: Some CUDA programs periodically launch a dummy kernel to reset the TDR watchdog. But this requires the persistent kernel to exit and relaunch, which is circular.
- **Compute-only GPU**: If the GPU is not driving a display, TDR may not apply. But this requires a dual-GPU setup, which most developers don't have.
- **WDDM → TCC mode**: NVIDIA GPUs can be switched to TCC (Tesla Compute Cluster) mode, which disables TDR. But TCC is only available on Tesla/datacenter GPUs, not consumer GeForce cards.

**Recommendation:** If the persistent kernel criterion stands, the epic MUST document TDR as a hard requirement and either: (a) accept it only works on Linux / modified Windows, or (b) redefine "persistent" to mean "long-running but finite" (e.g., processes N commands then exits).

---

## Hostcall Session Persistence — Deep Dive

**Current architecture (per test):**

```
1. HostcallBuffer::new() — allocates pinned memory, zeros it
2. Arc::new(hc_buf)
3. thread::spawn → hc_buf.listen(callback) — starts polling loop
4. kernel.launch() — GPU runs, uses hostcall
5. dev.synchronize() — wait for kernel exit
6. signal_shutdown() — stop listener
7. listener_handle.join() — wait for listener thread exit
8. [HostcallBuffer dropped — cuMemFreeHost]
```

**For session persistence, steps 1-3 happen once, steps 4-5 repeat, step 6-8 happen once at session end.**

**Problems:**

1. **listen() blocks on shutdown flag**: The listen loop checks `self.shutdown().load()` on every iteration. For session mode, we need a different exit condition — perhaps "shutdown only after the Nth kernel launch" or "shutdown on explicit session.close()."

2. **Listener thread state**: The listener owns a `HashMap<u64, File>` for open file handles. If Kernel A opens files and doesn't close them, those handles persist. Kernel B could potentially use Kernel A's file descriptors — but the GPU kernel has no way to know what FD numbers Kernel A opened. This could be a feature (pass FDs via persistent buffer) or a bug (leaked handles accumulate).

3. **Doorbell counter**: The listener uses `last_doorbell` to detect new packets. If the counter wraps or if Kernel B starts writing packets before the listener has drained Kernel A's packets, there could be missed or double-processed packets.

4. **Race between kernel exit and packet processing**: When `dev.synchronize()` returns, the GPU is done, but the host listener thread may still be mid-processing a packet. Starting Kernel B immediately could cause the listener to interleave processing of Kernel A's stale packets with Kernel B's new packets. This is a subtle concurrency bug.

---

## Cross-Launch State — Why It's Trivial

CUDA memory management 101:

```rust
// This already works:
let buf = alloc_mapped_u64_array(&dev, 1024);  // persists until free
launch_kernel_a(buf);
dev.synchronize();
launch_kernel_b(buf);  // sees kernel A's writes — guaranteed by CUDA
dev.synchronize();
free_mapped_u64_array(buf);
```

The `run_branching_pipeline_test` function already does exactly this pattern — it launches `branching_pipeline` twice, and the second launch sees the file that the first launch created. The state persistence is via the filesystem (hostcall file I/O), but the mechanism is identical for device buffers.

**What would make this non-trivial:**
- Typed buffer passing with compile-time safety (Kernel A writes `[f32; 1024]`, Kernel B expects `[f32; 1024]`, type mismatch is a compile error)
- Pipeline builder API with automatic synchronization insertion
- Zero-copy buffer sharing between kernels on different streams (requires explicit stream synchronization)

Without these, the criterion is "pass the same pointer to two kernels" — a one-line change in the host code.

---

## Scope and Overlap

**Overlap with existing code:**

| Criterion | Overlap | Assessment |
|-----------|---------|------------|
| Persistent kernel | None (new pattern) | Blocked by TDR on Windows |
| Cross-launch state | Trivially possible with existing infrastructure | Not research-worthy without type safety angle |
| GPU-driven workflow | Already demonstrated by 3 existing kernels | Criterion already met |
| Hostcall session persistence | Requires real engineering work | Genuine and useful |

**Assessment:** 2 of 4 criteria are either already met or trivially achievable. 1 is blocked by a platform limitation. Only 1 (hostcall session persistence) represents genuine engineering work.

**Overlap with other epics:**
- The "Multi-Kernel Pipeline" proposal from bs53 (Epic 2) covers cross-launch state and kernel composition. The `gpu-autonomous` epic partially duplicates this.
- The "GPU Debugging & Observability" proposal from bs53 (Epic 3) is orthogonal and would benefit any work done here.

---

## Counter-Proposals

### 1. Replace "Persistent Kernel" with "Multi-Command Kernel"

Instead of an indefinitely running kernel (TDR-blocked), define:
- **Multi-command kernel**: A kernel that processes a batch of N commands from a command buffer, then exits. The host can relaunch it with a new batch.
- This is TDR-safe (each launch processes a finite batch), demonstrates the command buffer pattern, and avoids the platform limitation.
- Success criterion: kernel processes 3+ different command types from a shared memory command buffer in a single launch. No requirement for indefinite persistence.

### 2. Replace "Cross-Launch Persistent State" with "Typed Pipeline Builder"

Instead of proving that CUDA memory persists (which is a given):
- **Typed pipeline builder**: A `Pipeline<S>` type that tracks buffer types through stages. `pipeline.stage::<KernelA>().stage::<KernelB>()` where `KernelB::Input = KernelA::Output` is enforced at compile time.
- This is genuinely novel (no CUDA framework does this in Rust) and provides real engineering value.

### 3. Narrow "GPU-Driven Workflow" to Something Not Already Done

Since the existing `file_transform_pipeline`, `branching_pipeline`, and `autonomous_pipeline` already demonstrate GPU-driven autonomous workflows, the criterion should target something genuinely new:
- **Data-dependent iteration**: A kernel that reads data, processes it, and decides whether to loop (e.g., iterative solver that checks convergence). The existing demos do linear or branching pipelines, but none do loops.
- **Multi-file orchestration**: GPU opens file A, reads a list of filenames, then opens and processes each file in the list. This tests dynamic hostcall count (not known at compile time).

### 4. Keep Hostcall Session Persistence — It's the Best Criterion

This is the only criterion with:
- Real engineering complexity
- No existing solution
- Clear practical value (eliminates per-launch overhead)
- Testable success criteria (multiple launches, same listener thread, no buffer leak)

Expand this into the primary focus of the epic.

---

## Summary Verdict

The `gpu-autonomous` v2 epic as currently defined has significant problems:

1. **Criterion 1 is blocked** by Windows TDR. Either the scope must change or the criterion must acknowledge it only works on Linux.
2. **Criterion 2 is trivial** — CUDA memory persistence is not a feature to build.
3. **Criterion 3 is already done** — three existing kernels demonstrate GPU-driven autonomous workflows with branching.
4. **Criterion 4 is the only genuinely valuable criterion** — hostcall session persistence requires real work and delivers real value.

**Recommendation:** Reshape the epic to focus on (a) multi-command batch processing (TDR-safe persistent kernel pattern), (b) typed cross-launch state with compile-time safety, (c) genuinely new GPU-driven patterns (loops, data-dependent iteration), and (d) hostcall session persistence as the centerpiece.
