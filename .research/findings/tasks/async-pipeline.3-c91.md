# async-pipeline.3: README overhaul — showcase the vision
**Cycle**: 91 | **Theme**: async-pipeline | **Kind**: design | **Status**: done

## Summary
Complete README rewrite to lead with the GPU autonomy vision. Previous README buried the value proposition under research structure details. New README opens with "What if the GPU could drive its own I/O?", shows the actual demo output immediately, explains the architecture, and provides runnable quick-start instructions.

## Findings
### Q: What is the most compelling way to present the project's value proposition?
A: Lead with the paradigm shift (GPU as autonomous compute environment), show the demo output as the first concrete thing the reader sees, then explain the "how." The pipeline diagram (open→read→transform→write→close) communicates the vision in one glance.

**Confidence**: high

### Q: How should the demo be structured so a user can run it immediately?
A: The demo runs as part of `cargo run --release` in `crates/gpu-host/`. No separate setup needed — the test creates `gpu_input.txt`, launches the kernel, verifies `gpu_output.txt`, and cleans up. Quick-start section gives the 3-command build sequence.

**Confidence**: high

## Key Changes
- Title: "GPU as Autonomous Compute Environment" (was: generic "Rust Async/Await on GPU")
- Opening: Hook question + one-sentence value prop
- Pipeline diagram: Shows 8 steps at a glance
- Demo output: Real terminal output from the test run
- "What Works" table: Capability matrix replaces verbose prose
- Simplified architecture diagram
- Trimmed performance section to essential data
- Moved research details to bottom
- Removed acknowledgments (integrated into Research section)

## Impact on Downstream Tasks
None — this is the final task in the async-pipeline theme.
