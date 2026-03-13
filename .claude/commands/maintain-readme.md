# /maintain readme — Update outdated README.md sections

Check README.md against current project state and fix stale sections.

## Language
- Conversation: 繁體中文 | Files: English

## Steps

1. **Read** `README.md`
2. **Check each section**:
   - **Quick Start**: Do script paths exist? (`examples/hello-gpu/run.sh`, not `./run-hello-gpu.sh`)
   - **Capabilities**: Does it mention all working features? (e.g., std::fs, error propagation, KV cache)
   - **Limitations**: Are resolved limitations still listed? (e.g., "No KV cache" if KV cache now works)
   - **Examples/Demos**: Are all `examples/*/` represented?
   - **Architecture diagram**: Does crate list match actual `crates/`?
   - **Performance numbers**: Are they still accurate?
3. **Fix** only stale sections — do NOT rewrite the whole README
4. **Report**: `[FIX] Updated: {sections}` or `[OK] README is current`

## Rules
- Keep existing style and tone
- Only update factually wrong or outdated content
- Do NOT add new sections unless something major is missing
- Do NOT change performance numbers unless you have newer data
