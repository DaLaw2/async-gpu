# /maintain patches — Regenerate std patches from patched-std/

Ensure `std-patches/` is up-to-date with the current `patched-std/` directory.

## Language
- Conversation: 繁體中文 | Files: English

## Steps

1. **Check** if `patched-std/` exists. If not → print `[SKIP] patched-std/ not present` and stop.
2. **Run** `bash scripts/gen-std-patches.sh`
3. **Verify** output:
   - `std-patches/*.patch` files exist
   - `std-patches/PATCHES.md` was generated
   - `scripts/apply-std-patches.sh` was regenerated
4. **Check** git diff on `std-patches/` — if patches changed, report what changed
5. **Report**: `[FIX] Regenerated N patches` or `[OK] Patches up-to-date`

## Rules
- Only run if `patched-std/` directory exists (it's gitignored, may not be present)
- Do NOT modify patched-std/ — only regenerate patches FROM it
- If gen-std-patches.sh fails, print `[ERR]` with the error and stop
