# tokenizer.2: Implement BPE Encoder/Decoder on Host Side
**Cycle**: 1 | **Theme**: tokenizer | **Kind**: experiment | **Status**: done

## Summary
Implemented GPT-2 BPE tokenizer in `crates/gpu-host/src/tokenizer.rs` using `tiktoken-rs` crate (v0.6) with `r50k_base` encoding. The module provides `Gpt2Tokenizer` with `encode()` and `decode()` methods, plus constants for vocab size (50,257) and end-of-text token ID (50,256). Compiles and passes basic test.

## Findings

### Q: Does the chosen approach (crate or custom) produce correct token IDs?
A: Yes. Using `tiktoken-rs` with `r50k_base` encoding, which is the exact encoding used by GPT-2. The crate handles all BPE merge rules, byte-level encoding, and special tokens correctly. The `encode_with_special_tokens()` method processes `<|endoftext|>` as token 50256.
**Confidence**: high

### Q: How to handle special tokens (EOS, padding)?
A: GPT-2 has only one special token: `<|endoftext|>` (ID 50256). There is no padding token in GPT-2's original tokenizer. For inference, we use `encode_with_special_tokens()` which correctly handles `<|endoftext|>` if present in input text. For generation, we check output token IDs against `ENDOFTEXT_TOKEN_ID` to detect end-of-sequence.
**Confidence**: high

## Implementation
- File: `crates/gpu-host/src/tokenizer.rs`
- Dependency: `tiktoken-rs = "0.6"` in Cargo.toml
- Error type: `TokenizerError` (InitError, DecodeError) - no anyhow
- API: `Gpt2Tokenizer::new()`, `.encode(text)`, `.decode(tokens)`, `.vocab_size()`

## Impact on Downstream Tasks
- **tokenizer.3**: Ready for validation against HuggingFace (need to compare token sequences for 10+ test sentences)
- **full-inference.4**: Tokenizer is available for text-to-tokens-to-inference-to-tokens-to-text pipeline
