//! GPT-2 BPE tokenizer — host-side only.
//!
//! Wraps `tiktoken-rs` with the `r50k_base` encoding (GPT-2's byte-level BPE).

use std::fmt;

/// GPT-2 vocabulary size: 256 base bytes + 50,000 merges + 1 special token.
pub const GPT2_VOCAB_SIZE: usize = 50257;

/// Token ID for the `<|endoftext|>` special token.
pub const ENDOFTEXT_TOKEN_ID: u32 = 50256;

/// Error type for tokenizer operations.
#[derive(Debug)]
pub enum TokenizerError {
    /// Failed to initialize the tiktoken encoding.
    InitError(String),
    /// Failed to decode token IDs back to text.
    DecodeError(String),
}

impl fmt::Display for TokenizerError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TokenizerError::InitError(msg) => write!(f, "tokenizer init error: {msg}"),
            TokenizerError::DecodeError(msg) => write!(f, "tokenizer decode error: {msg}"),
        }
    }
}

impl std::error::Error for TokenizerError {}

/// GPT-2 BPE tokenizer wrapping tiktoken-rs `r50k_base` encoding.
pub struct Gpt2Tokenizer {
    bpe: tiktoken_rs::CoreBPE,
}

impl Gpt2Tokenizer {
    /// Create a new GPT-2 tokenizer.
    pub fn new() -> Result<Self, TokenizerError> {
        let bpe =
            tiktoken_rs::r50k_base().map_err(|e| TokenizerError::InitError(format!("{e}")))?;
        Ok(Self { bpe })
    }

    /// Encode text into GPT-2 token IDs.
    pub fn encode(&self, text: &str) -> Vec<u32> {
        self.bpe.encode_with_special_tokens(text)
    }

    /// Decode token IDs back to text.
    pub fn decode(&self, tokens: &[u32]) -> Result<String, TokenizerError> {
        self.bpe
            .decode(tokens.to_vec())
            .map_err(|e| TokenizerError::DecodeError(format!("{e}")))
    }

    /// Return the GPT-2 vocabulary size (50,257).
    pub fn vocab_size(&self) -> usize {
        GPT2_VOCAB_SIZE
    }
}
