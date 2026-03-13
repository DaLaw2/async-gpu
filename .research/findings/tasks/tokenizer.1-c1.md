# tokenizer.1: GPT-2 Tokenizer Format Analysis
**Cycle**: 1 | **Theme**: tokenizer | **Kind**: investigation | **Status**: done

## Summary

GPT-2 uses byte-level BPE with two data files: `vocab.json` (50,257-entry token-to-ID map) and `merges.txt` (~50,000 merge rules). The encoding pipeline is: text -> regex pre-tokenization -> UTF-8 bytes -> byte-to-unicode mapping -> iterative BPE merges -> token IDs. Several high-quality Rust crates exist; `tiktoken-rs` is the most straightforward for GPT-2 (it calls GPT-2's encoding `r50k_base`), while HuggingFace `tokenizers` is the most flexible. A custom implementation is feasible in ~300-500 lines but unnecessary given mature crate options.

## Findings

### Q: What format are vocab.json and merges.txt?

**vocab.json**: A JSON object mapping token strings to integer IDs. Keys are BPE token strings (using the byte-to-unicode encoding, so bytes that are not printable ASCII appear as special unicode characters like `\u0120` for space-prefixed tokens). Values are integers 0-50256. Total entries: 50,257 (256 base byte tokens + 50,000 merged tokens + 1 special token `<|endoftext|>`).

Example entries:
```json
{
  "!": 0,
  "\"": 1,
  ...
  "Ġthe": 262,
  ...
  "<|endoftext|>": 50256
}
```

**merges.txt**: A text file starting with a header line `#version: 0.2`, followed by ~50,000 lines. Each line contains a space-separated pair of tokens representing a merge rule. Line position determines merge priority (earlier = higher priority). The pairs use the same byte-to-unicode encoded strings as vocab.json.

Example lines:
```
#version: 0.2
Ġ t
Ġ a
h e
i n
r e
...
```

**Confidence**: high

### Q: How does byte-level BPE work for GPT-2?

The encoding algorithm has these steps:

**Step 1: Pre-tokenization with regex**

Input text is split into chunks using the regex pattern:
```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

This pattern matches:
- English contractions: `'s`, `'t`, `'d`, `'m`, `'ll`, `'ve`, `'re`
- Optional space + letters: ` ?[a-zA-Z...]+`
- Optional space + digits: ` ?[0-9]+`
- Optional space + punctuation: ` ?[^\s\p{L}\p{N}]+`
- Trailing whitespace: `\s+(?!\S)` and `\s+`

This ensures merges never cross word/number/punctuation boundaries. "don't" becomes `["don", "'t"]`. Spaces are attached to the following word: "hello world" -> `["hello", " world"]`.

**Step 2: Byte-to-unicode mapping**

Each pre-tokenized chunk is UTF-8 encoded, then each byte is mapped to a unicode character via `bytes_to_unicode()`:

```python
def bytes_to_unicode():
    # Printable ranges map to themselves:
    bs = list(range(ord("!"), ord("~")+1))      # 33-126 (94 chars)
       + list(range(ord("¡"), ord("¬")+1))      # 161-172 (12 chars)
       + list(range(ord("®"), ord("ÿ")+1))      # 174-255 (82 chars)
    # Total: 188 "safe" bytes map to themselves
    # Remaining 68 bytes (0-32, 127-160, 173) map to chr(256+n)
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    cs = [chr(n) for n in cs]
    return dict(zip(bs, cs))
```

This avoids control characters and whitespace that would interfere with BPE processing. For example, byte 0x20 (space) maps to `\u0120` (`Ġ`), so " world" becomes `Ġworld` in the BPE token space.

**Step 3: BPE merge iteration**

The unicode-mapped string is treated as a sequence of characters. The algorithm:
1. Find all adjacent character pairs in the sequence
2. Look up each pair in `bpe_ranks` (built from merges.txt line positions)
3. Select the pair with the lowest rank (highest priority)
4. If no pair has a rank, stop — remaining sequence elements are the final tokens
5. Merge all occurrences of that pair into a single token
6. Repeat from step 1

```python
while True:
    bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float('inf')))
    if bigram not in self.bpe_ranks:
        break
    # merge all occurrences of bigram
    first, second = bigram
    # ... merge into first+second, rebuild word tuple
```

**Step 4: Token ID lookup**

Each resulting BPE token string is looked up in `vocab.json` (the `encoder` dict) to get the integer token ID.

**Decoding** is the reverse: token IDs -> token strings (via `decoder` dict) -> concatenate -> apply `byte_decoder` (unicode-to-byte reverse mapping) -> decode UTF-8 bytes to text.

**Confidence**: high

### Q: Can we use an existing Rust crate?

Yes. There are several mature options. Recommendation: **use `tiktoken-rs`** for simplicity, or **`tokenizers`** for flexibility. A custom implementation is feasible but unnecessary.

**Confidence**: high

## Tokenizer File Formats

### vocab.json
| Property | Value |
|----------|-------|
| Format | JSON object |
| Keys | BPE token strings (byte-to-unicode encoded) |
| Values | Integer token IDs (0-50256) |
| Size | 50,257 entries |
| Composition | 256 base bytes + 50,000 merges + 1 special (`<\|endoftext\|>`) |
| Download | `https://huggingface.co/openai-community/gpt2/resolve/main/vocab.json` |

### merges.txt
| Property | Value |
|----------|-------|
| Format | Text file, one merge per line |
| Header | `#version: 0.2` (first line) |
| Content | Space-separated token pairs |
| Lines | ~50,000 merge rules |
| Priority | Line position (earlier = higher priority) |
| Download | `https://huggingface.co/openai-community/gpt2/resolve/main/merges.txt` |

### Special Tokens
| Token | ID | Notes |
|-------|----|-------|
| `<\|endoftext\|>` | 50256 | End-of-text delimiter |
| (no padding token) | — | GPT-2 has no default pad token |
| (no unknown token) | — | Byte-level BPE can represent any input |

## Rust Crate Comparison

| Crate | Version | Pros | Cons | Recommendation |
|-------|---------|------|------|----------------|
| `tiktoken-rs` | 0.9.1 | Direct GPT-2 support via `r50k_base()`; MIT license; simple API; well-maintained | Bundles OpenAI's BPE data as embedded resources; limited to OpenAI model tokenizers | **Best for GPT-2 specifically** |
| `tokenizers` (HuggingFace) | 0.22.2 | Full pipeline (normalize, pre-tokenize, BPE, post-process); `BPE::from_file(vocab, merges)`; `from_pretrained("gpt2")`; Apache-2.0 | Heavier dependency tree (rayon, regex, aho-corasick); more complex API | **Best for flexibility / multiple models** |
| `bpe` (GitHub) | latest | ~4x faster than tiktoken, ~10x faster than HF; O(n) worst-case; incremental encoding; MIT license | Primarily targets newer OpenAI models (o200k); GPT-2 support unclear; newer/less battle-tested | **Best for performance-critical paths** |
| `kitoken` | latest | Multi-format compatibility (SentencePiece, HF, tiktoken, Mistral Tekken) | Less widely adopted | Consider if multi-format needed |
| Custom impl | — | Full control; minimal dependencies; ~300-500 LOC | Maintenance burden; must handle edge cases; regex dependency still needed | **Only if no external deps allowed** |

## Implementation Estimate (Custom)

If building from scratch:

- **Lines of code**: ~300-500 for core encode/decode
- **Key data structures**:
  - `HashMap<String, u32>` — vocab (token string -> ID)
  - `HashMap<u32, String>` — decoder (ID -> token string)
  - `HashMap<(String, String), u32>` — bpe_ranks (merge pair -> priority)
  - `[char; 256]` — byte_encoder lookup table
  - `HashMap<char, u8>` — byte_decoder reverse lookup
  - `HashMap<String, Vec<String>>` — BPE cache for memoization
- **Dependencies**: `regex` crate (for Unicode-aware pre-tokenization pattern), `serde_json` (for vocab.json parsing)
- **Complexity**: O(n * m) per token where n = token length, m = number of applicable merges; with caching, amortized O(n) for repeated tokens

## Impact on Downstream Tasks

- **tokenizer.2** (if created): Implementation task — can use `tiktoken-rs` with `r50k_base()` for immediate GPT-2 tokenization, or `tokenizers` with `Tokenizer::from_pretrained("gpt2")` for the HuggingFace pipeline approach.
- **Data files**: `vocab.json` and `merges.txt` can be downloaded from HuggingFace (`openai-community/gpt2`) or embedded via crate.
- **Token count**: GPT-2 context window is 1024 tokens; tokenizer output directly determines how much text fits.
- **No training needed**: We only need inference (encode/decode), not BPE training — this simplifies the implementation significantly.
