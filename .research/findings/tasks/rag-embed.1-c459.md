# rag-embed.1: How to embed text chunks for GPU vector search
**Cycle**: 459 | **Theme**: rag-embed | **Kind**: investigation | **Status**: done

## Summary
Three approaches for text chunk embeddings: (1) GPT-2 forward_features() for 768-dim semantic
embeddings, (2) wte bag-of-words averaging for fast approximate embeddings, (3) pre-computed
Python embeddings. Recommended: Option 2 (wte averaging) for speed, Option 1 for quality.

## Findings
### Q: How to produce chunk embeddings for similarity search?
A: GPT-2 exposes `forward_features()` returning [seq_len, 768] normalized hidden states. Mean-pool
across sequence length to get a fixed 768-dim embedding per chunk. Alternatively, average wte
(token embedding table, 50257×768) lookups per token — no transformer forward pass needed, much
faster but less semantic.
**Confidence**: high

### Q: What infrastructure exists?
A: - `Gpt2Tokenizer` (tiktoken-based BPE, 50257 vocab)
   - `Embedding` layer with wte/wpe on GPU
   - `embedding_lookup` kernel in ops/reshape.rs
   - `forward_features()` on Gpt2Model
   - GEMM kernels for batched cosine similarity (query × matrix = similarity scores)
**Confidence**: high

## Design Decision
For the RAG demo, use **wte averaging** (Option 2):
- Pre-embed 1000+ text chunks by averaging their token embeddings
- Store as [N, 768] matrix on GPU
- Query embedding computed the same way
- Cosine similarity = normalize → matmul query[1,768] × store[768,N] → scores[1,N]
- Top-K via CPU sort (N=1000 is trivial)

This avoids the 68ms/token GPT-2 forward pass for each chunk, making the embedding
step near-instant. GPT-2 is reserved for the generation step.

## Impact on Downstream Tasks
- rag-embed.2: implement cosine similarity + top-K
- rag-embed.3: vector store with pre-embedded chunks
