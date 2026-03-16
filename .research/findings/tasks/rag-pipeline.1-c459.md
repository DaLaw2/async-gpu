# rag-pipeline.1: How to orchestrate RAG in a single kernel with hostcall
**Cycle**: 459 | **Theme**: rag-pipeline | **Kind**: investigation | **Status**: done

## Summary
The hostcall framework fully supports multi-step sequential calls within a single kernel launch.
Both synchronous (spin-wait) and async (WarpFuture) patterns work. For the RAG demo, the
synchronous approach is simplest: kernel issues sequential hostcalls for query input, context
fetch, and result output, with GPU compute (embedding, search, GPT-2) between them.

## Findings
### Q: Can a GPU kernel make multiple sequential hostcall requests?
A: YES. Each `gpu_hostcall_request()` is independent — pop packet, fill, submit, spin-wait
for response, release. The kernel naturally blocks at each hostcall until the host responds.
Existing tests do 6+ sequential hostcalls (open, write, close, open, read, close).
**Confidence**: high

### Q: What's the best pattern for RAG orchestration?
A: Synchronous sequential pattern:
1. `gpu_hostcall_read(buf, query_fd, query_buf)` → get query text
2. GPU: tokenize → wte lookup → average → query embedding
3. GPU: matmul query × vector_store → similarity scores
4. GPU: top-K selection → retrieve context chunk indices
5. `gpu_hostcall_read(buf, context_fd, context_buf)` → fetch context text (or read from pre-loaded GPU memory)
6. GPU: GPT-2 inference on query + context → generated text
7. `gpu_hostcall_write(buf, result_fd, output_buf)` → return result

Alternatively: store all chunk texts in GPU memory (pre-loaded via hostcall at init), avoiding
step 5 hostcall entirely. Only steps 1 and 7 need hostcall I/O.
**Confidence**: high

### Q: Sideband buffer for large data?
A: Sideband (1MB) available for data >56 bytes. Chunks of text (up to 512 tokens × 4 bytes = 2KB)
and generated output easily fit. Use SERVICE_BULK_READ/WRITE for transfers.
**Confidence**: high

## Design Decision
Use **hybrid approach**:
- Pre-load vector store (embeddings + raw text) into GPU memory at kernel launch
- Kernel receives query via hostcall, does embedding + search + generation on GPU
- Returns generated text via hostcall
- Only 2 hostcall round-trips needed (query in, result out)

## Impact on Downstream Tasks
- rag-pipeline.2: implement the actual kernel
- rag-pipeline.3: standalone example
