Reading time: ~1 hour 45 min

---

TLDR: CacheBlend reuses precomputed KV caches for Retrieval-Augmented Generation (RAG) inputs by **selectively recomputing a small fraction of tokens (High-KV-Deviation tokens) per layer** and pipelining that recompute with KV loading — yielding large TTFT and throughput gains close to full prefill quality while enabling KV caches to be stored on slower, cheaper storage.

---

## 1. Motivation & Background

- **Context for the problem:**
  - Many LLM applications (e.g., RAG, QA) prepend multiple retrieved text chunks to a user query to provide relevant context.
  - Before generation, LLMs run a *prefill* over the entire input to produce the KV cache, which dominates **time-to-first-token (TTFT)** for long contexts.

- **Domain:**  
  - LLM inference / serving for retrieval-augmented workflows — efficient KV cache reuse and fusion across multiple context chunks.

- **Why this matters:**
  - Long prefill delays hurt latency-sensitive services and limit throughput and cost-efficiency of serving LLMs with large retrieved contexts.

- **Status quo:**
  - **Full prefill/recompute:** Compute KV for the entire concatenated input → highest quality, slowest latency.
  - **Prefix caching:** Reuse KVs only when a chunk is a prefix → limited reuse.
  - **Full KV reuse:** Reuse precomputed KVs without recomputing cross-attention → fast, but causes severe quality loss.

- **Gap in prior work:**
  - Real RAG inputs often concatenate **non-prefix chunks**, where naive KV reuse fails to deliver both speed and quality.
  - Existing methods either cannot handle non-prefix contexts or degrade output quality due to missing cross-attention effects.

---

## 2. Key Insight

- **Core realization:**  
  Cross-attention errors from KV reuse are **concentrated in a small subset of tokens**. Only about 10–20% of tokens (per layer) cause large KV deviations relative to a full prefill. By identifying and selectively recomputing these **High-KV-Deviation (HKVD)** tokens, CacheBlend achieves near-prefill generation quality at a fraction of the compute.

- **Why previous approaches fail:**  
  - *Full reuse:* Ignores cross-attention → high speed but low quality.  
  - *Full recompute:* Accurate but slow.  
  - *CacheBlend:* Bridges the gap — selectively recomputes only HKVD tokens and pipelines recompute with KV loading, preserving quality while reducing latency.

---

## 3. The CacheBlend Approach

### 3.1 Architecture / Method Overview

- **High-level design:**  
  CacheBlend stores precomputed KVs for text chunks, and at request time **fuses** multiple retrieved chunk KVs into one KV cache for the concatenated input.  
  It does this by:
  1. Loading cached KVs layer-by-layer.
  2. Selectively recomputing HKVD tokens for each layer.
  3. Pipelining KV loading and recompute to hide recompute latency.

- **Key components:**
  - **KV Cache Store / Manager:**  
    Hash-indexed storage of per-chunk KVs with LRU eviction.
  - **Loading Controller:**  
    Decides recompute ratio (`r%`) and storage tier (GPU, CPU RAM, or NVMe) using offline profiled compute and IO metrics.
  - **Fusor:**  
    Layer-wise executor that overlaps KV loading with HKVD recompute.
  - **LLM Inference Engine:**  
    Runs decoding using the final fused KV cache (built on top of vLLM).

### 3.2 Core Techniques

1. **KV deviation and attention deviation**  
   - For token *j* at layer *i*:  
    $\Delta_{KV}(i, j) = |KV_i[j] - KV^{full}_i[j]$

   - Goal: minimize **attention deviation** with minimal recomputation.

2. **Selective KV recompute (per layer)**  
   - Compute Q/K/V **only for selected tokens**, not all tokens.
   - Expand K/V for unselected tokens by reusing cached entries.
   - Attention operates over all tokens, preserving context integrity.

3. **HKVD token selection**  
   - Tokens with highest KV deviation are recomputed first.
   - Ground-truth deviation is unavailable → CacheBlend gradually **filters down** across layers:
     - Start with larger set early, reduce ratio at deeper layers (`r1% > r2% > ...`).
   - ~15% recompute per layer (`r*%`) balances quality and speed.

4. **Pipelining recompute with KV loading**  
   - If recompute delay ≤ KV load delay, recompute cost is fully hidden.
   
   - **Controller model:**
     - Recomputation delay:  
      $T_{recompute}(r\%, LLM, L) = r\% \times T_{prefill}(LLM, L)$

     - Load delay:  
       $T_{load}(LLM, L, device) = \frac{KV_{size} \times L}{Throughput_{device}}$
     
	 - Choose `r%` and device where \( T_{recompute} \approx T_{load} \).

---

## 3.3 Serving System

### Deployment & Placement Strategy
- Implemented on **vLLM** (~3K lines, PyTorch 2.0).
- Single-node focus with KVs stored in one tier: GPU memory, CPU RAM, or NVMe SSD.

### Scheduling & Load Balancing
- Per-request controller optimizes recompute vs IO.
- No multi-node or global scheduling yet — left for future work.

### Request Flow
1. **Retriever** fetches relevant chunks.
2. **Loading Controller** checks cache and selects recompute ratio (`r%`).
3. **KV Cache Store** fetches per-layer KVs into GPU queues.
4. **Fusor** pipelines:
   - Recompute HKVD tokens for current layer.
   - Load next layer KVs concurrently.
5. **Fused KV Cache** is finalized and passed to LLM decoder.

---

## 4. Performance & Evaluation

- **Headline improvements:**
  - **TTFT reduced by 2.2–3.3×** vs full prefill.
  - **Throughput increased by 2.8–5×** vs full prefill.

- **Quality:**
  - Near-identical to full recompute (negligible drop in F1 / Rouge-L).
  - Substantially higher quality than naive full KV reuse.

- **Default recompute ratio:**
  - ~15% (`r*%`) per layer is a good balance for speed and quality.

- **Storage tradeoffs:**
  - Enables KV caches to live on slower devices (e.g., NVMe) without hurting TTFT.
  - Example: recompute per layer = 3 ms vs NVMe load per layer = 16 ms → recompute hidden under load.

- **Datasets & models:**
  - Evaluated on 4 datasets (e.g., Musique, 2WikiMQA).
  - Tested with Mistral-7B, Yi-34B, Llama-70B.
  - Typical RAG input: ~6 retrieved chunks × 512 tokens each.

---

## 5. Limitations & Unimplemented Features

- Single-node focus; no multi-node shared KV caches yet.
- Transformer-specific assumption; other architectures untested.
- Depends on accurate profiling for controller decisions.
- Does not address cluster-level scheduling or straggler mitigation.
- Limited exploration of quantization and diverse model types.

---

## 6. Broader Impacts & Future Directions

- **Implications:**
  - Makes low-latency RAG feasible for production systems.
  - Reduces serving costs by leveraging cheaper storage tiers.
  - Improves scalability of assistant-like services and enterprise search.

- **Future work:**
  - Integrate into distributed systems for multi-node serving.
  - Combine with KV compression techniques.
  - Explore adaptive recompute strategies based on real-time workload metrics.

- **Sustainability:**
  - By increasing throughput and reducing hardware costs, CacheBlend can lower energy consumption per request.

---

## 7. Takeaway Summary

- **Core Idea:**  
  Recompute only the small set of tokens whose cached KVs cause the largest cross-attention deviation and pipeline that recompute with KV loading, yielding near-prefill quality at much lower cost.

- **Key Contributions:**
  - Selective KV recompute guided by High-KV-Deviation token selection.
  - Loading controller to match recompute and IO latency.
  - Full vLLM-based system implementation with pipelined fusor.
  - Extensive evaluation across models and datasets showing significant speedups.

- **Benefits:**
  - 2.2–3.3× TTFT reduction.
  - 2.8–5× throughput improvement.
  - Near-prefill output quality.
  - Supports cheaper KV storage tiers (e.g., NVMe).

- **Limitations:**
  - Single-node focus, no distributed scheduler.
  - Transformer-only design.
  - Requires accurate profiling for optimal results.

---

## References

- [CacheBlend: Fast Large Language Model Serving for RAG with Cached Knowledge Fusion (arXiv)](https://arxiv.org/abs/2407.14057)
- [CacheBlend GitHub Code](https://github.com/SJTU-IPADS/CacheBlend)
