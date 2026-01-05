
Reading time: ~35 min

TLDR: Ring Attention extends the Blockwise Parallelism concept to distributed settings. While BPT solves the memory constraint, Ring Attention solves the Communication constraint. By organizing GPUs in a logical ring and overlapping the transfer of Key/Value blocks with the computation of the Attention mechanism, it allows context lengths to scale linearly with the number of devices (reaching millions of tokens) without the communication overhead of standard All-Gather methods.

---

## 1. Motivation & Background

- **Context for the problem:**
    
    - BPT (the previous paper) successfully broke the **Memory Wall** by processing attention in blocks. You can now fit infinite context... _if_ you have infinite time.
        
    - However, BPT on a single GPU is too slow for 1M+ tokens. You must distribute the sequence across multiple GPUs ($N_{gpu}$).
        
- **Status quo (The Bottleneck):**
    
    - Standard Sequence Parallelism (e.g., Megatron-SP) partitions the sequence but requires an **All-Gather** operation to reconstruct the full $K$ and $V$ matrices on every GPU before computing attention.
        
    - **The Cost:** For a 1M token sequence, the $K, V$ cache is Terabytes in size. Moving Terabytes across the network via `All-Gather` (quadratic communication complexity in some topologies, or just massive volume) becomes the new bottleneck.
        
- **Gaps or inefficiencies:**
    
    - The network is idle while the GPU computes. The GPU is idle while the network transmits. We need to do both simultaneously.
        

---

## 2. Key Insight

- **The core realization:**
    
    - We don't need _all_ Keys and Values on _every_ GPU at the same time.
        
    - GPU 1 only needs Block $K_2, V_2$ for a brief moment to compute the attention score between its Query ($Q_1$) and those keys. Once done, it can discard them.
        
    - **Topology:** If we arrange GPUs in a **Ring**, GPU 1 can pass its current $K, V$ block to GPU 2, while simultaneously receiving a new block from GPU $N$.
        
- **Why previous approaches fall short:**
    - `All-Gather` is "bursty" and global. Ring Attention is "streaming" and local (P2P).
---

## 3. The Ring Attention Approach

### 3.1 Architecture / Method Overview

- **Setup:**
    
    - Input sequence length $L$ is split across $N$ devices.
    - Each device holds a local query block $Q_i$ and a local KV block $K_i, V_i$.
    - Same as flash attention and BPT, which both Q, K, V are splited along row index, leading to $B_q * d$ for Q and $B_k * d$ for K, V.
        
- The Protocol (The Loop):
    
    The algorithm runs for $N$ steps (number of devices):
    
    1. **Compute:** Calculate Attention($Q_{local}, K_{current}, V_{current}$) using the BPT blockwise logic.
        
    2. **Communicate (Async):** Send $K_{current}, V_{current}$ to the **Next** GPU in the ring. simultaneously **Receive** the new block from the **Previous** GPU.
        
    3. **Update:** The "Received" block becomes the "Current" block for the next iteration.
        
    4. **Repeat:** Until the KV blocks have rotated all the way around the ring.
    5. **Finalize:** After $N$ iterations, each GPU has computed attention against all KV blocks for given sequence segment and pass to next layers. (thanks for the FFN as pointwise operation, no cross-GPU communication needed))

### 3.2 Core Techniques

- **Computation-Communication Overlap:**
    
    - This is the "secret sauce." Because the computation of a large block (e.g., matrix multiplication of 4096 tokens) takes significant time, the network transmission of the _next_ block can happen entirely in the background.
        
    - **Zero-Overhead Goal:** If $T_{compute} > T_{communicate}$, the communication cost is effectively **zero** (perfectly hidden), to achieve the zeero-overhead, it required 1) $4dc^2/F ≥ 4cd/B$ where d is hidden size, c is block size, F is FLOPS per GPU, B is network bandwidth; 2) $s > 6c$ 
        
- **Blockwise Computing (Inherited from BPT):**
    
    - Inside the "Compute" step, it uses the BPT logic (fused FFN, nested loops) to ensure that the _local_ memory never spikes.

---

## 4. Performance & Evaluation

- **Metrics:**
    
    - **Context Length:** Validated up to **4 Million Tokens** (on Llama-13B) and potentially higher.
        
    - **MFU (Model FLOPs Utilization):** How efficient is the hardware usage?
        
- **Key Results:**
    
    - **Linear Context Scaling:** If you double the number of GPUs, you can process 2x the context length with roughly the same latency.
        
    - **Beating Megatron-SP:** Ring Attention achieved **1.1x to 1.4x higher throughput** than Megatron-LM's sequence parallelism because it eliminated the `All-Gather` stop-and-wait overhead.
        
    - **Near-Infinite Potential:** The paper demonstrated that the only limit to context length is the number of GPUs you can afford.
        
---

## 5. Limitations & Unimplemented Features

- **Load Imbalance (Causal Masking):**
    
    - In a standard ring, every GPU does the same work. But for **Causal (Decoder-only)** models, tokens can only attend to past tokens.
        
    - _The Problem:_ The GPU holding the first 10k tokens has very little work (few past tokens). The GPU holding the last 10k tokens has to attend to _everything_.
        
    - _Result:_ GPU 1 finishes early and sits idle waiting for GPU $N$. (The authors propose a "Striped" load balancing technique to mitigate this, but it adds complexity).
        
- **Network Bandwidth Requirement:**
    
    - For the overlap to work ($T_{compute} > T_{communicate}$), you need fast interconnects (NVLink or InfiniBand). On standard Ethernet, the communication time might dominate, making the ring stall.
        

---

## 6. Broader Impacts & Future Directions

- **The "Needle in a Haystack" Solved:** This paper (along with BPT) is the engineering foundation that enabled **Gemini 1.5 Pro (1M context)** and **Claude 3**.
    
- **Video/Code Understanding:** Enabling 1M+ context allows passing entire hour-long videos or entire repositories into the prompt, changing how we interact with LLMs.
    

---

## 7. Takeaway Summary

- **Core Idea:** Pass KV blocks in a circle. Compute on the current block while receiving the next one.
    
- **Key Contributions:**
    
    - Applied **Ring Topology** to Transformer Attention.
        
    - Achieved **Overlap** of Comm and Compute for sequence parallelism.
        
    - Validated **multi-million token** training.
        
- **Comparison to BPT:**
    
    - **BPT:** "How do I fit this on one GPU without OOM?" (Memory)
        
    - **Ring Attention:** "How do I split this across 100 GPUs without network lag?" (Latency/Throughput).
        

---

## References

- **Paper:** [arXiv:2310.01889](https://arxiv.org/abs/2310.01889)
    
- **Code:** [GitHub - lmsys/ring-attention](https://www.google.com/search?q=https://github.com/lmsys/ring-attention)