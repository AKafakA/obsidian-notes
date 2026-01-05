
Reading time: ~35 min

TLDR: BPT is a memory-optimization technique that prevents OOM errors when training long sequences. Unlike FlashAttention (which optimizes the $N^2$ attention matrix), BPT addresses the linear memory cost $O(N)$ of the Feed-Forward Networks (FFN) and intermediate activations. By fusing the FFN computation inside the blockwise attention loop, it caps peak memory usage to a small fixed block size ($B$), enabling context lengths 4x larger than FlashAttention on the same hardware.

---

## 1. Motivation & Background

- **Context for the problem:**
    
    - Training with long sequences ($N$) causes **Activation Memory** to explode.
        
    - **The Hidden Bottleneck:** While **FlashAttention** successfully reduced the Attention Matrix memory from $O(N^2)$ to linear $O(N)$, it did _not_ address the memory required for the rest of the transformer layer.
        
    - Specifically, the **Feed-Forward Network (FFN)** (which follows Attention) requires storing activations for all $N$ tokens to compute gradients. For a 100k sequence, these "linear" activations are massive and cause OOM even if the attention calculation is efficient.
        
- **Status quo:**
    
    - **Gradient Checkpointing:** Recomputes layers during backward pass. However, standard checkpointing works at the **Layer granularity**. You must store the _entire_ input sequence for a layer to recompute it. If $N=1M$, the input tensor itself is too big for VRAM.
        
- **Gaps or inefficiencies:**
    
    - There was no method to compute a Transformer layer without materializing the full sequence of intermediate embeddings ($N \times d$) in memory between the Attention and FFN sub-layers.
        

---

## 2. Key Insight

- **The core realization:**
    
    - The dependency structure of a Transformer allows for **Operator Fusion** across the Sequence Dimension.
        
    - **FFN is Position-wise:** The FFN for token $i$ depends _only_ on token $i$. It does not need to wait for token $j$'s attention to be finished (conceptually).
        
    - Therefore, we can compute the Attention output for a small **Block** of tokens, immediately feed it into the FFN, compute the FFN output, and then **discard the intermediate data** before moving to the next block.
        
- **Why previous approaches fall short:**
    
    - Standard implementations (PyTorch) execute "Breadth-First": Compute Attention for all $N$, then compute FFN for all $N$. BPT executes "Depth-First" on blocks: Compute Attn+FFN for Block 1, then Attn+FFN for Block 2.
        

---

## 3. The BPT Approach

### 3.1 Architecture / Method Overview

- **Components:**
    1. **Blockwise Attention:** Calculates attention scores for a specific Query block $Q_i$ against all Key/Values.
    2. **Fused FFN:** The FFN computation is injected _inside_ the attention's outer loop.
### 3.2 Core Techniques

- **Nested Loop Computation:**
    
    - BPT restructures the calculation into nested loops over blocks:
        
        - **Outer Loop ($i$):** Load Query Block $Q_i$.
            
        - **Inner Loop ($j$):** Load $K_j, V_j$, update attention scores for $Q_i$.
            
        - **Fusion Step:** Once the Inner Loop finishes for $Q_i$, we have the valid attention output for these tokens. **Immediately** run the FFN and Residual Add for this block.
            
        - **Discard:** Save only the lightweight boundary statistics (for backward pass) and discard the heavy activation tensors.
            
- **Memory Cost Analysis:**
    
    - **Standard:** $O(N \cdot d)$ (Must store embeddings for all tokens).
        
    - **BPT:** $O(B \cdot d)$ (Only store embeddings for the current block).
        
    - Since $B \ll N$, memory usage is effectively constant regardless of sequence length.
        

---

## 4. Performance & Evaluation

- **Metrics:**
    
    - **Max Context Length:** Longest trainable sequence before OOM.
        
    - **Throughput:** Tokens/sec compared to vanilla/FlashAttention.
        
- **Comparisons:**
    
    - Compared against **Vanilla PyTorch** and **FlashAttention (v1)**.
        
- **Key Results:**
    
    - **32x Longer Context:** Compared to vanilla transformers.
        
    - **4x Longer Context:** Compared to FlashAttention.
        
    - **Throughput Parity:** Despite the complex looping, BPT achieves comparable throughput to FlashAttention because it keeps data in high-bandwidth SRAM and reduces HBM reads/writes for the large activation tensors.
        

---

## 5. Limitations & Unimplemented Features

- **The "Single-Device" Trap:**
    
    - BPT effectively solved the _Memory_ constraint (fitting the model). However, computing 1 Million tokens on a _single_ GPU is painfully slow (Latency constraint).
        
    - **Communication Bottleneck:** While BPT allows you to split the sequence across GPUs (sharding), it didn't optimize the _communication topology_. Naive sharding requires `All-Gather`, which is slow.
        
    - _Note:_ This specific limitation is what led the same authors to develop **Ring Attention** months later, which took the "Blockwise" concept of BPT and arranged the blocks in a communication-hiding ring.
        

---

## 6. Broader Impacts & Future Directions

- **Precursor to Infinite Context:** BPT proved that _memory_ is no longer the bottleneck for context length; only _compute_ and _communication_ are.
    
- **Adoption:** The "Blockwise FFN" pattern is now standard in high-performance inference engines (like vLLM and TGI) to prevent memory spikes during the decoding of long prompts.

---

## 7. Takeaway Summary

- **Core Idea:** Fuse the FFN into the Attention loop. Process the sequence in chunks (Blocks) to ensure that we never hold $N$ tokens' worth of activations in memory at once.
    
- **Key Contributions:**
    
    - Extended the "Flash" concept from just the Attention Matrix to the entire Transformer Layer.
        
    - Decoupled activation memory from sequence length.
        
- **Benefits:**
    
    - Massive increase in trainable context length on limited hardware.
        
    - No approximation (Exact math).
        

---
## References
- **Paper:** [arXiv:2305.19370](https://arxiv.org/abs/2305.19370)    
- **Code:** [GitHub - lmsys/fastchat](https://github.com/lm-sys/FastChat)
