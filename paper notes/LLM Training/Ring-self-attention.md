 Sequence Parallelism: Long Sequence Training from System Perspective

Reading time: ~15 min

TLDR: The paper proposes Sequence Parallelism (SP) to enable training Transformers with infinitely long sequences by splitting the sequence dimension across devices. By introducing Ring Self-Attention (RSA), it breaks the memory bottleneck of holding full sequences on a single GPU, achieving 3x longer sequences and 13.7x larger batch sizes compared to Tensor Parallelism.

---

## 1. Motivation & Background

- **Context for the problem:**
    
    - **Large Language Model (LLM) Training:** Training Transformers requires handling increasingly long sequences (e.g., for document understanding, genomic data, or high-res images).
        
    - **Memory Bottleneck:** Self-attention has quadratic memory complexity $O(L^2)$, and activations (intermediate outputs) scale linearly with sequence length $L$. For very long sequences, a single GPU simply runs out of memory (OOM).
        
- **Status quo:**
    
    - **Data Parallelism (DP):** Replicates the model; splits the batch. Fails if a single sequence doesn't fit in memory.
        
    - **Model Parallelism (MP):** Includes **Tensor Parallelism (TP)** (splits hidden/head dimensions) and **Pipeline Parallelism (PP)** (splits layers).
        
    - **Baselines:** Megatron-LM is the standard for TP, but TP splits the calculation _within_ a token (hidden dimension) rather than _across_ tokens (sequence dimension).
        
- **Gaps or inefficiencies:**
    
    - **TP Limitations:** Even with TP, each GPU must store the activations for the _entire_ sequence for operations like LayerNorm and Dropout. This prevents scaling to truly long sequences.
        
    - **Communication Overhead:** Splitting small dimensions (like attention heads in TP) creates significant communication overhead that doesn't scale well for long contexts.
        

---

## 2. Key Insight

- **The core realization:**
    
    - We can partition the **sequence dimension itself** across multiple GPUs.
        
    - For element-wise operations (like Dropout) or reductions (like LayerNorm), operations can be done locally on partial sequences with minimal synchronization.
        
    - **The "Aha!" moment:** For Self-Attention, we don't need to gather the full Query, Key, and Value matrices on one device. Instead, we can circulate the Key and Value blocks in a ring topology, computing partial attention scores and accumulating them.
        
- **Why previous approaches fall short:**
    
    - Tensor Parallelism (e.g., Megatron-LM) reduces memory for _weights_ effectively but fails to reduce the memory footprint of _activations_ proportional to sequence length. SP makes activation memory per GPU independent of the total sequence length (linear scaling $O(L/N)$).
        

---

## 3. The Sequence Parallelism Approach

### 3.1 Architecture / Method Overview

- **High-Level logic:** The input sequence is split into $N$ chunks (where $N$ is the number of GPUs). Each GPU holds only $1/N$ of the sequence. The model processes these chunks in parallel for most layers, only communicating when strictly necessary (i.e., Attention).
    
- **Components:**
    
    1. **Sequence-Parallel Layers:** LayerNorm and Dropout are adapted to work on partitioned sequences.
        
    2. **Ring Self-Attention (RSA):** A distributed attention mechanism that rotates K and V blocks between devices.
        

### 3.2 Core Techniques

- **Technique 1: Ring Self-Attention (RSA)**
    
    - **How it works:**
        
        1. Each GPU holds a local chunk of Queries ($Q$), Keys ($K$), and Values ($V$).
            
        2. Devices form a logical ring.
            
        3. In step 1, GPU $i$ computes attention using its local $Q_i$ and local $K_i, V_i$.
            
        4. In step 2, GPU $i$ sends its $K_i, V_i$ to neighbor $i+1$ and receives $K_{i-1}, V_{i-1}$ from neighbor $i-1$.
            
        5. GPU $i$ computes scores with $Q_i$ and the received $K, V$ block.
            
        6. Repeat until all $K, V$ blocks have circulated the ring.
            
    - **Impact:** Eliminates the need to ever materialize the full $N \times N$ attention matrix or gather full $K, V$ tensors on one node. Computation and communication are overlapped (pipelined).
        
- **Technique 2: 4D Parallelism Compatibility**
    
    - **How it works:** The system is designed to compose with Data, Tensor, and Pipeline parallelism.
        
    - **Impact:** Allows training massive models (billions of parameters) on massive sequences (millions of tokens) by leveraging all available hardware dimensions.
        

---

## 4. Performance & Evaluation

- **Metrics:**
    
    - Maximum Batch Size (before OOM).
        
    - Maximum Sequence Length (before OOM).
        
    - Throughput (TFLOPS).
        
- **Baselines:**
    
    - **Megatron-LM (Tensor Parallelism):** The industry standard for distributed Transformer training.
        
- **Key Results:**
    
    - **Result 1 (Batch Size):** Achieved **13.7x** larger maximum batch size compared to Megatron-LM (TP) when scaling to 64 NVIDIA P100 GPUs.
        
    - **Result 2 (Sequence Length):** Enabled training with **3.0x** longer sequences than Megatron-LM on the same hardware.
        
    - **Result 3 (Infinite Context):** Demonstrated capability to train on sequences with over **114,000 tokens** (limited only by the number of GPUs available).
        
    - **Result 4 (Efficiency):** Maintained linear scalability; adding more GPUs linearly increases the maximum trainable sequence length.
        

---

## 5. Limitations & Unimplemented Features

- **Constraint A:** **Communication Bandwidth:** RSA relies on passing large Key/Value tensors between GPUs. While overlapped with compute, this requires high-bandwidth interconnects (like NVLink/InfiniBand) to prevent slowdowns.
    
- **Constraint B:** **Causal Masking Complexity:** Implementing causal masking (for decoder-only models like GPT) in a ring topology is non-trivial compared to standard attention, though solvable.
    
- **Overhead:** There is a slight setup overhead for the ring communication topology compared to pure Data Parallelism on short sequences.
    

---

## 6. Broader Impacts & Future Directions

- **Impact:**
    
    - This paper (and the resulting **Colossal-AI** framework) popularized Sequence Parallelism as a distinct dimension of distributed training.
        
    - It paved the way for "Context Parallelism" and "Ring Attention" techniques now used to train long-context models (e.g., GPT-4-128k, Claude 100k).
        
- **Future Work:**
    
    - Integrating SP natively with massive-scale heterogeneous clusters.
        
    - Optimizing the communication primitives further to handle low-bandwidth, high-latency networks (e.g., training across geodistributed clusters).
        

---

## 7. Takeaway Summary

- **Core Idea:** Don't just split the model weights (Tensor Parallelism); split the _data sequence_ itself to distribute the memory cost of activations.
    
- **Key Contributions:**
    
    - **Sequence Parallelism (SP):** A novel parallel strategy splitting along the sequence dim.
        
    - **Ring Self-Attention:** A communication-efficient algorithm to compute exact attention without gathering full sequences.
        
    - **Colossal-AI Integration:** Released as part of a widely used open-source system.
        
- **Benefits:**
    
    - Linearly reduces activation memory with more GPUs.
        
    - Compatible with Data, Pipeline, and Tensor parallelism ("4D Parallelism").
        
- **Limitations:**
    
    - Heavy reliance on P2P GPU communication bandwidth.
        

---

## References

- **Paper:** [arXiv:2105.13120 - Sequence Parallelism: Long Sequence Training from System Perspective](https://arxiv.org/abs/2105.13120)
    
- **Code:** [GitHub - HPC-AI Tech / ColossalAI](https://github.com/hpcaitech/ColossalAI)
    

---

## (Optional) Appendix: Deep Dive

### Ring Self-Attention Pseudocode Logic

Python

```
# Simplified Logic for GPU rank 'r' in a world of size 'P'
local_Q = split(Q, rank=r)
local_K = split(K, rank=r)
local_V = split(V, rank=r)

# Initialize local output
local_O = zeros_like(local_Q)

# Current K, V blocks start as local
curr_K = local_K
curr_V = local_V

for step in range(P):
    # 1. Compute partial attention with current blocks
    # Note: This is a standard local attention calc
    scores = matmul(local_Q, curr_K.T)
    attn_probs = softmax(scores) # *Simplified (requires global max trick for exact softmax)
    partial_O = matmul(attn_probs, curr_V)
    
    # 2. Accumulate result
    local_O += partial_O
    
    # 3. Async Communication (Ring Shift)
    # Send current K, V to (r + 1) % P
    # Receive next K, V from (r - 1) % P
    next_K = comm.send_recv(curr_K, dest=next_rank, src=prev_rank)
    next_V = comm.send_recv(curr_V, dest=next_rank, src=prev_rank)
    
    curr_K, curr_V = next_K, next_V

return local_O
```