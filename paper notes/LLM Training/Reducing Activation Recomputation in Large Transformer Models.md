Reading time: ~45 min

TLDR: Training large models (500B+) is memory-bottlenecked even with 3D parallelism. The authors introduce Sequence Parallelism (partitioning pointwise ops like LayerNorm across GPUs to eliminate redundancy) and Selective Activation Recomputation (recomputing only memory-heavy/compute-light ops) to cut activation memory by 5$\times$1. This speeds up training by ~30% compared to standard full-recomputation baselines22.

---

## 1. Motivation & Background

- **Context for the problem:**
    
    - **Large Model Training:** Training massive models (e.g., MT-NLG 530B) requires **3D Parallelism**: Data Parallelism (DP), Tensor Parallelism (TP), and Pipeline Parallelism (PP)3333.
        
    - **Memory Redundancy in TP:** In standard Tensor Parallelism, operations like **LayerNorm and Dropout** are replicated on every GPU in the TP group4. This results in **duplicated storage** of large activation maps, wasting significant memory5.
        
- **Status quo:**
    
    - **3D Parallelism (Baseline):** The standard approach (Megatron-LM) combines Data, Tensor, and Pipeline Parallelism.
        
    - **Pipeline Parallelism (PP):** Splits layers into stages. To minimize bubbles, schedulers (like 1F1B) require storing activations for $p$ microbatches in the first stage, creating a massive memory bottleneck.
        
    - **Full Activation Recomputation:** To fit this memory footprint, the baseline checkpoints the **input to every Transformer layer** (not just the stage)7. During the backward pass, it fully recomputes the layer.
        
- **Gaps or inefficiencies:**
    
    - **Inefficient Recomputation:** Full layer recomputation is a "blunt instrument"8. It re-executes expensive GEMMs (Matrix Multiplications) just to regenerate cheap activation maps (like Gelu/Softmax), adding 30-40% compute overhead9.
        
    - **Wasted Memory:** Standard TP ignores the redundancy of pointwise operations, causing activation memory to bloat unnecessarily as the sequence length increases10.
        

---

## 2. Key Insight

- **The core realization:**
    
    - **Insight 1: Pointwise Independence (Sequence Parallelism).** Operations like LayerNorm and Dropout are pointwise (independent along the sequence dimension)11. Instead of duplicating them (as in standard TP), they can be partitioned along the sequence dimension12.
        
    - **Insight 2: Cost Asymmetry (Selective Recomputation).** Not all activations are equal13.
        
        - **GEMMs:** High Compute, Low Memory output. $\rightarrow$ **Store these** (Checkpoint).
            
        - **Non-Linearities (Softmax, Gelu, Norms):** Low Compute, High Memory output. $\rightarrow$ **Recompute these** (Drop).
            
- **Why previous approaches fall short:**
    
    - **Standard TP:** Duplicates storage for the input of pointwise operations (LayerNorm/Dropout) across all GPUs14.
        
    - **Standard Full Recomputation:** Wastes compute by re-running heavy GEMMs. It hurts training speed significantly to save memory15.
        
    - **Selective Recomputation (Proposed):** Solves this by only recomputing the memory-bound operations, paying a negligible cost in FLOPs to save massive memory16.
        

---

## 3. The Approach

### 3.1 Architecture / Method Overview

- **High-Level logic:** The method modifies the Transformer block to shard replicated operations (Sequence Parallelism) and cache only high-value tensors (Selective Recomputation).
    
- **Components:**
    
    1. **Sequence Parallelism (SP):** Groups pointwise operations into a sequence-parallel block.
        
    2. **Selective Activation Recomputation (SAR):** Caches specific expensive tensors.
        

### 3.2 Core Techniques

- **Technique 1: Sequence Parallelism**
    
    - **How it works:** The authors identify regions of the Transformer (LayerNorm, Dropout) that are typically replicated. They partition these along the sequence dimension.
        
    - **The Converters ($g$ and $\bar{g}$):** To transition between Tensor Parallel blocks and Sequence Parallel blocks, they replace standard communication primitives:
        
        - **$g$ (Forward):** **All-Gather**. Converts partitioned sequence data (SP) to full duplicated data for Tensor Parallelism17.
            
        - **$\bar{g}$ (Forward):** **Reduce-Scatter**. Converts full Tensor Parallel outputs back to partitioned sequence data (SP)18.
            
    - **Impact:** Replaces the standard `All-Reduce` (2x data movement) with `Reduce-Scatter` + `All-Gather` (2x data movement), so communication volume is neutral19. However, it reduces activation memory for these layers by a factor of $t$ (TP size)20.
        
- **Technique 2: Selective Activation Recomputation**
    
    - **How it works:** instead of `checkpoint(layer)`, the model explicitly:
        
        - **Stores:** Outputs of Linear layers (GEMMs).
            
        - **Drops & Recomputes:** Outputs of Attentions ($QK^T$), Softmax, Gelu, Dropout, and LayerNorm21.
            
    - **Impact:** Reduces memory usage to near "Full Recomputation" levels ($sbh(10 + 24/t)$ vs $2sbh$) but incurs only **~2%** FLOP overhead instead of 30-40%22.
        

---

## 4. Performance & Evaluation

- **Metrics:**
    
    - **Throughput:** End-to-end iteration time.
        
    - **Model Flops Utilization (MFU):** Efficiency of hardware usage.
        
    - **Memory Usage:** Activation memory per GPU.
        
- **Baselines:**
    
    - **Megatron-LM (Full Recomputation):** Standard 3D parallelism with layer-wise checkpointing23.
        
- **Key Results:**
    
    - **Result 1 (Memory):** Reduced activation memory by **5$\times$** compared to the baseline (TP without recompute)24.
        
    - **Result 2 (Speed):** On a 530B model (2240 A100s), achieved **54.2% MFU**, a **29.7% speedup** over the full recomputation baseline25252525.
        
    - **Result 3 (Overhead):** Reduced recomputation overhead from 36% to **2%** for large models26.
        

---

## 5. Limitations & Unimplemented Features

- **Constraint A:** **Interconnect Dependencies.** Sequence Parallelism relies on `Reduce-Scatter` and `All-Gather`. While the volume is the same as `All-Reduce`, splitting the ops can expose latency if not overlapped or if interconnects (NVLink) are slow27.
    
- **Constraint B:** **Pipeline Imbalance.** The first stage of the pipeline remains the memory bottleneck because it must store activations for $p$ microbatches. The authors suggest a "Microbatch Level Recomputation" heuristic in Appendix C to mitigate this28.
    
- **Constraint C:** **Code Intrusiveness.** Requires specific implementation of the forward/backward pass (defining $g$ and $\bar{g}$), rather than a simple wrapper.
    

---

## 6. Broader Impacts & Future Directions

- **Impact:** This paper standardized the "recompute cheap, store expensive" philosophy. The techniques (SP and SAR) are now core components of **Megatron-LM** and **NeMo**29.
    
- **Future Work:** Further reducing memory pressure on the first pipeline stage and handling memory fragmentation30.
    

---

## 7. Takeaway Summary

- **Core Idea:** Eliminate duplicated memory in Tensor Parallelism using Sequence Parallelism (via $g$/$\bar{g}$ converters) and optimize the recomputation trade-off by only replaying cheap, memory-heavy operations.
    
- **Key Contributions:**
    
    - **Sequence Parallelism:** Shards LayerNorm/Dropout across the sequence dimension.
        
    - **Selective Recomputation:** Saves GEMM outputs, recomputes non-linearities.
        
- **Benefits:**
    
    - 5$\times$ Memory reduction.
        
    - ~30% Training speedup.
        
- **Limitations:**
    
    - Complexity of implementing custom communication primitives ($g, \bar{g}$).
        

---

## References

- **Paper:** [arXiv:2205.05198](https://arxiv.org/abs/2205.05198)
    
- **Code:** [NVIDIA/Megatron-LM](https://github.com/NVIDIA/Megatron-LM)
    

---

## (Optional) Appendix: Deep Dive

Communication Primitives in Sequence Parallelism

Standard Tensor Parallelism uses All-Reduce (which is Reduce-Scatter + All-Gather).

Sequence Parallelism splits this:

1. **Forward:** `Linear` $\rightarrow$ `Reduce-Scatter` ($\bar{g}$) $\rightarrow$ `LayerNorm/Dropout` (on shards) $\rightarrow$ `All-Gather` ($g$) $\rightarrow$ `Linear`.
    
2. Backward: The operations are conjugates ($g$ becomes Reduce-Scatter, $\bar{g}$ becomes All-Gather)31.
    
    This ensures communication volume is identical to the baseline but memory storage for the intermediate LayerNorm/Dropout is divided by $t$.