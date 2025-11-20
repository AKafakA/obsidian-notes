Reading time: ~45 min 

TLDR: Fiddler is an inference system for Mixture-of-Experts (MoE) models that orchestrates CPU and GPU resources to minimize data movement. It achieves fast inference on memory-constrained devices by executing specific experts on the CPU rather than transferring heavy weights to the GPU. 

## 1. Motivation & Background
- **Context for the problem:**
    - The paper addresses the deployment of Large Language Models (LLMs) based on **Mixture-of-Experts (MoE)** architectures (e.g., Mixtral-8x7B) in resource-constrained environments.
    - This is critical for local deployment (edge devices, personal PCs) where GPU memory is limited and cannot hold the entire model parameters (often >90GB for Mixtral).
- **Status quo:**
    - Existing solutions ("offloading" systems like DeepSpeed-MII or llama.cpp offloading) store model weights in CPU memory and transfer them to the GPU for computation on demand.
- **Gaps or inefficiencies:**
    - **PCIe Bottleneck:** Moving large expert weights from CPU to GPU over PCIe is slow and dominates inference latency.
    - **Underutilization:** These systems typically treat the CPU solely as a storage device, ignoring its available computational capabilities.
    - **Inefficiency for Small Batches:** The overhead of weight transfer is particularly dominating in single-batch or low-latency inference scenarios (like interactive chat).

---

## 2. Key Insight
- **The core realization:**
    - For small batch sizes (common in local inference), the data volume of **weights** (which must be moved from CPU to GPU in standard offloading) is orders of magnitude larger than the volume of **token activations**.
    - Therefore, it is faster to move the small activations to the CPU and compute the layer there, rather than moving the massive weights to the GPU.
- **Why previous approaches fall short:**
    - Prior methods optimized for "GPU-only compute," assuming GPU execution is always faster. They failed to account for the massive data transfer penalties that outweigh the compute speedup for sparse MoE layers.

---

## 3. The Fiddler Approach

### 3.1 Architecture / Method Overview
- **Hybrid CPU-GPU Orchestration:** Fiddler utilizes both CPU and GPU for computation.
- **Weight Distribution:**
    - **Non-expert layers:** Placed permanently on GPU memory (these are usually dense but small enough, e.g., <2GB total).
    - **Expert layers:** A subset of "hot" (frequently accessed) experts are pinned to GPU memory if space allows (as much as possible); the vast majority remain in CPU memory.

### 3.2 Core Techniques
- **Dynamic Execution Strategy:**
    - Fiddler decides *per layer* and *per token* where to execute the computation:
        1.  **GPU Hit:** If the required expert is already on GPU, execute on GPU.
        2.  **GPU Miss:** If the expert is on CPU, Fiddler compares the cost of:
            - (a) Loading expert weights to GPU (Traditional offloading).
            - (b) Sending token activations to CPU and computing there (Fiddler's approach).
- **Latency Modeling:**
    - Uses a runtime profiler to estimate `cpu_lat(batch_size)`, `gpu_lat(batch_size)`, and `transfer_lat`.
    - Since CPU compute scales linearly with batch size while weight transfer is constant but heavy, Fiddler prefers CPU execution for small batches and GPU offloading for larger batches.
- **Expert Management:**
    - Prioritizes keeping popular experts on the GPU to maximize the "hit rate."

---

## 3.3 Serving System
*This system focuses on single-node, memory-constrained inference.*

#### Deployment & Placement Strategy
- **Target Environment:** Single GPU setups with limited VRAM (e.g., 24GB VRAM for a 90GB model).
- **Weight Placement:** Static placement of non-expert layers on GPU; dynamic or heuristic-based placement of experts based on affinity/usage frequency.

#### Scheduling & Load Balancing
- **Orchestration Algorithm:**
    - For every token in the prefill or decode phase, the system consults the gating network.
    - Based on the selected expert's location and the current batch size, it dynamically routes the computation to the optimal resource (CPU or GPU).
- **Parallelism:** Capable of processing different tokens on different hardware simultaneously if they route to different experts.

#### Request Flow
- **Step-by-step:**
    1.  **Input Embedding:** Processed on GPU.
    2.  **Non-Expert Layers:** Executed on GPU (Attention, Norm, etc.).
    3.  **Gating Layer:** Executed on GPU to identify required experts.
    4.  **Orchestration Decision:** System checks expert location.
        - *If on GPU:* Execute immediately on GPU.
        - *If on CPU:* Move activation -> CPU, compute on CPU, move result -> GPU.
    5.  **Next Layer:** Repeat until output generation.

---

## 4. Performance & Evaluation
- **Metrics:**
    - Token generation throughput (tokens/sec).
    - Latency for single-batch and beam search.
- **Comparisons:**
    - **Baselines:** DeepSpeed-MII, standard Mixtral offloading implementation (e.g., Eliseev & Mazur).
    - **Hardware:** Tested on NVIDIA L4 (24GB) and RTX 6000 (24GB) with standard CPUs.
- **Key Results:**
    - **Single Batch:** ~1.26x speedup over optimized baselines; >10x speedup over naive offloading.
    - **Beam Search:** **11.57x speedup**. Beam search increases memory pressure significantly; Fiddler avoids the thrashing caused by constant weight swapping.
    - **Throughput:** Generates >3 tokens/s for uncompressed Mixtral-8x7B on a single 24GB GPU (usable for real-time chat).

---

## 5. Limitations & Unimplemented Features
- **Hardware Assumptions:** Assumes non-expert layers fit entirely in GPU memory (might be an issue for extremely large models on very small GPUs).
- **CPU Dependency:** Heavy reliance on CPU compute performance; performance degrades significantly if the CPU lacks modern vector instructions (e.g., AVX512) or has slow RAM bandwidth.
- **Scope:** Primarily optimized for single-user / low-batch latency. It is not designed for high-throughput batch serving (where GPU offloading/pipelining typically scales better).

---

## 6. Broader Impacts & Future Directions
- **Democratization of AI:** Allows consumers and researchers to run massive MoE models on commodity hardware (e.g., gaming PCs) without needing expensive H100 clusters.
- **Privacy:** Enables local processing of sensitive data by removing the need for API-based model serving.
- **Energy Efficiency:** Reduces the massive data movement overhead, which is a significant source of energy consumption in PCIe-bound offloading systems.

---

## 7. Takeaway Summary
- **Core Idea:** Move the data (activations) to the compute (CPU), rather than moving the compute (weights) to the data (GPU), when the batch size is small.
- **Key Contributions:**
    - Proposed a CPU-GPU orchestration strategy for MoE inference.
    - Developed a latency model to dynamically select the optimal execution path.
    - Demonstrated >3 tokens/s on consumer-grade hardware for Mixtral-8x7B.
- **Benefits:**
    - Significantly lower latency for single-batch inference.
    - Better utilization of idle CPU compute resources.
- **Limitations:**
    - Performance is tightly coupled with CPU capabilities (RAM bandwidth, AVX support).

---

## References
- Paper PDF (Local)
- Arxiv Preprint (2402.07033)
- GitHub Code (efeslab/fiddler)