Reading time: ~45 min --- TLDR: NanoFlow is a high-throughput LLM serving framework that exploits "Intra-Device Parallelism." By splitting requests into operation-level "nano-batches" and co-scheduling compute, memory, and network resources, it achieves up to 1.91x throughput boost over state-of-the-art systems (like vLLM and TensorRT-LLM) on NVIDIA GPUs while maintaining strict latency SLOs. ---

## 1. Motivation & Background
- **Context for the problem:**
    - LLM serving requires massive scale (tens of thousands of GPUs). Throughput (tokens/sec/device) is the critical metric defining cost-efficiency.
    - Standard serving engines (vLLM, Deepspeed-MII) focus on **inter-device parallelism** (Pipeline/Tensor Parallelism) but treat the execution *within* a single device as a sequential stream of operations.
- **Status quo:**
    - Current systems execute operations sequentially: `Load Weights -> Compute -> Network Transfer`.
    - While one resource (e.g., Compute/Tensor Cores) is busy, others (e.g., High Bandwidth Memory, NVLink) often sit idle.
- **Gaps or inefficiencies:**
    - **Sequential Dependency:** The strict dependency chain of LLM inference prevents resource overlap.
    - **Resource Underutilization:** Despite LLMs being "memory bound" in theory, end-to-end serving often leaves compute units underutilized (~40% utilization) because heterogeneous operations do not run concurrently.

---

## 2. Key Insight
- **The core realization:**
    - You can break the strict sequential dependency by **Nano-Batching**: splitting a single batch of requests into smaller sub-batches (nano-batches) at the *operation level*.
    - While Nano-Batch 1 is doing "Compute" (using Tensor Cores), Nano-Batch 2 can be doing "Memory Load" (using HBM), and Nano-Batch 3 can be doing "Communication" (using NVLink).
- **Why previous approaches fall short:**
    - Existing "Pipeline Parallelism" (inter-device) only pipelines across GPUs. It does not pipeline the internal units of a *single* GPU. NanoFlow brings pipelining *inside* the device.

---

## 3. The NanoFlow Approach

### 3.1 Architecture / Method Overview
- **Intra-Device Parallelism:** The central engine that manages overlapping resources.
- **Components:**
    1.  **Nano-Batching:** Divides the input batch into fine-grained pieces (e.g., splitting 2048 requests into chunks of 128).
    2.  **Execution Unit Scheduling:** Assigns specific GPU Streaming Multiprocessors (SMs) to different tasks to prevent interference.
    3.  **Automated Pipeline Search:** An offline algorithm that compiles the optimal execution schedule.

### 3.2 Core Techniques: Two-Stage Auto-Search
- **Problem:** The search space for "how many nano-batches" and "which kernel to run when" is exponential. NanoFlow solves this via a **Two-Stage Mixed-Integer Linear Programming (MILP)** approach.
    - **Stage 1: Pipeline Structure Search (MILP):**
        - **Goal:** Determine the *structure*—the number, size, and execution order of nano-operations.
        - **Assumption:** Assumes kernels are "interference-free" (i.e., they run at full speed even when overlapped).
        - **Output:** A structural pipeline skeleton that minimizes theoretical execution time by removing pipeline bubbles.
    - **Stage 2: Resource Allocation Refinement (MILP):**
        - **Goal:** Refine the pipeline to account for *real-world interference*.
        - **Method:** Uses an offline-profiled **Interference Table** (mapping Resource Utilization $R$ to Performance $P$). It re-solves the MILP to decide exactly *which* kernel implementation to use and *how much* of the GPU (e.g., 40% of SMs) to allocate to each overlapping operation.
        - **Output:** A practical, robust schedule that guarantees QoS for concurrent kernels.

---

## 4. Performance & Evaluation
- **Metrics:**
    - **Throughput:** Tokens generated per second per GPU.
    - **Latency:** Normalized latency per token.
    - **SLO Capacity:** Maximum request rate sustainable under a strict latency constraint (e.g., 200ms/token).
- **Models:** Llama-2-70B, Mixtral 8x7B, LLaMA-3-8B.
- **Comparisons:**
    - Compared against **vLLM**, **DeepSpeed-FastGen**, and **TensorRT-LLM**.
- **Key Results:**
    - **1.91x Throughput Boost:** Significantly outperforms baselines by effective pipelining.
    - **Approaching Optimality:** Achieves **59% - 72%** of the theoretical hardware roofline (optimal throughput), whereas standard systems often stall at <40%.
    - **Higher Capacity under SLOs:**
        - At a strict **200ms normalized latency SLO**, NanoFlow sustains a **1.64x higher request rate** than the best baseline (TensorRT-LLM).
        - For the LMSYS-Chat-1M dataset, it handles ~35 req/s before violating SLOs, whereas baselines crash/violate at ~20 req/s.
    - **Tail Latency:** The 99th-percentile latency is only **1.07x** of the average latency, demonstrating extreme stability due to constant dense batch sizing.

---

## 5. Limitations & Unimplemented Features
- **Hardware Dependency:** The specific "Resource vs. Performance" interference profiles are highly sensitive to the GPU architecture (A100 vs. H100). Porting requires re-profiling.
- **Complexity:** The "Auto-Search" is an offline compilation step. You cannot dynamically change the model structure at runtime without re-running the MILP solver (~10 mins).
- **Latency Trade-off at Low Load:** For extremely low request rates (Batch Size $\approx$ 1), NanoFlow has slightly higher latency than vLLM because the pipeline setup overhead dominates. It is designed for high-throughput regimes.

---

## 6. Broader Impacts & Future Directions
- **Efficiency:** Drastically reduces the "GPU Tax" for serving companies. 2x throughput means half the GPUs needed for the same traffic.
- **Hardware Design:** Suggests future GPUs should explicitly support "Quality of Service" (QoS) for intra-device partitioning to make this easier than software-based SM masking.

---

## 7. Takeaway Summary
- **Core Idea:** Don't let the Memory unit sleep while the Compute unit works. Split batches into tiny pieces ("nano-batches") and pipeline them inside the GPU using a compiled MILP schedule.
- **Key Contributions:**
    - **Intra-Device Parallelism:** Overlapping Compute, Memory, and Network ops.
    - **Two-Stage Auto-Search:** A rigorous MILP-based compiler to generate interference-aware schedules.
    - **1.91x Throughput:** Massive gains in capacity without sacrificing strict latency SLOs.
- **Benefits:**
    - nearly 2x throughput improvement.
    - Higher hardware utilization efficiency.
    - Robust tail latency.
- **Limitations:**
    - Complex offline setup (profiling + MILP search).

---

## References
- **Paper:** [arXiv:2408.12757](https://arxiv.org/abs/2408.12757) / OSDI 2025
- **Code:** [GitHub - efeslab/Nanoflow](https://github.com/efeslab/Nanoflow)

## Appendix: Cost Model & Optimal Throughput Analysis

### 1. The Cost Model: Decomposition of Latency
NanoFlow models the latency of a single LLM serving iteration (processing one batch) by decomposing it into three primary resource constraints: Memory, Compute, and Network.

* **Memory Latency ($T_{mem}$):**
    The time required to load all model weights from High Bandwidth Memory (HBM) into the GPU registers/cache.
    $$T_{mem} = \frac{\text{MemSize}}{\text{MemBW}}$$
    * **Insight:** LLM inference is unique because the model weights are too large to stay in the cache (long reuse distance). Therefore, for every iteration, the entire model must be loaded from HBM, making this cost constant regardless of batch size (assuming the batch fits in memory).

* **Compute Latency ($T_{compute}$):**
    The time required to perform the dense matrix multiplications (GEMMs) for projections (Q, K, V, Output) and FFN layers (Up, Gate, Down).
    $$T_{compute} \approx \frac{2 \cdot B_{Dense} \cdot P_{Model}}{\text{Compute}_{\text{FLOPS}}}$$
    * **$B_{Dense}$:** The total token count in the batch (Prefill tokens + Decode tokens).
    * **$P_{Model}$:** The number of model parameters.
    * **Insight:** Unlike memory latency, compute latency scales linearly with the batch size ($B_{Dense}$). As batch size increases, compute time grows while memory time stays fixed.

* **Network Latency ($T_{net}$):**
    The time required for collective communication (All-Reduce/All-Gather) to synchronize activations across GPUs in Tensor Parallelism.
    $$T_{net} \approx 4 \cdot \frac{N_{GPU} \cdot B_{Dense} \cdot D_{model} \cdot S_{type} \cdot L}{\text{NetBW}}$$
    * **Insight:** Like compute, network latency scales with the dense batch size ($B_{Dense}$) because activations must be synchronized for every token.

### 2. Bottleneck Analysis: Why Modern Serving is Compute-Bound
Standard wisdom assumes LLMs are memory-bound. However, the paper introduces the ratio $T_R$ to determine the actual regime:
$$T_R = \frac{T_{mem}}{T_{compute}}$$

The analysis proves that end-to-end serving is actually **Compute-Bound** ($T_R < 1$) for two key reasons:
1.  **Grouped Query Attention (GQA):** Modern models (Llama-3, Qwen2) use GQA, which drastically reduces the KV-cache size. This allows significantly larger batch sizes ($B_{Dense}$) to fit in GPU memory.
2.  **Large Batch Sizes:** Serving systems maximize throughput by saturating memory with requests. With a batch size of ~2048 (common for 70B models), the compute time ($T_{compute}$) significantly exceeds the memory load time ($T_{mem}$).

**Empirical Validation:**
On Llama-2-70B (8x A100), the calculated Total Compute Time (114 ms) far exceeds Memory Time (45 ms) and Network Time (31 ms).

### 3. Deriving Optimal Throughput (The Roofline)
Since the workload is compute-bound, the theoretical maximum throughput is limited strictly by the GPU's arithmetic capability, not its bandwidth.

**Optimal Throughput Formula:**
$$\text{Throughput}_{\text{optimal}} = \frac{B_{Dense}}{T_{compute}} = \frac{\text{Compute}_{\text{FLOPS}}}{2 \cdot P_{Model}}$$

* **Derivation:** This formula is derived by substituting $T_{compute}$ into the throughput definition.
* **Implication:** Optimal throughput depends *solely* on the aggregate GPU compute capacity and the model size. Other factors like memory bandwidth or input/output length do not restrict the theoretical maximum.
* **Example:** For Llama-2-70B on 8x A100 GPUs, the optimal throughput is calculated as **1857 tokens/s/GPU**.

### 4. The Gap: Sequential Execution Bubbles
Existing systems (vLLM, TensorRT-LLM) fail to reach this optimal throughput (achieving only ~22-37% of it) because they execute operations sequentially.
* **The Bubble Problem:** When the GPU is computing, the Memory bandwidth is idle. When the GPU is communicating (Network), the Compute units are idle.
* **NanoFlow's Solution:** By splitting batches into "Nano-Batches," NanoFlow overlaps these distinct phases. If the overlap is perfect, the total time becomes $\max(T_{mem}, T_{compute}, T_{net})$ rather than the sum, effectively hiding the memory and network costs behind the dominant compute cost.