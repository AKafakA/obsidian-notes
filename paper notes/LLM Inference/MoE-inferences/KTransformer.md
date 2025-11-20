Reading time: ~45 min --- TLDR: KTransformers is a high-performance inference system for Mixture-of-Experts (MoE) models that unlocks the potential of CPU/GPU hybrid computing by using specialized AMX-optimized CPU kernels and an "Expert Deferral" mechanism to hide latency, enabling massive models (like DeepSeek-671B) to run efficiently on commodity hardware. ---

## 1. Motivation & Background
- **Context for the problem:**
    - MoE models (e.g., DeepSeek-V3, Mixtral) are becoming massive, often exceeding the VRAM capacity of single or even multiple GPUs (e.g., DeepSeek-V3 is 671B parameters).
    - Local users and researchers want to run these models for privacy or analysis but are limited by hardware costs.
- **Status quo:**
    - Existing hybrid solutions (like llama.cpp or standard offloading) use the CPU primarily as a "backup" memory pool, swapping weights to the GPU or running unoptimized scalar code on the CPU.
    - Previous works like **Fiddler** improved this by executing experts on the CPU, but they often hit a "compute wall" where the CPU processes too slowly to keep up with the GPU, or they suffer from high synchronization overheads.
- **Gaps or inefficiencies:**
    - **CPU Compute Bottleneck:** Standard CPU kernels (even with AVX) are often too slow to process expert layers in real-time, leaving the GPU idle while waiting for results.
    - **Sync Overhead:** Constant communication between CPU and GPU for every layer introduces latency bubbles that kill throughput.

---

## 2. Key Insight
- **The core realization:**
    - Modern CPUs (especially Intel Sapphires Rapids with AMX) have significant untapped compute power for matrix math that is rarely utilized in LLM inference.
    - Instead of treating the CPU as just "slow memory," KTransformers treats it as a specialized "Sparse Processor."
- **Why previous approaches fall short:**
    - Other systems failed to fully saturate the CPU's compute units or overlap the workloads effectively. They treated the CPU and GPU execution as sequential steps (wait for CPU -> run GPU), rather than a parallel pipeline.
- **KTransformers' Solution:**
    - Optimize the "compute" on the CPU using hardware-specific instructions (AMX/AVX-512) and optimize the "schedule" using a novel deferral strategy to maximize parallel execution.

---

## 3. The KTransformers Approach

### 3.1 Architecture / Method Overview
- **Heterogeneous Split:**
    - **GPU (Hot/Dense):** Handles the "dense" components that require high bandwidth and constant access, specifically the **Attention** mechanism and **Shared Experts** (if applicable).
    - **CPU (Cold/Sparse):** Handles the **Routed Experts** (the majority of the parameter count). Since only a few experts are active per token, the CPU's slower bandwidth is less of a bottleneck if the compute is fast enough.

### 3.2 Core Techniques
- **AMX-Specialized Kernels:**
    - Leverages Intel **AMX (Advanced Matrix Extensions)** instructions to drastically speed up matrix multiplications on the CPU.
    - Uses a custom memory layout (block-wise quantization, tiling-aware access) to maximize cache hits and memory bandwidth efficiency.
- **Expert Deferral (The "Magic"):**
    - A scheduling innovation where the system doesn't wait for the CPU to finish the expert computation immediately.
    - Instead, it "defers" the dependency, allowing the GPU to proceed with other independent tasks (like processing the next layer's attention for a different token or part of the batch) while the CPU crunches the expert numbers in the background.
    - This increases CPU utilization from typically <75% to nearly **100%**.
- **Asynchronous Scheduling:**
    - Implements a non-blocking handoff between CPU and GPU to minimize the "launch overhead" (the time cost of telling the hardware to start working).
- **YAML-Based Optimization Composition:** 
	- KTransformers introduces a flexible, rule-based configuration system (via YAML). - Users define rules to map specific PyTorch modules to optimized kernels (e.g., "Replace all Linear layers in the MLP block with Q4_K_M kernels"). 
	- **Benefit:** This decouples the model logic from the optimization backend, allowing for rapid iteration, easy testing of different quantization schemes, and convenient usage for developers.

---

## 3.3 Serving System
*This system allows running 600B+ parameter models on setups with only 24GB-48GB VRAM.*

#### Deployment & Placement Strategy
- **Placement Rule:** "Arithmetic Intensity-Aware." Operators with high arithmetic intensity (dense computations) use AMX kernel at CPU, CPU hold the routed experts while GPU have shared experts.
- **CUDAGraphs:** extensively used to reduce the CPU overhead of launching GPU kernels, which is critical when the CPU is already busy calculating experts.

---

## 4. Performance & Evaluation
- **Metrics:**
    - Prefill speed (processing prompt) and Decoding speed (generating text).
- **Comparisons:**
    - Compared against **llama.cpp**, **Fiddler**
- **Key Results:**
    - **DeepSeek-V3/R1 (671B):** Capable of running on a dual-socket server with modest GPUs, where other systems OOM (Out of Memory) or crawl at <0.1 tokens/s.
    - **Speedups:**
        - **Prefill:** 4.62x – 19.74x faster than baselines.
        - **Decoding:** 1.25x – 4.09x faster than baselines.
    - **Efficiency:** The Expert Deferral mechanism alone added ~1.45x throughput by simply hiding latency.

---

## 5. Limitations & Unimplemented Features
- **Hardware Dependency:** The "peak" performance numbers rely heavily on **Intel AMX** (Sapphire Rapids or newer CPUs). While it supports AVX-512/AVX2, the performance drop-off on older consumer CPUs is significant.
- **Bandwidth Wall:** Ultimately limited by DDR5 RAM speeds; no amount of compute optimization can fix the bottleneck if the RAM cannot feed data fast enough for the largest models.
- **Evaluating with batch size as 1**, which is simpler local cases but may not reflect high-concurrency scenarios.

---

## 6. Broader Impacts & Future Directions
- **Local Research on SOTA Models:** Enables researchers to "dig into the internals" of massive open-weights models (like DeepSeek) without needing an H100 cluster.
- **SGLang Integration:** The technology is being upstreamed into serving engines like SGLang, meaning it will likely become a standard backend for hybrid inference in production.
- **Cost Reduction:** Proves that CPU DRAM (cheap) + Weak GPU is a viable alternative to High-VRAM GPU (expensive) for low-concurrency serving.

---

## 7. Takeaway Summary
- **Core Idea:** Don't just offload to CPU; *accelerate* the CPU. By using AMX instructions and asynchronous "deferral" scheduling, the CPU becomes a capable co-processor rather than a bottleneck.
- **Key Contributions:**
    - Developed high-performance AMX kernels for MoE experts.
    - Introduced "Expert Deferral" to overlap CPU/GPU execution perfectly.
    - Delivered usable inference speeds for 600B+ models on commodity workstations.
    - Configurable YAML-based kernel injection system.
- **Benefits:**
    - Massive cost savings on hardware.
    - High throughput for single-batch/local use cases.
-