Reading time: 1 hr 17 min

---

TLDR: First paper purpose the concepts of pd dis-aggregation by splitting LLM inference into two distinct phases and allocating them to different hardware resources optimised for their specific needs. 

## 1. Motivation & Background

- LLM inference consists of two distinct phases:
  - **Prompt computation** (Prefill): Compute-intensive, parallel processing of input tokens to generate the first output token.
  - **Token generation** (decoding): Sequential, memory-intensive phase, dominated by KV-cache usage and lower computational demand.  

And Prefill and decoding phase has totally different hardware requirements, leading to:
	- Prefill: High compute (FLOPs) demand, benefits from high-end GPUs (e.g., H100).
	- Decoding: Memory bandwidth-bound, can run efficiently on lower-cost or power-capped GPUs (e.g., A100).

- Existing systems colocate both phases on the same hardware, leading to:
  - Inefficient GPU utilisation
  - Over-provisioning to meet latency SLOs
  - Sub-optimal cost and energy efficiency  

---

## 2. Key Insight
- **Prompt computation** requires high compute (e.g., H100), while **token generation** is bounded by memory bandwidth and can run efficiently on cheaper or power-capped GPUs.  
- This divergence opens the possibility of **phase-specific resource allocation** for improved performance-per-dollar/watt.  

---

## 3. The Splitwise Approach

- **Splitwise** separates LLM inference phases onto different machines:
  - **Prompt Pool**: Machines tailored for compute-heavy prompt processing.
  - **Token Pool**: Machines optimised for memory-bound token generation, potentially using lower-power or older GPU models (e.g., A100).  
  - **Mixed Pool**: Machines serving LLM still using the monolithic approach (chunked-prefill) as backup for phase-splited pools
  -
- **State Transfer**: Efficient KV-cache transfer between prompt and token machines via fast back-plane interconnects (e.g., InfiniBand), adding negligible latency overhead (~0.8% of end-to-end latency).   The layer-level KV-cache is transferred concurrently during prefilling  to hide transfer latency compared with sequentially transfer after prefilling.  
- 
- **Scheduling Architecture**:
  - A **cluster-level scheduler (CLS)** that allocates requests between prompt and token pools and manages pool sizing. Request was scheduled by Join the Shortest Queue (JSQ) algorithm  to prompt and token simultaneously to allow concurrent layer-level KV-cache transferring.
  - A **machine-level scheduler (MLS)** that handles batching and mixed scheduling within each machine.  

---

## 4. Performance & Efficiency Gains

- **Throughput and Cost**:
  - Achieved **1.4× higher throughput** at **~20% lower cost** compared to traditional homogeneous GPU clusters.  
  - Or, **2.35× throughput** under same cost and power budget.  
- **Latency Impact**:
  - KV-cache transfer overhead is minimal (< 1%), keeping latency-sensitive metrics like TTFT and TBT largely unaffected.  
- **Robustness & Optimization**:
  - Supports both homogeneous and heterogeneous cluster designs.
  - Employs simulation to optimize cluster sizing under various workloads (e.g., coding vs. conversational traces), power caps, and hardware costs.  

Code opensourced at: https://github.com/vllm-project/vllm/pull/2809
And the simulator for evaluation https://github.com/mutinifni/splitwise-sim

---

## 5. Broader Impacts & Future Directions

- **Architectural Implications**:
  - Splitting the phases enables independent optimization of hardware design for each stage, enabling future custom accelerators (e.g., high-bandwidth memory GPUs for token gen, high-FLOPs GPUs for prompt).  
  
- **Hardware Flexibility**:
  - Token generation could target less powerful or older GPUs, improving hardware recycling and reducing procurement costs.  

- **Generality & Limitations**:
  - Though focused on A100 and H100 GPUs, the approach is portable to other architectures and power profiles.
  - Requires fast interconnects for KV-cache transfer; heterogeneous hardware in token/prompt pools may introduce complexity in real deployments.  


---

## 6. Takeaway Summary
- **Splitwise** introduces a novel architectural partitioning of LLM inference:
  - **Core idea**: Decouple prompt and token phases, allocate them to phase-appropriate hardware.
- **Benefits**: Improved throughput, reduced cost, and energy efficiency with minimal latency overhead.
- **Significance**: Launches the paradigm of *PD disaggregation*, influencing future works like Arrow, Block, and ConServe with scheduling and granularity enhancements.

---

