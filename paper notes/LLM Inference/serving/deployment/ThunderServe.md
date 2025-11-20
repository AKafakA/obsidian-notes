Reading time: ~1 hr 11 min 

---

TLDR: ThunderServe is a cloud-oriented LLM serving system that couples phase-splitting (prefill vs. decode) with a two-level, tabu-search-based scheduler and lightweight re-scheduling to maximize SLO attainment and cost-efficiency on heterogeneous GPUs. 

---

## 1. Motivation & Background

- **Context for the problem:**
  - Serving LLMs in the cloud often means mixing GPU types and non-uniform interconnects (PCIe/Ethernet), which complicates high-performance, cost-efficient deployment. 

- **Domain / area:** 
  - LLM inference serving in **heterogeneous cloud environments**.

- **Why this matters:** 
  - Heterogeneous resources can be far cheaper/more available than homogeneous high-end clusters; smart scheduling and phase-aware placement can cut latency and raise throughput under a fixed budget.

- **Status quo / baselines:**
  - Systems like DistServe (phase disaggregation), vLLM (homogeneous clusters), HexGen (heterogeneous-aware) are key references; cloud links are typically slower than NVLink/InfiniBand, making KV transfer a bottleneck.
  
- **Gaps in prior work:**
  - Limited joint optimization of (i) GPU grouping, (ii) phase designation, (iii) per-group parallelism, and (iv) prefill/decode orchestration under realistic cloud network constraints.

---

## 2. Key Insight

- **Core realization:** 
  - Treat deployment as a **two-level hierarchical optimization** over heterogeneous GPUs: upper-level groups GPUs and assigns phases; lower-level picks per-group parallelism and orchestrates prefill/decode routing to maximize end-to-end SLO attainment. 
  
- **Why previous approaches fall short & how this fixes it:**
  - Prior work often assumes uniform/fast interconnects or does not co-optimize phase designation with parallelism; ThunderServe explicitly models network limits and phase characteristics (compute-bound prefill vs memory-bandwidth-bound decode) and adapts online via **lightweight re-scheduling**. 

---

## 3. The ThunderServe Approach

### 3.1 Architecture / Method Overview

- **High-level design / components:**
  - **Scheduler:** produces a deployment plan (GPU grouping, phase designation, per-group parallel config, orchestration rules).
  - **Workload Profiler:** tracks prompt/output length distributions and detects shifts.
  - **Task Coordinator:** dispatches requests to prefill and decode replicas based on the DistServe SLO estimator, handles KV transfer. 

- **Interaction:**
  - Initial plan from the scheduler → coordinator instantiates replicas → profiler monitors workload → scheduler triggers **lightweight re-scheduling** on shifts/failures without model reloads.
### 3.2 Core Techniques

- **Two-level scheduling algorithm:**
  - **Upper level:** **tabu search** partitions heterogeneous GPUs into serving groups and assigns each group to **prefill or decode** to maximize SLO attainment.
  - **Lower level:** for each group, chooses parallelism (TP/PP/DP as feasible) and **orchestrates prefill↔decode routing** to optimize overall SLOs under the group’s compute/memory/network constraints. 

- **KV cache compression for inter-phase transfer:**
  - Quantize-for-transport and **immediate dequantize on receipt** (compute still uses 16-bit KV), reducing transfer cost on cloud links while maintaining model quality. 

- **Implementation details:**
  - Uses NCCL async Send/Recv (and cudaMemcpy) with prebuilt communication groups; integrates FlashAttention and PagedAttention; batching strategy follows prior art. 

---

## 3.3 Serving System (Optional)

#### Deployment & Placement Strategy
- Partition heterogeneous GPUs into **model-serving groups**; mark each as **prefill** or **decode**; select feasible TP/PP/DP respecting topology/bandwidth. 

#### Scheduling & Load Balancing
- Coordinator routes each request through a prefill replica, then transfers compressed KV to a decode replica per orchestration rules; separate batching behaviors per phase to avoid interference. 
#### Scaling Strategy
- **Lightweight re-scheduling:** adjusts phase designations and orchestration **without reloading model parameters but just fliping between Prefill and Decoding** , enabling quick adaptation to workload shifts or GPU changes. 

#### Request Flow
1) Profiler summarizes workload; 2) Scheduler emits a deployment plan; 3) Coordinator instantiates replicas; 4) Request → prefill → KV transfer → decode → response; 5) On shift/failure, lightweight re-schedule updates the plan
---

## 4. Performance & Evaluation

- **Setups:**
  - Heterogeneous cloud with multiple GPU types vs. homogeneous in-house baseline.

- **Main metrics:** end-to-end latency (TTFT, TPOT, SLO attainment), throughput (tokens/s), cost under equal budget.

- **Headline results:**
  - Up to **2.1×** (avg **1.7×**) **throughput** gain; up to **2.5×** (avg **1.5×**) **latency** reduction vs. SOTA under the same price budget.
  - Under equal budget, ThunderServe can deploy **~3× more replicas** in cloud settings than an in-house 8×A100 server, improving parallel capacity despite slower single-GPU performance. 
  - **Workload-aware phase ratios:** coding (long prompts, short outputs) → more prefill replicas; conversation (short prompts, long outputs) → more decode replicas; example optimal ratios (e.g., 5:3 vs 3:5) shown empirically
  - **Ablations:** removing KV compression adds ~**1.3×** per-request overhead; disabling orchestration adds another ~**4×** performance penalty. **Lightweight re-scheduling** cuts reconfig time from ~**157 s** (full) to ~**13 s** (no reload). 
- **Representative table/figures to look at in the paper:** 
  - Figure 2 (batching effects per phase), Table 3 (deployments by GPU type/TP/PP), Figures 8–12 (SLO, throughput, ablations), Figure 14–15
 ------
  
## 5. Limitations & Unimplemented Features

- Assumes clear **prefill/decode split** and typical bottlenecks (compute vs. memory bandwidth); applicability to non-autoregressive or atypical architectures is not addressed explicitly.
- Extreme low-bandwidth inter-node links may still bottleneck KV transfer even with compression.
- Fault tolerance beyond quick **lightweight re-scheduling** (e.g., broader failure modes, preemption policies) is not deeply explored.
- Focus is **inference only** (no training/fine-tuning).
- Full from-scratch scheduling still incurs seconds-scale search (acceptable pre-deploy; mitigated online via lightweight re-scheduling).

---

## 6. Broader Impacts & Future Directions

- **Impacts:** more accessible cloud LLM serving using mixed/cheaper GPUs; encourages topology-aware, phase-aware schedulers in industrial serving stacks.
- **Future work:** deeper fault tolerance and preemption handling; adaptive/learned KV compression; richer multi-tenant/multi-model scheduling; finer-grained network/topology adaptation. (Discussion consistent with paper’s scope and conclusions.)

---

## 7. Takeaway Summary

- **Core Idea:** Jointly optimize phase-splitting, heterogeneous GPU grouping, per-group parallelism, and orchestration—with **tabu-search scheduling** and **lightweight re-scheduling**—to improve SLOs and cost-efficiency for cloud LLM serving. :contentReference[oaicite:23]{index=23}

- **Key Contributions:**
  - Two-level hierarchical scheduler (upper: grouping + phase designation via tabu search; lower: per-group parallelism + orchestration). :contentReference[oaicite:24]{index=24}
  - **Lightweight re-scheduling** that adjusts plans without model reloads. :contentReference[oaicite:25]{index=25}
  - **KV cache compression** (quantize-for-transport, dequantize-on-receive) for cloud interconnects. :contentReference[oaicite:26]{index=26}
  - System implementation with NCCL-based comms, FlashAttention/PagedAttention, and extensive evaluation.

- **Benefits (measured):**
  - Up to **2.1×** throughput (avg **1.7×**) and up to **2.5×** latency improvement under equal budget vs. SOTA. :contentReference[oaicite:27]{index=27}

- **Limitations:**
  - Dependent on clear phase characteristics; limited discussion of extreme network constraints and comprehensive fault tolerance.

---

## References

- [ArXiv Preprint](https://arxiv.org/abs/2502.09334)
