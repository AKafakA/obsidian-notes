**Reading time:** ~ 1 hr 30 min

---

## TLDR  
Mooncake introduces a disaggregated LLM serving architecture centered around a global, shared KVCache pool (across CPU/DRAM/SSD/RDMA), separates prefill and decode stages, and uses a KVCache-aware scheduler plus early-rejection under overload to greatly increase effective throughput while meeting latency SLOs (TTFT and TBT).  

---

## 1. Motivation & Background

- **Context for the problem:**
  - The domain is Model-as-a-Service (MaaS) LLM serving (inference) for large conversational / chat systems, particularly supporting long context lengths.
  - Key latency metrics are **Time to First Token (TTFT)** and **Time Between Tokens (TBT)** — both critical to user experience.
  - Modern GPU clusters have substantial underutilized **CPU, DRAM, SSD, and NIC (RDMA)** resources, which could be leveraged for caching.

- **Baseline / status quo:**
  - Prefill and decoding stages are often colocated and run entirely on GPUs.
  - KVCache (prefix cache) is typically **local-only**, so reuse is limited to a single machine.
  - Systems like **vLLM** support optimizations like *prefix caching* and *chunked prefill*, but without global sharing.
  - Load balancing is usually random or simple queue-based, not cache-aware.

- **Gaps motivating this paper:**
  - **Long context prompts** make prefill expensive; without reuse, computation is wasted.
  - Lack of **global KVCache** leads to low cache hit rates and poor throughput.
  - Simple scheduling fails under **strict TTFT/TBT SLOs**, leading to degraded performance or SLO violations.
  - No proactive mechanism to **reject requests early** under overload, causing wasted work and poor latency.

---

## 2. Key Insight

- Treat **KVCache as a first-class global resource**, disaggregated and shared across nodes, to maximize reuse of expensive prefix computation.
- Build a **KVCache-aware scheduler** that:
  - Understands cache locality,
  - Predicts TTFT (per-request) and TBT(system-level) under current cluster conditions,
  - And can reject requests early if SLOs cannot be met.
- By separating **prefill and decode stages**, GPUs can be used more efficiently:
  - Prefill = compute-intensive,
  - Decode = latency-sensitive.

---

## 3. The Mooncake Approach

### 3.1 Architecture / Method Overview

- **High-level design:**
  - Two distinct GPU pools:
    - **Prefill pool:** Processes prompts and generates KVCache.
    - **Decode pool:** Streams output tokens using the KVCache.
  - **Mooncake Store (Global KVCache):** A distributed KVCache system spanning CPU/DRAM/SSD with RDMA for fast transfer.
  - **Transfer Engine:** High-throughput, topology-aware KVCache transfer layer leveraging multiple NICs and zero-copy RDMA.
  - **Conductor (Scheduler):** Allocates requests to prefill/decode pools, optimising for cache hit ratio and SLO adherence.
- **Techniques to reduce latency:**
	- **Chunked Pipeline Parallelism (CPP):** Break long prompts into chunks processed across GPUs in a pipeline.
	- **Layer-wise prefill streaming:** Begin transferring usable KVCache before prefill fully completes.
---

### 3.2 Core Techniques

- **KVCache Management:**
- KVCache divided into **paged cache blocks**, each with a unique key (hash-id).
- **Global matching:** Identify longest matching prefix across all stored blocks for new requests.
- **Eviction policy:** LRU with protection for blocks currently in use.

- **Scheduling Algorithm:**
- Prefill scheduling considers:
  1. Cache reuse potential,
  2. Instance load and queue times,
  3. SLO predictions for TTFT and TBT.
- **Hot-spot replication:** Frequently accessed blocks are replicated across nodes to reduce contention.
- **Early rejection under overload:**
  - Predict latency for a new request.
  - Reject it immediately if meeting TTFT/TBT is impossible, preventing wasted computation.

- **Computation & Transfer Overlap:**
- As partial KVCache is produced, it is streamed to the decode node, overlapping compute and communication.

---

## 3.3 Serving System

### Deployment & Placement Strategy
- Prefill and decode GPU pools have different hardware optimizations.
- Global KVCache uses **RDMA** over a fabric with up to **8×400 Gbps NICs per node** for high transfer throughput.
- [High-performance Transfer Engine](https://kvcache-ai.github.io/Mooncake/design/transfer-engine.html) performs topology-aware path selection.
### Scheduling & Load Balancing
- **Conductor**:
- Matches each request to optimal prefill and decode nodes.
- Factors in cache hit ratio, load, and queue delays.
- Cache-aware load balancing prevents single-node hot spots.
- Replicates hot blocks to maintain low latency.

### Request Flow
1. Request arrives
2. Conductor predicts TTFT/TBT
3. Select prefill/decode node based on predicted TTFT/TBT → load cached KV blocks from Mooncake Store if no early rejected due to SLO unmet (Algorithm 1)
4. Cache updated, eviction or replication as needed.
---

## 4. Performance & Evaluation

- **Setup:**
- Real trace from **Kimi** service, anonymized with token counts and timestamps.
- Dummy model matching **LLaMA2-70B** architecture.
- Cluster: 16 nodes, each with 8×A800 GPUs.

- **Key metrics:**
- **Effective request capacity** under SLOs.
- TTFT and TBT compliance rates.
- Prefill GPU time reduction.
- Global vs local cache hit rates.

- **Results:**
- Effective capacity improvement: **+59% to +498%** over vLLM baselines depending on SLOs.
- Production deployment:
  - **115%** more requests on A800 clusters.
  - **107%** more on H800 clusters.
- Prefill GPU time reduced up to **3.33×** compared to vLLM with prefix caching.
- Global cache improves hit rates by **2.36×**, reducing prefill time by up to **48%**.
- Transfer Engine achieves near-RDMA line rate, significantly faster than TCP.

---

## 5. Limitations & Unimplemented Features

- Experiments use **dummy models**, so absolute performance may differ in production.
- Limited support for **elastic scaling** of GPU pools; relies mainly on static pools + scheduling.
- **Network-heavy design:** Requires high-bandwidth RDMA; performance may degrade in limited network environments.
- **Fault tolerance not deeply addressed:** No detailed recovery plan for cache node failures.
- Prediction-based rejection depends on accurate latency models; mispredictions could harm performance.
- Benefits shrink when workloads have **low prefix reuse**.

---

## 6. Broader Impacts & Future Directions

- **Research and industry impact:**
- Encourages use of **storage/network resources** to offset GPU computation cost.
- Provides a blueprint for **long-context LLM serving**.
- Cache-aware scheduling concepts applicable to other distributed inference systems.

- **Future work:**
- KVCache compression or quantization to reduce storage and transfer cost.
- More dynamic resizing of prefill vs decode pools.
- Advanced fault tolerance for the global cache store.
- Integration with specialized hardware for attention/KVCache handling.

- **Sustainability:**
- Reduces redundant computation → lower energy and infrastructure cost per token.
- Utilizes idle resources (CPU, DRAM, SSD) effectively.

---

## 7. Takeaway Summary

- **Core Idea:**  
Use a **global KVCache pool** with cache-aware scheduling and overload rejection to minimize redundant prefill computation and maximize throughput while meeting strict latency SLOs.

- **Key Contributions:**
- Disaggregated architecture separating prefill and decode stages.
- Global KVCache store leveraging idle cluster storage and RDMA networking.
- Chunked and layer-wise prefill with compute/transfer overlap.
- KVCache-centric scheduling algorithm optimizing for cache hits and SLOs.
- Early rejection mechanism under overload to prevent wasted work.

- **Benefits:**
- Up to **498% increase** in request capacity under SLOs.
- **115%+ production improvement** in request handling capacity.
- Dramatic reductions in prefill GPU time and cache misses.

- **Limitations:**
- Relies on accurate latency prediction for scheduling and rejection.
- Requires substantial high-bandwidth networking and storage capacity.
- Fault tolerance and dynamic scaling not fully addressed.

---

## References
- [Arxiv PDF](https://arxiv.org/pdf/2407.00079)
- [USENIX FAST '25 Presentation](https://www.usenix.org/conference/fast25/presentation/qin)
- [Mooncake GitHub Repository](https://github.com/kvcache-ai/Mooncake)

