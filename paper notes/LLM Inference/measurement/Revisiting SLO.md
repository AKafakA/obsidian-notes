

Reading time: ~40 min

---

TLDR: The paper argues that existing SLO and goodput metrics in LLM serving do not adequately capture user experience, and proposes a unified metric called **smooth goodput** (with a revised token-level SLO) which weights both the number of tokens generated and the waiting time users suffer, to give a more realistic evaluation.

---

## 1. Motivation & Background

- **Context for the problem:**
  - Large Language Models (LLMs) are increasingly deployed in services such as chatbots and assistants, with inference served in a **streaming** (autoregressive) fashion.  
  - Key metrics today include TTFT (Time-to-First-Token), TBT (Time Between Tokens), TPOT (Time per Output Token), and E2E latency. At the system level, SLO attainment and goodput are widely used to measure performance.

- **Domain or area addressed:**
  - LLM serving and inference systems, especially streaming generation where tokens are delivered incrementally.

- **Why this problem matters:**
  - User experience degrades when there are long pauses or stalls in token delivery. Even if throughput is high, users may abandon the service if they have to wait without content.  
  - Current metrics can incentivize undesirable behaviors such as:
    - Dropping requests likely to violate SLOs to inflate goodput.
    - Delaying tokens to artificially smooth TBT tails, while worsening actual user waiting time.

- **Status quo / baseline:**
  - Classical metrics focus on per-token latency (e.g., TBT) or binary pass/fail SLO attainment at the request level.  
  - Goodput credits only fully successful requests, counting partially successful ones as zero.

- **Gaps motivating this paper:**
  - **Gap 1:** TBT-based metrics penalize any irregularity, even when the user has buffered tokens and does not perceive a problem.  
  - **Gap 2:** Goodput ignores partial usefulness of requests that fail strict SLOs, failing to account for partially delivered content.

---

## 2. Key Insight

- Metrics should reflect how token timing aligns with **user consumption behavior**.  
- The real problem is **user idle latency** — intervals when the user has nothing new to read.  
- Even failed requests often provide value through earlier tokens, which should be counted.  
- The paper proposes:
  1. A **revised token-level SLO** where each token has a deadline relative to the first token and user reading speed.
  2. A **smooth goodput metric** that balances total tokens generated against penalties for user idle time, rewarding partial progress instead of binary success/failure.

---

## 3. The Smooth Goodput Approach

### 3.1 Architecture / Method Overview

- This work is a **metric framework**, not a new serving system.  
- Components:
  1. **Token-level SLO** with deadlines based on assumed user reading speed.  
  2. **User Idle Latency** metric to measure time users wait without content.  
  3. **Benefit Function** that combines total tokens and idle latency penalty.  
  4. **Smooth Goodput** as the aggregate metric across all requests.

### 3.2 Core Techniques

- **Token-level SLO:**
  - Deadline for token \(i\):  
    $$
    d_i = v \cdot i
    $$
    where \(v\) = user reading speed (tokens/sec).  
  - Each token must arrive before its deadline to avoid idle time.

- **User Idle Latency \(l_r\):**
  $$
  l_r = \max_{1 \leq i \leq n} (t_i - d_i)
  $$
  where \(t_i\) = actual generation time of token \(i\).

- **Benefit of a single request:**
  $$
  \text{benefit}(r) = n_r - \alpha \cdot f(l_r)
  $$
  - \(n_r\) = total tokens for request \(r\)  
  - \(α\) = penalty weight  
  - \(f(l_r)\) = mapping from idle latency to penalty fraction.

- **Smooth Goodput over time \(T\):**
  $$
  \text{smooth goodput} = \frac{\sum_{r \in R} \text{benefit}(r)}{T}
  $$
  - Unlike traditional goodput, this gives **partial credit** to partially successful requests.

---

## 3.3 Serving System (Optional)

#### Deployment & Placement Strategy
- Experiments use **vLLM v0.5.3.post1** on a single NVIDIA A100 40GB GPU.

#### Scheduling & Load Balancing
- vLLM’s throughput-optimal scheduler and **chunked prefill** strategy are evaluated.
- Request arrivals follow a Poisson distribution.

#### Request Flow
- A request undergoes a prefill phase, then generates tokens sequentially.
- Each token’s arrival time is compared to its deadline to compute idle latency and benefit.

---

## 4. Performance & Evaluation

- **Experimental Setup:**
  - Model: **Qwen2-7B**
  - Dataset: **ShareGPT_gpt4** conversations and concatenated long conversations (~1600 tokens).
  - Hardware: Single NVIDIA A100 GPU.

- **Metrics Compared:**
  - Existing: Throughput, TTFT, tail TBT, etc.
  - Proposed: Smooth goodput with default user reading speed of 20 tokens/sec.

- **Key Findings:**
  1. **Chunked prefills** appear beneficial under traditional metrics but do **not always improve smooth goodput**, showing misalignment between TBT tail reduction and real UX gains.
  2. Smooth goodput grows with QPS in unsaturated regimes but drops sharply when overload causes excessive idle latency.
  3. The new token-level SLO highlights problematic requests more accurately than strict per-token deadlines.

---

## 5. Limitations & Unimplemented Features

- Requires calibration of parameters (user speed \(v\), penalty weight \(α\), penalty function \(f(·)\)) for each deployment.  
- Evaluations are limited to single-GPU setups; no distributed inference scenarios explored.  
- Simplistic user model assumes constant reading speed and immediate consumption of tokens.  
- Network delivery latency and frontend delays are not included.  
- Primarily applies to streaming interactive workloads; less relevant for offline batch inference.

---

## 6. Broader Impacts & Future Directions

- **Implications:**
  - Provides a UX-aligned performance metric for LLM serving.
  - Could guide scheduler design and admission control to prioritize early token delivery.

- **Extensions:**
  - Apply to distributed, multi-node serving systems with network latency.
  - Incorporate richer user behavior models (variable speeds, abandonments).
  - Weight tokens by semantic importance.

- **Sustainability:**
  - By valuing partial results, smooth goodput can reduce wasted computation from dropped requests.
  - However, optimizing for responsiveness may increase hardware utilization.

---

## 7. Takeaway Summary

- **Core Idea:**  
  Introduces smooth goodput and a revised token-level SLO to better align LLM serving metrics with user-perceived experience by balancing token throughput with idle waiting time.

- **Key Contributions:**
  - Identifies flaws in traditional SLO/goodput and token-interval metrics.
  - Defines deadlines for tokens relative to first token using user reading speed.
  - Introduces user idle latency and smooth goodput metric.
  - Evaluates existing strategies like chunked prefills under the new framework.

- **Benefits:**
  - Captures partial utility of partially completed requests.
  - Encourages systems to prioritize early, continuous token delivery.
  - Reveals cases where classical optimizations misalign with user experience.

- **Limitations:**
  - Needs careful parameter tuning.
  - Limited to streaming settings; less relevant for batch inference.
  - Does not model network latency or complex user behaviors.

---

## References

- [Arxiv Preprint](https://arxiv.org/abs/2410.14257)
