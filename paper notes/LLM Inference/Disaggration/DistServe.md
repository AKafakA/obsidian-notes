Reading time: 1 hr 25 min

---

TLDR: **DistServe** introduces *Prefill-Decoding (PD) disaggregation* for LLM serving by splitting the **prefill stage** (prompt computation) and **decoding stage** (token generation) across separate GPU pools.  
This separation eliminates interference between the two phases and allows **phase-specific optimization** to improve both throughput and latency under **strict latency SLOs** for **TTFT** (*Time to First Token*) and **TPOT** (*Time Per Output Token*).  
DistServe uses a **simulation-driven optimizer** to determine GPU allocation, parallelism strategy, and placement under realistic hardware constraints.

---

## 1. Motivation & Background

- LLM inference naturally has **two distinct computational phases**:
  - **Prefill (Prompt Computation)**  
    - Processes the entire input prompt **in parallel** to generate the first output token.  
    - **Highly compute-intensive**, well-suited for large batches.
  - **Decoding (Token Generation)**  
    - Sequentially generates tokens, one step at a time, dependent on previous outputs.  
    - **Memory bandwidth-bound**, dominated by KV-cache operations and inherently harder to parallelize.

- **Mismatch in hardware needs**:
  - Prefill: FLOPs-bound → benefits from cutting-edge GPUs like H100.
  - Decoding: Latency-sensitive and memory-bound → can run efficiently on older or cheaper GPUs like A100.

- **Problem with current colocated designs**:
  - Existing serving systems colocate prefill and decoding on the **same GPU**:
    - Prefill’s large batch jobs **delay decoding**, creating latency spikes.
    - Mixed workloads prevent either phase from achieving optimal batching.
    - Hard to meet both TTFT and TPOT simultaneously.
    - Leads to **over-provisioning** and wasted resources.

---

## 2. Key Insight

> Prefill and decoding have **fundamentally different performance characteristics**, and colocating them creates interference and inefficiency.

- **Decouple prefill and decoding into separate GPU pools**, so each can be optimized independently:
  - Prefill GPUs: optimized for high-throughput, large-batch parallelism.
  - Decoding GPUs: optimized for latency-sensitive sequential token generation.

- Introduce **goodput** as the optimization target:
  - **Goodput** = number of requests served per GPU **while meeting latency SLOs** (both TTFT and TPOT).
  - Unlike raw throughput, goodput reflects real-world user experience.

---

## 3. The DistServe Approach

DistServe has two main components:
1. **Disaggregated serving architecture** with separate GPU pools.
2. **Simulator-based optimizer** to decide GPU allocation, parallelism, and placement under constraints.

### 3.1 Disaggregated GPU Pools
- **Prefill Pool**
  - Dedicated to prompt computation.
  - Uses **tensor parallelism (intra-operator parallelism)** for highly batched operations.
  - Focus: maximize throughput of compute-heavy workloads.

- **Decoding Pool**
  - Dedicated to sequential token generation.
  - Uses smaller tensor splits or pipeline parallelism.
  - Focus: minimize per-token latency and queueing delay.

### 3.2 KV-Cache Transfer and Placement
- After prefill finishes, its **KV-cache** must be transferred to a decoding GPU to continue generation.
- **Placement decisions depend on cluster topology**:
  - **With NVLINK (intra-node)**:
    - Prefill and decoding stages for a request are placed on the same node to leverage high-bandwidth NVLINK for fast transfers.
  - **With InfiniBand or RDMA (cross-node)**:
    - Prefill and decoding stages can be distributed across nodes.
    - Simulator models network latency to ensure transfer overhead stays negligible.
  - **Without high-speed interconnect**:
    - the same stage for pipeline parallelism from Prefill and decoding instances must run on the **same node** to avoid slow PCIe/Ethernet transfer.

- **Transfer mechanism**:
  - Performed **after prefill finishes**, not streamed progressively.
  - Overhead is negligible (<0.1% of total latency) when NVLINK or InfiniBand is available:

### 3.3 Parallelism Strategy
- **Prefill** → **Tensor parallelism** (large-batch efficiency).
- **Decoding** → Flexible:
  - Pipeline parallelism for very large models.
  - Smaller tensor splits for low-latency generation.
- DistServe allows **different parallelism strategies** per pool — something monolithic systems cannot do.

### 3.4 Simulator-Driven Resource Optimization
- A lightweight simulator explores different configurations to **maximize goodput** given:
  - Latency SLOs for TTFT and TPOT.
  - Hardware topology (NVLINK, InfiniBand availability).
  - Workload characteristics (prompt lengths, token distributions).

- **Outputs of the simulator**:
  1. GPU count split between prefill and decoding pools.
  2. Optimal parallelism type and degree for each pool.
  3. Placement plan that minimizes KV-cache transfer bottlenecks.

- **Runtime**:
  - Simulation runs in **under 1.3 minutes**, enabling rapid reconfiguration for changing workloads.


### 3.5 Scheduling and Load Balancing

DistServe uses **two-level scheduling**:
#### **Cluster-Level Scheduler (CLS)**
- Responsible for routing incoming requests to GPU instances.
- Uses **Join-the-Shortest-Queue (JSQ)** independently for each pool:
  - Assigns a request to the **prefill instance** with the shortest queue.
  - After prefill finishes, assigns it to the **decoding instance** with the shortest queue.
- This balances workloads and prevents stragglers but **does not reserve decoding slots ahead of time**.
  - If decoding GPUs are overloaded, a completed prefill job may wait briefly before decoding can start.

#### **Machine-Level Scheduler (MLS)**
- Runs on each node to:
  - Manage batching:
    - Prefill → large batches for throughput.
    - Decoding → small batches for low latency.
  - Coordinate execution for efficient GPU utilization.

### 3.6 Scaling Strategy
- GPU ratios between pools are **dynamically adjusted** based on workload mix:
  - Prompt-heavy → allocate more GPUs to prefill pool.
  - Token-heavy → allocate more GPUs to decoding pool.
- Adjustments are guided by the simulator and real-time latency telemetry.

---

## 4. Performance & Efficiency Gains

- **Throughput and Latency**:
  - Up to **7.4× more requests served** compared to colocated systems like vLLM.
  - **>90% of requests** meet both TTFT and TPOT SLOs.
  - Achieves **12.6× tighter SLO compliance**.

- **Workload-specific results**:
  - **Chatbots (short prompts)** → 1.8–3.2× better latency adherence vs vLLM.
  - **Code completion (OPT-66B)** → 5.7× higher request rate, 1.4× tighter SLOs.
  - **Summarization (long prompts)** → 4.3× throughput improvement, 12.6× tighter SLOs.

- **Simulation accuracy**:
  - Predictions match actual performance within **2% error**.

---

## 5. Limitations & Unimplemented Features

1. **No Preemption**
   - Prefill and decoding jobs cannot be interrupted mid-execution.
   - Latency-sensitive decoding jobs may suffer during heavy prompt loads.

2. **Independent Scheduling**
   - Prefill and decoding stages are scheduled separately.
   - No atomic reservation → completed prefill jobs may wait for decode GPUs.

3. **Static Scaling Interval**
   - Pool sizes are adjusted periodically, not continuously.

4. **High-Bandwidth Dependency**
   - Optimal performance depends on NVLINK or InfiniBand.
   - Without these, prefill and decoding must colocate, limiting flexibility.

5. **Limited Fault Tolerance**
   - Failures in one pool may stall requests; robust failover is not yet implemented.

---

## 6. Broader Impacts & Future Directions

- **Paradigm shift**:
  - Establishes **Prefill-Decoding disaggregation** as a core design for LLM serving.
  - Opens the door to follow-up work like Arrow and ConServe for finer-grained disaggregation.

- **Hardware specialization**:
  - Future accelerators could target:
    - Prefill: FLOPs-intensive, batch-optimized chips.
    - Decoding: latency-focused, high-bandwidth chips.

- **Sustainability**:
  - Decoding can run on **older or lower-cost GPUs**, reducing hardware costs and improving utilization.

---

## 7. Takeaway Summary

- **Core Idea**:
  Separate prefill and decoding into distinct GPU pools, optimizing each for its unique workload characteristics.

- **Key Contributions**:
  - Disaggregated design with phase-specific batching and parallelism.
  - Simulation-based optimizer for goodput and latency compliance.
  - JSQ-based load balancing for independent prefill and decoding pools.

- **Benefits**:
  - Eliminates phase interference.
  - 7.4× throughput improvement, 12.6× tighter SLOs.
  - >90% of requests meet latency SLOs.

- **Limitations**:
  - No preemption or cross-phase reservation.
  - Heavy reliance on high-speed interconnects.



