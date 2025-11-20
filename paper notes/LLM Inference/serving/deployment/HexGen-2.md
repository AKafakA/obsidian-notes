# HexGen-2: Disaggregated Generative Inference of LLMs in Heterogeneous Environment

Reading time: ~45 min  

---

**TLDR:** HexGen-2 uses **graph-based disaggregated scheduling** to optimally place prefill/decoding across heterogeneous GPUs, formulating placement as a **max-flow problem** and refining allocations through **edge-swapping**, achieving significant throughput and cost efficiency.  

---

## 1. Motivation & Background

- **Context for the problem:**
  - Modern LLM serving mixes prefill-heavy (compute-intensive) and decode-heavy (latency-sensitive) requests.
  - Data centers increasingly have **heterogeneous GPUs** with diverse compute/memory/network profiles.
  - Efficient scheduling across such environments is non-trivial.

- **Status quo & limitations:**
  - Systems like vLLM, Orca, DistServe, and HexGen-v1 assume **homogeneous GPU clusters** or use static partitioning.
  - Leads to **inefficient GPU utilization** in real-world mixed hardware clusters.

- **Gap addressed:**
  - Lack of a **general-purpose scheduler** that (1) adapts to heterogeneous GPUs, (2) jointly optimizes prefill & decode placement, and (3) maximizes **end-to-end throughput**.  

---

## 2. Key Insight

- **Observation:**  
  Prefill and decode stages form a **bipartite dependency structure**: large-batch prefill results must flow to decode nodes, and throughput is bounded by the slowest cut.  

- **Core idea:**  
  Model heterogeneous serving as a **graph partition + max-flow problem**. Use **graph merging for partitioning**, then solve placement with **max-flow optimization**, and refine with **edge-swapping** for balancing.  

---

## 3. The HexGen-2 Approach  

### 3.1 Architecture / Method Overview

- **Two disaggregated services:**  
  - **Prefill pool:** heavy matrix multiplications (batched, high compute).  
  - **Decode pool:** small sequential steps (latency-sensitive, lower compute).  

- **Scheduling challenge:**  
  - Prefill GPUs ↔ Decode GPUs must be mapped efficiently while considering heterogeneous hardware capacities and network bandwidth.  

---

### 3.2 Scheduling Algorithm  

#### Step 1: Graph Construction  

- **Input:**  
  - Heterogeneous GPU cluster (raw devices with compute, memory, and bandwidth profiles).  
  - Workload request set (prefill and decode tasks).  

- **Process:**  
  - Build a graph where **each GPU is an individual node**.  
  - Edges represent potential KV-cache transfer between GPUs.  
  - No GPU is yet assigned to “prefill” or “decode” roles.  
  - Edge weights capture interconnect bandwidth and communication cost.  

- **Output:**  
  - Fine-grained graph \( G = (V, E) \) with GPU nodes and possible communication links.  

- **Target:**  
  - Encode the heterogeneous cluster into a graph without premature role assignment.  

---

#### Step 2: Node Partitioning via Graph Merging  

- **Input:**  
  - Graph \( G \) from Step 1.  

- **Process:**  
  - Merge GPUs with **similar resource characteristics** into **super-nodes**.  
  - Similarity is measured via profiling (TFLOPS, memory BW, NVLink/PCIe).  
  - After merging, each **super-node** is assigned a role: **prefill** or **decode**, based on aggregated profile.  

- **Output:**  
  - Coarse-grained graph \( G' \) with super-nodes labeled as prefill or decode.  

- **Target:**  
  - Simplify the scheduling problem and enable role assignment at the partition level.  

---

#### Step 3: Placement as Max-Flow Problem  

- **Input:**  
  - Partitioned graph \( G' \) with super-nodes labeled as prefill or decode.  
  - Capacity constraints per super-node (throughput, memory).  

- **Process:**  
  - Model request flow from prefill super-nodes to decode super-nodes.  
  - Formulate as a **max-flow optimization problem**:  

$$
\max \sum_{(p,d) \in E} f(p,d) \quad \text{s.t. } f(p,d) \leq c(p,d), \ \sum f \leq \text{supernode capacity}
$$

  - Solve using standard max-flow algorithms.  

- **Output:**  
  - Initial placement of requests across prefill and decode super-nodes.  

- **Target:**  
  - Maximize system throughput under heterogeneous resource constraints.  

---

#### Step 4: Iterative Edge Swapping (Load Rebalancing)  

- **Input:**  
  - Initial max-flow assignment.  
  - Real-time utilization feedback.  

- **Process:**  
  1. Detect overloaded decode super-nodes.  
  2. Identify alternative underutilized super-nodes.  
  3. Re-route requests (swap edges).  
  4. Repeat until balanced or convergence.  

- **Output:**  
  - Refined mapping with balanced utilization.  

- **Target:**  
  - Mitigate stragglers and maintain high throughput while honoring SLOs.  


---

## 3.3 Serving System  

#### Deployment & Placement Strategy
- **Prefill nodes:** mapped to compute-optimized GPUs (A100/H100).  
- **Decode nodes:** mapped to cost-efficient GPUs (V100/T4).  
- Graph-based scheduling ensures balanced flow across heterogeneous devices.  

#### Scheduling & Load Balancing
- Global scheduler computes flow assignments.  
- Edge swapping dynamically rebalances workloads.  

#### Scaling
- Independent scaling of prefill vs. decode pools.  
- Scheduler recomputes flow when scaling events occur.  

#### Request Flow
1. Request enters prefill service.  
2. KV-cache produced and transferred to decode pool.  
3. Decode proceeds token-by-token until response is completed.  

---

## 4. Performance & Evaluation

- **Benchmarks:** GPT-style LLMs with varied workloads (short & long prompts).  
- **Results:**  
  - **Throughput:** Up to *2.1× higher* than homogeneous baseline.  
  - **Latency:** Comparable or better (especially under load).  
  - **Cost efficiency:** Mix of GPU types lowers serving cost significantly.  
- **Baselines:** vLLM, Orca, DistServe, HexGen-v1.  
- **Cluster setup:** Mix of A100, V100, L40, and T4 GPUs.  

---

## 5. Limitations & Unimplemented Features

- Relies on **high-bandwidth interconnects** (InfiniBand/PCIe Gen4).  
- Added **scheduling complexity** compared to homogeneous systems.  
- No explicit **fault tolerance** or **preemption** mechanisms.  

---

## 6. Broader Impacts & Future Directions

- **Research:** Extends disaggregation to **graph-optimized heterogeneous scheduling**.  
- **Industry:** Enables cost savings in GPU-diverse data centers.  
- **Future:**  
  - Multi-tenant fairness scheduling.  
  - Energy-aware graph scheduling.  
  - Extensions to multi-modal workloads with diverse compute profiles.  

---

## 7. Takeaway Summary

- **Core Idea:**  
  Use **graph-based max-flow scheduling** with iterative edge swapping to optimally map prefill/decoding across heterogeneous GPUs.  

- **Key Contributions:**  
  - Graph merging for node partitioning.  
  - Max-flow formulation for placement optimization.  
  - Edge-swapping refinement for balancing throughput.  
  - Demonstrated gains over homogeneous-serving systems.  

- **Benefits:**  
  - 2.1× throughput improvement.  
  - Cost reduction via GPU heterogeneity.  
  - Low tail latency under bursty workloads.  

- **Limitations:**  
  - Communication overhead from KV transfer.  
  - Requires fast interconnects.  
  - Deployment complexity.  

---

## References

- [Arxiv Preprint](https://arxiv.org/abs/2408.XXXX)  
- [GitHub Code (Optional)](#)  
