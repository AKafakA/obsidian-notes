
# Supplementary Notes: System Design Decisions & Flaw Mitigations for Cara

This document summarizes the architectural decisions, identified theoretical/engineering flaws in baseline approaches, and the specific mitigations adopted for the Cara system design.

## 1. Performance Prediction Strategy
**Decision:** Adoption of **Black-Box Representation Learning (LSTM)** over Simulation.

* [cite_start]**Identified Flaw in Baselines:** Simulation-based approaches (e.g., Vidur, Mooncake) are "forward-incompatible" and brittle[cite: 149]. [cite_start]They rely on static kernel profiling (e.g., attention on A100) that breaks with software updates, kernel optimizations, or hardware degradation[cite: 148, 150, 151].
* [cite_start]**Our Solution:** We treat performance prediction as a sequence-to-value regression task using an LSTM[cite: 153, 154].
* [cite_start]**Advantage:** This approach automatically captures hidden system overheads (e.g., Python glue code latency) and adapts to heterogeneous hardware without manual re-profiling[cite: 152, 162].

## 2. Engineering Architecture: The "Sidecar" Pattern
**Decision:** Distributed Inference-Side Prediction.

* [cite_start]**Identified Flaw (The Serialization Bottleneck):** In a high-throughput cluster, transferring the full internal request queue (list of tuples) from every worker to a Central Router creates massive serialization and network overhead[cite: 141].
* [cite_start]**Our Solution:** We deploy the Performance Predictor as a co-located **local sidecar** on each worker instance[cite: 126, 137].
* **Implementation:**
    * [cite_start]**Input:** The Global Scheduler sends only the candidate request features (e.g., prompt length, prefix hash) to the worker[cite: 142].
    * [cite_start]**Process:** The sidecar accesses the local queue state directly in memory (zero-copy/low-latency) to run the prediction[cite: 143, 144].
    * [cite_start]**Output:** The sidecar returns a lightweight scalar (predicted latency score) rather than the full queue state[cite: 145].

## 3. Modeling "Batching Physics" & Capacity
**Decision:** Explicit Capacity Context Features.

* **Identified Flaw:** LSTMs model sequences serially (summing latencies), whereas vLLM utilizes continuous batching (parallel execution). A standard LSTM fails to predict the "Step Function" where latency spikes only *after* the hardware batch capacity is saturated.
* **Our Solution:** We augment the LSTM input sequence with explicit **Hardware Context Tokens**.
    * **Input Structure:** `[Context_Vector, Req_1, Req_2, ...]`
    * **Context Vector:** `{Max_Batch_Size, Current_Free_Slots, Total_GPU_Memory}`.
* **Result:** The model learns the non-linear relationship between queue depth and latency relative to the specific instance's capacity (handling heterogeneity).

## 4. Cache-Awareness Mechanism
**Decision:** Local **Shadow Radix Tree** (SGLang-style) over Remote Queries.

* **Identified Flaw:** Querying a centralized LMCache Controller API (HTTP/gRPC) for every routing decision introduces unacceptable latency (2-5ms) to the critical path, negating the speed benefits of cache hits.
* **Our Solution:** The Scheduler maintains a local **Shadow Radix Tree** (or Hash Map) to approximate the cache state of each worker without network calls.
* **Drift Mitigation:**
    * **Local Eviction:** The Scheduler implements a local LRU eviction policy mirroring the GPU's capacity to minimize state drift.
    * **Inter-Queue Dependency:** The predictor includes a boolean feature `Shared_Prefix_In_Queue` to account for "Ghost Hits"—requests currently in the waiting queue that will warm the cache for subsequent requests before they execute.

## 5. Handling P-D Disaggregation (Prefill-Decode)
**Decision:** Path-Vector Input Representation.

* **Identified Flaw:** In a Prefill-Decode (P-D) disaggregated cluster, "Latency" is a composite function of two distinct physical instances. Feeding the predictor only the state of the "Prefill Instance" is theoretically insufficient to predict end-to-end latency.
* **Our Solution:** Concatenated Path Inputs.
    * **Input:** `[State_Prefill_Instance, State_Decode_Instance, Request_Features]`.
* **Mechanism:** The LSTM learns to predict the total time: $T_{prefill}(Inst_A) + T_{transfer} + T_{decode}(Inst_B)$.

## 6. Data Collection & Distribution Shift
**Decision:** Randomized (Epsilon-Greedy) Data Generation.

* **Identified Flaw:** Training solely on production logs (FCFS) creates a biased dataset where "Deep Queue" always equals "High Latency." A smart router may create novel queue orderings (e.g., placing short jobs first) that represent Out-Of-Distribution (OOD) states for the model.
* **Our Solution:** The training data pipeline includes **Randomized Scheduling** (approx. 20% of samples). [cite_start]This acts as exploration, forcing the system into "bad states" (e.g., cache misses, mismatched queues) to teach the model the true penalty of sub-optimal routing[cite: 159, 160].

## 7. Scheduling Algorithm Choice
**Decision:** Greedy Mini-Batch Scheduling.

* [cite_start]**Justification:** While Mixed-Integer Linear Programming (MILP) provides optimal assignment, it is NP-hard and computationally infeasible for real-time serving[cite: 208, 209].
* [cite_start]**Approach:** We utilize a Greedy Mini-Batch Scheduler (analogous to beam search with beam size=1)[cite: 212, 223]. [cite_start]This balances decision quality with the strict low-latency requirements of the routing layer[cite: 211].