Reading time: 1 hr

---

TLDR: HEXGEN is a flexible distributed inference engine that supports asymmetric partitioning of LLM generative inference over tensor model parallelism and pipeline parallelism for deployment across heterogeneous GPUs and networks, with a scheduling algorithm to optimize assignments for cost reduction and performance improvement.

---

## 1. Motivation & Background

- Context for the problem:  
  Serving generative inference requests for large language models (LLMs) in heterogeneous and cross-datacenter environments to mitigate high inference costs.

- Which domain or area the paper addresses:  
  LLM inference serving, distributed computing, and heterogeneous systems.

- Why this problem matters (e.g., scalability, cost, accuracy):  
    Centralized homogeneous deployments are expensive, limiting democratization of LLMs; heterogeneous setups reduce costs via affordable options like spot instances, serverless computing, or decentralized platforms while ensuring low-latency inference.

- Describe the **status quo** or baseline solutions:  
    LLM inference is hosted in centralized data centers with homogeneous high-performance GPUs; frameworks like HuggingFace Accelerate and FastTransformer support symmetric tensor model parallelism (TP) and pipeline parallelism (PP), where all TP groups have the same degree and PP stages handle equal layers.

- Identify **gaps or inefficiencies** in prior work that motivate this paper:  
    Existing frameworks assume homogeneous clusters, enforcing symmetric parallelism that fails in heterogeneous environments (e.g., varying GPU capacities and connections), leading to out-of-memory errors, underutilization, or high communication overheads.

---

## 2. Key Insight

- The **core realization** or conceptual breakthrough of the paper:  
    Asymmetric TP and PP, where each PP stage can have different layer counts and TP degrees, combined with a two-phase scheduling algorithm (genetic for global partitioning, DP for local optimization) that models heterogeneous costs to maximize service level objectives (SLO).

- Explain **why previous approaches fall short** and how this paper’s idea addresses those shortcomings:  
    Symmetric implementations limit performance in heterogeneous setups by uniform distribution, causing OOM or excessive comm; HEXGEN's asymmetry flexibly matches workloads to hardware, and scheduling optimizes via cost models (Table 1), achieving 2-19× speedups in case studies.

---

## 3. The HEXGEN Approach

### 3.1 Architecture / Method Overview

- High-level design or methodology:  
    Distributed inference engine with asymmetric parallelism over TP (intra-stage) and PP (inter-stage), scheduled via constrained optimization to partition GPUs into pipeline groups (replicas) and stages.

- Key components and how they interact:  
    Asymmetric implementation modifies pipeline init/comms; genetic algorithm partitions GPUs globally into groups; DP optimizes intra-group assignments; layer adjustments in genetic loop; SLO simulation evaluates.

- If applicable, note the role of algorithms, models, or frameworks:  
    Built on FlashAttention; genetic with mutations (merge/split/swap); DP (Algorithm 1); cost models (Table 1, Equations 1-3); AlpaServe simulator.

### 3.2 Core Techniques

- Step-by-step explanation of the paper's main techniques or methods:  
    1. Asymmetric parallelism: Initialize PP groups with varying TP degrees/layers per stage; select leader GPUs for low-latency inter-stage comm; broadcast activations within TP groups.  
    2. Formalize scheduling: Maximize $$ E_{T \sim P} [\text{SLO}(C_{\text{comm}}(\sigma) + C_{\text{comp}}(\sigma))] $$ subject to memory limits (Equation 1), with costs per Table 1.  
    3. Genetic phase: to partition GPUs into pipeline groups and set of model stages(layers) for each groups need to infer
    4. DP phase: generate the execution plan for each pipeline group for fixed group/layers, minimize pipeline cost (Equation 2) via DP (Equation 3, Algorithm 1); have same-type/machine TP heuristic constraints to reduce the search space and also reduce the potential inter-node communication cost.

- Include:  
    Algorithms: Genetic mutations, DP recursion.  
    Data structures: DP buffer 
    Models introduced: Cost formulations (comp, TP/PP comm, memory).  
    Theoretical foundation or equations if relevant:  
    Global opt:  
    $$ \sigma^* = \arg\max_{\sigma \in \Sigma} E_{T \sim P} [\text{SLO}(C_{\text{comm}}(\sigma) + C_{\text{comp}}(\sigma))] $$  
    s.t. $$ C^d_{\text{mem}}(\sigma) \leq M_d \quad \forall d \in D $$ (Equation 1).  
    Sub-opt:  
    $$ C^i_{\text{comp}}(\{d_{i,\sim}\}) + C^i_{\text{comm}}(\{d_{i,\sim}\}) = \sum_{j=1}^{S_i} C^{i,j}_{\text{comp}}(d_{i,j}) + \sum_{j=1}^{S_i} C^{i,j}_{\text{comm-tp}}(d_{i,j}) + \sum_{j=1}^{S_i-1} C^{i,j}_{\text{comm-pp}}(d_{i,j}) $$  
    s.t. $$ C^d_{\text{mem}}(\{d_{i,\sim}\}) \leq M_d \quad \forall d \in \bigcup_j d_{i,j} $$ (Equation 2).  
    DP transition:  
    $$ \text{DP}[j; \tau] = \min_{\tau_k \cdot e_k \subset d_{i,\sim}} \{\text{DP}[j-1; \tau - \tau_k \cdot e_k] + C^{i,j}_{\text{comp}}(\tau_k \cdot e_k) + C^{i,j}_{\text{comm}}(\tau_k \cdot e_k)\} $$ (Equation 3).

- Clear explanation of how these techniques improve over prior methods:  
    Asymmetry avoids symmetric failures; DP $$ O(S_i \cdot 4^{N_T}) $$ vs. exponential; genetic explores partitions efficiently, yielding 1.8-4× better SLO vs. baselines.

---

## 3.3 Serving System (Optional)

#### Deployment & Placement Strategy

- How the system is deployed across nodes or GPUs:  
    On heterogeneous cloud GPUs (e.g., AWS/FluidStack)

- Topology-aware placement rules (e.g., NVLINK, InfiniBand):  
    Communication-cost matrices between GPUs used for scheduling; TP same-machine/type; PP min latency links.

- Mapping of models or stages to hardware resources:  
    Genetic to groups (replicas); DP to stages (TP degrees); layers adjusted to memory.

#### Scheduling & Load Balancing

- How requests or workloads are distributed across GPU instances or replicas:  
    Coordinator routes to replicas; concurrent via multiple groups.

- Algorithms used (e.g., Join-the-Shortest-Queue, weighted fair scheduling):  
    Genetic + DP static; re-run on changes.

- Handling of bottlenecks, stragglers, and latency-sensitive workloads:  
    Cost models minimize max comp/comm; balanced assignments.

#### Scaling Strategy

- Dynamic mechanisms for reallocating GPUs/CPUs between different pools or services:  
    Re-run genetic on dynamics (<30s for 4 offline GPUs).

- Predictive modeling or simulation-based optimization used for scaling:  
    AlpaServe simulator for estimating SLO.

#### Request Flow

- Step-by-step description of how a request moves through the system:  
    1. To coordinator → route to pipeline group.  
    2. Prefill: Prompt through stages (PP sequential, TP concurrent within).  
    3. Decode: Token-by-token, PP inter-stage via leaders, TP intra-stage AllReduce.

---

## 4. Performance & Evaluation

- Summarize experimental results:  
    2.3× lower deadlines or 4× higher rates vs. homogeneous at same budget; matches at half budget; 1.8×/2× over symmetric HEXGEN; 3.5×/10× over Petals.

- Comparisons to baseline systems or methods:  
    Vs. FlashAttention (homo), symmetric HEXGEN, Petals, TGI; scheduling vs. random (faster convergence).

- Highlight the most significant figures or tables:  
    Figure 1 (case study 2-19× speedup); Figure 2 (SLO curves); Table 1 (costs).

- Mention datasets, benchmarks, or workload conditions:  
    Llama2-70B on LMSYS prompts; Poisson rates 0.125-10/s; seq 32-128; SLO % within A100 multiples.

---

## 5. Limitations & Unimplemented Features

- Clearly state **what the paper does NOT address** or remaining challenges:  
    No batching, fault tolerance, preemption, or dynamic migration; evaluation with static workloads only.

- Examples:  
    Missing preemption/faults; assumes Poisson; hardware deps on measured latencies; genetic scalability for large pools.

---

## 6. Broader Impacts & Future Directions

- Potential implications for research and industry:  
    Cost-effective LLM serving, enabling decentralization.

- Opportunities for extending or generalizing this work:  
    Add batching, dynamics, other parallelisms; extend to training/models.

- Environmental or sustainability considerations, if applicable:  
    Not addressed.

---

## 7. Takeaway Summary

- **Core Idea:**  
    Asymmetric TP/PP with two-phase scheduling for heterogeneous LLM inference.

- **Key Contributions:**  
    - Asymmetric implementation.  
    - Scheduling optimization (genetic + DP).  
    - Evaluation of cost-performance.

- **Benefits:**  
    2.3× lower deadlines/4× rates vs. homo at same budget; matches at half.

- **Limitations:**  
    No batching/faults; static re-run on changes.

---

## References

- [Arxiv Preprint](https://arxiv.org/abs/2311.11514)

- [GitHub Code (Optional)](https://github.com/Relaxed-System-Lab/HexGen)Reading time: 1 hr

---

