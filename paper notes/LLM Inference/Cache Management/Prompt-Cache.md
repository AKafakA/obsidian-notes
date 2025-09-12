Reading time: 1h 20 min

---

TLDR: Prompt Cache accelerates LLM inference by precomputing and reusing modular attention states across overlapping prompt segments, achieving up to 8× latency reduction on GPUs and 60× on CPUs without modifying the model or degrading output quality.

---

## 1. Motivation & Background

- Context for the problem:  
  Large language model (LLM) prompts often contain reusable text segments, such as system messages, overlapping documents in long-context applications (e.g., legal analysis, healthcare, education), and prompt templates in robotics or tool learning.

- Which domain or area the paper addresses:  
  Efficient inference for generative LLMs, focusing on reducing time-to-first-token (TTFT) latency in long-context tasks.

- Why this problem matters (e.g., scalability, cost, accuracy):  
  Inference latency scales quadratically with sequence length due to attention computation, impacting user experience in real-time applications and increasing costs for long prompts (e.g., 4K–10K tokens); reducing TTFT enables faster interactions while maintaining accuracy.

- Describe the baseline solutions:  
  The KV Cache reuses key-value (KV) attention states within a single prompt during autoregressive generation, reducing operations from ~6nd² + 4n²d to 6d² + 4nd (where n is sequence length, d is hidden dimension), but it only handles intra-prompt reuse.

- Identify **gaps or inefficiencies** in prior work that motivate this paper:  
  Prior work like paged attention enables limited inter-request prefix sharing within batches but is scenario-specific and lacks general modularity; there is no efficient way to reuse attention states across arbitrary prompts, especially handling position-dependent computations and recognizing reusable segments.

---

## 2. Key Insight

- The **core realization** or conceptual breakthrough of the paper:  
  LLM prompts have frequent reusable structures (e.g., shared modules like system prompts or documents), and LLMs can process attention states with discontinuous position IDs, allowing precomputation and modular reuse of these states across prompts.

- Explain **why previous approaches fall short** and how this paper’s idea addresses those shortcomings:  
  KV Cache and paged attention limit reuse to within a prompt or batch prefixes, recomputing attention for overlapping segments across requests; Prompt Cache addresses this by ***explicitly defining reusable "prompt modules" via a schema, enabling inter-request reuse that reduces quadratic attention costs to linear memory copies.**

---

## 3. The Prompt Cache Approach

### 3.1 Architecture / Method Overview

- High-level design or methodology:  
  Prompt Cache extends KV Cache to inter-request reuse by defining prompts in a Prompt Markup Language (PML) schema that specifies reusable text segments as "prompt modules" with assigned position IDs; during inference, cached KV states for modules are retrieved and concatenated, with new segments computed on top.

- Key components and how they interact:  
  PML schema defines modules (e.g., via `<module>` tags) and their relative positions; prompts import modules (e.g., `<miami/>`) and add uncached text; encoder precomputes KV states for modules; during inference, parser extracts modules, retrieves/concatenates KV states, and computes remaining attention using the cached prefix as KV Cache.

- If applicable, note the role of algorithms, models, or frameworks:  
  Supports LLMs like Llama2, MPT, and Falcon; uses lookup tables for position embeddings (RoPE, ALiBi) to handle discontinuous IDs; compatible with Hugging Face Transformers for encoding and inference.

### 3.2 Core Techniques

- Step-by-step explanation of the paper's main techniques or methods:  
  1. Define schema in PML: Specify modules with `<module>` tags, parameters via `<param>`, unions via `<union>` for alternatives, and nesting for hierarchies; assign unique, contiguous position IDs based on module order and sizes.  
  2. Derive prompts: Users write prompts importing schema modules (e.g., `<prompt schema="legal"> <miami/> <user>Query here</user> </prompt>`), replacing parameters (e.g., duration=3 days) with `<unk>` during caching.  
  3. Compute attention states (caching generation): Extract token sequences from schema; for parameterized modules, replace parameters with a fixed number of tokens equal to the len attribute (trailing whitespace does not affect semantics if actual length is shorter); encode modules separately with assigned position IDs and module-specific causal masking; compute KV states using the LLM, storing in CPU/GPU memory (e.g., 2.5 GB per 1K tokens for Llama 70B). This precomputation treats as placeholders to enable reuse across different parameter values
  4. Reuse during inference: Parse prompt to identify modules and positions, verifying alignment with schema; retrieve and concatenate precomputed KV states (e.g., (k_C, v_C) = concat(k_A, k_B)); compute KV for uncached segments (actual parameters, new text) at runtime, adopting the position IDs previously assigned to tokens; use concatenated KV as prefix cache for full attention. If position mismatch occurs (e.g., due to invalid schema alignment), the system falls back on verification during parsing to ensure usability, leveraging discontinuous position IDs to preserve relative positions without quality loss.
  5. Optional scaffolding: Extend attention spans across modules by duplicating keys/values, increasing memory but improving quality for dependent modules.

- Include:  
  - Algorithms, data structures, or models introduced:  
    PML as a markup language for modularity; Python API for automatic PML derivation from prompt programs (e.g., if-statements to unions).  
  - Theoretical foundation or equations if relevant:  
    Attention reuse reduces TTFT from O(n²d) (full computation) to O(nd) (linear copy + small computation), with quadratic gains as n grows.  
  - Clear explanation of how these techniques improve over prior methods:  
    Unlike KV Cache's intra-prompt reuse, Prompt Cache enables cross-prompt sharing via explicit modules, reducing recomputation for overlaps (e.g., 50–80% of tokens in LongBench); discontinuous IDs and masking allow flexible, position-accurate reuse without model changes.

---

## 4. Performance & Evaluation

- Summarize experimental results:  
  Prompt Cache reduces TTFT latency by 1.5×–10× on GPUs (5×–10× with GPU memory, 1.5×–3× with CPU memory) and 20×–70× on CPUs, with memory overhead of 0.2–4.5 MB/token depending on model size; output quality matches baseline (e.g., <2.5% accuracy difference in F1/Rouge L/Acc).

- Comparisons to baseline systems or methods:  
  Vs. KV Cache baseline: Up to 8× GPU speedup and 60× CPU speedup on long prompts (5K tokens); outperforms paged attention by generalizing reuse beyond prefixes/batches, enabling larger effective batch sizes via reduced memory.

- Highlight the most significant figures or tables:  
  Figure 3: GPU TTFT bars show 5×–10× reductions across 8 LongBench datasets on A100/A40/RTX 4090.  
  Figure 4: CPU TTFT shows 20×–70× gains on i9-13900K/Ryzen 7950X.  
  Table 1: Accuracy metrics (e.g., F1 scores) comparable to baseline across Llama2 7B/13B, MPT 7B, Falcon 7B.  
  Figure 5: Plots quadratic latency scaling vs. linear copy overhead, emphasizing gains for n > 1K.

- Mention datasets, benchmarks, or workload conditions:  
  LongBench (21 datasets, 6 categories: multi-doc QA, summarization, code; 4K–10K tokens); single-GPU/CPU setups; deterministic sampling; CPU memory for large caches (TB-scale), GPU for low-latency access.

---

## 5. Limitations & Unimplemented Features

- Clearly state **what the paper does NOT address** or remaining challenges:  
	Requires prompts to use PML schema, limiting ad-hoc flexibility; assumes modular structure, which may not fit all prompts without manual derivation; precomputed caches with for parameters confine attention to modules via masking, potentially ignoring cross-attention (self-attention) between parameters and other tokens/modules, as this precomputation treats placeholders separately without accounting for interactions (e.g., subsequent tokens attend to instead of actual values, leading to quality issues in dependent scenarios)
- Examples:  
	  - Missing features like preemption or fault tolerance: No eviction policy detailed (future work mentions GPU cache replacement); no handling of concurrent multi-request sharing beyond batch pointers.  
	  - Strong assumptions or hardware dependencies: Relies on CPU/GPU memory bandwidth; discontinuous positions may degrade quality without scaffolding (extra memory cost); tested only on single-device setups, not distributed inference.  
	  - Scalability or deployment constraints: GPU memory limits cache size (e.g., 40 GB caps ~8K tokens for Llama 70B); host-to-device copies add overhead for CPU storage.

---

## 6. Broader Impacts & Future Directions

- Potential implications for research and industry:  
  Enables faster LLM deployment in latency-sensitive apps (e.g., real-time QA, dialogue); reduces costs for long-context tasks in cloud/edge settings by minimizing compute.

- Opportunities for extending or generalizing this work:  
  Integrate with serving systems for batch-level sharing; apply to RAG for in-context retrieval; automate schema generation from unstructured prompts.

- Environmental or sustainability considerations, if applicable:  
  Lowers energy use via reduced computation, especially on CPUs for non-GPU environments.

---

## 7. Takeaway Summary

- **Core Idea:** Prompt Cache uses a modular schema to precompute and reuse attention states across LLM prompts, cutting inference latency quadratically with sequence length.

- **Key Contributions:**  
  - Introduces Prompt Markup Language (PML) for explicit definition of reusable prompt modules with position IDs and parameters.  
  - Enables inter-request KV state reuse with discontinuous positions and masking, compatible with existing LLMs.  
  - Demonstrates automatic PML derivation from prompt programs and optional scaffolding for quality.  
  - Evaluates on LongBench, showing 8× GPU and 60× CPU TTFT reductions without accuracy loss.

- **Benefits:**  
  - 1.5×–10× GPU TTFT speedup, 20×–70× CPU speedup for 4K–10K token prompts.  
  - Linear memory overhead (0.2–4.5 MB/token) vs. quadratic compute savings.  
  - Maintains output quality (e.g., comparable F1/Rouge L scores) and supports batch inference.

- **Limitations:**  
  - PML schema requirement reduces ad-hoc prompt flexibility.  
  - Memory scaling for large models; potential quality issues without scaffolding.  
  - No built-in eviction or distributed support; CPU-GPU copy overhead.

---

## References

- [Arxiv Preprint](https://arxiv.org/abs/2311.04934)
- [Project Page](https://github.com/yale-sys/prompt-cache/tree/main)]