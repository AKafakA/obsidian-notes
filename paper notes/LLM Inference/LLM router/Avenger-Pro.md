
Reading time: ~35 min --- TLDR: Avengers-Pro is a test-time routing framework that optimizes the trade-off between performance and cost by dynamically routing queries to an ensemble of heterogeneous LLMs. By introducing a tunable parameter ($\alpha$) into a cluster-based routing score, it achieves a Pareto frontier that outperforms single "giant" models like GPT-5-medium in both accuracy and efficiency. ---

## 1. Motivation & Background
- **Context for the problem:**
    - The advancement of LLMs faces a fundamental dilemma: the trade-off between **Performance** (accuracy/capability) and **Efficiency** (cost/latency).
    - Proprietary giants like "GPT-5" attempt to solve this with internal routing (switching between an efficient model and a "thinking" model), but these internal routers are often opaque and fixed.
- **Status quo:**
    - Existing open-source routing methods either focus purely on maximizing performance (ignoring cost, like the original *Avengers*) or use complex trained neural networks that are hard to adapt to new models.
- **Gaps or inefficiencies:**
    - **Lack of Granular Control:** Users cannot easily tune the "willingness to pay" vs. "quality requirement" trade-off in existing systems.
    - **Suboptimal Pareto Frontiers:** Single models (even routed ones like GPT-5) often lie below the optimal efficiency frontier because they are limited to their own model family.

---

## 2. Key Insight
- **The core realization:**
    - A diverse ecosystem of models (e.g., Claude, Gemini, GPT, Llama) offers a "Market of Intelligence" where different models offer better value (performance/dollar) for different types of queries.
    - By explicitly modeling both **Accuracy** and **Cost** in the routing score formulation, we can trace a **Pareto Frontier** that is superior to any single model family.
- **Why previous approaches fall short:**
    - Previous "Avengers" work focused only on *collective intelligence* (beating the giant at all costs) with model blending/aggragation. *Avengers-Pro* realizes that for real-world adoption, cost is as critical as accuracy, and introduces a mechanism to navigate this trade-off dynamically with single model get queried.

---

## 3. The Avengers-Pro Approach

### 3.1 Architecture / Method Overview
- **Framework:** A training-free, cluster-based routing system.
- **Workflow:**
    1.  **Embedding:** Map queries to vector space.
    2.  **Clustering:** Group queries into semantic clusters (unsupervised).
    3.  **Profiling:** Measure both Accuracy ($P$) and Cost ($Q$) for every model in every cluster.
    4.  **Routing:** Select the model that maximizes a weighted score of $P$ and $Q$.

### 3.2 Core Techniques
- **Performance-Efficiency Score ($S$):**
    - The router ranks models using a unified score derived from cluster-specific statistics:
      $$S_{i,j} = \alpha \cdot \tilde{P}_{i,j} + (1 - \alpha) \cdot (1 - \tilde{Q}_{i,j})$$
    - Where:
        - $\tilde{P}_{i,j}$: Normalized Accuracy of Model $i$ in Cluster $j$.
        - $\tilde{Q}_{i,j}$: Normalized Cost of Model $i$ in Cluster $j$.
        - $\alpha$: A user-defined **Trade-off Parameter** ($0 \le \alpha \le 1$).
    - **Effect of $\alpha$:**
        - High $\alpha$ ($\approx 1.0$): "Max Quality Mode" (like original Avengers).
        - Low $\alpha$ ($\approx 0.0$): "Max Savings Mode" (Frugal routing).
- **Robust Routing:**
    - Instead of assigning a query to just the *single* nearest cluster, it aggregates scores from the **Top-$p$ nearest clusters**. This smooths out noise in the embedding space and prevents "edge cases" from being misrouted.

---

## 4. Performance & Evaluation
- **Metrics:**
    - **Accuracy:** Tested on 6 benchmarks (MMLU, MATH, etc.).
    - **Cost:** Measured in USD (token pricing).
    - **Pareto Efficiency:** The curve of Max Accuracy possible for a given Cost budget.
- **Models:** Evaluated using 8 heterogeneous models, including **GPT-5-medium**, **Gemini-2.5-pro**, and **Claude-opus-4.1**.
- **Key Results:**
    - **Beat the Giant:** Surpassed GPT-5-medium by **+7%** accuracy while maintaining comparable cost ($\alpha \approx 0.5$).
    - **Massive Savings:** Matched GPT-5-medium's accuracy with **27% lower cost**.
    - **Budget Option:** Achieved ~90% of GPT-5's performance (comparable to Gemini-2.5-pro) at **63% lower cost**.
    - **Pareto Dominance:** The Avengers-Pro curve consistently lies *above* the data points of all individual single models, proving it is the most efficient way to consume intelligence.

---

## 5. Limitations & Unimplemented Features
- **Offline Profiling Overhead:** Requires a labeled validation set to pre-calculate the accuracy profiles ($\tilde{P}$) for every model in every cluster. If the query distribution drifts (OOD), these profiles may become stale.
- **Latency:** The routing step (embedding + search) is fast, but using API-based models (Claude, Gemini) introduces network latency that the router cannot control.
- **Complexity:** Managing API keys and async calls for 8+ different providers is significantly more complex than calling a single "GPT-5" endpoint.

---

## 6. Broader Impacts & Future Directions
- **Model Arbitrage:** Creates a mechanism for "Compute Arbitrage," where the router automatically exploits price drops from any provider (e.g., if DeepSeek lowers prices, the router automatically shifts traffic there).
- **Decoupling:** Further decouples the *application layer* from the *model layer*. Developers write prompts, and the router handles the vendor selection.

---

## 7. Takeaway Summary
- **Core Idea:** Use a tunable parameter $\alpha$ to explicitly weight the trade-off between Accuracy and Cost within a semantic clustering router.
- **Key Contributions:**
    - Introduced the **Performance-Efficiency Score** formula.
    - Demonstrated a **Pareto Frontier** that outperforms GPT-5-medium.
    - Achieved 7% better accuracy or 27% lower cost than the state-of-the-art single model.
- **Benefits:**
    - Flexible control (tune $\alpha$ at runtime).
    - No neural network training required.
- **Limitations:**
    - Relies on accurate offline profiling of model capabilities.

---

## References
- **Paper:** [arXiv:2508.12631](https://arxiv.org/abs/2508.12631)
- **Code:** [GitHub - ZhangYiqun018/AvengersPro](https://github.com/ZhangYiqun018/AvengersPro)