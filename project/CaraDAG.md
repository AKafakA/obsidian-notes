 1Proposal: CaraDAG — Workflow-Aware Scheduling for Agentic LLM Serving (Post-Cara Extension)

## TL;DR
Extend the **Cara-style “router + scheduler over heterogeneous instances”** from single-turn requests to **agent workflows** (dynamic DAGs of LLM/tool steps). The scheduler becomes *workflow-aware*: it schedules ready DAG nodes to model instances using **critical-path / JCT objectives** plus **KV/session locality** and (optionally) **mixed prefill/decode disaggregation**.

---

## Motivation
Single-turn LLM serving assumes each request is independent. Agent workloads break this:
- A “user request” expands into **multiple dependent LLM calls** + tool calls + retries/branches.
- **KV/session locality** spans multiple steps and can dominate cost/latency if ignored.
- The performance target shifts from per-request latency to **workflow completion time (JCT)** and tail behavior across workflows.

This suggests a natural generalization of Cara: keep the same *pluggable scheduling layer*, but change the **unit of scheduling** from “prompt” to “ready workflow node.”

---

## Key idea
Represent each incoming job as a **partially revealed workflow DAG**. At runtime, the system maintains:
- a **ready set** of LLM nodes (deps satisfied),
- per-instance **capability + queue state + cache state**,
- and a **workflow-aware scoring function** that chooses (node → model → mode → instance) jointly.

---

## Problem statement
Given:
- A set of workflows, each a dynamic DAG `G_w = (V_w, E_w)`.
- Each LLM node `v ∈ V_w` has metadata: `(agent_id, prompt_ref, est_prefill_tokens, est_decode_tokens, SLO/priority, node_type, kv_lineage_id, required_model(optional))`.
- A cluster of instances `i ∈ I`, each described by:
  - `hardware` (GPU type, memory, bandwidth),
  - `model` (family/size/quantization),
  - `mode` ∈ {integrated, prefill-only, decode-only} (optional),
  - `cache_state` (KV residency / pinned sets),
  - `queue_state` (current backlog, predicted completion times).

Goal:
- Minimize **workflow-level objectives** (e.g., average/p95 JCT, deadline miss rate),
- under constraints: GPU compute, memory, KV capacity, bandwidth, and concurrency limits.

---

## System sketch (minimal changes to Cara architecture)
### Components
1. **Workflow Frontend / DAG Expander**
   - Accepts `(workflow_spec, agent_id, initial_prompt, tool registry)`.
   - Emits DAG nodes incrementally as agent logic executes (tool results, branches).

2. **Ready-Queue Manager**
   - Tracks dependencies, pushes ready LLM nodes to the scheduler.
   - Maintains workflow metadata: estimated critical path, slack, fanout.

3. **Predictors (Sidecars)**
   - Predict per-(node, instance) latency/cost:
     - prefill time, decode time, queue delay,
     - memory footprint / KV growth,
     - (optional) KV transfer cost if disaggregated.

4. **CaraDAG Scheduler (core)**
   - Jointly decides:
     - **Model routing** (which model for this node),
     - **Mode** (integrated vs P/D split; optional),
     - **Placement** (which instance(s)),
   - while optimizing workflow-level objective.

5. **Execution Runtime**
   - Dispatches node execution, collects outputs.
   - Updates cache/session metadata and triggers DAG expansion.

---

## Scheduling objective (starting point)
Use a weighted score per candidate assignment `(v → i)`:

`Score(v,i) = + w_cp * Criticality(v)
              - w_q  * PredictedQueueDelay(i)
              - w_lat* PredictedExecLatency(v,i)
              + w_kv * KVReuseBenefit(v,i)
              - w_mem* MemoryPressurePenalty(i)
              - w_xfer* (optional) KVTransferCost(v,i)`

Where:
- `Criticality(v)` approximates whether `v` lies on the workflow’s predicted critical path
  (or a proxy: remaining depth, downstream fanout, slack to deadline).
- `KVReuseBenefit(v,i)` is high if `kv_lineage_id` is present/pinned on `i`.

**MVP**: integrated-only mode, no cross-instance migration, no disaggregation.
**Extensions**: mixed integrated + P/D, proactive KV prefetch, bounded state movement.

---

## Algorithm (MVP)
1. Every scheduling tick:
   - Gather ready nodes `R` across workflows.
   - For each `v ∈ R`, enumerate feasible instances `Feasible(v)`:
     - compatible model(s),
     - memory headroom constraints.
2. Run a **batch greedy / beam** assignment:
   - prioritize high-criticality nodes first (or use a bipartite matching / min-cost flow if needed),
   - select placements maximizing total score, respecting per-instance concurrency limits.
3. Dispatch, update states, repeat.

This mirrors Cara’s “predict → score → solve small batch” loop, but with DAG-aware prioritization.

---

## Optional extension: Mixed Prefill/Decode Disaggregation
Allow a node to be executed as:
- integrated on one instance, OR
- prefill on a prefill pool + decode on decode pool with KV handoff.

Represent a “virtual assignment” as `(v → (i_p, i_d))` with:
- `PredLatency = prefill(v,i_p) + xfer(KV,i_p→i_d) + decode(v,i_d)`
- plus interference terms from queue states.

This becomes a natural additional dimension in the scheduler without changing the DAG semantics.

---

## What is novel (intended contribution)
1. **Workflow-aware scheduling layer** that optimizes **workflow JCT / critical path**, not just per-request latency.
2. **State-locality aware placement** via `kv_lineage_id` and cache-status modeling.
3. **Unified decision** across *routing (model choice) + placement (+ optional P/D mode)* under heterogeneity.
4. A practical, modular design that fits a “Cara-like” predictor + scheduler architecture.

---

## Evaluation plan (after Cara submission)
### Workloads
- Agent coding workflows (multi-step tool use + retries).
- Tool-calling benchmarks (function-calling heavy).
- Multi-agent DAGs (planner → worker → verifier patterns).

### Baselines
- Request-level schedulers/routers (treat each LLM call independently).
- Agent runtime without workflow-aware placement (random / round-robin / least-queue).
- KV-agnostic vs KV-aware placement.

### Metrics
- Workflow JCT (p50/p95/p99), deadline miss rate.
- GPU utilization, queueing delay, throughput.
- KV hit rate / KV transfer volume (if disaggregated).
- Cost-per-workflow under fixed SLO.

---

## Risks & mitigations
- **DAG unpredictability** (branches/retries): start with robust heuristics + online re-estimation of criticality.
- **Too many knobs** (routing + placement + P/D): ship MVP integrated-only first; add P/D as a second stage.
- **Model-quality coupling**: initially constrain to same model family (sizes) and treat routing as cost/latency only.

---

## Milestones (aligned with “finish Cara first”)
1. **Design doc + interface spec** (workflow schema, node metadata, instance tuple).
2. **MVP simulator** using Cara-style predictors to validate scheduling logic on traces.
3. **Prototype runtime integration** (agent framework → DAG expander → scheduler → serving backend).
4. **P/D mixed mode** + KV transfer modeling (optional second paper / appendix).

---

## Deliverables
- A short systems paper: “From Single-Turn Routing to Workflow DAG Scheduling in Agent Serving”
- Open-source scheduler module + trace-driven simulator + evaluation scripts.
