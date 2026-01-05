Reading time: ~45 min --- TLDR: ZeRO is a memory optimization technology that enables training models with trillions of parameters by eliminating data redundancy in distributed training. By partitioning optimizer states, gradients, and parameters across data-parallel GPUs instead of replicating them, it scales memory usage linearly with the number of devices while maintaining the communication efficiency of Data Parallelism. ---

## 1. Motivation & Background
- **Context for the problem:**
    - Training massive models (100B+ parameters) requires massive memory.
    - **Data Parallelism (DP)** is efficient but wastes memory because every GPU holds a *full copy* of the model weights, gradients, and optimizer states.
    - **Model Parallelism (MP)** (like Megatron-LM) splits the model but is complex to implement, has high communication overhead, and requires rewriting model code.
- **Status quo:**
    - To train a 1.5B parameter model (GPT-2 XL), you need 32GB memory just for the weights and states. For 100B, you need Terabytes.
    - DP hits a "Memory Wall": adding more GPUs doesn't help because each GPU still needs to fit the whole model.
- **Gaps or inefficiencies:**
    - **Redundancy:** In DP with $N$ GPUs, the model is replicated $N$ times. This is $N-1$ times redundant. And for the mix-precision training, the optimizer states (like Adam's momentum and variance) take up 2x the memory of the weights themselves due to storing FP32 copies for accumlative operations.
    - **Memory Pie Chart:** For large models (using Adam optimizer), the **Optimizer States** (momentum, variance) consume **75%** of the memory, while Gradients and Weights consume the rest. Standard DP ignores this massive redundancy.

---

## 2. Key Insight
- **The core realization:**
    - We don't need to replicate the full model states on every GPU. We can **partition** them (sharding) such that each GPU owns only $\frac{1}{N}$ of the data.
    - Crucially, we can do this *without* losing the communication efficiency of DP. By using collective communication primitives (Scatter-Reduce / All-Gather) at the right time, every GPU can get the data it needs just-in-time for computation and discard it immediately after.
- **Why previous approaches fall short:**
    - They viewed "Model State" as a monolithic block that had to exist entirely on a device to compute a forward/backward pass. ZeRO treats state as transient: "Fetch, Compute, Discard."

---

## 3. The ZeRO Approach (The 3 Stages)

### 3.1 Architecture / Method Overview
ZeRO is implemented in three incremental stages, each removing a specific type of memory redundancy.

### 3.2 Core Techniques

#### A. ZeRO-DP: Data Parallelism Optimizations (Model States)
*Focuses on reducing the memory footprint of Parameters, Gradients, and Optimizer States.*

- **Stage 1: Optimizer State Partitioning ($P_{os}$)**
    - **Idea:** Instead of every GPU holding the full Adam optimizer state (momentum + variance), split it. GPU 1 updates weights $W_{1}$, GPU 2 updates $W_{2}$, etc.
    - **Impact:** Reduces memory usage by **4x**. Communication volume remains exactly the same as standard Data Parallelism (DP).

- **Stage 2: Gradient Partitioning ($P_{os+g}$)**
    - **Idea:** After the backward pass, each GPU normally holds a full gradient vector to do All-Reduce. ZeRO instead does a **Reduce-Scatter**: each GPU only keeps the averaged gradients for the portion of weights it is responsible for updating (from Stage 1).
    - **Impact:** Reduces memory usage by **8x** combined with Stage 1. No extra communication cost.

- **Stage 3: Parameter Partitioning ($P_{os+g+p}$)**
    - **Idea:** Partition the **Model Parameters** themselves across GPUs.
    - **Workflow:**
        1.  **Forward Pass:** Before Layer 1 runs, all GPUs **All-Gather** the weights for Layer 1. They compute, then immediately **discard** the weights. Repeat for Layer 2.
        2.  **Backward Pass:** Fetch weights again to compute gradients, then discard.
    - **Impact:** Memory usage scales **linearly** with $N$ (number of GPUs). With enough GPUs, you can train *any* size model.
    - **Trade-off:** Increases total communication volume by ~50% compared to DP, but enables Trillion-parameter scale.

- **ZeRO-Offload (Extension):**
    - **Idea:** Moves Optimizer States and Gradients to **CPU RAM** (which is cheap and massive, e.g., 1TB) to free up GPU HBM for even larger batch sizes or models.

#### B. ZeRO-R: Residual Memory Optimizations
*Focuses on "Residual" memory consumed by Activations, temporary buffers, and fragmentation, which becomes the bottleneck once Model States are optimized.*

- **1. Partitioned Activation Checkpointing ($P_a$):**
    - **Problem:** Standard activation checkpointing replicates the stored input tensors on every GPU. For large batch sizes, this replication wastes huge amounts of memory.
    - **Solution:** ZeRO-R partitions the stored checkpoints across GPUs. When the backward pass needs them, it performs an All-Gather to reconstruct the checkpoint, computes the gradient, and discards it.
    - **Impact:** Reduces activation memory by a factor of $N$ (scaling linearly with DP degree).

- **2. Constant Size Buffers ($C_B$):**
    - **Problem:** Collective operations (like All-Reduce) usually require temporary buffers that scale with model size (e.g., flattening all gradients into one huge bucket).
    - **Solution:** ZeRO-R enforces a fixed-size temporary buffer. Messages are sent in chunks that fit this buffer.
    - **Impact:** Prevents temporary memory spikes from causing Out-Of-Memory (OOM) errors on large models.

- **3. Memory Defragmentation ($M_D$):**
    - **Problem:** Training allocates tensors of varying lifespans (short-lived activations vs. long-lived weights), causing memory fragmentation (holes in memory that are too small to use).
    - **Solution:** ZeRO-R pre-allocates contiguous memory chunks for long-lived tensors and periodically defragments the memory space during training steps.
    - **Impact:** Ensures that "available memory" is actually usable, preventing OOM even when the theoretical free memory is sufficient.

---

## 4. Performance & Evaluation
- **Metrics:**
    - **Max Model Size:** Largest trainable model parameter count.
    - **Throughput:** TFLOPS per GPU.
    - **Scalability:** Throughput vs. Number of GPUs (Weak Scaling).
- **Scale:**
    - Demonstrated training a **1 Trillion Parameter** model on 1024 GPUs.
    - Trained **Turing-NLG (17B)**, which was SOTA at the time.
- **Key Results:**
    - **Super-Linear Speedup:** Sometimes ZeRO is *faster* than standard DP because saving memory allows for larger Batch Sizes, which utilizes GPU cores better.
    - **10x Scale:** Enabled training 100B+ models on clusters that previously could only handle 10B.
    - **Usability:** Unlike Megatron-LM (Model Parallelism), ZeRO requires almost **no code changes** (just a config flag in DeepSpeed).

---

## 5. Limitations & Unimplemented Features
- **Communication Intensive (Stage 3):** ZeRO-3 requires fetching weights constantly. On clusters with slow interconnects (e.g., standard Ethernet instead of InfiniBand/NVLink), this slows down training significantly.
- **Complexity:** Debugging partitioned states can be harder than standard replicated setups.
- **Quantization:** The original paper focused on FP16; later works (ZeRO++) added quantization to reduce the communication bandwidth pressure.

---

## 6. Broader Impacts & Future Directions
- **DeepSpeed:** ZeRO became the foundation of Microsoft's **DeepSpeed** library.
- **Democratization:** ZeRO-Offload allows researchers to finetune massive models (like Llama-70B) on a *single* GPU by offloading states to CPU RAM, making LLMs accessible to non-enterprise users.
- **FSDP:** PyTorch later adopted these ideas natively as **FSDP (Fully Sharded Data Parallel)**, which is essentially ZeRO-3.

---

## 7. Takeaway Summary
- **Core Idea:** Partition model states across GPUs to eliminate redundancy. Treat memory as a distributed shared pool.
- **Key Contributions:**
    - **3-Stage Partitioning:** Optimizer (Stage 1), Gradient (Stage 2), Parameter (Stage 3).
    - **ZeRO-Offload:** CPU offloading for single-GPU large model training.
- **Benefits:**
    - Linear memory scaling.
    - No model code rewrite required.
    - Enables Trillion-parameter training.
- **Limitations:**
    - Stage 3 requires high-bandwidth interconnects.

---

## References
- **Paper:** [arXiv:1910.02054](https://arxiv.org/abs/1910.02054)
- **Code:** [GitHub - microsoft/DeepSpeed](https://github.com/microsoft/DeepSpeed)