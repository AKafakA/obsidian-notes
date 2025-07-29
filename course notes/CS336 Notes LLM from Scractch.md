 
---

## Lecture 1: Tokenization

### Tokenization Strategy

Tokenization is the process of breaking down raw text into smaller units called tokens. The choice of tokenization strategy is a fundamental trade-off between vocabulary size, sequence length, and the ability to handle unseen words.

* **Character/Byte Tokenization:**
    * **Pros:** Very small vocabulary size. No "out-of-vocabulary" (OOV) words.
    * **Cons:** Produces very long sequences of tokens, which increases the computational cost for models like Transformers. It also splits meaningful units (like words) into less meaningful pieces.
* **Word Tokenization:**
    * **Pros:** Creates shorter, more semantically meaningful sequences.
    * **Cons:** Requires a massive vocabulary, leading to huge embedding tables. It struggles with OOV words, typos, and morphological variations (e.g., "run", "running", "ran").

### Byte Pair Encoding (BPE)

**BPE** is a subword tokenization algorithm that provides a great balance. It's the most common method used for modern LLMs.

1.  **How it Works:**
    * **Initialization:** Start with a vocabulary consisting of all individual characters (or bytes) present in the training corpus.
    * **Training (Iterative Merging):** Repeatedly count all adjacent token pairs and merge the most frequently occurring pair into a single new token. This new token is then added to the vocabulary.
    * **Example:** `t h e` might be merged into `th e`, and later `th e` might be merged into `the`. The pair `ug ly` might become `ugly`.
2.  **Key Advantages:**
    * **Good Compression:** It finds a middle ground between character-level and word-level tokenization, resulting in shorter sequences than characters but longer than words.
    * **Handles OOV:** Rare or unseen words can be broken down into known subword units (e.g., "cs336" might become `cs`, `33`, `6`), avoiding the OOV problem.

---

## Lecture 2: Resource Efficiency & Accounting

### 1. Numerical Precision & Tensors

#### Floating Point Formats

The choice of numerical precision is a trade-off between memory usage, computational speed, and model stability. A floating-point number is represented by a sign, an exponent, and a fraction (mantissa).

* **`fp32` (Single Precision):** The default. 8 bits for exponent, 23 bits for fraction. High precision and wide dynamic range, but memory and computationally intensive.
* **`fp16` (Half Precision):** 5 bits for exponent, 10 bits for fraction.
    * **Pros:** Uses 50% less memory than `fp32`, faster computation on compatible hardware (e.g., NVIDIA GPUs).
    * **Cons:** **Small dynamic range** due to fewer exponent bits. Prone to **overflow** (number is too large) or **underflow** (number is too small, becomes zero). This can cause gradient explosion/vanishing and training instability.
* **`bf16` (BFloat16):** 8 bits for exponent, 7 bits for fraction.
    * **Pros:** Same dynamic range as `fp32` (prevents over/underflow), making it much more stable for training than `fp16`. Still offers 50% memory savings over `fp32`.
    * **Cons:** Lower precision (fewer fraction bits) than `fp16`, but this is often an acceptable trade-off.
* **`fp8`:** Newest format supported on hardware like NVIDIA H100s, offering even greater speed and memory savings.

⭐ **Takeaway: Mixed Precision Training**
To get the best of both worlds, **mixed precision training** is the standard.
1.  **Forward/Backward Pass:** Operations are performed using `bf16` or `fp16` for speed.
2.  **Master Weights & Gradients:** A master copy of the model weights and the final gradients are stored in `fp32` to maintain numerical stability and precision for the weight update step.

#### Tensors in Memory

* A **tensor** is a multi-dimensional array. In memory, it's represented by a pointer to a storage block (the actual numbers) and metadata (like shape, data type, and **stride**).
* **Stride:** An integer array indicating the number of memory locations to skip to get to the next element along each dimension.
* **Contiguous vs. Non-Contiguous:**
    * A tensor is **contiguous** if its elements are laid out in memory in the same order a C-style array would be.
    * Operations like transposing (`.T`) can create a **non-contiguous** tensor (a "view") that points to the original data but has different strides.
    * Some operations require contiguous memory. Calling `.contiguous()` on a non-contiguous tensor will create a **copy** of the tensor with the correct memory layout.
* Element-wise operations (e.g., `a + b`) typically create a new tensor in memory.

### 2. Computational Cost (FLOPs)

* **FLOP:** A FLoating-point OPeration (e.g., one addition or one multiplication).
* **FLOPs** or **FLOP/s:** FLOPs per second, a measure of computational performance.
* **GPU Performance:** An NVIDIA A100 GPU offers ~312 TFLOP/s (`10^{12}`) of `fp16` performance. An H100 offers ~1979 TFLOP/s with sparse matrix and 879 if not.

#### FLOPs in Transformers

* **Matrix Multiplication:** The dominant cost. Multiplying two matrices of shapes $(N, D)$ and $(D, K)$ requires approximately $2 \times N \times D \times K$ FLOPs.
* **Training FLOPs Estimation (Rule of Thumb):** For a Transformer model, the total FLOPs for one full training pass (forward + backward) can be estimated as:
    $$ \text{Total FLOPs} \approx 6 \times (\# \text{parameters}) \times (\# \text{tokens}) $$
    * **Forward Pass:** $\approx 2 \times (\# \text{params}) \times (\# \text{tokens})$
    * **Backward Pass:** $\approx 2 \times (\text{Forward Pass FLOPs}) = 4 \times (\# \text{params}) \times (\# \text{tokens})$

#### Model FLOPs Utilization (MFU)

* **MFU** is a key efficiency metric:
    $$ \text{MFU} = \frac{\text{Actual Achieved TFLOP/s}}{\text{Theoretical Peak TFLOP/s}} $$
* It measures how effectively your code is using the GPU's potential. An MFU > 0.5 (50%) is generally considered good.

### 3. Memory Consumption

The total memory required to train a model is the sum of memory for several components.

$$ \text{Memory} \approx (\# \text{Params} + \# \text{Activations} + \# \text{Gradients} + \# \text{Optimizer States}) \times \text{Bytes per element} $$

* **Parameters:** The model's weights. Size is constant.
* **Gradients:** Stored for the backward pass. Same size as parameters.
* **Optimizer States:** Data stored by the optimizer. For **Adam/AdamW**, this includes momentum and variance terms, so it's typically **$2 \times$ the number of parameters**.
* **Activations:** Intermediate results from the forward pass (e.g., outputs of each layer) that must be stored for gradient calculation. This is highly dependent on `batch_size`, `sequence_length`, `d_model`, and `num_layers`.

### 4. Training Loop Essentials

* **Initialization:** Use methods like **Xavier/Glorot initialization** to set initial weights, which helps maintain variance across layers and aids stable training.
* **Reproducibility:** Always set a **random seed** to ensure experiments are deterministic and reproducible.
* **Data Loading:** For very large datasets that don't fit in RAM, use `np.memmap` to create a memory-mapped array on disk, allowing you to lazily load data into memory as needed.
* **Checkpointing:** Periodically save the full training state to disk. This includes the **model weights**, **optimizer states**, and other metadata like the current epoch and learning rate. This allows you to resume training from the last saved point in case of failure.
* **Quantization:** While training in lower precision is hard, a fully trained `fp32` model can be **quantized** to `fp16`, `int8`, or even lower precisions for inference, drastically reducing its memory footprint and increasing speed with minimal performance loss.

---

## Lecture 3: Model Architectures & Hyperparameters

### 1. Modern Transformer Architecture Changes

Modern LLMs (e.g., Llama, GPT-3/4) have evolved from the original "vanilla" Transformer architecture.

1.  **Pre-LayerNorm (`Pre-Norm`):** The LayerNorm is applied **before** the self-attention and FFN blocks, not after (`Post-Norm`). This leads to much more stable training, as it prevents the output magnitudes from exploding.
2.  **RMSNorm:** A simplified version of LayerNorm that only normalizes by the root mean square (the "scale") and omits the re-centering step (the "shift"). It's computationally cheaper and works just as well.
    $$ \text{LayerNorm}(x) = \frac{x - \mu}{\sigma} \gamma + \beta \quad \quad \text{RMSNorm}(x) = \frac{x}{\text{RMS}(x)} \gamma $$
3.  **SwiGLU Activation:** Gated Linear Units (GLU) variants like SwiGLU have replaced ReLU as the standard activation function in the FFN layer. https://arxiv.org/abs/2002.05202v1 
4.  **Rotary Position Embeddings (RoPE):** Replaces learned absolute or sinusoidal position embeddings. RoPE encodes absolute position by rotating the query/key vectors based on their position and encodes relative position naturally through the dot product interaction.
5.  **No Biases:** Many modern LLMs remove bias terms from their linear layers to save a small number of parameters and simplify the model.

### 2. Key Hyperparameters & Ratios

* **Feed-Forward to Model Dimension Ratio (`d_ff / d_model`):**
    * Vanilla Transformer: $d_{ff} = 4 \times d_{model}$.
    * With **GLU variants (like SwiGLU)**, the FFN layer is often down-projected by a factor of $2/3$. To compensate, the intermediate dimension is made larger: $d_{ff} = \frac{8}{3} \times d_{model} \approx 2.67 \times d_{model}$.
* **Attention Heads:** Typically, $n_{heads} \times d_{head} = d_{model}$. However, some models deviate from this.
* **Aspect Ratio (`d_model / n_layers`):**
    * **Wide and shallow** models (high aspect ratio) are easier to parallelize with **tensor parallelism**.
    * **Deep and thin** models (low aspect ratio) are better suited for **pipeline parallelism** but can have higher inference latency.
    * Overall validation loss is not heavily affected by this ratio for a fixed parameter count, but it can impact downstream task performance.
* **Vocabulary Size:** Has grown from ~30-50k to **~100-250k** for powerful multilingual models.
* **Regularization:**
    * **Dropout** is used much less frequently. With massive datasets, overfitting is less of a concern.
    * **Weight Decay** is still widely used, primarily to improve training stability rather than to prevent overfitting. https://arxiv.org/pdf/2310.04415

### 3. Generation and Evaluation

* **Sampling Strategies:** Instead of greedy decoding (always picking the most likely next token), sampling methods are used to produce more diverse and interesting text.
    * **Top-k Sampling:** Sample from the `k` most probable next tokens.
    * **Temperature Sampling:** The logits are divided by a temperature parameter $\tau$ before the softmax.
        * $\tau > 1$: Makes the distribution "flatter," increasing randomness.
        * $\tau < 1$: Makes the distribution "peakier," increasing determinism.
* **Evaluation:**
    * **Perplexity (PPL):** A measure of how well a probability model predicts a sample. Lower is better. **Limitation:** PPL is highly sensitive to the tokenizer used, so it's only comparable between models that share the exact same vocabulary and tokenizer.
    * **Downstream Task Metrics:** The ultimate evaluation is performance on specific tasks (e.g., accuracy on MMLU, score on HumanEval).

### 4. Efficiency & Stability Tricks

* **Grouped-Query (GQA) & Multi-Query Attention (MQA):** 
	* https://fireworks.ai/blog/multi-query-attention-is-all-you-need / https://arxiv.org/abs/1911.02150
    * **Problem:** During inference, the Key-Value (KV) cache for all previous tokens consumes a huge amount of memory. In standard multi-head attention (MHA), each head has its own K and V projection.
    * **Solution:** Share K and V projections across multiple query heads.
        * **MQA:** All `Q` heads share a single `K` and `V` head.
        * **GQA:** `Q` heads are grouped, and each group shares a `K` and `V` head.
    * **Benefit:** Drastically reduces the size of the KV cache, allowing for much longer context windows during inference with a small hit to model performance.
* **Training Stability:**
    * **QK Norm:** L2-normalizing the query and key vectors before the softmax can prevent large dot product values that destabilize attention.
    * **Z-loss:** An auxiliary loss term that penalizes large logits before the final softmax, which also helps improve stability.


# Lecture 4: Mixture of Experts (MoE)

---

## 1. What is a Mixture of Experts (MoE)?

A **Mixture of Experts (MoE)** is a model architecture that decouples the total number of parameters from the computational cost (FLOPs) for a single input. Instead of using the entire model for every token, an MoE layer activates only a small subset of its parameters.

An MoE layer consists of two main components:
1.  **A set of "Experts" ($E_1, E_2, ..., E_N$):** These are neural networks that process the input.
2.  **A "Gating Network" or "Router" ($G$):** This is a small network that decides which experts to send each input token to.

### Expert Implementation

* **Experts over FFN (Feed-Forward Network):** This is the most common and effective approach. In a standard Transformer block, the two FFN layers account for roughly 2/3 of the total parameters. Replacing the FFN block with an MoE layer containing many FFN experts allows for a massive increase in model parameters with only a small increase in inference FLOPs.
* **Experts over Attention:** It is also possible to have different attention-head experts, but this is less common as the FFN is the primary source of parameters.

### Mathematical Formulation

For a given input token embedding $x$, the router computes a score for each expert. In a sparse MoE with **top-k routing**, the router selects a set of $k$ experts ($I_k$) to process the token. The final output $y$ is a weighted sum of the outputs from these selected experts:

$$ y = \sum_{i \in I_k} G(x)_i \cdot E_i(x) $$

Here, $G(x)_i$ is the weight (gate value) assigned by the router to expert $E_i$.

## 2. Benefits of MoE

The primary benefit of MoE is achieving **better performance for a fixed computational budget**.

* **Dense Model:** A 7B parameter dense model uses ~7B parameters' worth of computation (FLOPs) for every token.
* **MoE Model:** An MoE model with 42B total parameters but using only 2 experts per token (e.g., Mixtral 8x7B) might only use ~12B parameters' worth of computation per token.

This means you can train a model with a vastly larger "knowledge capacity" (total parameters) while keeping the inference cost manageable, leading to lower validation loss compared to a dense model with the same FLOPs.

## 3. MoE Design: Routing Strategies

Routing determines how tokens are assigned to experts. The goal is to send a token to experts that are best suited for it while keeping all experts utilized.

### Token-to-Expert vs. Expert-to-Token

1.  **Token-to-Expert Routing (Standard Top-k):** Each token independently selects the top-k experts it has the highest affinity for.
    * **Pros:** Conceptually simple.
    * **Cons:** Can lead to **load imbalance**, where some "popular" experts are overwhelmed with tokens while others sit idle. This is computationally inefficient.
2.  **Expert Choice Routing:** Each expert "chooses" the top-k tokens it is most specialized for from the batch.
    * **Pros:** Guarantees perfect load balance across experts.
    * **Cons:** More complex to implement.
    * **Paper:** [Mixture-of-Experts with Expert Choice Routing](https://research.google/blog/mixture-of-experts-with-expert-choice-routing/)

### Advanced Routing Designs (from DeepSeekMoE)

* **Shared and Fine-Grained Experts:** To improve performance, some models combine routed experts with other types.
    * **Shared Experts:** One or more experts that are *always* selected for every token, running in parallel with the routed experts. This ensures a baseline of knowledge is always applied.
    * **Fine-Grained Experts:** Breaking down large experts into smaller, more specialized sub-experts.
    * **Paper:** [DeepSeekMoE: Towards Ultimate Expert Specialization](https://arxiv.org/pdf/2401.06066)

## 4. Training MoE Models

Training MoE models presents unique challenges, primarily related to routing.

### The Differentiability Problem

The act of selecting the `top-k` experts is a **discrete choice**, which is not differentiable. You cannot backpropagate a gradient through a hard selection like `argmax`. This means the router cannot be trained directly from the main model's loss function.

### Solution: Heuristic Load Balancing Loss

The standard solution is to add an **auxiliary loss function** that encourages the router to spread tokens evenly across all experts, ensuring efficient hardware utilization.

* **Switch Transformer Loss:** A common auxiliary loss that penalizes load imbalance. It encourages:
    1.  The fraction of tokens dispatched to each expert to be roughly equal.
    2.  The router's confidence (softmax output for the chosen experts) to be high.
* **DeepSeek-V2 Loss Evolution:**
    1.  Introduced a **per-device balance loss** to improve expert parallelism across GPUs.
    2.  Added a **communication balancing loss** to minimize data transfer between devices.
* **Auxiliary-Loss-Free Balancing (DeepSeek-V3):** The latest research aims to achieve load balancing through architectural changes and clever routing algorithms, removing the need for an explicit auxiliary loss, which simplifies training.
    * **Papers:** [DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434),  [Auxiliary-Loss-Free Load Balancing](https://arxiv.org/abs/2408.15664)

### Upsycling Training

A common and effective technique is to **upsycle** a pre-trained dense model into an MoE model. The weights of the dense model are used to initialize the experts in the new MoE model, providing a much stronger starting point for training, used by miniCPM/Qwen model

## 5. MoE Characteristics and Issues

* **Inherent Stochasticity:** MoE models can produce different results even with `temperature=0`. This is because tiny floating-point variations in the router's calculations can cause a token to be sent to a different set of experts on different runs, leading to a different output. This can also be caused by **token dropping** if an expert's processing capacity is exceeded.
* **Training Instability:** MoE models are notoriously difficult to stabilize during training. Techniques like applying **z-loss** to the router's softmax output are used to control the magnitude of the logits and improve stability.
* **Overfitting:** Due to their massive parameter count, MoE models are highly prone to overfitting, especially on smaller datasets. They require web-scale data to train effectively.

## 6. DeepSeek-V3 Innovations

DeepSeek has pushed the boundaries of MoE architectures with several new techniques:

1.  **Sigmoid Gating:** Using `sigmoid` instead of `softmax` for the gate. With softmax, the gate values for all experts must sum to 1. With sigmoid, each expert's gate value is independent (between 0 and 1), allowing for more flexible routing logic.
2.  **MLA (Multi-head Latent Attention):** A technique to address the KV cache bottleneck. Instead of storing the full Key-Value cache, it is compressed into a smaller **latent matrix**. This dramatically reduces memory usage, but can be incompatible with position encodings like **RoPE**, as positional information might be lost during compression.
    * **Code:** [FlashMLA](https://github.com/deepseek-ai/FlashMLA) [vllm-trition-MLA](https://github.com/monellz/vllm/commit/feebaa7c063be6bfb590a876741aeef1c5f58cf8#diff-2baa97f0f579db22d121ddbbf44cf4c556c07f4f8823292bac9e0e83c391a1e3) 
3.  **MTP (Multi-Token Prediction):** Training the model to predict 2 (or more) tokens simultaneously instead of just one. This aims to increase the throughput and speed of inference.