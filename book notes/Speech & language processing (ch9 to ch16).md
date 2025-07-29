

---

## 1. The Transformer (Chapter 9)

### Key Architectural Concepts

* **Encoder-Decoder Structure:** The original Transformer was designed for machine translation and had an **encoder** to process the source text and a **decoder** to generate the target text. Most modern LLMs (like GPT) use only the **decoder** block.
* **Unembedding / Language Model Head:** At the end of the Transformer, the final hidden state (with shape `[batch, seq_len, d_model]`) must be converted into a probability distribution over the vocabulary. This is done by a linear layer (also called unembedding layer) that projects the hidden state from dimension $d_{model}$ to the vocabulary size $|V|$  as logits and then apply a softmax to do the normalization to convert logits to probablity.
* **KV Caching in Auto-regressive Generation:** During inference (generating one token at a time), the decoder uses the final hidden state of the **last token** to predict the next one. To make this efficient, we use a **KV Cache**.
    * For each new token, its Query vector ($Q$) needs to attend to the Key ($K$) and Value ($V$) vectors of **all previous tokens**.
    * Instead of re-computing these past $K$ and $V$ vectors every time, we cache them in memory. This drastically speeds up generation by turning a quadratic operation into a linear one at each step.

---

## 2. Large Language Models (LLMs) (Chapter 10)

### Generation & Decoding

To generate text, we need to convert the model's output logits into a sequence of tokens.

* **Greedy Decoding:** Always pick the token with the highest probability. This is fast but leads to repetitive and boring text.
* **Sampling Strategies:** Introduce randomness for more creative and human-like output.
    * **Temperature Sampling:** Rescales the logits before the softmax function using a temperature parameter $\tau$.
        $$ \text{softmax}(z_i / \tau) = \frac{e^{z_i / \tau}}{\sum_j e^{z_j / \tau}} $$
        * $\tau > 1.0$: Increases randomness (flattens the distribution).
        * $\tau < 1.0$: Decreases randomness (makes the distribution "peakier").
        * $\tau \to 0$: Approaches greedy decoding.
    * **Top-k Sampling:** At each step, restrict the sampling pool to the `k` most probable tokens.

### Evaluation & Scaling

* **Evaluation Metrics:**
    1.  **Perplexity (PPL):** An intrinsic measure of how well a model predicts a test set. Lower is better. **Crucial Caveat:** PPL is highly dependent on the tokenizer. You can only compare PPL scores between models that share the **exact same vocabulary and tokenizer**.
    2.  **Downstream Task Performance:** The most important evaluation. Measure performance on standardized benchmarks like MMLU (general knowledge), HumanEval (code), or GLUE/SuperGLUE (NLU tasks).
* **Parameter Counting (Rule of Thumb):** For a standard decoder-only Transformer, the number of non-embedding parameters can be approximated as:
    $$ \text{Parameters} \approx 12 \times n_{layers} \times d_{model}^2 $$
* **Parameter-Efficient Fine-Tuning (PEFT):**
    * **LoRA (Low-Rank Adaptation):** Instead of fine-tuning all model weights (which is extremely memory-intensive), we freeze the original LLM and inject small, trainable "low-rank" matrices into its layers. We only train these much smaller matrices, dramatically reducing the computational and memory cost of fine-tuning.
* **Risks and Harms:** LLMs can produce incorrect information (**hallucinations**), reflect and amplify societal biases, and generate toxic or harmful content. Responsible development requires significant effort in safety tuning and mitigation.

---

## 3. Masked Language Models (MLMs) (Chapter 11)

### Contextual Embeddings (BERT)

While auto-regressive models like GPT are trained to predict the *next* word, models like BERT are **bidirectional encoders** trained to understand context from both left and right.

* **Static vs. Contextual Embeddings:**
    * **Static (e.g., word2vec):** A word has a single, fixed embedding regardless of context. "Bank" in "river bank" and "investment bank" would have the same vector.
    * **Contextual (e.g., BERT):** The embedding for a word is generated based on the entire sentence it appears in. This allows the model to solve **word sense disambiguation** and capture nuanced meaning.
* **Anisotropy:** A common issue where the contextual embeddings from models like BERT occupy a narrow cone in the vector space, hurting their performance on tasks like semantic similarity. This can be mitigated by post-processing steps like **z-scoring** standardization.

### Pre-training Objectives

BERT-like models are pre-trained on two tasks simultaneously:

1.  **Masked Language Modeling (MLM):** 15% of the tokens in the input are replaced with a special `[MASK]` token, and the model's goal is to predict the original tokens. This forces the model to learn bidirectional context.
2.  **Next Sentence Prediction (NSP):** The model is given two sentences, A and B, and must predict whether B is the actual sentence that follows A. This was intended to help with sentence-pair tasks. (Later models like **RoBERTa** found that dropping NSP improved performance).

### Fine-Tuning for Downstream Tasks

Once pre-trained, the BERT encoder can be fine-tuned for various tasks:
* **Sequence/Sentence-Pair Classification:** Add a classification head on top of the output for the special **`[CLS]` token**. Used for sentiment analysis, natural language inference, etc.
* **Named Entity Recognition (NER):** Add a classification head on top of **every token's output** to predict its entity tag (e.g., `B-PER`, `I-LOC`, `O`).