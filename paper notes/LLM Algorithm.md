
[GPT-1](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf)
Use decoder-only transformers as the pretrained language models with maxmize the likelihood of tokens to capture the sentence-level pattern with semi-supervised learning

And fine tune the pretrained model with linear layers for supervised language tasks.
A big improvement with previous work is using transformers instead of LSTM with performance improvement (6 points on average scores)

-----------------------------------------------------
[GPT-2](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) and [GPT-3](https://arxiv.org/abs/2005.14165)
Extend the study of GPT from supervised fine tuning to zero/few shot learning
And for few shot learning, which can be regarded as the start of prompts, as the new data points as part of prompts to the model

Model architect keep the same (as GPT-3 applied the sparser transformers)
Model size increased from 1.5B to 170B and beat/approaching the supervised fine-tuning with few/zero shot settings

-----------------------------------------------------
[Bert](https://arxiv.org/abs/1810.04805) 
encoder-only pre-trained model which trained with musk LM and next sentence predictions. 
musk LM: randomly musk 15% tokens and trained the model to predict it

next sentence predictions: concatenate  the sequences with SEP splitter. training the model to predict the output of the SEP to represent if next sentences or not 

Fine tuning still required for the down-stream tasks
100M trainable parameters for bert-base and 300M for bert-large

-----------------------------------------------------

[RoBERTa](https://arxiv.org/abs/1907.11692)
Further study on bert.
Same model as Bert by training with large data, longer training and removing NSP (next sequence prediction) replacing with larger batch size, dynamic masking.

-----------------------------------

[T5](https://arxiv.org/abs/1910.10683)
empirical study with T5 (text to text) tasks with different settings. Study about the transfer learning and allowing the fine-tuning with supervised labels

1. model architects 
	The model is designed as prefix-llm as the prefix keep fully visible but causal masked used for left text.
	encoder-decoder model usually own 2 * L parameters but keep the same computation cost as decoder-only model with L layers

2. Training objects
	Denosing (masked LM) with Bert styles outperformed with prefixed LM and deshuffling

3. Dataset
	pretraining on in-domain data (even unlabeled) improve the downstream 
	
	tasks. Also, corruption ratio for mask LM cause little impact on the final performances. But small data with high deduplication can cause big performance degradation. And corruption with span (3 words) can either benefit the model performance and training speed

4. Fine tuning
	adaptive layers and gradual unfreezing can speed up the training but causing performance degradation compared with full training


-------------

[LoRA](https://openreview.net/forum?id=nZeVKeeFYf9)

LoRA was proposed for fine tuning pre-trained model for downstream tasks.

Compared with full fine-tuning, with high cost of training, and adaption layer/bias, introduce the extra inference latency with hurting the inference parallelism.

LoRA allows the pre-trained model to remain frozen while introducing two low-rank trainable matrices that learn the distribution of downstream tasks as an auxiliary. - The trainable parameter matrix is decomposed as:  $W_0x + \Delta W x = W_0x + BAx$

LoRA shows comparable performance of model but saving 1000x gpu memory during training without extra inference overhead.

--------------------

[Block-wise quantization] [https://arxiv.org/pdf/2110.02861]
Motivated by large memory assigned for optimzer states during training (For 32-bit states, Momentum and Adam consume 4 and 8 bytes per parameter)
8-bit quantization to compress the state of optimiser for storage and de-quantitation for updates.
Each tensor are divided into blocks based on the dimension and normalization and no-linear quantization will be conduct over each blocks individually. dynamic quantization used for all positive cases by fixing the bits for fraction.
A stable embedding layer trained with 32 bits is required.
Shows promising results with no-side effect of model metrics but able to reduce 8.5 gb memory for 1.5B training. 


----

RAG


----

Mamba 


----
HyperMamaba


---------------------------------

[Informer](https://arxiv.org/abs/2012.07436)


----
TimeLLM



