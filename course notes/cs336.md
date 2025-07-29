

Lecture 1

BPE tokenization (vs word/byte/character tokenization):

1) large compression ratio
2) training the tokenizer by auto-merging the token-pairs 


Lecture 2

resource efficiency/accounting

1. Tensor
			1.float points 
				1. fp32 - (full precision/default)
				2. fp16 - smaller dynamic range (less exponents -5 bits) and potential under/overflow. And can cause instability due to gradient vanishing and exploding during training
				3. bf16- exponents -8 bits, so larger dynamic range but less fraction and precision
				4. fp8 H100 support only
				5. take away:
				 training with less floats means less memory but more instability 
				 Solution: mixed precision training
			2. tensor definition
				1.  pointers and meta data (stride to get number of skip to next floats)
				2. tensor is references/view so can mutable across different operation, pointing to the same storage
				3. contiguous vs uncontiguous
					1. https://stackoverflow.com/questions/48915810/what-does-contiguous-do-in-pytorch
					2. tensor.contiguous make a copy
				4. element-wise  create the new tensor
				5. einops via jaxtyping
					1. https://docs.kidger.site/jaxtyping/
				6. tensor operation cost
					1. FLOP basic operation over floats
					2. FLOPS vs FLOP/s
					3. A100 312 e12 (TFLOPS/triilion) H100 1979/2 TFLOPS on FP16
					4. Matrix multiplication of (B, D) (D, K), FLOPS is 2 * B * D * K -> for transformers training, 1 forward pass = 2 (# tokens) (# parameters)
				7. model flops utilization (MFU)
					actual FLOPS / promised FLOPS
					>0.5 considered as good
				8. gradient 
					transformers training, 1 backward pass = 2* 2 (# tokens) (# parameters)
					--> so the total FLOPS for one training pass is  6  * (# tokens) (# parameters)
2. Parameter
	1. initialization 
		 invariant to hidden dim -> Xavier initialization  (https://cs230.stanford.edu/section/4/)
	2. training loop
		1. have the randomness seed as parameter for determinstic/experiment reproducible
	3. data loading
		1. use np.memmap for lazily loading data to memory
	4. optimizer
		1. Adan grad
		2. Adam optimiser
		3. data
			1. optimizer state -- the meta data like history data or momentun values
			2. data/grad -- the real values
	5. memory  = (number of activation  + number of parameters + number of gradients + number of optimizor state) * number of bytes per float
	6. checking point
	    periodically saving model into disk with model weight and optimizor states
	7. training is difficult with lower precisions (even with mixed precision training), but trained model could be quanization via lower precisions




Lecture 3 Model architectures and Hyperiparameter

1. Transformer recap (https://web.stanford.edu/~jurafsky/slp3/9.pdf)
	1. Unembedding layer is used to convert the hidden state (1 * d) back to (1 * |V|), as reverse operation of embedding layers, whose output marked as logit lens
	2. language model head (the linear build at top of final transformer blocks) is supposed to read the transformer output of last word during inferences. So, it explain why no Query cache as it will only be used once to predict the next words
2. LLM (https://web.stanford.edu/~jurafsky/slp3/10.pdf)
	1. top-k sampling instead of greedy decoding for generationg
	2. temperature sampling as simply dividing the logit by a temperature parameter τ before passing it through the softmax. So, it kind of auto-adjustment and reduce the gap between the large/small probability (less than means enlarging the gap and then increase the deterministic)
	3. Evaluation on the language model with 1) Perplexity (but can be affected by tokenizer so can only be used between LLMs sharing the tokenizers) 2) downstream task metrics
	4. scaling raw and parameter of LLM can be inferred as 12 *  num_layer * d_model ^2
	5. PEFT with LoRA 
	6. Risk of LLM like hallucination and toxic information and misinformation
3.  Musked LLM (https://web.stanford.edu/~jurafsky/slp3/11.pdf) 
	1. MLM with Bidirectional Encoder for word contextual representation (ELMo -> Bert) 
	2. For training, musk word and next sequence prediction (dropped by Roberta) with CLS/SEP token
	3. To train with Multilingual models, the auto adjustment based on language frequency
	4. Word Contextual Embedding can be used to overcome the word sense disambiguation compared with static word embedding like word2vec
	5. However the word contextual embedding have anisotropy issues and could be mitigated by z-scoring standardization
	6. downstream task examples
		1. sequence /sequence-pair classification - add a extra classifier head only applied for CLS token output
		2. Named entity recognition: BIO tagging and entity classification over each token outputs
	
4.  Model change from vanilla transformer paper 
	1. use pre-norm instead post norm before Attention
	2. use RMSNorm instead of layerNorm https://docs.pytorch.org/docs/stable/generated/torch.nn.modules.normalization.RMSNorm.html
	3. Use gated relu version, like SwitGLU https://arxiv.org/abs/2002.05202v1 
	4. Use Rope for position embedding https://arxiv.org/abs/2104.09864
	5. Some work try serial vs Parallel layers (run MLP and attention in parallel)
	6. No biased FFN
5. Hyper parameters
	1. feedforward/model ratio
		1. d_ff = 4*d_model
		2. exceptions if GLU variants used for nolinearlity, as it will scale down 2/3 scales, so d_ff = (8/3) *d_model ~= 2.6* d_model
	2. head-dim*num-heads to model-dim ratio
		1. num_head * head_dim = d_model ~ most models do follow this guideline
		2. but it doesn’t have to be true as map to higher dimensions for self-attention and then projected back, as T5 and PaLM
	3. Aspect ratios
		1. defined as d_model/n_layers. To define the model is deep/ wide 
		2. wide model can be parallized by tensor parallsim, and deep ones can be speed up by pipeline parallisims.  However, Extremely deep models are harder to parallelize and have higher latency
		3. For model presepective, the validation loss would not be largely affected by the ratio (if total parameters is fixed) but could affect downstream task performances
		4. vocabulary sizes keep increasing from 30k~50k to 100k~250k level for multilingual LLMs
		5. overfitting is not a issue for LLM any more, so regularization like drop out get applied little. But weight decay still widely applied with learning scheduler primarily on stability of training to get lower training loss. (https://arxiv.org/pdf/2310.04415) 
6. Other tricks
	1. z-loss used for mitigating the instability introduced by softmax
	2. QK norm used for mitigating the instability introduced by attention
	3. Logit soft-capping is also used to improve the stability but with model performance cost (compared the perplexity)
	4. GQA/MQA: reduce the memory movement for KV cache by reducing the k/v matrics size (only keep single or fewer KV vector shared by all query embedding for each tokens) and increase the arithmetic intensity. https://fireworks.ai/blog/multi-query-attention-is-all-you-need / https://arxiv.org/abs/1911.02150
		1. with formal MHA without KV cache,  Total arithmetric operations (𝑏𝑛𝑑^2 ), total memory accesses (𝑏𝑛𝑑 + 𝑏ℎ𝑛^2 + 𝑑^2) ( X, softmax, and projection) (b-batch size, n - number of tokens, d - model dimensions)
		2. with formal MHA with KV cache,  Total arithmetric operations (𝑏𝑛𝑑^2 ), total memory accesses (𝑏𝑑𝑛^2 + 𝑛𝑑^2 )
		3. with MQA, Total memory access (𝑏𝑛𝑑 + 𝑏(𝑛^2)𝑘 + 𝑛𝑑^2 )



Lecture 4 MoE

1) MoE defination
	1) Expert over FFN (preferred)
	2) Expert over attention
2) Mode Benefits: can achieve the lower validation loss with same amount of FLOPS for inferences
3) MoE design (https://arxiv.org/pdf/2401.06066 deepseekMoE paper)
	1) Routing function (select top-k and then aggregation over sum)
		1) token preferences  (preferred)
			1) shared experts which always been assigned beside the routered experts (OLMOE studies no gain from shared expert in their setup)
			2) fine-grained experts
				1) re-divided experts to m smaller experts within one segmentation with finer grain.
				2) prove very useful
		2) expert preferences -- better balance for experts
		3) Global optimisations
		4) example: https://research.google/blog/mixture-of-experts-with-expert-choice-routing/
		5) baseline: hashing vs top-k
		6) others: RL-learned router, linear assignment (via matching modeling)
4) MoE training
	1) major issue: the gating function is not differentiable, solutions:
		1) RL training works but hard to train
		2) stochastic approximation (stochastic routing tricks) but still abandoned
		3) heuristic balancing loss
			1) switch transformer
			2) deepseek (v1-v2) loss balance two sub-losses
				1) per-expert balance (similar as switch transformer)
				2) per-device balance (improve the expert parallisim) 
				3) communication balancing loss (deepseek v2) + top-M device routing due to number of experts/parameters exploded
			3) auxiliary-loss free balancing (deepseek v3) https://arxiv.org/abs/2408.15664 / https://arxiv.org/pdf/2412.19437v1
	2) upsycling training, initialize MoE training with dense model (miniCPM model/Qwen) 
5) MoE introduced extra randomness/stochasticity besides sampling
	1) As GPT4 can still return different results with temperature as 0
	2) token dropping due to token imbalacing so causing different results
6) MoE issues
	1) mititgate stability issue during training, on softmax with z-loss for router
	2) MoE can easily overfittting with small data
7) new for DeepSeekV3
	1) use sigmod instead of softmax normalizer for gate top-k selections
	2) use aux-loss-free and seq-wise aux (avoid expert get overwheelmed during inferences)
	3) no more communication loss
	4) Besides MoE, Deepseek v3, use MLA https://github.com/deepseek-ai/FlashMLA  use  kv cache with laten matrix compression/decompression
		1) no compatiable with RoPe due to the positioning encoding could be loss after latent compression
	5) MTP to predic 2 tokens instead of single tokens
	
