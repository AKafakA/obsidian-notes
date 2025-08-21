

[TurboTransformers](https://arxiv.org/abs/2010.05680)
The serving framework proposed at 2020
1. kernel fusion
2. memory allocation based on memory pooling to reduce the memory footprint 
3. shared the same memory space for multiple tensors at different life cycling. Conflicting with kv caching. 
4. batching with sequence length awared and pre-profiling  
	1. all request will be executed together with padding for variable length
	2. a DP based approaches to decide the batch
	3. Profiling required to detect the cost function from the (request_length, batch size) -> cost, and target is to minimize the global cost. Cost is the latency of execution of the whole request. So, each batch will keep running at request-level (not token level)
	4. All requests will be sorted by the length so conflicting with FCFS also.
5. Request(Prompt) caching

-----------------------------------------------------

[LightSeq] [[https://arxiv.org/abs/2010.13887]
The serving framework proposed at 2021, which targeting to speed up the inferences through 3 approaches
1. Kernel fusing. Implement the fused customised kernel for self-attention blocks, replacing the PyTorch/torch implementation based on the standard kernel library (even after openXLA optimisations)
    
2. Improve the auto-regressive searching by Hierarchical Auto Regressive Search (use grouping and retrieval to filter the candidates first. However, haven’t explain how this can be efficient compared withe the global ranking)
    
3. Dynamic GPU Memory Reuse, required to pre-define all the maxima length of each request. (However, haven’t explain how this length get defined)

------------
[Petals] [https://arxiv.org/abs/2209.01188]

Distributed inference and peft via Model parallelism across servers.

Client keep the embedding layers and fine-tuned adaptive layers and servers hold transformers layers, which keep locked during inference/fine tuning, with KV-cache enabled.
A flow/chain get auto-generated for each session from client to servers covered all blocks for inferences

System techniques used for speeding up the system / lower hardware requirement
1. dynamic block-wise quantization
2. quantization from bf16 to bf8
3. hivemind for node management
4. load balance for the server
	1. distributed hashtable as shared states across nodes keeps
		1. subset of serving layers/blocks
		2. workload/throughput
	2. all nodes periodically do the rebalance checking to switch layers to improve the global throughput
	3. client determine the inference server flows via "ping nearby servers to measure latency and then find the path with minimal time via beam search"
	4. data parallelism is applied during fine-tuning as one batch of data get split into multiple servers sets

Beat the offload approaches almost 10x for single batch inferences

-----------------------------

Orca 

---

vLLM

----

SGLang

----
LightLLM

---
Chunked prefill

-----

Split-wise

----
Mooncake

----
DistServe

----
FlashAttention v1, v2, v3


----
FlashInfer

-----
PODAttention

----
ProMoE

-----
AlpaServe

----
RAGCache

---
CacheBlend

---
ServerlessLLM

