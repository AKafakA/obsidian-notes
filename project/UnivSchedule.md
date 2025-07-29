
***
Implementation

Scheduler Algorithm
	 Interface 
		 Latency optimal
		 Pricing optimal
		 Relevance optimal
		 Cost optimal
		 baseline:
			 random
			 round-robin
			 esscot + round-robin
			 llm routers + round-robin

System Scheduler and instance API
	Scheduler Architecture
		 Server API - [x]
		 Train Phase - [x]
		 Serving Phase - [x]
	Instance Implementation
		Instance
			Query
			Predictor
		 Ollama
			 API -[X]
			 Local client - [x]
				 convert no-stream input to stream input to support the ttft metrics - Ollama-python client
		 vLLM
		 	API -[X]
			Local client - [x]
	Auo-scaling


Predictor Model:
	LSTM latency predictor:
		Training  - [x]  
		 Inference - [x]
		End-to-end test
	Memory ratio predictor (optional) :  Allow to predict the average memory utilisation during inference
		 Training
		 Inference
		 End-to-end test


Develop 
		benchmark scripts
			1. Adapt the new changes from Block benchmarks with new length restriction 
			2. Scripts for collect training data
				1. scripts - [x]
				2. experiments for unpredictability
					1. for same request
						1. same instance, check the variance
						2. different instance, check the variance and latency
					2. for request with same length at same instance, latency and output variance
			3. scripts to generate the uniformed data for all the source trace - [x]
			4. scripts for serving with new uniformed training data and include the new API
				1. auto count the request timed-out / out-of-budget before decoding.
		end-to-end test
			 prediction model
				 latency model
				 memory model
				 length prediction model
			 Integration test

Experiments
		1. Train the length prediction model
			1. Llama family (Lmsys 1m + shareGPT)
				1. collect data
				2. training model
			2. Qwen family (Lmsys)
				1. collect data
				2. training model
		 2. Training latency prediction model
			1.  Llama family
			2. Qwen family
		 3. Dataset preparation and released 
		 4. End-to-end test 
			 1. latency
			 2. relevance (llm judger) (https://huggingface.co/flowaicom/Flow-Judge-v0.1)
			 3. cost
			
****
Writing

Introduction

Motivation
	Uncertainty
		Experiment
			1. random select the request with same prompt length and check the mean/variance of generated response length / latency / memory ratio during 1-pass workload replay
			2. Rerun the workload n times with shuffled order and record the mean/variance latency / memory ratio for selected requests of each rounds
	Heterogeneity: 
			1 hardware
				GPU architectures can cause performance degradation / restriction of software and models. e.g. v100 do not support bf16 and p100 do not have cudnn and tensor-core
			2 software
			3 SLA 
				as large timeout/stricted SLA for bigger models
			4 model
				size / quantisation /  price (input and output)
					Interesting observation is : 
						1) current model API provider does not provide all combination of the model, but based on the model popularity and hardware setting (for example, qwen-2.5 7b with bf16 but 35 and 72b with bf8 only)
						2) Also, the price can be separated into input and output, which grant more flexibility of the model selector
							1) as Llama-2-7b price is 0.05/0.25 but qwen-2.5 is 0.1/0.5, so for request expecting with longer answer, using small size model can significantly reducing more budget but also expect to see more relevance dropped
						
					

			


System Design

Implementation

Evaluation

Related Work