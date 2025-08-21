
[Apollo](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-boutin_0.pdf) 
shared-state scheduler with lazy correction to handle the scheduling conflict / rely on the estimated task completion time reported by replica for scheduling. divided tasks into regular/opportunistic tasks to use the idle resources. 

Can be a good references to Dodoor as 1) proved the necessity of distributed scheduler for production 2) both applied the task completion time for scheduling 3) same execution order at server sides.

-----------------------------------------------------

[Quincy](https://dl.acm.org/doi/10.1145/1629575.1629601) 
centralized MCMF for task scheduling to minimize the cost of data transferring (or maximize the locality) with fairness guarantee. The scheduling overhead can be around 1 seconds when scheduling 100 jobs over 2500 computers. 

-----------------------------------------------------
[Firemant](https://www.usenix.org/conference/osdi16/technical-sessions/presentation/gog)
still MCMF but reduced the scheduling latency from minutes to hundred of milliseconds for large scale data center compared to Quincy.  
  
By rewriting the MCMF solver with heuristic tricks, implemented the incremental cost scaling and relaxation, runned both in parallel and pick the first results. (As relaxation is usually work better in common cases but can perform degraded dramatically e.g. heavy load in cluster )
  
Evaluate on Google cloud track with simulation for the placement efficient and develop a network-awared scheduler for on-cluster testing.

---
Lava
