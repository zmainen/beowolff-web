## 10. Privacy-Preserving ML: Differential Privacy, Federated Learning, and Machine Unlearning

The program commits to "right-to-delete that actually propagates." This is a stronger claim than it might appear. GDPR gives users the right to have their data deleted, and most platforms comply by removing rows from a database. But when those rows were used to train a model — when the user's browsing history shaped gradient updates across millions of parameters — deleting the rows does not delete the influence. The model still carries traces of that user's data in its weights. Getting those traces out is the problem of machine unlearning, and it is genuinely hard.

This chapter covers three privacy-preserving ML techniques the program invokes: differential privacy, federated learning, and machine unlearning via SISA training. The honest summary is that only one of these is production-ready at the scale and quality level the program requires, and it is the least elegant: scheduled clean retrains. The others are either too costly (differential privacy), too immature (federated learning at this scale), or practical but engineering-intensive (SISA). Understanding why each technique falls where it does on the tractability spectrum is essential for evaluating the program's privacy commitments — which are genuine but should be understood as a roadmap, not a delivered capability.

### 10.1 Differential privacy

Differential privacy is the strongest formal guarantee available for privacy in machine learning. It was introduced by Dwork et al. (2006) and has since become the gold standard in the privacy research community, though its adoption in production recommender systems remains limited for reasons that will become clear.

**The definition.** A randomised mechanism $\mathcal{M}$ satisfies $(\varepsilon, \delta)$-differential privacy if for any two datasets $D$ and $D'$ that differ in exactly one record, and for any set of outputs $S$:

$$\Pr[\mathcal{M}(D) \in S] \leq e^{\varepsilon} \cdot \Pr[\mathcal{M}(D') \in S] + \delta$$

The intuition: an adversary looking at the model's output cannot reliably determine whether any particular individual's data was included in the training set. The output distributions with and without that individual are nearly indistinguishable.

**What epsilon controls.** Think of the adversary as running a hypothesis test: "was this person's data present or absent?" The two distributions — model output with the person's data, model output without — are the null and alternative hypotheses. $\varepsilon$ controls how far apart those distributions are, or equivalently, how much statistical power the adversary has. At $\varepsilon = 0$, the distributions are identical and the test has no power — perfect privacy. As $\varepsilon$ increases, the distributions separate and the adversary's ability to distinguish them improves. The parameter $\delta$ is the probability of a catastrophic failure — an event where the guarantee simply does not hold, where the mechanism produces an output so far in the tail that even small $\varepsilon$ cannot mask it.

**How it works in ML training.** The dominant method is DP-SGD (Abadi et al., 2016): during training, each per-example gradient is clipped to a maximum norm and then Gaussian noise is added before aggregation. The clipping bounds individual influence; the noise blurs it. The privacy budget $\varepsilon$ accumulates across training steps via a composition theorem — each batch of gradient updates spends some privacy, and the total spend is tracked.

**The privacy-utility tradeoff.** This is where the difficulty lives. Meaningful privacy requires small $\varepsilon$ — typically $\varepsilon \leq 1$ for strong guarantees. But the noise added to achieve small $\varepsilon$ directly degrades model quality. For a recommendation system, where the whole point is to learn fine-grained distinctions between user preferences, the noise floor at $\varepsilon \leq 1$ typically destroys the signal the model needs.

How bad is the degradation? In the best reported results for DP recommendation models (e.g., Chien et al., 2021; Gopi et al., 2022), models trained with $\varepsilon = 1$ show 10-30% drops in recommendation quality compared to non-private baselines. At $\varepsilon = 8$ — which many privacy researchers consider too loose to be meaningful — the degradation is 3-8%. For a platform where recommendation quality directly determines revenue, even 3% is significant, and 30% is a different product.

The current state of practice reflects this tradeoff. Apple uses $\varepsilon = 2-8$ for device-level analytics. Google uses $\varepsilon = 1-3$ for Chrome usage statistics. LinkedIn reports $\varepsilon = 6.5$ for ad targeting. These are aggregate statistics tasks where moderate noise is tolerable. No major recommendation platform has published results showing production deployment of DP training at $\varepsilon \leq 3$ for personalised recommendations. The technique works in principle; at the quality level a competitive recommender needs, the cost is currently too high.

### 10.2 Federated learning

Federated learning is a training paradigm where the model comes to the data rather than the data coming to the model. User data stays on user devices (or in user-controlled storage); the model sends its parameters to each device, computes local updates, and aggregates the updates centrally — without ever seeing the raw data.

**FedAvg.** The foundational algorithm is FedAvg (McMahan et al., 2017). The protocol:

1. The server sends the current global model to a sample of clients.
2. Each client runs several steps of SGD on its local data.
3. Each client sends its updated model parameters (or parameter deltas) back to the server.
4. The server averages the updates, weighted by each client's data size.
5. Repeat.

The result is a model trained on the union of all clients' data without that data ever leaving the devices. In principle, this provides strong privacy: the server never sees raw data, only aggregated gradient updates. (In practice, gradient updates can leak information about the training data — this is where secure aggregation and DP noise on the updates come in, but that is a second-order concern for understanding the basic architecture.)

**Communication costs.** The practical challenge for federated learning is communication. Each round requires sending and receiving a full model's worth of parameters — or at least the deltas — to and from every participating client. For a ViT-L model with ~300M parameters, that is roughly 1.2 GB per client per round. With thousands of clients and hundreds of rounds, the bandwidth costs are substantial. Compression techniques (gradient quantisation, sparsification) help but do not eliminate the problem.

**Why the program lists federated user tower as aspirational (Phase 5+).** The program's architecture has an item tower and a user tower. The item tower is trained on artwork data that Beowolff owns or licenses — there is no privacy motivation for federating it. The user tower is trained on user behavioural data, which is where the privacy concern concentrates. Federating the user tower means: each user's device maintains a local copy of the user-tower model, trains on the user's own interaction history, and sends only the updated parameters to the server.

This is architecturally clean and privacy-preserving. It is also research-grade for a system at this scale, for three reasons. First, art browsing is sparse — many users have few interactions, and local training on a handful of data points produces noisy updates. Second, the interaction data includes server-side events (collection groupings, purchase inquiries) that do not naturally live on the client. Third, the two-tower architecture requires the user-tower output to live in the same embedding space as the item tower, which means the user tower's training must be tightly coordinated with item-tower updates — a synchronisation problem that federated settings make harder.

The program is right to flag this as aspirational and to architect so the user tower can eventually migrate to client-side. The architecture should support it; the Phase 2-4 deployment should not depend on it.

### 10.3 Machine unlearning and SISA training

Here is the core problem. A neural network trained by gradient descent does not store individual training examples in identifiable locations. Each gradient update is a function of the current example *and* the current parameter state, which was itself shaped by all previous examples. After training, every parameter in the model carries a superposition of influences from the entire training set. Removing one example's influence is not like deleting a row from a database — it is like removing one instrument's contribution from a finished recording. The sounds have been mixed.

This is the entanglement problem. It is fundamental to how gradient-based optimisation works, not an artifact of a particular architecture.

**Naive unlearning approaches and why they fail.** The simplest response to a deletion request is: delete the data and retrain from scratch on the remaining data. This is correct — the retrained model provably does not contain the deleted user's influence — but at the scale the program targets (millions of items, weeks of training), retraining from scratch for every deletion request is prohibitively expensive.

Approximate methods — fine-tuning the existing model to "forget" the deleted data, or using gradient ascent to reverse the deleted data's influence — are faster but lack formal guarantees. The model might still encode the deleted data in subtle ways that approximate unlearning does not remove. For a system that commits to "right-to-delete that actually propagates," approximate unlearning is insufficient as a sole mechanism.

**SISA training.** Bourtoule et al. (2021) proposed SISA (Sharded, Isolated, Sliced, Aggregated) as a middle path. The key insight: if you partition the training data into disjoint shards and train a separate model on each shard, then deleting one example requires retraining only the shard that contained it — not the entire model.

The framework has two levels of partitioning:

- **Sharding**: divide the training set into $k$ disjoint shards. Train $k$ independent models, one per shard. At inference time, aggregate their predictions (e.g., average the output embeddings or logits). To unlearn example $x$, identify which shard contains $x$, remove $x$ from that shard, retrain only that shard's model. Cost: $1/k$ of a full retrain.

- **Slicing**: within each shard, further divide the data into ordered slices $s_1, s_2, \ldots, s_m$. Save checkpoints after training on each slice. To unlearn example $x$ in slice $s_j$, reload the checkpoint from slice $s_{j-1}$ and retrain on slices $s_j$ through $s_m$ (minus $x$). This further reduces the retraining cost to a fraction of the shard.

**The tradeoffs.** SISA introduces real costs:

- *Model quality.* An ensemble of $k$ models each trained on $1/k$ of the data will generally underperform a single model trained on all the data. Each shard model sees less data and learns a less complete representation. The degradation depends on $k$ and on the data distribution — if shards are heterogeneous (different art periods in different shards), the per-shard models may miss cross-shard patterns.

- *Engineering complexity.* Managing $k$ independent training pipelines, $m \times k$ checkpoints, shard-to-example mappings, and an aggregation layer adds substantial infrastructure. For an attribution-grade system that already requires deterministic training and provenance logging, the additional complexity is not trivial.

- *Shard assignment is a design choice.* How you assign data to shards matters. Random assignment minimises correlation between deletion requests and shard membership. But for art data, random assignment might put all of a particular artist's works across many shards, meaning an artist requesting deletion triggers retraining across most shards. Assigning by contributor (all of one artist's works in one shard) concentrates the retraining cost but creates heterogeneous shards.

**Comparison to scheduled clean retrains.** The realistic baseline for the program's privacy commitments is simpler than SISA: scheduled clean retrains. Maintain a queue of deletion requests. At each scheduled retraining cycle (say, monthly), apply all pending deletions to the training corpus and retrain from scratch. Between retrains, the model still carries the deleted users' influence — but the latency is bounded and disclosed.

This is what most production ML systems actually do. It is not elegant, and the gap between deletion request and influence removal is a genuine privacy concern. But it is correct (each retrain provably removes the deleted data's influence), simple to implement, and compatible with the program's existing retraining cadence for Shapley computation.

### 10.4 What is tractable today

An honest assessment of where these techniques sit for a system at Beowolff-Embed's scale:

**Differential privacy.** Not tractable at the quality level needed for competitive recommendation. The noise required for meaningful $\varepsilon$ values degrades fine-grained preference learning too severely. Monitor the research frontier — techniques are improving yearly, and the gap is narrowing — but do not promise DP training in the near-term roadmap.

**Federated learning.** Architecturally sound for the user tower in principle, but research-grade for this application. The combination of sparse interaction data, server-side events, and tight item-tower coupling makes it impractical before the system has a large, engaged user base with rich on-device interaction histories. Phase 5+ is the right timeline.

**SISA training.** Practical but expensive in engineering terms. Worth implementing if deletion volume is high enough that scheduled clean retrains become too infrequent to meet privacy commitments. The shard-assignment problem (by contributor vs. random) requires a design decision that depends on the expected deletion pattern.

**Scheduled clean retrains.** The realistic baseline. Compatible with the program's existing need for periodic retraining (for Shapley computation). The latency between deletion request and influence removal must be disclosed and bounded. Monthly retraining is a reasonable starting point; the cadence can be shortened if deletion volume or regulatory pressure demands it.

**The practical recommendation.** Start with scheduled clean retrains. Architect the training pipeline so that SISA sharding can be introduced later without redesigning the data flow. Track the deletion-request rate; if it exceeds the level where monthly retrains provide acceptable latency, implement SISA. Defer DP and federated learning to the research roadmap. Be transparent about what the system does and does not guarantee at each phase.

The program's commitment to "right-to-delete that actually propagates" is genuine and important. The gap between the commitment and current tractability is also genuine. Naming that gap is not a criticism of the program — it is a prerequisite for designing a credible roadmap to close it.
