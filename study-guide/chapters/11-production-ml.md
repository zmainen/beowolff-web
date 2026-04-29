## 11. Production ML Operations: From Trained Model to Deployed Feature

The program references "training pipelines," "model versioning," "model cards," "eval methodology," and "A/B testing" as though each is a single concept. Each is a substantial engineering domain with its own failure modes. This chapter explains what actually happens between "the model converged" and "users see recommendations" — the infrastructure that turns a trained model into a production feature. The gap between a working research prototype and a deployed system is where most ML projects stall, and understanding that gap is essential for evaluating whether the program's commitments (especially attribution-grade logging and reproducible training) are engineering-feasible or aspirational.

The bridge to your existing knowledge is through experimental methodology. A/B testing for recommenders is a variant of randomised controlled trials with the same concerns about sample size, multiple comparisons, and confounds — plus domain-specific problems that clinical trials do not face. And the reproducibility discipline the program requires is conceptually identical to the reproducibility crisis in experimental science: the question is not whether you intend to be reproducible, but whether the tools and practices actually achieve it.

### 11.1 Training pipelines

"Pipeline" is production ML's term for what an experimentalist would call a protocol — a sequence of steps, each with defined inputs and outputs, that transforms raw materials into a finished product. The word carries a specific connotation: each step should be automated, deterministic, and independently testable.

A typical training pipeline for an embedding model like Rasa has six stages:

**Data ingestion.** Raw data enters the system: images from museum APIs, metadata from curatorial databases, behavioural logs from the platform, price records from auction archives. Ingestion includes validation (is the image corrupt? is the metadata schema correct?), deduplication (the same Rembrandt appears in WikiArt, the Met, and Rijksmuseum — are they merged or kept separate?), and provenance logging (where did this data come from, when, under what license?).

**Preprocessing.** Raw data is transformed into the format the model expects. Images are resized and normalised. Text is tokenised. Structured metadata is mapped to vocabulary indices. Behavioural sequences are sorted chronologically, truncated to maximum length, and padded. Preprocessing must be deterministic — the same raw input must always produce the same preprocessed output — because any randomness here propagates into the training and makes reproducibility impossible.

**Feature extraction.** For transfer-learning architectures like the program's, some features are extracted from pretrained models and cached rather than computed during training. DINOv3 features for images, SigLIP-2 features for text — these can be precomputed once and stored, since the pretrained models are frozen during early training phases. Caching saves enormous compute (no need to run the full ViT forward pass at every training step) but creates a versioning dependency: if the pretrained model changes, all cached features must be recomputed.

**Training.** The actual gradient descent. This is the step most researchers think of when they hear "training," but in production it is one of six. It consumes preprocessed data and (optionally) cached features, runs the optimisation loop, and produces model checkpoints at regular intervals. For attribution-grade training, the program requires deterministic data ordering and fixed seeds — we will examine what this actually entails in Section 11.2.

**Evaluation.** Each checkpoint is evaluated on a held-out test set. The program specifies Recall@K and NDCG, stratified by item warmth. Evaluation must use a temporally split test set (train on data before time $T$, evaluate on data after $T$) to avoid information leakage. The evaluation results are recorded against the checkpoint hash and the training corpus hash in the Filix ledger — this is what makes Shapley computation possible, because every $v(S)$ evaluation references a specific model version and a specific eval set.

**Checkpoint storage.** Model weights, optimiser state, training configuration, and evaluation results are stored together as a versioned artifact. The checkpoint must be self-describing: anyone who retrieves it should be able to determine exactly what data it was trained on, what hyperparameters were used, what evaluation scores it achieved, and what the previous checkpoint in the lineage was. This is the model-versioning discipline described in Section 11.3.

**Why determinism matters for attribution.** The Shapley computation described in Chapter 9 requires evaluating $v(S)$ — the model's performance when trained on data subset $S$. If training is non-deterministic (different runs on the same data produce different models), then $v(S)$ is not a function but a random variable, and the Shapley values inherit that randomness. Some variance is tolerable (Chapter 13 discusses how much), but uncontrolled non-determinism makes the attribution framework unreliable.

### 11.2 Reproducibility in practice

Reproducibility in ML has a specific meaning: given the same data, the same code, and the same configuration, the training process produces the same model. "Same" is where the difficulty hides.

**What you can control:**

- *Data ordering.* Fix the random seed for the data loader's shuffle. This ensures the same examples appear in the same order in every batch. Most frameworks (PyTorch, TensorFlow) support this through `torch.manual_seed()` or equivalent.

- *Parameter initialisation.* Fix the random seed for weight initialisation. Combined with deterministic data ordering, this means the first gradient update is identical across runs.

- *Library versions.* Pin every dependency to an exact version — not just PyTorch, but CUDA, cuDNN, NCCL (for multi-GPU communication), NumPy, and the operating system's math libraries. A minor version bump in cuDNN can change the implementation of a convolution kernel, producing different floating-point results.

- *Containerisation.* Package the entire environment in a Docker container with a fixed base image. This freezes the software stack, ensuring that the same container produces the same results regardless of the host machine's software environment.

**What breaks reproducibility in practice:**

- *GPU non-determinism.* This is the most intractable source of irreproducibility. GPU operations like `atomicAdd` (used in scatter operations and gradient accumulation) are non-deterministic by default — the order in which parallel threads contribute to a sum is not guaranteed, and floating-point addition is not associative ($a + b + c \neq a + c + b$ in general, due to rounding). PyTorch offers `torch.use_deterministic_algorithms(True)`, which forces deterministic implementations, but at a significant performance cost (some operations are 2-10x slower in deterministic mode, and some have no deterministic implementation at all).

- *Floating-point non-associativity.* Even on CPU, reducing a sum across multiple threads can produce different results depending on the reduction order. The difference per operation is at the level of machine epsilon ($\sim 10^{-7}$ for float32), but these differences accumulate across millions of operations and thousands of training steps. Two runs that are identical in every respect can diverge measurably after a few hundred steps.

- *Multi-GPU communication.* Distributed training across multiple GPUs or nodes introduces additional non-determinism through the all-reduce operations that synchronise gradients. The order of arrival of gradients from different GPUs is not guaranteed, and the reduction is again subject to floating-point non-associativity.

- *Library updates.* Even with pinned versions, rebuilding a container from scratch may pull different compiled binaries if the build environment has changed. True reproducibility requires storing the built container image, not just the Dockerfile.

**The gap.** "Reproducible in principle" means: if you control all the sources of randomness listed above and accept the performance penalty of fully deterministic GPU operations, you can achieve bitwise identical results. "Reproducible in production" means: you accept that bitwise identity is impractical at scale, define a tolerance (e.g., model performance within $\pm 0.1\%$ of the reference run), and verify empirically that your pipeline stays within that tolerance. Chapter 13 (Section 13.6) discusses what happens to attribution when this tolerance is non-zero.

### 11.3 Model versioning and model cards

A model card (Mitchell et al., 2019) is a structured document that accompanies a trained model and describes what it is, what it can do, and what it should not be used for. The concept originated in the fairness and accountability literature, but its utility extends to any system where multiple stakeholders need to understand a model's provenance and limitations.

**What a model card contains:**

- *Model description.* Architecture, parameter count, training objective.
- *Training data.* Description of the training corpus — size, sources, preprocessing, known biases. Not the data itself, but enough information to understand what the model was trained on.
- *Evaluation results.* Performance on held-out test sets, broken down by relevant strata (for Rasa: warm vs. cold items, art period, medium, geographic origin).
- *Intended use.* What the model is designed for (art recommendation, similarity search) and what it is not designed for (aesthetic quality judgment, authentication, market prediction).
- *Limitations and biases.* Known failure modes, underrepresented populations in the training data, domains where performance degrades.
- *Versioning metadata.* Version number, training date, parent version (what model this was fine-tuned from), changelog.

**Why versioning matters for attribution.** Each Shapley computation references a specific model version — "what was the performance of model $v_k$ on eval set $E$, given training corpus $D_k$?" If model versions are not tracked with their corresponding training corpora and evaluation results, the Shapley computation is impossible. The Filix ledger provides the storage layer; the model card provides the semantic layer. Together they answer the question: "for any given model checkpoint, what went into it and what came out?"

**Connection to the Filix ledger.** In the program's architecture, the model card is not a standalone document — it is a ledger entry. The checkpoint hash, the training corpus hash, the evaluation results, and the model card metadata are all content-addressed and linked. This means the model card is tamper-evident: changing any field changes the hash, which breaks the chain. This is the infrastructure that makes attribution-grade claims verifiable rather than rhetorical.

### 11.4 A/B testing for recommenders

A/B testing is the randomised controlled trial of production ML. The logic is identical to what you know from experimental science: randomly assign subjects to conditions, measure outcomes, test whether the difference exceeds what you would expect by chance. The domain-specific complications are what make recommender A/B testing its own discipline.

**The basic setup.** Users are randomly assigned to a control group (current model) or treatment group (new model). Both groups interact with the platform normally; the only difference is which model generates their recommendations. After a predetermined period, outcomes are compared.

**What to measure.** This is the first place where recommender A/B testing diverges from standard practice. The obvious metric — engagement (clicks, time spent) — is available but dangerous as a sole criterion. A recommender optimised for engagement can learn to show sensational or controversial content that keeps users clicking without serving their actual interests. For an art platform with transactional revenue, the metrics that matter are:

- *Conversion.* Did the recommendation lead to a purchase inquiry or sale? This is the metric most aligned with the platform's revenue model.
- *Discovery.* Did the user interact with artists or periods they had not previously engaged with? This measures whether the recommendation system is expanding taste rather than reinforcing existing preferences.
- *Retention.* Did users in the treatment group return to the platform at higher rates? This captures long-term satisfaction that short-term engagement metrics miss.
- *Cold-start performance.* Did the new model successfully recommend cold-start items (works with no prior interaction history)? This is the program's stated differentiator.

**Duration and sample size.** Recommendation effects take time to manifest. A user who receives better recommendations today may not convert for weeks. Short A/B tests (days) capture only immediate engagement effects and miss the retention and conversion signals that matter most. Industry practice for recommender A/B tests is typically 2-4 weeks, with sample sizes in the tens of thousands per arm — enough to detect a 1-2% lift in conversion with adequate power.

The standard power calculation applies: for a two-proportion test with baseline conversion rate $p_0$, minimum detectable effect $\Delta$, significance level $\alpha$, and power $1-\beta$:

$$n \geq \frac{(z_{\alpha/2} + z_\beta)^2 \cdot [p_0(1-p_0) + p_1(1-p_1)]}{\Delta^2}$$

where $p_1 = p_0 + \Delta$. For art platforms where baseline conversion rates are low (often $< 1\%$), detecting meaningful lifts requires large sample sizes or longer test durations.

**The network-effects problem.** This is the domain-specific complication that makes recommender A/B testing harder than standard RCTs. In a clinical trial, treating Patient A does not change Patient B's outcome. In a recommender system, it can: if the treatment model recommends Artwork X to User A and they buy it, Artwork X is no longer available — the recommendation set for User B has changed. This is an interference effect, analogous to the stable unit treatment value assumption (SUTVA) violation in causal inference.

For unique goods like art, this interference is particularly acute. Every sale removes an item from the market. If the treatment model is better at matching buyers to artworks, it will selectively deplete the best matches, making the control group's experience worse than it would have been without the experiment. The standard mitigation is to run the A/B test on a scale where the treatment effect on inventory is negligible relative to total inventory — which, for a platform with millions of artworks, is usually satisfied.

**Temporal evaluation vs. online evaluation.** The program specifies temporal evaluation splits (Chapter 6) for offline model selection and A/B testing for online validation. These answer different questions. Temporal evaluation asks: "on historical data, does the new model predict future interactions better?" A/B testing asks: "in the live system, do users have better outcomes with the new model?" Both are necessary because offline metrics do not perfectly predict online performance — a model that scores well on historical data may fail in production due to latency issues, serving bugs, or distributional shift between the eval set and the live user population.

### 11.5 The engineering layers

Between a trained model and a user seeing a recommendation, several engineering layers intervene. Understanding them matters for evaluating whether the program's latency and quality commitments are feasible.

**Model serving.** The trained model must be deployed as a service that accepts queries and returns predictions. Frameworks like TorchServe (PyTorch's serving solution) or TensorFlow Serving handle the mechanics: loading model weights into GPU memory, batching incoming requests for efficient GPU utilisation, managing model versions (so a new model can be deployed without downtime). For the program's architecture, model serving means: given a user's interaction history, compute the user-tower embedding; given an artwork, compute the item-tower embedding. The embeddings are the products; the recommendations are assembled downstream.

**Two-stage architecture: candidate generation and ranking.** At production scale, scoring every item against every user at query time is infeasible. A corpus of millions of artworks and a latency budget of 100-200 milliseconds forces a two-stage approach:

- *Candidate generation.* Use approximate nearest-neighbour search (FAISS with HNSW, as described in Chapter 8) to retrieve the top 100-1000 items closest to the user's embedding. This is fast (sub-millisecond for a well-indexed corpus) but coarse — the ANN index trades precision for speed.

- *Ranking.* A more expensive model re-scores the candidates using features not available in the embedding space: recency, inventory status, business rules (e.g., promote gallery partners), diversity constraints (do not show five Monets in a row). The ranker produces the final ordered list the user sees.

This two-stage architecture is standard across YouTube, Pinterest, Spotify, and every other large-scale recommendation system. The candidate generator determines *what the user might see*; the ranker determines *what the user actually sees*. The embedding quality matters most for the candidate generator — a great ranker cannot rescue candidates that were never retrieved.

**Caching.** Item embeddings change only when the model is retrained. Between retrains, they can be precomputed and cached, eliminating the need to run the item tower at query time. User embeddings change with every new interaction, but in practice they can be recomputed periodically (e.g., every few minutes) rather than on every request. The freshness-latency tradeoff — how recently updated the user embedding is — affects recommendation quality at the margins but allows significant compute savings.

**Batching.** GPU inference is most efficient when processing multiple queries simultaneously. Serving frameworks accumulate incoming requests into batches (typically 8-64 queries) and process them together. This introduces a small latency (waiting for the batch to fill) but dramatically improves throughput. The batch-size-latency tradeoff is tuned empirically based on traffic patterns.

**Latency budgets.** The total time from user request to displayed recommendation is typically budgeted at 100-300 milliseconds for a responsive user experience. This budget must cover: user embedding computation (or cache lookup), ANN retrieval, ranking, business logic, and rendering. Each component has its own sub-budget. If the embedding model is too large to compute within its budget, it must be distilled to a smaller serving model — which introduces a gap between the training model (used for Shapley computation) and the serving model (used for recommendations). The program does not discuss this gap; it may not arise if the model is small enough to serve directly, but it is worth monitoring.

**The attribution logging layer.** Unique to the program's architecture: every recommendation served must be logged with sufficient provenance to support after-the-fact attribution. This means recording the model version, the user embedding, the candidate set, the ranking scores, and the final displayed list — all content-addressed and linked to the Filix ledger. This logging adds latency (disk writes) and storage costs (potentially gigabytes per day at scale). The program treats this as non-negotiable infrastructure for attribution; the engineering cost is real but bounded.
