## 12. Threat Models: Distillation Attacks, Membership Inference, and the Open/Closed Defence

The program describes a three-layer architecture: open protocol (Filix, Shapley library, HAAK platform), private model and data (trained weights, licensed corpus, user histories), and user-owned portable artifacts (contribution records, attribution receipts, personal agents). This chapter asks whether that boundary holds when someone actively tries to break it.

The adversary is not a script kiddie. It is a well-funded competitor — an existing art platform or a technology company entering the art market — with engineering talent, computational resources, and strategic patience. The question is not whether such an adversary could theoretically breach the boundary, but what it would cost them and whether the program's defences make the cost exceed the value.

### 12.1 Model distillation attacks

Model distillation — also called model extraction or model stealing — is the process of training a surrogate model to mimic a proprietary model's behaviour by querying it through its API. The attacker does not need access to the model's weights, architecture, or training data. They only need input-output pairs.

**How it works.** The attacker constructs or collects a set of inputs (images of artworks, user profiles, or query embeddings), submits them to the target model's API, and collects the outputs (recommended artworks, similarity scores, or embedding vectors). They then train their own model to map the same inputs to the same outputs. If the target model's API returns rich outputs — full embedding vectors rather than just top-K recommendations — the attack is easier because the surrogate has more signal to learn from.

**How much query access is needed.** This depends on the complexity of the target model and the richness of the output. For embedding models, Tramèr et al. (2016) showed that a model returning confidence scores or embedding vectors can be distilled with a number of queries roughly proportional to the model's parameter count — typically millions of queries for a production model. For models returning only top-K ranked lists (no scores), the attacker needs more queries but the attack is still feasible with techniques like active query selection (choosing inputs that maximise the information gained per query).

For the program's architecture specifically: if the API returns full embedding vectors (as it would for a similarity-search endpoint), a competitor could collect embedding vectors for a large sample of artworks, train their own encoder to reproduce those embeddings, and use the result as a free-riding substitute. If the API returns only ranked recommendation lists, the attack is harder but not infeasible — learning-to-rank from pairwise preferences is a well-studied problem.

**Defences.**

- *Rate limiting.* Cap the number of queries per user, per API key, per IP address. This forces the attacker to either slow down (increasing the time cost) or use many accounts (increasing the coordination cost). Rate limiting is easy to implement and hard to circumvent at scale, but a determined attacker with thousands of accounts and months of patience can accumulate enough queries.

- *Output perturbation.* Add calibrated noise to the output — slightly perturb embedding vectors or randomly swap items in ranked lists. The noise is small enough that legitimate users do not notice degraded quality, but large enough that the surrogate model trained on noisy outputs is measurably worse than the original. The tradeoff is direct: more noise means better defence but worse user experience. Section 12.4 discusses this in detail.

- *Watermarking.* Embed a detectable signal in the model's outputs that survives distillation. If the surrogate model reproduces the watermark, this provides evidence of copying. Watermarking is useful for legal enforcement (proving the surrogate was trained on stolen outputs) but does not prevent the attack itself.

- *Terms of service.* Contractually prohibit systematic querying for the purpose of model extraction. This provides a legal enforcement mechanism but no technical prevention.

- *Tiered access.* Return richer outputs (full embeddings, confidence scores) only to authenticated partners with contractual relationships, and return only ranked lists to public API users. This limits the information available to anonymous attackers while preserving functionality for trusted consumers.

**Are these sufficient?** Against a casual attacker, yes. Against a well-resourced competitor willing to invest months and millions of queries, no single defence is sufficient, but the combination raises the cost substantially. The honest assessment: if a major technology company decides to clone the recommendation model, they probably can, given enough time and resources. The question is whether it is worth their effort — which depends on whether the model is the primary source of competitive advantage, or whether the data and relationships are. We return to this in Section 12.3.

### 12.2 Membership inference

Membership inference is a different class of attack: the adversary's goal is not to steal the model but to determine whether a specific data point was in the training set. Was this particular artwork used to train the model? Was this user's behavioural data included?

**How it works.** The basic intuition (Shokri et al., 2017): a model trained on a specific data point tends to be more confident on that point than on similar points it has not seen. The attacker trains "shadow models" on data drawn from a similar distribution, notes the confidence patterns on included vs. excluded data, and uses these patterns to classify the target model's predictions as "member" or "non-member."

For an embedding model, the analog is: an artwork in the training set will have a "better-fitting" embedding — closer to its expected neighbours, more consistent with its metadata — than a similar artwork not in the training set. The attacker queries the model with a target artwork, examines the embedding's properties (distance to cluster centre, consistency with known metadata), and infers membership.

**Why this matters.** For artwork membership, the privacy concern is moderate — whether a publicly available artwork was used in training is commercially sensitive (it reveals something about the training corpus composition) but not personally identifying. For user membership, the concern is sharper: confirming that a specific user's behavioural data was in the training set reveals that the platform has data on that user, which may itself be sensitive.

**Connection to differential privacy.** Differential privacy (Chapter 10) provides the formal defence against membership inference: if the model satisfies $(\varepsilon, \delta)$-DP, the adversary's advantage in the membership inference game is bounded by $e^{\varepsilon} - 1$. At $\varepsilon = 1$, the advantage is at most $\sim 1.7\times$ over random guessing — detectable in aggregate but unreliable for individual inferences. Without DP, membership inference is often feasible for overfit models and becomes easier as the model grows more expressive.

Since the program does not currently plan DP training at production quality levels (Chapter 10), the practical defence against membership inference relies on the model being well-regularised (not overfitting to individual training examples) and on the output perturbation described in Section 12.4. These provide empirical but not formal protection.

### 12.3 The open/closed boundary under adversarial pressure

The program's three-layer architecture is designed to be simultaneously open (for trust) and closed (for competitive advantage). The open layer — the protocol, the Shapley library, the HAAK platform — is meant to be auditable, forkable, and trust-generating. The closed layer — the trained model, the data corpus, the relationships — is meant to be the source of competitive advantage. The question is whether this separation is stable under pressure from a sophisticated adversary.

**Where would a competitor probe?**

*Attempt 1: Clone the model.* Use distillation attacks (Section 12.1) to replicate the recommendation quality. As discussed, this is feasible in principle but expensive. More importantly, the cloned model is a snapshot — it does not improve as the original model is retrained on new data. The competitor must continuously distill, which means continuously querying at volume, which rate limiting and monitoring can detect and throttle.

*Attempt 2: Replicate the data.* License or scrape the same data sources. Museum image corpora (WikiArt, Met, Rijksmuseum) are publicly available. Artnet auction data is commercially licensable. Artsy works and behavioural data are proprietary but could potentially be obtained through a partnership or acquired through a corporate transaction. The question is whether the data corpus is the moat.

*Attempt 3: Replicate the relationships.* Onboard the same galleries, sign the same artists, build the same collector base. This is the hardest part to replicate and takes years. Relationships are the true moat in most marketplace businesses, and this is likely true for Beowolff as well.

*Attempt 4: Fork the open layer and compete on attribution.* Take the Filix protocol, the Shapley library, and the HAAK platform. Build a competing art intelligence platform with the same attribution infrastructure but different data and relationships. This is explicitly permitted by the open-source architecture — and the program argues it should be.

**Why the model is not the primary moat.** The program's competitive advantage does not rest primarily on the model architecture (which is built from published components: ViT, SigLIP, two-tower retrieval, contrastive training) or on the training procedure (which the study guide you are reading explains in full). It rests on three things a competitor cannot easily replicate:

1. *The data corpus.* The combination of Artsy's artwork database, Artnet's auction archive, and the platform's own behavioural data. Each source is individually obtainable; the union under a single training pipeline, with cross-source entity resolution and quality control, represents years of engineering and relationship-building.

2. *The relationships.* Gallery onboarding, artist deposits, auction-house integrations, collector accounts. These are commercial network effects that compound over time.

3. *The attribution track record.* Once Shapley-based attribution is running and contributors are receiving payments, the platform has a demonstrated history of fair compensation. This is a trust asset that a new entrant cannot replicate on day one — they can promise attribution, but they cannot prove a track record of it.

**Why commensurability is a feature.** The program notes that each platform's Shapley scores are computed against its own value function and are not transferable. A Shapley score of 0.03 on Beowolff means something different from a Shapley score of 0.03 on a competitor's platform, because the value functions differ. This might seem like a limitation — shouldn't attribution be comparable across platforms? — but it is actually a feature. It means a competitor cannot free-ride on Beowolff's attribution computations or claim equivalence without doing the work. Each platform must earn its own attribution credibility.

Conversely, if the program publishes attribution receipts (Open Question 4 in the program file) — signed records of "contributor X was attributed value Y and paid Z" — these serve as portable reputation signals. A contributor can take their receipts to another platform and say: "I was valued here; what will you offer me?" This is the contributor-friendly version of data portability, and it makes the open protocol layer work for contributors even if different platforms are not directly commensurable.

### 12.4 Watermarking and output perturbation

These are the two technical defences the program can deploy without degrading the core training process.

**Embedding watermarking.** The idea: during training, embed a detectable pattern in the model's output space — a statistical signature that is invisible in normal use but detectable by the model owner. If a competitor distills the model, the watermark survives in the surrogate's outputs, providing evidence of copying.

Techniques include:

- *Trigger-based watermarking.* Train the model to produce a specific output pattern on designated trigger inputs (e.g., a set of artworks that the model maps to predetermined positions in the embedding space). A distilled model that reproduces these trigger-output mappings is almost certainly derived from the original.

- *Distribution-based watermarking.* Impose a subtle statistical constraint on the output distribution — for example, ensuring that the last few dimensions of every embedding vector satisfy a particular parity condition. This is harder for a distiller to detect and remove.

The limitation of watermarking is that it proves copying after the fact; it does not prevent it. Legal enforcement requires detecting the watermark in the competitor's model, which may require querying the competitor's API — creating a symmetrical access problem.

**Output perturbation.** Add calibrated noise to API responses. For embedding endpoints, this means adding a small random vector to each returned embedding. For ranked-list endpoints, this means occasionally swapping adjacent items in the ranking. The noise is calibrated so that:

- Legitimate users experience negligible quality degradation (similarity search returns slightly noisier results; rankings are slightly suboptimal).
- A distillation attacker receives noisy training signal, degrading the surrogate model's quality.

The tradeoff is direct and quantifiable. If the noise standard deviation is $\sigma$ and the embedding dimension is $d$, the expected distortion in cosine similarity between the true and perturbed embeddings is approximately $\sigma / \sqrt{d}$. For $d = 768$ and $\sigma = 0.1$, this is about 0.0036 — negligible for recommendation quality but sufficient to introduce meaningful noise into a distillation training set of millions of queries.

The subtlety is that fixed-magnitude noise can be averaged out: if the attacker queries the same input multiple times and averages the responses, the noise cancels. The defence is to return the same perturbed output for the same input (deterministic perturbation keyed to the input hash) rather than independent noise on each query. This prevents averaging without introducing inconsistency.

**The combined defence posture.** No single defence is sufficient against a determined adversary. The program's realistic defence is layered:

1. *Rate limiting* raises the time cost of distillation.
2. *Output perturbation* raises the quality cost (the surrogate is measurably worse).
3. *Watermarking* provides evidence for legal enforcement.
4. *Tiered access* limits information available to anonymous attackers.
5. *Terms of service* provide contractual enforcement.
6. *The data and relationship moat* makes even a successful distillation insufficient to compete, because the model is only one component of the product.

This is the standard defence posture for production ML systems — not impenetrable, but raising the cost of attack above the value of success. For an art intelligence platform, where the market is niche and the value of the model is intertwined with the data and relationships behind it, this posture is likely adequate. The most important investment is not in technical defences but in building the data corpus and relationship network that make the model valuable in the first place — and that no amount of API querying can replicate.
