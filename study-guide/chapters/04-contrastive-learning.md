## Contrastive Learning: Teaching a Model What "Similar" Means

Chapter 2 established that the loss function determines the geometry of the embedding space. This chapter is about the specific loss function the program uses: **contrastive learning**, and specifically the **InfoNCE** objective. Contrastive learning is the mechanism by which the embedding space acquires its semantic structure — it is how the model learns that a Rothko should be near a Newman and far from a Vermeer. Every claim the program makes about the embedding's quality depends on this training procedure working well.

The idea is deceptively simple: given a pair of items that should be similar (a "positive pair"), push their representations together; given items that should be dissimilar ("negative pairs"), push their representations apart. But every word in that sentence hides a design choice that the program has had to make. What counts as similar? What counts as dissimilar? How hard to push? Across how many comparisons at once? The answers to these questions matter more than the choice of network architecture.

### The contrastive objective: deriving InfoNCE

The learning dynamic is worth stating explicitly before the derivation. The model maintains a current estimate of how similar two items are (their embedding distance). The training signal provides ground truth about which pairs should be similar and which should not. The loss function computes the discrepancy between the model's estimate and the ground truth, and the gradient update closes the gap. Items that should be close but are currently far apart get pushed together; items that should be far apart but are currently close get pushed apart. The model is learning a similarity function over pairs, and updating it based on prediction errors — the difference between where items are in the current geometry and where they should be.

Now the derivation. Suppose we have an **anchor** item $x$ with embedding $\mathbf{z} = f(x)$, a **positive** item $x^+$ with embedding $\mathbf{z}^+ = f(x^+)$, and $N-1$ **negative** items $\{x_1^-, \ldots, x_{N-1}^-\}$ with embeddings $\{\mathbf{z}_1^-, \ldots, \mathbf{z}_{N-1}^-\}$. The model should learn to rate the positive pair as more similar than any negative pair.

Define similarity as the cosine of the angle between normalized embeddings (or equivalently, the dot product of L2-normalized vectors):

$$\text{sim}(\mathbf{z}_i, \mathbf{z}_j) = \frac{\mathbf{z}_i \cdot \mathbf{z}_j}{\|\mathbf{z}_i\| \|\mathbf{z}_j\|}$$

The InfoNCE loss (Oord et al., 2018) frames this as a classification problem. Given the anchor, can the model identify which of the $N$ candidates is the positive? It defines a probability distribution over candidates using a softmax:

$$p(\text{positive} = k \mid \mathbf{z}) = \frac{\exp(\text{sim}(\mathbf{z}, \mathbf{z}_k) / \tau)}{\sum_{j=1}^{N} \exp(\text{sim}(\mathbf{z}, \mathbf{z}_j) / \tau)}$$

where $\tau$ is the **temperature** parameter (discussed below). The loss is the negative log probability of correctly identifying the positive:

$$\mathcal{L}_\text{InfoNCE} = -\log \frac{\exp(\text{sim}(\mathbf{z}, \mathbf{z}^+) / \tau)}{\exp(\text{sim}(\mathbf{z}, \mathbf{z}^+) / \tau) + \sum_{j=1}^{N-1} \exp(\text{sim}(\mathbf{z}, \mathbf{z}_j^-) / \tau)}$$

This is a softmax cross-entropy loss over similarities. Minimizing it does two things simultaneously: it maximizes the similarity between the anchor and the positive (numerator up), and it minimizes the similarity between the anchor and all negatives (denominator down). The model learns to concentrate similarity mass on true positives.

**Why "InfoNCE"?** The name comes from Noise-Contrastive Estimation. Oord et al. showed that this loss function provides a lower bound on the mutual information between the anchor's input and the positive's input: $\mathcal{L}_\text{InfoNCE} \geq \log N - I(x; x^+)$. Maximizing InfoNCE (minimizing the loss) maximizes a lower bound on mutual information. Intuitively, two items that share a lot of information (same image under different augmentations, same artist's works, same collector's purchases) should have high mutual information, and InfoNCE trains the model to capture this.

The connection to **noise-contrastive estimation in language models** is worth noting: word2vec's skip-gram objective is a special case of InfoNCE where the positive pair is (word, context word) and the negatives are randomly sampled words. The embeddings that word2vec produces — where "king - man + woman = queen" — are a contrastive learning outcome. The Beowolff-Embed program is applying the same principle to artworks instead of words, with a richer definition of what constitutes a positive pair.

### Temperature: the sharpness dial

The temperature parameter $\tau$ in the InfoNCE loss controls how sharply the model discriminates between similar and dissimilar pairs. To understand it, consider what happens to the softmax distribution as temperature changes.

**Worked example.** Suppose the anchor has cosine similarities of 0.8 to the positive, 0.5 to negative 1, 0.3 to negative 2, and 0.1 to negative 3. The softmax probabilities assigned to the positive item under different temperatures are:

**At $\tau = 0.5$ (cold):**

$$\text{logits} = (0.8/0.5,\; 0.5/0.5,\; 0.3/0.5,\; 0.1/0.5) = (1.6,\; 1.0,\; 0.6,\; 0.2)$$

$$\text{softmax} = \frac{(e^{1.6}, e^{1.0}, e^{0.6}, e^{0.2})}{e^{1.6} + e^{1.0} + e^{0.6} + e^{0.2}} = \frac{(4.95, 2.72, 1.82, 1.22)}{10.71}$$

$$p(\text{positive}) = 4.95 / 10.71 = 0.462$$

**At $\tau = 0.05$ (very cold):**

$$\text{logits} = (0.8/0.05,\; 0.5/0.05,\; 0.3/0.05,\; 0.1/0.05) = (16,\; 10,\; 6,\; 2)$$

$$\text{softmax} \approx \frac{(8.9 \times 10^6,\; 2.2 \times 10^4,\; 403,\; 7.4)}{8.9 \times 10^6}$$

$$p(\text{positive}) \approx 0.998$$

**At $\tau = 0.5$ (warm):**

$$\text{logits} = (0.8/0.5,\; 0.5/0.5,\; 0.3/0.5,\; 0.1/0.5) = (1.6,\; 1.0,\; 0.6,\; 0.2)$$

This is the same as the first example. Now let's try even warmer:

**At $\tau = 2.0$ (hot):**

$$\text{logits} = (0.4,\; 0.25,\; 0.15,\; 0.05)$$

$$\text{softmax} = \frac{(1.49, 1.28, 1.16, 1.05)}{4.99}$$

$$p(\text{positive}) = 1.49 / 4.99 = 0.299$$

At high temperature, the distribution is nearly uniform — the model barely distinguishes positive from negative. At very low temperature, the distribution is nearly one-hot — the model is forced to make sharp discriminations.

**What this means geometrically.** Low temperature forces the model to push positives very close and negatives very far away. The embedding space becomes "tight" — clusters are compact, and the space between clusters is empty. High temperature allows the model to be lazy — positives and negatives can be at moderate distances and the loss is still low.

**The risk of too-low temperature.** If $\tau$ is too small, the loss gradient becomes dominated by the hardest negatives — the negatives most similar to the anchor. Every training step pulls the model toward maximally separating the anchor from its closest false match, which can cause training instability and "mode collapse" — the model might solve the problem by mapping everything to a few tight clusters, losing fine-grained structure within clusters.

**The risk of too-high temperature.** If $\tau$ is too large, the gradients are too weak to learn anything. The model does not need to separate positives from negatives because the soft distribution forgives large errors.

**How it is set in practice.** In CLIP, $\tau$ is a learnable parameter initialized at 0.07 and constrained to stay above 0.01. In many contrastive learning papers, it is fixed at 0.07 or 0.1 based on empirical tuning. Some systems use a cosine schedule that starts warm (easy problem) and cools during training (progressively harder discriminations). The program file does not specify the temperature, but the range 0.05-0.1 is standard for the kind of multi-modal contrastive training it describes.

### In-batch negatives and why batch size matters

Where do the negative examples come from? In the InfoNCE formula, we need $N-1$ negatives for each anchor. The most common approach — and the one the program uses — is **in-batch negatives**: the other items in the same training batch serve as negatives.

If the batch has 8,192 items, each anchor has 8,191 negatives. If the batch has 32,768 items, each anchor has 32,767 negatives. The program file mentions "8K-32K in-batch" — this is not a hardware detail, it is a first-order statistical decision.

**Why larger batches give better gradients.** The InfoNCE loss is a categorical cross-entropy over $N$ classes. With 100 negatives, the model needs to pick the positive out of 101 options — a relatively easy task. With 32,000 negatives, it needs to pick the positive out of 32,001 options — much harder. The harder task forces the model to learn finer-grained representations. This is not just intuition; it follows from the mutual-information bound: $I(x; x^+) \leq \log N + \mathcal{L}_\text{InfoNCE}}$. Larger $N$ gives a tighter bound.

Think of it this way. With 100 random negatives, many of them are "easy" — obviously different from the anchor (a landscape paired against a portrait). The model barely updates from these. It only learns from the negatives that are close enough to be confusing. With 32,000 negatives, the probability of having several confusing negatives in the batch is much higher. Each training step teaches the model something useful.

**Computational tradeoffs.** Larger batches require more GPU memory (every item in the batch must have its embedding stored simultaneously for the loss computation) and more data parallelism (distributing the batch across multiple GPUs). The 8K-32K range the program targets requires multi-GPU training. On 8x A100 with 80GB each, fitting 8K items with ViT-L embeddings is feasible; 32K pushes toward gradient accumulation or communication-heavy distributed training. This is why the program file mentions that "compute is not a constraint" — the batch-size decision becomes a hardware budget decision.

**Statistical tradeoffs.** Larger batches also mean fewer gradient updates per epoch. A dataset of 20M images trained with batch size 32K gets only 625 gradient updates per epoch. The optimizer sees each example less frequently, which can slow convergence. In practice, this is mitigated by the much higher quality of each gradient update. The consensus in the literature is that for contrastive learning, larger batches are strictly better until hardware limits or diminishing returns intervene — and 32K is well within the productive range.

### Hard-negative mining: where the real lift comes from

Here is a claim that might be surprising: **the largest gains in contrastive learning at scale come not from architectural improvements but from better negative sampling.** The reason goes back to the gradient structure of InfoNCE.

The gradient of InfoNCE with respect to the anchor embedding is a weighted sum of the differences between the anchor and each item:

$$\frac{\partial \mathcal{L}}{\partial \mathbf{z}} \propto -\left(\mathbf{z}^+ - \sum_{j} w_j \mathbf{z}_j^-\right)$$

where $w_j$ is the softmax weight of negative $j$. Crucially, these weights are exponential in similarity: a negative that is very similar to the anchor gets a large weight, and a negative that is very different gets a negligible weight. **The gradient is dominated by the hardest negatives** — the ones most similar to the anchor.

With random in-batch negatives at the start of training, most negatives are easy. The model quickly learns to separate obviously different items (landscapes from portraits, Old Masters from contemporary). But the easy negatives contribute almost nothing to the gradient — the model already knows they are different. What the model needs to learn next is the hard discriminations: Monet from Renoir, early Picasso from late Picasso, one Dutch Golden Age interior from another. Random sampling is an inefficient way to find these hard pairs.

**Hard-negative mining** is the practice of deliberately selecting negatives that are close to the anchor in the current embedding space. Three strategies, in order of increasing aggressiveness:

**Semi-hard negatives.** Select negatives that are farther from the anchor than the positive, but still within a margin. These are negatives the model would get right, but only barely. They provide a moderate learning signal without the risks of very hard negatives.

**Hardest-in-batch.** Take the negative in the current batch with the highest similarity to the anchor. This is free — you've already computed all pairwise similarities to evaluate the InfoNCE loss — and it concentrates the gradient on the most informative comparison. The risk: the hardest in-batch negative might not actually be very hard (if the batch is randomly sampled from 20M images, even the hardest random negative might be easy).

**Global FAISS-based mining.** Periodically re-embed the entire corpus, build a FAISS index, and for each anchor, retrieve its $k$ nearest neighbors in the current embedding space. Use these as negatives in subsequent training batches. This is computationally expensive (re-embedding 20M images takes hours on a GPU cluster) but provides truly hard negatives — items that the current model thinks are similar to the anchor. The program file mentions "hard-negative mining via FAISS at scale" as a Phase 2+ technique, after initial convergence with in-batch contrastive.

**The risk of false negatives.** Here is the subtlety that makes hard-negative mining dangerous. A "hard negative" — an item very similar to the anchor in the current embedding — might actually be a true positive that the training data has failed to label as such. Two works by the same school, the same period, the same Iconclass subject might look very similar and be meaningful recommendations for each other — but if they happen to be in different positions in the training data, the contrastive loss treats them as negatives and pushes them apart.

This is the **false negative problem**, and it is especially acute for art. Consider two works: a Pieter Claesz vanitas still life and a Willem Claesz Heda vanitas still life. They look nearly identical — same genre, same period, same visual vocabulary of skulls, hourglasses, and half-eaten food. If the positive-pair definition is "same artist," these are negatives. If the positive-pair definition is "same Iconclass branch," they are positives. If the hard-negative miner selects Heda as a hard negative for a Claesz anchor and the loss pushes them apart, the model is learning something wrong.

Mitigation strategies include: using metadata to filter out likely false negatives before mining (never use an item from the same school/period as a hard negative); maintaining a "safe negative" margin (only use hard negatives that are above a similarity threshold but below a higher threshold — excluding items so similar they might be positives); and the multi-positive approach described in the next section.

### Multi-positive contrastive: the program's specific variant

Standard contrastive learning has one positive per anchor. The Beowolff-Embed program uses **multiple positives per anchor**, drawn from the metadata graph, with hierarchical weighting. This is a significant departure from vanilla contrastive learning and deserves careful attention.

For a given anchor artwork, the positive set includes:

1. **Same artwork, different augmentation** — the standard self-supervised positive. Weight: highest.
2. **Same artist** — another work by the same artist. Weight: high.
3. **Same school** — another work from the same school/movement. Weight: medium.
4. **Same Iconclass branch** — another work with the same iconographic subject. Weight: lower.

The weights reflect the specificity of the relationship. Two works by the same artist share more than two works from the same school, which share more than two works with the same Iconclass label. The model should pull same-artist pairs closer together than same-school pairs, but all of them closer than random pairs.

How is this implemented? There are two common approaches:

**Weighted InfoNCE.** Modify the loss so that the positive term is a weighted sum over multiple positives:

$$\mathcal{L} = -\log \frac{\sum_{k \in \text{pos}} w_k \exp(\text{sim}(\mathbf{z}, \mathbf{z}_k) / \tau)}{\sum_{k \in \text{pos}} w_k \exp(\text{sim}(\mathbf{z}, \mathbf{z}_k) / \tau) + \sum_{j \in \text{neg}} \exp(\text{sim}(\mathbf{z}, \mathbf{z}_j) / \tau)}$$

where $w_k$ is the hierarchical weight for positive $k$. Same-artist positives get $w = 1.0$, same-school gets $w = 0.5$, same-Iconclass gets $w = 0.25$ (the exact weights are hyperparameters to be tuned).

**Separate loss terms.** Compute a separate contrastive loss for each level of the hierarchy and sum them with level-specific weights. This gives more control but is more expensive.

**What multi-positive buys.** Standard single-positive contrastive can only learn from one relationship at a time. If the positive is always "same artist," the model learns artist similarity but not stylistic or iconographic similarity. If positives rotate between relationship types across batches, the model gets conflicting signals (this batch says "Monet and Renoir are different"; next batch says "Monet and Renoir are similar because they're both Impressionists"). Multi-positive with hierarchical weighting resolves this: in every batch, the model sees all levels of similarity simultaneously, with appropriate weights. The geometry reflects a hierarchy — same-artist pairs are closer than same-school pairs, which are closer than same-Iconclass pairs, which are closer than random pairs.

**How this differs from standard practice.** Most contrastive learning systems in production (CLIP, SimCLR, DINO) use a single positive per anchor. Pinterest's ItemSage uses multi-positive contrastive with co-occurrence on boards as the positive signal, which is closer to what Beowolff proposes. The hierarchical weighting based on metadata-graph distance is, as far as I can tell, the program's specific contribution rather than an off-the-shelf method.

**The risk.** Hierarchical positives create a multi-scale optimization problem. If same-artist positives are weighted too heavily, the model collapses artist clusters — all Rothkos at one point, all Newmans at another — with no internal structure within an artist's oeuvre. If same-Iconclass positives are weighted too heavily, the model produces a coarse semantic map where all "female saints" are at one point regardless of artist or period. The weighting scheme determines the balance between specificity and generality in the embedding, and getting it wrong produces a model that is good at one level of the hierarchy but bad at others. This is tunable, but it is a large hyperparameter space — the interaction between hierarchy weights, temperature, batch size, and mining strategy produces a combinatorial optimization landscape that requires systematic ablation to navigate.

### Putting it all together: how a training step works

To make the full procedure concrete, here is what one training step looks like for the Beowolff-Embed system:

1. **Sample a batch** of 8,192 artworks from the corpus. For each artwork, look up its metadata: artist, school, period, Iconclass subjects.

2. **Construct positive sets.** For each artwork in the batch, find other artworks in the batch that share an artist, a school, or an Iconclass branch. Weight these relationships hierarchically.

3. **Encode each artwork** through the image tower (ViT-L/14, initialized from DINOv3), the entity tower (learned embeddings for artist/school/period/Iconclass, aggregated by a small transformer), and the text tower (biography and description text). These tower outputs are projected to a shared 768-d space and L2-normalized.

4. **Compute all pairwise similarities** within the batch. This is an 8,192 x 8,192 matrix of cosine similarities. On a GPU, this is a single matrix multiplication — fast, even at this scale.

5. **Evaluate the weighted InfoNCE loss** for each anchor, using its hierarchical positive set and all non-positive items in the batch as negatives.

6. **Backpropagate** the loss through all three towers. The gradients tell each tower how to adjust its parameters to make positives more similar and negatives less similar.

7. **Update parameters** via the optimizer (typically AdamW with cosine learning-rate schedule and warmup).

8. **Periodically** (every few thousand steps), re-embed a sample of the corpus with the current model, build a FAISS index, and mine hard negatives to supplement in-batch negatives for subsequent steps.

Each training step adjusts the geometry of the embedding space. Early in training, the adjustments are large — the model is learning basic category structure. Late in training, the adjustments are small — the model is refining fine-grained distinctions within categories. The temperature schedule, learning rate schedule, and hard-negative mining schedule are all coordinated to match this progression: start warm with random negatives (learn the easy structure), then cool with mined negatives (learn the hard structure).

### What to watch for when evaluating the program

The contrastive training pipeline has several failure modes that the program should explicitly monitor:

1. **Embedding collapse.** All items map to the same point (or a small number of points). Detected by monitoring the average pairwise distance across a validation set. If it drops toward zero, training has collapsed. Cause: temperature too low, learning rate too high, or insufficient diversity in positives.

2. **Uniformity without alignment.** Items spread evenly across the hypersphere but positives are not closer than negatives. The embedding has high uniformity (good) but low alignment (bad). Detected by separately measuring alignment (average distance between positives) and uniformity (KL divergence from uniform on the sphere). This framework, due to Wang & Isola (2020), is the right diagnostic for contrastive embeddings.

3. **Dimensional collapse.** The embedding uses only a few dimensions of the available 768, effectively reducing to a low-rank space. This is subtler than full collapse — items are spread out, but the effective dimensionality is much lower than 768. Detected by computing the rank of the embedding covariance matrix (or its effective rank via the eigenvalue spectrum). Caused by insufficient negative diversity or over-regularization.

4. **Hierarchy inversion.** Same-school pairs are closer than same-artist pairs, or Iconclass pairs are closer than same-school pairs. The hierarchical weighting is not producing the intended distance ordering. Detected by measuring average within-level distances across the validation set. Caused by incorrect weight ratios or insufficient same-artist pairs in the training data (rare artists contribute few same-artist positives).

Each of these failure modes is recoverable if detected early. The program's Phase 1 evaluation suite should include explicit checks for all four, measured at every checkpoint and recorded against the ledger state — both because they are diagnostically important and because the attribution mechanism requires that model quality be tracked with this granularity.

The next chapter covers how multiple modalities — not just images but text, entities, and eventually behavior — are aligned into the same space. That is where CLIP, SigLIP, and the program's structured-entity tower enter the picture.
