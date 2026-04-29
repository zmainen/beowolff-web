## Attribution: From Theory to Practice

### The question attribution answers

When a work sells on the platform, the revenue event triggers a question: who contributed to this sale, and how should the proceeds be distributed? The artist made the work. The buyer found it because the system surfaced it. The system surfaced it because the embedding placed it in a navigable neighborhood. The neighborhood exists because other artists' works defined that region of the manifold. The algorithm that trained the embedding was built by engineers, shaped by researchers, governed by policy decisions.

Each link in that causal chain has a different character, involves a different number of contributors, and calls for a different measurement method. The original Beowolff attribution whitepaper proposed a unified Shapley framework for all of it. This chapter argues — on the basis of both the theory and the practical literature — that a layered architecture is more honest and more implementable: Shapley values for one specific layer, simpler mechanisms for the rest, and a provenance ledger (Filix) as the shared audit trail.

### Cooperative game theory and the Shapley value

The starting point is a formal result from 1953. Lloyd Shapley considered a cooperative game: $n$ players form coalitions, and each coalition $S$ produces a value $v(S)$. The question is how to distribute $v(N)$ — the value of the grand coalition of all players — among the individual players.

Shapley proved that there is exactly one distribution rule satisfying four axioms:

- **Efficiency.** The shares sum to the total: $\sum_i \phi_i = v(N)$.
- **Symmetry.** If two players make identical marginal contributions in all coalitions, they receive equal shares.
- **Null player.** A player who contributes nothing to any coalition receives nothing.
- **Linearity.** If the game's value function is a sum of two component games, the attribution decomposes accordingly.

The unique solution is:

$$\phi_i(v) = \sum_{S \subseteq N \setminus \{i\}} \frac{|S|! \, (|N| - |S| - 1)!}{|N|!} \left[ v(S \cup \{i\}) - v(S) \right]$$

The term $v(S \cup \{i\}) - v(S)$ is the marginal contribution of player $i$ to coalition $S$: what changes when $i$ joins. The Shapley value averages this marginal contribution over all possible orderings in which the players might have arrived — equivalently, over all subsets $S$ that might have preceded $i$.

#### A worked example

Three artists contribute training corpora to a model: Monet (M), Hokusai (H), Picasso (P). There are $3! = 6$ orderings:

| Ordering | Monet's marginal | Hokusai's marginal | Picasso's marginal |
|:---------|:-----------------|:-------------------|:-------------------|
| M → H → P | $v(\{M\}) - v(\emptyset)$ | $v(\{M,H\}) - v(\{M\})$ | $v(\{M,H,P\}) - v(\{M,H\})$ |
| M → P → H | $v(\{M\}) - v(\emptyset)$ | $v(\{M,P,H\}) - v(\{M,P\})$ | $v(\{M,P\}) - v(\{M\})$ |
| H → M → P | $v(\{H,M\}) - v(\{H\})$ | $v(\{H\}) - v(\emptyset)$ | $v(\{H,M,P\}) - v(\{H,M\})$ |
| H → P → M | $v(\{H,P,M\}) - v(\{H,P\})$ | $v(\{H\}) - v(\emptyset)$ | $v(\{H,P\}) - v(\{H\})$ |
| P → M → H | $v(\{P,M\}) - v(\{P\})$ | $v(\{P,M,H\}) - v(\{P,M\})$ | $v(\{P\}) - v(\emptyset)$ |
| P → H → M | $v(\{P,H,M\}) - v(\{P,H\})$ | $v(\{P,H\}) - v(\{P\})$ | $v(\{P\}) - v(\emptyset)$ |

Each artist's Shapley value is the average of their column: their average marginal contribution across all orderings.

Notice what this buys over the simpler "leave-one-out" approach. Leave-one-out would only compute $v(N) - v(N \setminus \{i\})$ — the marginal contribution in one context, where everyone else is present. Shapley averages over all contexts. Monet's contribution to a model that already has 50 Impressionists (small — redundant) is weighted equally with Monet's contribution to a model with only Cubists and Ukiyo-e (large — opens a new region). The averaging is what makes the attribution "fair" in the axiomatic sense.

### Why leave-one-out might actually be enough

The standard argument against leave-one-out is context dependence: a contribution's value depends on what else is in the coalition. In classic Data Shapley — millions of training images, each contributing a tiny signal — this matters genuinely.

But the Beowolff sale-attribution problem has different structure. When work X sells, the question is not "what was every training point's contribution to overall model quality?" It is "what caused this specific sale?" That is a causal chain, not a coalition game. The chain has a small number of links, each involving a different kind of contributor. And for the one layer where Shapley is genuinely needed — manifold-shaping credit among corpora — the number of relevant contributors per region is small (tens, not millions), so even exact Shapley is tractable.

This is the key insight: the regime matters. Shapley's permutation-averaging is essential when you have many symmetric players making small contributions and you need to divide a pie among all of them fairly. It is overkill when you have a few large, structurally distinct contributors whose effects are measurable by simpler methods.

### The layered attribution architecture

Attribution decomposes into four layers, each with its own mechanism:

#### Layer 1: The work itself (property rights)

When work X sells, the artist who made X has a claim that is moral, legal, and contractual — not statistical. No computation is needed. This is the dominant share (~90% of the attribution pool for traditional sales). It is governed by the commission agreement between the artist, the gallery, and the platform.

This layer is trivial in the traditional case but becomes interesting in two extensions: system-assisted creation (where the platform's tools influenced the creative process) and system-generated works (where the system is a co-creator). We return to these below.

#### Layer 2: Manifold-shaping credit (corpus-level Shapley)

This is the layer where Shapley earns its keep. The question: which training corpora shaped the region of the embedding that made this sale's recommendation possible?

When a recommendation surfaces work X to a buyer, X's position in the embedding space determined that it appeared in the buyer's candidate set. That position exists because the model was trained on a corpus that includes X's neighborhood — other Impressionist landscapes, other works by the same school, other pieces from the same period and medium. If those corpora were removed, X might embed somewhere less coherent, and the recommendation might not have fired.

The relevant contributors are corpus blocks, not individual images. The number of corpora that shape any given neighborhood is small — typically 5 to 20 artist corpora, plus the museum collections that provide the dense background. At this scale, you can compute Shapley exactly (for 10 corpora: $2^{10} = 1024$ subsets, each requiring a model retrain or an influence-function approximation). You can also use leave-one-out at the corpus level as a reasonable approximation, since the context-dependence that makes leave-one-out biased at the image level is much weaker at the corpus level — removing one artist's 200 paintings from a training set of 690,000 is a small perturbation whose context-dependence is correspondingly small.

The **manifold-shaping credit** from the pitch deck is a special case of this layer: an artist whose corpus defines a region of the manifold receives Shapley credit when any work the model recommends from their region sells, because the model's ability to surface that region was shaped by their contribution. Artists get paid for the taste they teach the model, not just for their own inventory turning over.

**Droit de suite as a special case.** Several jurisdictions (France, EU, UK) have resale royalty laws — a fixed percentage of secondary-market sales flows to the artist or estate. Corpus-level Shapley attribution is a more principled generalization: it pays for measurable contribution to the system's recommendation capability rather than imposing a flat percentage. The Shapley share may be higher or lower than the statutory droit de suite rate, and it's grounded in a computation rather than a political negotiation.

#### Layer 3: Recommendation path (referral tracking)

When a sale follows a recommendation, there is a traceable path: the buyer browsed works A and B, the system recommended X based on the buyer's state, the buyer purchased X. The referring works — A and B — are part of the causal chain.

But the referring works are typically fungible. If the buyer was browsing Impressionist landscapes and the system recommended X (another Impressionist landscape), many other Impressionist landscapes could have served the same referral role. The credit belongs to the neighborhood, not to the individual referrer.

This layer is closer to an advertising referral system than to a Shapley problem. The mechanisms are well-understood: attribution windows, referral fees, multi-touch attribution models. The literature review (Chapter 9 appendix) surveys how advertising, affiliate marketing, and streaming platforms handle this.

#### Layer 4: Infrastructure (aggregate A/B testing)

Code, method documents, governance decisions, algorithm versions — these affect every sale, not one. Their contribution is aggregate: over a quarter, did revenue go up or down relative to a counterfactual where this code change or policy decision hadn't happened?

This is standard causal inference via randomized experiment. Run algorithm v7 and v8 side by side, measure conversion rates over weeks, attribute the difference. The number of infrastructure contributions is small (tens to hundreds of git commits, a handful of governance decisions) and their effects compound in ways that make per-sale factorization meaningless. A single git commit doesn't cause a single sale — it shifts the system's overall conversion rate by some amount, measurable only in aggregate.

The appropriate allocation is: a fraction of total platform revenue goes to infrastructure, sized by the aggregate A/B-tested contribution of infrastructure improvements over the measurement period. This fraction is a governance parameter, not a Shapley computation.

### Factoring difference and meaning

A contribution's value decomposes into two components:

- $\Delta_i$ — the **representational effect**: how much did contribution $i$ change the embedding? This is objective, computable, and shared across all stakeholders. Measured by representational similarity metrics: CKA (Centered Kernel Alignment — Kornblith et al. 2019), RSA (Representational Similarity Analysis — Kriegeskorte et al. 2008), or neighborhood-structure metrics like the fraction of k-nearest neighbors that change.

- $\omega$ — the **meaning function**: was the change valuable? This is perspective-dependent and context-dependent. Different stakeholders might apply different meaning functions to the same $\Delta$:

| Meaning function | What it measures | Who cares |
|:-----------------|:-----------------|:----------|
| Curatorial coherence | Do clusters match art-historical categories? | Curators, art historians |
| Recommendation quality | Did Recall@K improve? | Platform, engineers |
| Revenue | Did sales follow from the changed region? | Artists, investors |
| Discovery | Were previously invisible works surfaced? | Emerging artists, long-tail advocates |

Each meaning function, applied to the same $\Delta$, produces different Shapley values. This is a feature: it makes the evaluative choice explicit rather than burying it inside $v(S)$.

The architecture this implies: compute and store $\Delta$ once (expensive but objective). Compute Shapley values on demand for whatever meaning function $\omega$ the context calls for. The Filix ledger holds $\Delta$. The meaning function is a parameter of the query, not a property of the ledger.

**The political question is visible.** Who chooses $\omega$? That's a governance decision — and in the attribution framework, governance decisions are themselves attributable. The meaning function is a contribution whose effect on attribution can be measured. The recursion is productive, not circular: it means the system's value alignment is auditable, not hidden.

### The neuroscience bridge: credit assignment

The attribution problem maps directly to credit assignment in reinforcement learning — a problem the reader knows well.

In temporal-difference learning, an agent receives a reward at the end of a sequence of actions and must assign credit to the individual actions that produced it. The reward is analogous to the sale; the actions are analogous to the contributions (training data, algorithm, recommendations) that led to the sale. TD learning solves credit assignment by propagating value backward through the sequence: the value of a state is the expected future reward from that state, and each action's credit is the temporal difference — the change in expected value it produced.

The layered attribution architecture does something similar. The sale event is the reward signal. The recommendation path (Layer 3) is the action sequence. The manifold structure (Layer 2) is the state representation — the value function that determines which actions (recommendations) are available and how valuable each state (neighborhood) is. Infrastructure (Layer 4) is the learning algorithm itself — the meta-level that determines how the value function is updated.

Credit assignment in RL also faces the same decomposition problem: the value of an action depends on the policy (analogous to the algorithm version), the state representation (analogous to the embedding), and the reward function (analogous to the meaning function $\omega$). Trying to assign credit simultaneously across all three levels is intractable. The standard solution is to fix two levels and attribute at the third — which is exactly what the layered architecture does.

### System-generated works

The layered architecture shifts dramatically when the system participates in creation.

**Traditional sales.** Layer 1 dominates. The artist made the work that sold. Attribution is mostly property rights, with a small supplementary pool for manifold-shaping and infrastructure.

**System-assisted creation.** The artist used the embedding model as a creative tool — exploring the manifold to find gaps, getting feedback on positioning, generating variations. The final work is the artist's, but the system influenced the process. Layer 1 still dominates, but the platform's contribution is real — handled as a platform fee or a terms-of-service percentage, not as a per-sale computation. You cannot run the counterfactual (what would the artist have made without the tool?).

**System-generated works.** The system produces a curated composition, a style transfer, a generated image synthesizing elements from the training corpus. Now there is no single artist. Layer 1 disappears — or rather, the "artist" is the system, and the system is composed of contributions. The entire sale revenue (minus operating costs) enters the attribution pool. The relevant contributors are:

- **Training corpora** that shaped the generative model's output space. If the system generates an image in the Post-Impressionist region, the artists who defined that region have a direct claim. Manifold-shaping credit becomes the primary mechanism.

- **Method contributions.** The generative architecture, the training objective, the curation process, the prompt or exploration that led to this particular output. In a traditional sale, infrastructure is background. In a generative sale, method is foreground. These are testable by aggregate A/B: run the same corpus through method A and method B, measure the difference in output quality over time.

- **The factoring becomes essential.** For generative works, you cannot run per-instance counterfactuals. Removing Cézanne's corpus and regenerating doesn't produce a worse version of the same image — it produces a completely different image. The counterfactual doesn't produce a comparable object. Attribution must be aggregate: over a population of generative outputs, what is the measured effect of including versus excluding each corpus block?

| Sale type | Layer 1 (creator) | Layer 2 (manifold shapers) | Layer 3 (referral) | Layer 4 (infrastructure) |
|:----------|:------------------|:---------------------------|:-------------------|:-------------------------|
| Traditional | Artist: ~90% (contract) | Small pool: corpus-level Shapley | Referral tracking | Aggregate A/B, platform fee |
| System-assisted | Artist: reduced share (ToS) | Same as traditional | Same | Larger platform share |
| System-generated | No single creator — full pool | Primary mechanism | Same | Major share: method A/B |

### Data Shapley and its computational descendants

For the corpus-level Shapley computation in Layer 2, several methods are available. The choice depends on the number of corpora, the cost of retraining, and the required precision.

**Data Shapley** (Ghorbani & Zou, 2019). The foundational method. Treats model performance on a held-out set as $v(S)$, uses Monte Carlo sampling of random permutations to approximate the Shapley value. Convergence rate $O(1/\sqrt{T})$ where $T$ is the number of sampled permutations. For corpus-level attribution with 10-50 contributors, $T = 1000$ is typically sufficient.

**Beta-Shapley** (Kwon & Zou, 2022). Relaxes the efficiency axiom by weighting coalition sizes with a Beta distribution. Concentrates attention on coalition sizes where the signal is strongest — typically small coalitions, where adding one contributor has large marginal effect. Empirically, 2-5x better sample efficiency than standard Data Shapley.

**KNN-Shapley** (Jia et al., 2019). Exploits the structure of k-nearest-neighbor classifiers to compute exact Shapley in polynomial time. Directly relevant because our recommendation system is essentially k-NN in embedding space: find the $k$ nearest items to the user's state vector.

**Influence functions** (Koh & Liang, 2017). Approximate the leave-one-out effect using the model's gradient, without retraining. Cheap at query time. The approximation breaks down for highly non-convex loss landscapes (which deep networks have), but refinements — TracIn (Pruthi et al. 2020), TRAK (Park et al. 2023), DataInf (Kwon et al. 2023) — have extended the approach to large Transformers and contrastive models including CLIP.

**For Beowolff specifically:** at the corpus level (tens of contributors, not millions), the computation is tractable by any of these methods. The practical choice is probably TRAK or DataInf for the initial approximation (cheap, validated on CLIP-like models), with periodic full Monte Carlo Shapley as a calibration check.

### The Filix ledger as audit trail

The Filix ledger's role in the layered architecture is not "the thing that makes Shapley computable" — it is the audit trail that makes every layer of attribution verifiable.

What Filix records:

- **Training provenance.** Which corpora were in the training set for each model version. Which algorithm version. Which hyperparameters. Cryptographic hash of the training state, so any contributor can verify their data was (or was not) included.

- **Recommendation provenance.** For each recommendation, which model version served it, what the user's state was, what the candidate set was, and why this item was ranked where it was. Influence-function scores for the top-K attributions.

- **Sale events.** Which recommendation (if any) preceded the sale. The recommendation path. Timestamps.

- **A/B test results.** Which algorithm versions were tested, over what period, with what outcomes.

- **Attribution computations.** The Shapley values (or leave-one-out estimates) computed for each corpus block at each retraining. The meaning function used. The resulting distribution.

The Merkle-CRDT structure provides tamper evidence: any modification to a historical entry invalidates all subsequent entries. Contributors can verify their own history without seeing others'. The CRDT component handles concurrent writes from multiple agents.

The ledger's value is not computational — it is trust. A contributor considering whether to share their data with the platform needs to believe that their contribution will be tracked honestly and compensated proportionally. The ledger makes that belief verifiable rather than promissory.

### What the literature supports and what remains novel

**Well-established:** Shapley values are the uniquely fair attribution mechanism (Shapley 1953). They can be applied to ML data valuation (Ghorbani & Zou 2019). Efficient approximations exist at scale (TRAK, DataInf, Beta-Shapley). Influence functions work for Transformers (Schioppa et al. 2022).

**Adjacent but not directly precedented:** Shapley attribution for contrastive models (Deng et al. 2024 handles non-decomposable losses). TRAK validated on CLIP (Park et al. 2023). Data valuation for diffusion models trained on art (Lin et al. 2024).

**Genuinely novel in this proposal:**
- Using embedding geometry (CKA, neighborhood structure) as the Shapley value function rather than classifier accuracy.
- Decomposing attribution into representational effect ($\Delta$) and meaning ($\omega$).
- The layered architecture: property rights + corpus-level Shapley + referral tracking + aggregate A/B, rather than unified Shapley.
- Extending attribution to system-generated works where per-instance counterfactuals don't exist.

**What remains risky:** The signal-to-noise ratio of individual corpus-level attribution (Nguyen et al. 2023 found that individual training point influence is often below stochastic noise — corpus-level attribution should be above this threshold but needs empirical validation). The sensitivity to hyperparameters (Wang et al. 2025). The gap between attribution that is approximately correct and attribution that is credible enough to underpin revenue distribution.

### Designing the validation

The right way to test whether corpus-level attribution works is a controlled experiment on a toy problem where the answers are known. The design:

Take 1,000 artworks across 10 clearly distinct styles (100 each). Train an embedding. The manifold should show 10 clean clusters.

**Experiment 1: Remove a style.** Pull all Ukiyo-e works, retrain. The manifold loses a cluster. Shapley value of the Ukiyo-e corpus should be high.

**Experiment 2: Remove one work from a dense cluster.** Pull a single Monet from 100 Impressionists. Shapley value should be near zero — redundant contribution.

**Experiment 3: Add a bridging artist.** Cézanne between Impressionism and Cubism. Shapley value should be high relative to a generic Impressionist — structurally non-redundant.

**Experiment 4: Noisy data.** Add 50 photographs to the Impressionism cluster. Shapley value should be negative — harmful contribution.

**Experiment 5: The meaning function.** Apply different $\omega$ to the same $\Delta$. Under "curatorial coherence," Ukiyo-e is highly valuable. Under "recommendation accuracy for a Western collector who never browses Ukiyo-e," the same corpus has near-zero value. Same difference, different Shapley values.

At 10 corpus-level contributors, exact Shapley requires $2^{10} = 1024$ model evaluations — tractable. The toy validates the framework before production deployment.
