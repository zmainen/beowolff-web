## 13. Where the Proposal Could Be Wrong

This is the most important chapter in the guide. Most ML proposals fail not because the architecture is wrong but because an assumption upstream of the architecture turns out to be false. The architecture can be elegant, the math can be correct, and the system can still fail because the data does not behave as assumed, or the metric does not measure what matters, or a constraint that seemed manageable proves intractable.

The program makes at least seven testable assumptions that could break. This chapter names each one, explains what failure would look like, describes the early warning sign, and proposes a mitigation. For each, it is honest about whether the mitigation is sufficient.

The goal is not to undermine the program. It is to make its risk profile visible and discussable. After reading this chapter, you should be able to say: "I accept risk X because mitigation Y is credible, but I am not willing to accept risk Z because mitigation W is insufficient, and here is what I would need to see before proceeding."

### 13.1 Multi-head loss instability

**The assumption.** The program trains a single embedding space against five heterogeneous loss terms: image contrastive, entity contrastive, text contrastive, behavioural prediction, and price regression. The assumption is that these losses can coexist productively — that a single embedding space can simultaneously encode visual similarity, curatorial relationships, textual semantics, user preferences, and market value.

**What failure looks like.** Multi-objective optimisation is hard. Each loss term produces gradients that pull the shared parameters in a different direction. When the gradients are aligned (all losses want the same parameter update), training proceeds smoothly. When they conflict (the image loss wants to push two works apart because they look different, but the behaviour loss wants to pull them together because the same users like both), the optimiser must compromise, and the compromise is often mediocrity — a model that is slightly good at everything and excellent at nothing.

The problem is compounded by gradient magnitude imbalance. Different loss terms produce gradients of vastly different scales. A regression loss on log-price percentile might produce gradients 100x larger than a contrastive loss on Iconclass subjects, simply because the loss landscapes have different curvatures. Naive weighting (equal coefficients on all loss terms) lets the largest-gradient loss dominate training, effectively ignoring the smaller-gradient losses.

**Early warning sign.** Validation metrics diverging across modalities during training. Specifically: if image retrieval quality improves while entity retrieval quality degrades (or vice versa) over the course of training, the losses are in conflict and the model is trading off between them rather than jointly improving. Track per-modality metrics on a shared validation set at every checkpoint. Plot them on the same time axis. If the curves cross, you have a problem.

A subtler warning: the model achieves reasonable aggregate metrics but performs poorly on cases where modalities disagree — works that look different but share deep curatorial connections, or works that look similar but belong to completely different traditions. These cross-modal edge cases are where loss conflicts manifest most acutely.

**Mitigation.** Three techniques, in order of complexity:

1. *GradNorm* (Chen et al., 2018). Dynamically adjust the weight of each loss term during training so that all tasks train at the same rate. The insight: rather than weighting losses equally, weight them so that the *gradient norms* are equal. This prevents any single loss from dominating the gradient signal. GradNorm adds a small overhead (computing per-task gradient norms at each step) but is well-tested in multi-task learning.

2. *Progressive modality addition.* Do not start with all five losses. Train on image + entity first (Phase 1). Add text. Add behaviour (Phase 2). Add price (Phase 3). At each stage, the model has already learned a stable base representation from the earlier modalities, and the new modality's loss is adjusting a representation that is already useful rather than fighting with other modalities for control of a random initialisation. This is the program's phased architecture, and it is not just a project-management convenience — it is a mitigation strategy for loss instability.

3. *Ablation budget.* Reserve compute for systematic ablation studies: train with each subset of losses and measure the marginal contribution of each. If adding the price loss degrades image retrieval by more than it improves price-aware queries, the price loss may need a smaller weight, a different functional form, or removal. The ablation budget is an investment in understanding the loss landscape before committing to the final configuration.

**Is this sufficient?** Probably, if the progressive addition strategy is followed faithfully. Multi-task learning with 2-3 related tasks is well-established; with 5 heterogeneous tasks it is less proven, but the phased approach decomposes the problem into manageable stages. The key risk is impatience — skipping ahead to all-five-losses training before the earlier phases are stable. Recommendation: do not add a new loss term until the existing loss terms are jointly stable for at least one full training cycle.

### 13.2 Cold-start metric noise

**The assumption.** The cold-start metric (Recall@K on items with no behavioural history) is the program's stated differentiator — "the cold-start metric is the differentiator and where biography text and entity structure earn their keep." The assumption is that this metric can be measured with sufficient precision to guide architectural decisions and demonstrate competitive advantage.

**What failure looks like.** Cold-start evaluation sets are small by definition. A "cold-start item" is one with no behavioural history, which means it was recently added or rarely interacted with. In a temporal evaluation split (train on data before $T$, evaluate on data after $T$), the cold-start eval set consists of items that first appeared after $T$. Depending on the platform's growth rate and the time window, this set might be a few hundred to a few thousand items.

Recall@10 on a set of 500 items is a noisy metric. A single item moving in or out of the top-10 retrieval changes the metric by 0.2%. Architectural decisions that appear to improve cold-start recall by 5% may be within the noise band.

**Early warning sign.** High variance across evaluation splits. Run the same evaluation with different values of $T$ (different temporal split points) and compute the standard deviation of the cold-start metric. If the standard deviation is comparable to the effect sizes you are trying to detect, the metric is too noisy to be informative.

**Mitigation.**

1. *Bootstrap confidence intervals.* For every cold-start metric report, compute bootstrap confidence intervals (resample the cold-start eval set with replacement, recompute the metric 1000 times, report the 95% CI). This makes the noise visible. Do not report a point estimate without its CI.

2. *Minimum eval-set sizes committed in advance.* Define a minimum cold-start eval set size below which the metric is not considered reliable. A reasonable rule of thumb: Recall@10 on $n$ items has a standard error of approximately $\sqrt{p(1-p)/n}$ where $p$ is the true recall rate. For $p = 0.3$ and a desired standard error of 0.01, you need $n \geq 2100$. If the cold-start eval set is smaller than this, either extend the evaluation window or supplement with synthetic cold-start items (warm items with their behavioural data masked).

3. *Paired comparisons.* When comparing two models on cold-start performance, use a paired test — evaluate both models on exactly the same cold-start items and test whether the per-item differences are significant. This controls for item-level variance and is more powerful than comparing aggregate metrics.

**Is this sufficient?** Yes, if the discipline is followed. The risk is not that the problem is unsolvable — it is that in the pressure of development, noisy metrics get treated as reliable and architectural decisions get made on insufficient evidence. The mitigation is procedural: commit to the statistical discipline before you need the results.

### 13.3 Biography text quality

**The assumption.** The program assumes that artist biographies from Wikipedia and Artwiki will provide useful signal for the entity tower, following Spotify's precedent of using music journalism to solve cold-start. The assumption is that biography text carries discriminative information — that two artists with different biographies will occupy different regions of the embedding space, and that these regions will correlate with meaningful aesthetic and contextual differences.

**What failure looks like.** Artist biographies vary enormously in quality, length, and informativeness. A well-known artist like Rembrandt has a 15,000-word Wikipedia article covering training, influences, technique, historical context, major works, and critical reception. An emerging contemporary artist might have a 200-word stub — or no Wikipedia article at all. Artwiki coverage is better for some periods and regions but has its own gaps.

When biography quality is bimodal (rich for famous artists, sparse for obscure ones), the biography tower learns to discriminate among famous artists but provides no signal for the artists who need it most — the cold-start cases. The Spotify analogy breaks here: music journalism covers emerging artists relatively quickly (a new album gets reviewed within weeks), while art criticism covers emerging artists slowly or not at all.

**Early warning sign.** In ablation studies, the biography tower's contribution to cold-start Recall@K is near zero. Specifically: compare the full model (image + entity + biography) against the model without biography (image + entity only) on the cold-start eval set. If the difference is not statistically significant (using the paired comparisons from Section 13.2), the biography signal is not carrying its weight.

A subtler warning: the biography tower's contribution is positive in aggregate but driven entirely by famous artists — the biography helps discriminate among Rembrandts and Vermeers but not among unknown contemporary painters. Stratify the ablation by artist fame (measured by, e.g., number of works in the training corpus or Wikipedia article length). If the biography effect disappears for the bottom quartile, the feature is not solving the problem it was designed to solve.

**Mitigation.**

1. *Biography quality scoring.* Compute a quality score for each biography — length, lexical diversity, presence of key information fields (birth date, training, medium, influences, exhibitions). Use this score as a modality-confidence weight during training: full weight for rich biographies, reduced weight for stubs, zero weight for absent biographies. This prevents noisy or uninformative biographies from degrading the entity embedding.

2. *Fallback to structured metadata.* For artists without useful biographies, the entity tower should rely on structured metadata alone (period, medium, school, exhibition history, gallery affiliations). The architecture already supports this — the structured-entity tower operates independently of the biography tower — but the training procedure should explicitly verify that the entity embedding for a biography-absent artist is at least as good as the purely metadata-derived embedding.

3. *Supplementary sources.* Gallery websites, exhibition catalogues, artist statements, auction-house lot descriptions. These are not as standardised as Wikipedia articles but may provide signal for artists that Wikipedia misses. The engineering cost of ingesting heterogeneous text sources is non-trivial, and the quality variance is high, but for the specific problem of cold-start biography coverage, the marginal value could be significant.

**Is this sufficient?** Partially. The quality scoring and fallback mechanisms are straightforward and should prevent the biography tower from actively hurting performance. But they do not solve the underlying problem: for the artists who most need cold-start help, biography text may simply not exist. The honest answer is that biography text will help for some fraction of cold-start cases (artists with decent Wikipedia coverage but no behavioural history) and will not help for others (emerging artists with no public text footprint). The structured-entity tower — period, medium, school, Iconclass — is the true cold-start backbone, and the biography tower is supplementary.

### 13.4 Behaviour data sparsity

**The assumption.** The program assumes "millions of users" with dense enough interaction histories to train a two-tower model with a transformer-based user tower. The assumption is that art browsing generates enough behavioural signal per user to learn a meaningful taste representation.

**What failure looks like.** Art browsing is sparser than music listening. A Spotify user might play 50 songs a day; a Pinterest user might save 20 pins per session. An art platform user might browse 10-20 artworks in a session and visit the platform once a week. Many users will have total interaction histories of fewer than 50 events. A transformer-based user tower trained on 50-event sequences is learning from very little data per user.

The consequence is a heavy-tailed user distribution: a small number of power users (collectors, dealers, curators) generate dense interaction histories, while the majority of users generate sparse histories. If the user tower is trained without accounting for this distribution, the gradients will be dominated by power users, and the model will learn representations that are excellent for active collectors and useless for casual browsers. This is the opposite of the program's goal.

**Early warning sign.** User-tower gradient analysis reveals that a small fraction of users (top 5-10% by interaction count) contribute most of the gradient magnitude. Equivalently: the user-tower's recommendation quality, stratified by user activity level, shows a steep dropoff — high Recall@K for active users, near-random for sparse users. If the model is effectively ignoring the majority of users, the behaviour tower is not generalising.

**Mitigation.**

1. *Minimum-interaction thresholds.* Define a minimum number of interactions below which a user is classified as "cold" and served by the content tower alone (image + entity + biography), without the behaviour tower. This is not a failure — it is the designed fallback. The content tower exists precisely for cases where behavioural signal is absent or insufficient.

2. *Frequency-weighted sampling.* During training, up-weight sparse users and down-weight dense users so that the effective training distribution is more uniform across activity levels. This is the user-side equivalent of the program's inverse-popularity weighting for items. The weighting prevents power users from dominating the gradient and forces the model to learn from sparse interaction histories.

3. *Session-level rather than history-level modelling.* For sparse users, a single session might be all the data available. The user tower should be able to produce a useful embedding from a single session of 5-10 interactions, not only from a full history of hundreds. Architecturally, this means the transformer should work well on short sequences — which requires appropriate positional encoding and attention patterns. PinnerSage's session-based approach is directly relevant here.

4. *Content-behaviour interpolation.* For users with intermediate activity levels (enough to have some behavioural signal but not enough for the user tower to be confident), interpolate between the content-only recommendation and the behaviour-based recommendation. Weight the interpolation by a confidence measure — e.g., the entropy of the user-tower's output distribution, or simply the number of interactions.

**Is this sufficient?** Yes, if the minimum-interaction threshold and the content-tower fallback are implemented faithfully. The risk is not that sparse users will get bad recommendations — the content tower provides a reasonable baseline. The risk is that the *incremental value* of the behaviour tower over the content tower is small for the majority of users, which would mean the two-tower architecture adds substantial complexity for limited benefit. The early test is: at what interaction count does the two-tower model begin to outperform the content-only model? If the answer is "50+ interactions" and 80% of users have fewer than 50, the two-tower architecture is serving only 20% of users. This is a legitimate architectural outcome, but it should be discovered through evaluation, not assumed away.

### 13.5 Attribution noise at scale

**The assumption.** The attribution framework requires Shapley values precise enough to support actual revenue distribution. With millions of data contributions, the Monte Carlo Shapley estimates for individual contributions must be reliable enough that contributors receive payments proportional to their actual marginal contributions, not proportional to estimation noise.

**What failure looks like.** Monte Carlo Shapley approximation converges at rate $O(1/\sqrt{T})$ where $T$ is the number of sampled permutations. For $T = 100{,}000$ permutations and $n = 1{,}000{,}000$ contributions, the variance of each individual Shapley estimate depends on the variance of the marginal contributions across permutations. In many data-valuation settings, the marginal contribution of a single data point is small (order $1/n$) and variable, which means the confidence interval around the Shapley estimate can be wider than the point estimate itself.

Concretely: if a contributor's true Shapley value is \$0.50 per month and the 95% confidence interval is [\$0.10, \$0.90], the attribution is not meaningful at the individual level. The contributor might be worth five times as much as the estimate says, or one-fifth as much. Distributing revenue based on these estimates is not "fair attribution" — it is noisy attribution with a fairness story.

**Early warning sign.** Compute confidence intervals for individual Shapley estimates. If the median ratio of confidence-interval width to point estimate exceeds 1.0 (the CI is wider than the estimate), individual attribution is unreliable. This should be tested on a synthetic dataset of known composition before deploying on production data — generate a dataset where the true Shapley values are known analytically, run the Monte Carlo approximation, and measure how many permutations are needed for the estimates to converge within a tolerable error.

**Mitigation.** The study guide's Chapter 9 covered the basic approximation methods. The key mitigation for the noise problem is architectural — changing what level of granularity attribution operates at:

1. *Corpus-level rather than image-level attribution.* Instead of computing a Shapley value for each individual artwork, compute Shapley values for each *contributor* (artist, gallery, data provider). A contributor's Shapley value is the marginal contribution of their entire corpus, not of each individual work. Since contributor-level marginal contributions are larger (an artist's 500 works collectively have a much larger and more stable marginal effect than any single work), the Monte Carlo estimates converge faster and have tighter confidence intervals.

2. *Layered attribution architecture.* Use a coarse-to-fine approach. First, compute contributor-level Shapley values (tractable, reliable). Then, within each contributor's corpus, use influence functions (Koh & Liang, 2017) to estimate per-item contributions — these are less precise but are used only for informational purposes (showing an artist which of their works contributed most), not for revenue distribution. Revenue flows at the contributor level; item-level breakdowns are informational.

3. *Beta-Shapley concentration.* Use the Beta-Shapley variant (Kwon & Zou, 2022), which concentrates sampling on small coalitions where individual marginal contributions are larger and more measurable. This improves sample efficiency — fewer permutations needed for the same precision — at the cost of a slight bias in the estimate.

4. *Minimum-attribution thresholds.* Define a minimum Shapley value below which attribution is pooled into a collective fund rather than distributed individually. This acknowledges that very small contributions cannot be attributed reliably and avoids the appearance of precision where none exists.

**Is this sufficient?** At the contributor level, probably yes. Contributor-level Shapley values are an order-of-magnitude more stable than per-item values because the marginal contributions are larger and the coalition space is smaller (thousands of contributors vs. millions of items). At the individual-artwork level, the honest answer is that precise Shapley attribution is likely not achievable with current methods at million-item scale. The layered architecture — revenue at the contributor level, informational breakdown at the item level — is a principled compromise. The risk is communicational: if the system promises "we track every artwork's contribution and pay accordingly" but actually distributes at the contributor level, contributors may feel the promise was overstated. Transparency about the granularity of attribution is essential.

### 13.6 Reproducibility as constraint vs. aspiration

**The assumption.** The program requires "same ledger state produces the same model" for attribution to work. This is because Shapley computation requires evaluating $v(S)$ — the model's performance when trained on subset $S$ — and this evaluation must be deterministic. If training on the same data twice produces different models with different performance, then $v(S)$ is a random variable rather than a function, and the Shapley values inherit the noise.

**What failure looks like.** Chapter 11 (Section 11.2) catalogued the sources of irreproducibility: GPU non-determinism, floating-point non-associativity, multi-GPU communication order, library version drift. In practice, two training runs on the same data with the same configuration will produce models that agree on most predictions but diverge on boundary cases. The divergence is typically small — model performance within $\pm 0.1-0.5\%$ — but it is real and irreducible without substantial performance penalties.

For Shapley computation, the question is: if $v(S)$ varies by $\pm 0.3\%$ across replicate runs, are the resulting Shapley values reliable? The answer depends on the magnitude of individual marginal contributions relative to the noise. If contributor A's marginal contribution is 2% and the training noise is 0.3%, the attribution is reliable (signal-to-noise ratio $\approx 7$). If contributor B's marginal contribution is 0.05% and the training noise is 0.3%, the attribution is dominated by noise.

**Early warning sign.** Retrain the model from the same data twice (with the same configuration but independent GPU execution) and compute Shapley values on both models. Compare the Shapley rankings. If the rank correlation (Spearman's $\rho$) is below 0.9, reproducibility noise is affecting attribution meaningfully. If individual contributors change by more than $2\times$ between runs, the attribution is unreliable at the individual level.

**Mitigation.**

1. *Define "reproducible" as "within tolerance $\varepsilon$."* Rather than demanding bitwise reproducibility, define a performance tolerance (e.g., $\varepsilon = 0.1\%$ in Recall@10) and verify empirically that replicate training runs stay within it. This shifts the standard from an impossibility (exact reproduction) to a testable claim (bounded deviation).

2. *Pin the tolerance before committing to attribution-grade claims.* The tolerance $\varepsilon$ should be chosen based on the magnitude of marginal contributions you need to detect. If the smallest contributor-level marginal contribution you care about is $\delta$, then $\varepsilon$ must be substantially smaller than $\delta$. If it is not, the attribution system cannot reliably distinguish that contributor from noise.

3. *Ensemble over replicate runs.* Average the Shapley values computed from $k$ replicate training runs. This reduces the noise by $\sqrt{k}$ at the cost of $k\times$ training compute. For $k = 5$, noise is reduced by $\sim 2.2\times$. This is expensive but may be necessary for high-stakes attribution decisions.

4. *Use deterministic GPU operations where feasible.* Accept the performance penalty of `torch.use_deterministic_algorithms(True)` for the specific training runs used in Shapley computation, even if production training uses non-deterministic operations for speed. This creates a separate "attribution-grade training mode" that is slower but more reproducible.

**Is this sufficient?** Conditionally. The tolerance-based approach is sound in principle — it converts an impossibility into an engineering constraint. But the constraint must be validated empirically, and the validation is expensive (multiple replicate training runs). The risk is that the tolerance needed for reliable attribution is tighter than what the training infrastructure can achieve without deterministic GPU mode, and deterministic GPU mode is too slow for the retraining cadence Shapley computation requires. If this tradeoff binds — if you cannot retrain often enough, deterministically enough, for Shapley estimates to converge — the attribution system needs to operate at a coarser granularity (fewer, larger retraining runs with higher per-run precision).

### 13.7 The meaning function is a political choice

**The assumption.** This is not a technical risk but a governance risk, and it may be the most consequential of the seven. The attribution framework computes Shapley values against a value function $v(S)$ — a function that maps a subset of contributions to a real-valued measure of performance. The program specifies Recall@K stratified by item warmth as the value function. But what counts as "performance" is not a neutral question. It is a political choice that different stakeholders will answer differently.

**What failure looks like.** Consider three stakeholders:

- *A gallery* wants the value function to weight sales revenue heavily. An artwork that drives a \$50,000 sale should contribute more to attribution than an artwork that drives a \$500 sale. The value function should be revenue-weighted Recall@K.

- *An emerging artist* wants the value function to weight discovery. An artwork that introduces 100 new users to the artist's oeuvre is more valuable — to the artist — than one that produces a single high-value sale. The value function should be diversity-weighted engagement.

- *A data provider* (say, an auction house contributing price records) wants the value function to weight prediction accuracy. Their contribution's value is in improving the model's price estimates, not in driving recommendations. The value function should include pricing accuracy as a component.

These three value functions will assign different Shapley values to the same contributions. The gallery's artwork is more valuable under the revenue-weighted function; the emerging artist's work is more valuable under the discovery-weighted function; the auction house's data is more valuable under the pricing-accuracy function. The choice of value function is not a technical parameter — it is a decision about whose interests the attribution system prioritises.

**Early warning sign.** Stakeholder complaints that the attribution system undervalues their contributions. More precisely: if different stakeholder classes consistently receive Shapley values that diverge from their self-assessed contribution, the value function does not reflect their understanding of "value." This is not a bug to be fixed but a tension to be governed.

**Mitigation.**

1. *Transparency about the choice.* The value function should be documented, justified, and published. Contributors should know what metric their Shapley values are computed against and why. "We use Recall@10 stratified by item warmth because it measures recommendation quality across both popular and obscure works" is a defensible position. Stating it explicitly invites challenge; not stating it invites suspicion.

2. *Multi-metric attribution.* Compute Shapley values against multiple value functions and report all of them. Contributor X has Shapley value $\phi_1$ under revenue-weighted performance and $\phi_2$ under discovery-weighted performance. The revenue distribution can be a weighted combination, with the weights set through governance rather than technical fiat.

3. *Governance mechanism for the weights.* The program file already identifies the attribution pool fraction as a "constitutional parameter, amendable through the platform's governance process." The value function — and the weights across multiple value functions — should receive the same treatment. This is not a parameter to be optimised; it is a political settlement to be negotiated.

4. *Decomposable attribution.* The Shapley linearity axiom (Chapter 9) guarantees that if the value function decomposes as $v = \alpha v_1 + \beta v_2 + \gamma v_3$, the Shapley values decompose accordingly: $\phi_i(v) = \alpha \phi_i(v_1) + \beta \phi_i(v_2) + \gamma \phi_i(v_3)$. This means the system can compute Shapley values against each component value function independently and combine them with governance-determined weights. The decomposition is not just mathematically convenient — it makes the political choice inspectable. Each stakeholder can see their attribution under each component and understand how the weighting affects them.

**Is this sufficient?** Technically yes — the decomposition and multi-metric approach are clean implementations of transparent governance. Whether it is sufficient *politically* depends on whether the stakeholders accept the governance process. The deeper risk is that the system promises "fair attribution" and delivers a specific political settlement disguised as mathematical necessity. Shapley values are uniquely fair *given a value function* — but the choice of value function is prior to the mathematics and carries all the political weight. If the system presents the value function as a technical parameter rather than a governance choice, it will face legitimacy challenges when stakeholders disagree.

The strongest position is to be explicit: "The Shapley value guarantees fairness given a definition of value. We cannot tell you what value means — that is a question for governance. What we can tell you is that once you agree on what value means, the attribution is the uniquely fair distribution. And we can show you exactly how changing the definition changes the attribution." This is honest, defensible, and — crucially — it makes the program's attribution mechanism a tool for negotiation rather than a substitute for it.

---

### Summary of risks

| Risk | Severity | Mitigation quality | Monitor via |
|:-----|:---------|:-------------------|:------------|
| Multi-head loss instability | Medium | Good (progressive addition, GradNorm) | Per-modality validation curves |
| Cold-start metric noise | Medium | Good (bootstrap CI, minimum eval sizes) | CI width vs. effect size |
| Biography text quality | Medium | Partial (quality scoring helps; coverage gap remains) | Ablation stratified by artist fame |
| Behaviour data sparsity | High | Good (content-tower fallback, thresholds) | Two-tower vs. content-only by activity level |
| Attribution noise at scale | High | Adequate at contributor level; insufficient at item level | Shapley CI width vs. point estimate |
| Reproducibility as constraint | Medium | Conditional (tolerance-based, needs validation) | Spearman $\rho$ of Shapley rankings across replicate runs |
| Value function as political choice | High | Technically good; politically depends on governance | Stakeholder divergence in self-assessed vs. computed value |

The two highest-severity risks — behaviour data sparsity and the value function's political nature — are also the two that are least amenable to purely technical solutions. The first requires the platform to build a large, engaged user base before the behaviour tower can justify its complexity. The second requires a governance process that the program has correctly identified as needed but not yet designed. Both are solvable, but neither is solvable by the ML engineering team alone. They require product decisions and institutional design, respectively.

The most important thing this risk analysis reveals is not any single risk but a pattern: the program's most ambitious claims — attribution-grade training, Shapley-based revenue distribution, right-to-delete that propagates — are all technically feasible but operate near the boundary of what current methods can achieve at production scale. Each one works in a controlled setting. Whether they work together, simultaneously, at the scale the program targets, with the noise levels the real world introduces, is a question that can only be answered by building the system and measuring. The phased architecture (Phases 0-4) is the right approach because it sequences the risks: you learn whether the content tower works before you add behaviour, and you learn whether behaviour works before you add attribution. Each phase is a test of the assumptions the next phase depends on. Trust that sequencing.
