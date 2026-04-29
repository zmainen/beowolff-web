## 13. What Could Go Wrong (Honestly)

This is the most important chapter. Most technology proposals fail not because the architecture is wrong but because an assumption upstream of the architecture turns out to be false. The math can be correct, the engineering can be sound, and the system can still fail because the data does not behave as expected, or the metrics do not measure what matters, or a constraint that seemed manageable proves intractable.

This chapter names seven risks. For each: what it means for you, what the early warning sign is, and what the mitigation is. Some of the mitigations are strong. Some are not. We say which.

### 1. The model might be mediocre at everything

**The risk.** The system trains a single model against five different objectives simultaneously: visual similarity, curatorial relationships, textual meaning, user preferences, and market value. Each objective pulls the model in a different direction. When the objectives agree (a Monet water lily should be near other Impressionist landscapes by every measure), training proceeds smoothly. When they conflict (two works look different but the same collectors buy both), the model must compromise. The compromise can be mediocrity -- slightly good at everything, excellent at nothing.

**What this means for you.** Recommendations that are adequate but uninspired. The model surfaces works that are vaguely related rather than precisely matched. The "I already knew about that" problem -- the system tells you things you could have found on your own.

**Early warning sign.** During development, the team tracks performance on each objective separately. If improving visual similarity degrades user-preference prediction (or vice versa), the objectives are in conflict. The curves should all improve together, or at worst hold steady while one improves. If they diverge, the model is trading off between them.

**Mitigation.** The system is designed to add objectives progressively, not all at once. Phase 1 trains on visual and curatorial signals only. Phase 2 adds user behavior. Phase 3 adds market data. At each stage, the model has a stable foundation before the next objective is introduced. This is not just a project-management convenience -- it is a deliberate strategy to avoid the multi-objective conflict. Additionally, techniques exist to dynamically balance the objectives during training so that no single one dominates.

**Is this sufficient?** Probably, if the progressive approach is followed faithfully. The risk is impatience -- trying to add everything at once before the earlier stages are stable.

### 2. The cold-start metric might be too noisy to trust

**The risk.** The system's stated differentiator is its ability to recommend works that have no interaction history -- new works by unknown artists, recently deposited pieces. The metric that measures this (how often the model correctly retrieves cold-start works) is computed on a small evaluation set. Cold-start works are, by definition, rare in the data. A small evaluation set means a noisy metric -- measured improvements might be statistical fluctuations rather than real gains.

**What this means for you.** If the cold-start metric is unreliable, the team might believe the biography text and entity structure are working when they are not, or might abandon a promising approach because a noisy measurement showed no effect.

**Early warning sign.** High variability when the same measurement is repeated under slightly different conditions. If the metric swings by five percentage points depending on which week's data is used as the evaluation set, it cannot support architectural decisions.

**Mitigation.** Statistical discipline: report confidence intervals, not point estimates. Define minimum evaluation set sizes below which the metric is considered unreliable. Compare models on exactly the same set of cold-start works (paired comparisons) rather than on aggregate numbers. These are standard practices in experimental science. The risk is not that the problem is unsolvable but that under development pressure, noisy metrics get treated as reliable.

**Is this sufficient?** Yes, if the discipline is followed. This is a procedural risk, not a fundamental one.

### 3. Artist biographies might be too low-quality to help

**The risk.** The system assumes that Wikipedia and Artwiki biographies carry useful information about artists -- their training, influences, medium, historical context. For well-known artists, this is true: Rembrandt has a 15,000-word Wikipedia article covering everything a model could want. For emerging contemporary artists -- the ones who most need cold-start help -- the biography might be a 200-word stub, or it might not exist at all.

**What this means for you.** The biography signal works for artists who do not need it (famous artists already discoverable through other channels) and fails for artists who do (emerging artists invisible to conventional recommendation).

**Early warning sign.** In testing, the biography signal improves recommendations for well-known artists but has no measurable effect for obscure ones. The improvement is real but concentrated where it matters least.

**Mitigation.** Quality scoring: weight biographies by how informative they are, so that uninformative stubs do not introduce noise. Supplementary sources: gallery websites, exhibition catalogues, artist statements, auction-house lot descriptions. These are less standardised but may provide signal for artists that Wikipedia misses. Fallback to structured metadata: for artists with no useful biography, the system relies on structured information (period, medium, school, exhibition history) which exists more consistently.

**Is this sufficient?** Partially. Quality scoring prevents harm (bad biographies will not make things worse). But for the artists who most need cold-start help, the text may simply not exist. The structured-entity information -- period, medium, school -- is the true cold-start backbone. The biography is supplementary. This is an honest limitation, not a failure of design.

### 4. Users might not browse enough to train the behavior model

**The risk.** The system's behavior model learns from user interaction histories -- what you browsed, saved, followed, purchased. Art browsing is sparser than music listening or social media scrolling. A Spotify user might play fifty songs a day. A collector on an art platform might browse twenty works in a session and visit once a week. Many users will have total interaction histories of fewer than fifty events.

**What this means for you.** The behavior model works well for power users (active collectors, dealers, curators) who generate dense histories, and poorly for casual browsers who generate sparse ones. If the majority of users are casual browsers, the behavior model is serving only a minority.

**Early warning sign.** Recommendation quality, measured by user activity level, shows a steep drop-off. Active users get excellent recommendations; casual users get recommendations no better than browsing the collection at random. The behavior model is effectively ignoring most of its audience.

**Mitigation.** The system is designed with a fallback. Users with insufficient behavioral data are served by the content model alone -- recommendations based on visual similarity, curatorial relationships, and biography text, without any behavioral signal. This is not a degraded mode; it is the designed cold-start path for users as well as for artworks.

For users with some but not much activity, the system can blend the content-based and behavior-based recommendations, weighting toward content when behavioral data is sparse and toward behavior as the history grows.

The deeper question is: at what point does the behavior model begin to outperform the content model? If the answer is "after fifty interactions" and 80% of users have fewer than fifty, the behavior model adds complexity for limited benefit. This is a legitimate architectural outcome that should be discovered through measurement, not assumed away.

**Is this sufficient?** Yes, as long as the content-model fallback is genuinely good. The risk is that the team treats the behavior model as the core product and under-invests in the content model. Both need to work. The content model is the foundation; the behavior model is the enhancement.

### 5. Attribution might be too noisy to support actual payments

**The risk.** The attribution system computes each contributor's share of the value they created. The computation involves estimating marginal effects -- how much worse would the model be without your contribution? -- across many possible scenarios. These estimates are statistical approximations with confidence intervals. If the confidence intervals are wider than the estimates themselves, the attribution is noise with a fairness story on top.

**What this means for you.** If your true attribution share is $50 per month but the system's estimate could be anywhere from $10 to $90, the payments you receive are not meaningfully tied to your actual contribution. The system is distributing money, but not fairly in any precise sense.

**Early warning sign.** When the system computes attribution scores, each score comes with a confidence interval. If the median confidence interval is wider than the median score, individual attribution is unreliable.

**Mitigation.** The key architectural choice: attribution operates at the contributor level (an artist's entire body of work, a gallery's complete inventory), not at the individual-artwork level. An artist's 500 works collectively have a much larger and more stable marginal effect than any single painting. Contributor-level attribution converges faster and has tighter confidence intervals.

Within each contributor's corpus, the system can estimate which individual works contributed most -- but these per-work breakdowns are informational, not the basis for revenue distribution. Revenue flows at the contributor level; work-level details help you understand your contribution but do not determine your payment.

Below a minimum threshold, attribution is pooled into a collective fund rather than distributed individually. This acknowledges that very small contributions cannot be attributed reliably and avoids false precision.

**Is this sufficient?** At the contributor level, probably yes. At the individual-artwork level, the honest answer is that precise attribution is not achievable with current methods at scale. The layered approach -- revenue at the contributor level, information at the work level -- is a principled compromise. The risk is communicational: if the system promises per-artwork precision and delivers per-contributor precision, contributors will feel misled. Transparency about the granularity of attribution is essential.

### 6. Reproducibility might be aspirational

**The risk.** Attribution requires that the same training data produces the same model. As discussed in the production chapter, perfect reproducibility is extremely difficult to achieve in modern ML systems due to the way parallel processors handle arithmetic. Two training runs on identical data can produce models that differ by small but measurable amounts.

**What this means for you.** If the model varies between runs, the attribution scores inherit that variation. A contributor whose true share is 2% might measure as 1.7% on one run and 2.3% on another. If the variation is small relative to the contribution, this is tolerable. If the variation is large relative to the contribution, the attribution is unreliable.

**Early warning sign.** Train the model twice on the same data. Compute attribution scores on both models. Compare them. If the scores change by more than a factor of two for individual contributors, reproducibility noise is overwhelming the attribution signal.

**Mitigation.** Define reproducibility as "within tolerance" rather than "exact." Verify empirically that repeated training runs produce models within that tolerance. Average attribution scores across multiple runs to reduce noise. Accept the performance cost of using fully deterministic training for the specific runs used in attribution computation, even if normal training uses faster non-deterministic operations.

**Is this sufficient?** Conditionally. The tolerance-based approach is sound, but it must be validated empirically -- which is expensive (multiple training runs). The risk is that the tolerance needed for reliable attribution is tighter than what the infrastructure can achieve without unacceptable slowdowns. If this trade-off binds, attribution may need to operate at a coarser granularity (fewer, larger contributors) where the signal is clear above the noise.

### 7. Different stakeholders will want different definitions of "value"

**The risk.** This is not a technical risk. It is a political one, and it may be the most consequential of the seven.

Every attribution computation requires a definition of value. What counts as a good outcome? Different stakeholders answer this differently:

A **gallery** wants the model to drive sales. A work that leads to a $50,000 sale should be valued more than a work that leads to a $500 sale. Value means revenue.

An **emerging artist** wants the model to drive discovery. A work that introduces 100 new people to the artist's practice is more valuable -- to the artist -- than a single high-value sale. Value means audience.

A **data provider** (an auction house) wants the model to predict prices accurately. Their contribution's value is in improving market intelligence. Value means prediction quality.

These three definitions produce different attribution scores for the same contributions. The gallery's inventory is more valuable under the revenue definition. The emerging artist's work is more valuable under the discovery definition. The auction house's data is more valuable under the prediction definition.

**What this means for you.** The system cannot use all three definitions simultaneously -- or rather, it can, but it must assign weights to each, and the weights determine whose interests are prioritised. The choice of weights is a political decision, not a technical one.

**Early warning sign.** Stakeholder complaints that the attribution system undervalues their contributions. Different classes of contributors consistently disagree about whether the system is fair. This is not a bug to be fixed but a tension to be governed.

**Mitigation.** The system computes attribution under each definition of value separately and transparently. You can see your score under the revenue definition, the discovery definition, and the prediction definition. The final revenue distribution uses a weighted combination, with the weights set through a governance process rather than technical fiat.

The Shapley value guarantees fairness given a definition of value. It does not choose the definition. That is the hardest problem this system faces, and it is a problem for people, not algorithms.

**Is this sufficient?** Technically, yes -- the decomposition and multi-metric approach are clean. Politically, it depends on whether stakeholders accept the governance process. The deepest risk is that the system presents attribution as mathematical necessity when it is actually a political settlement. The strongest position is honesty: the mathematics guarantees fairness within a given definition of value. The definition itself is yours to negotiate.

### The pattern across all seven risks

The two highest-severity risks -- behavior data sparsity and the political nature of the value definition -- are also the two that are least amenable to purely technical solutions. The first requires the platform to build a large, engaged user base. The second requires a governance process that the system has correctly identified as necessary but not yet designed. Both are solvable, but neither is solvable by the engineering team alone.

The most important thing this risk analysis reveals is a pattern: the system's most ambitious promises -- attribution-grade training, Shapley-based revenue distribution, privacy that propagates through the model -- all work in controlled settings. Whether they work together, simultaneously, at production scale, with the noise the real world introduces, is a question that can only be answered by building the system and measuring.

The phased architecture is the right response to this uncertainty. Phase 1 tests whether the content model works before adding behavior. Phase 2 tests whether behavior works before adding attribution. Each phase is a test of the assumptions the next phase depends on. If a risk materialises, it is caught at the phase where it matters, not after the entire system is built on a false premise.

That sequencing is the most honest form of risk management available: build, measure, and decide whether to proceed -- at every stage.

For the full technical treatment, see [Reader Chapter 13](../study-guide/#13).
