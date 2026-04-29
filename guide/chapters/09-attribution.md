## 9. Who Gets Paid, and Why

This is the chapter that matters most. Everything else in the system -- the model, the visualiser, the recommendation engine -- is infrastructure. The question this chapter answers is the one that determines whether artists, galleries, and data contributors will participate: when a work sells, who contributed to the sale, and how are the proceeds shared?

The standard answer in the art market is: the artist made the work, the gallery brokered the sale, and the commission split is negotiated between them. That answer is complete when the sale happens through personal relationships and gallery representation. It is incomplete when a machine learning model surfaces the work to a buyer who would never have found it otherwise.

The model exists because it was trained on data. The data includes not just the sold work but thousands of other works that taught the model what this region of aesthetic space looks like. The artist who made the sold work has an obvious claim. But the artists whose works taught the model to recognise that region also have a claim -- a subtler one, but real. Without their contributions, the model would not have learned to see this kind of art, and the recommendation that produced the sale would not have fired.

The attribution architecture is designed to handle this. It has four layers, each answering a different version of "who contributed?"

### Layer 1: The artist made the work

When a painting sells, the artist who painted it has a claim that is moral, legal, and contractual. No computation is needed. This is the dominant share -- roughly 90% of the attribution pool for a traditional sale, governed by the commission agreement between the artist, the gallery, and the platform.

This layer is straightforward. It is also the layer that changes most dramatically when the system itself participates in creation -- a case we return to below.

### Layer 2: Other artists shaped the neighborhood

This is where the architecture does something new.

When the model recommends a work to a buyer, that recommendation depends on where the work sits in the model's learned space. Its position was shaped not just by its own features but by the other works the model was trained on. An emerging painter working in geometric abstraction enters the model's space in a neighborhood defined by Mondrian, Malevich, Sol LeWitt, Bridget Riley. If those artists' works were removed from the training data, the geometric abstraction region would be less coherent, the model's understanding of that tradition would be weaker, and the recommendation might not have happened.

This is manifold-shaping credit: artists get paid not just when their own work sells, but when works in their region of the aesthetic space sell, because they taught the model to see that region. A collector buys a painting by a young abstract painter. The painter gets the primary share. But Mondrian's estate, and Riley's gallery, and every other contributor whose work defined that neighborhood, receive a smaller share from the attribution pool -- proportional to the measurable effect their contributions had on the model's ability to surface works in that region.

The measurement uses a concept from cooperative game theory called the Shapley value. Here is what you need to know about it, without the mathematics.

### The Shapley value in one paragraph

Imagine you are dividing the proceeds of a group project among the contributors. The fair question is not "what did you do?" -- it is "what would have been different without you?" And not just "what would have been different if you alone were missing," but "what would have been different in every possible combination of contributors?" The Shapley value averages your contribution across all these scenarios. If your work is essential -- if removing it always makes things worse -- your share is large. If your work is redundant -- if others could have provided the same signal -- your share is small. Shapley proved in 1953 that this is the only way to divide credit that satisfies basic fairness conditions: everyone's shares add up to the total, identical contributions get identical shares, and someone who contributes nothing gets nothing.

### Why the full Shapley framework is not what we actually use

Computing the Shapley value across millions of individual artworks would require an astronomical number of comparisons -- removing every possible subset of works and retraining the model to see what changed. This is not feasible.

But the attribution problem does not require that level of granularity. The question is not "what was the marginal contribution of this particular Monet water lily?" It is "what was the contribution of Monet's entire corpus to the model's ability to surface Impressionist works?" At the corpus level -- tens or hundreds of contributors rather than millions of individual works -- the computation becomes tractable. You can evaluate the effect of including or excluding an artist's entire body of work with a manageable number of model retraining runs. This is where Shapley earns its keep: among a manageable number of meaningfully different contributors, each with a measurable effect.

For the other layers of attribution, simpler mechanisms work better.

### Layer 3: The recommendation path

When a sale follows a recommendation, there is a traceable path: the buyer browsed certain works, the system surfaced the sold work based on that browsing pattern, and the buyer purchased it. The referring works -- the ones the buyer was looking at when the recommendation fired -- are part of the causal chain.

But the referring works are usually fungible. If the buyer was browsing Impressionist landscapes and the system recommended another Impressionist landscape, many different Impressionist landscapes could have served the same role. The credit belongs to the neighborhood, not to the individual referrer. This layer uses standard referral-tracking mechanisms -- the same kind of multi-touch attribution that advertising and affiliate marketing have used for decades.

### Layer 4: The infrastructure

The algorithm, the engineering, the governance decisions that shaped the platform -- these affect every sale, not any single one. Their contribution is aggregate: over a quarter, did the system's improvements increase revenue or not?

This is measured by A/B testing -- showing different groups of users different versions of the system and comparing outcomes over weeks. The infrastructure's share of total revenue is a governance parameter, sized by the measured aggregate improvement. It is not a per-sale computation.

### What "value" means -- and why that is a political question

Every attribution computation requires a definition of value. What does it mean for a contribution to be "valuable"? This is where the architecture becomes honest about something most systems hide.

The model's effect on the world can be measured in different ways. A contribution might be valuable because it improved recommendation quality (the model surfaces better matches). Or because it improved discovery (users find artists they would never have encountered). Or because it drove revenue (more sales at higher prices). Or because it improved curatorial coherence (the model's understanding of art history became more sophisticated).

These are different definitions. They produce different attribution scores for the same contributions. A well-known artist's corpus might score highly under revenue-weighted attribution but lower under discovery-weighted attribution, because their works would have sold regardless. An obscure artist's corpus might score highly under discovery but poorly under revenue.

The system handles this by separating two things that most attribution systems conflate:

**The representational effect** -- how much did a contribution change the model? This is objective and computable. It is the same regardless of who is asking. You can measure it: did the model's internal structure shift when this corpus was included?

**The meaning function** -- was that change valuable? This depends on what you are measuring, and different stakeholders will answer differently.

Think of it this way. An artist's corpus moved the manifold -- it changed the shape of the model's aesthetic space. Whether that movement was valuable depends on what you are optimising for. The system computes the movement once (expensive but shared) and lets different stakeholders apply different definitions of value to the same movement. A gallery might weight revenue. A curator might weight coherence. An artist might weight discovery. The attribution architecture makes these choices visible and negotiable rather than burying them inside a single opaque number.

This is not a bug. It is the system being transparent about the fact that "value" is a political choice, not a technical one. The Shapley value guarantees fairness given a definition of value. What it cannot do is choose the definition for you. That is a governance question -- one the platform's stakeholders need to negotiate.

### Droit de suite as a special case

Several jurisdictions -- France, the EU, the UK -- have resale royalty laws. When an artwork resells on the secondary market, a fixed percentage flows to the artist or their estate. The percentage is set by statute: typically 3-4%, regardless of how much the artist's work contributed to the market that made the resale possible.

Shapley attribution is a more principled generalisation. Instead of a flat percentage applied to all resales, it pays for measurable contribution. An artist whose work fundamentally shaped a region of the market receives more. An artist whose contribution was redundant -- because many other artists in the training data carry the same signal -- receives less. The attribution is grounded in computation rather than political negotiation, though the definition of value that the computation uses is itself a governance choice.

This does not replace droit de suite where it exists legally. It extends the principle to contexts where the law has not yet reached -- digital recommendations, cross-border transactions, and the model's own contribution to discovery.

### When the system creates

The layered architecture shifts dramatically when the AI itself participates in creation.

**Traditional sales.** Layer 1 dominates. The artist made the work. Attribution is mostly property rights, with a small supplementary pool for manifold-shaping and infrastructure.

**System-assisted creation.** The artist used the model as a creative tool -- exploring the manifold to find gaps, getting feedback on positioning, generating variations. The final work is the artist's, but the platform influenced the process. Layer 1 still dominates, but the platform's contribution is real, handled as a platform fee.

**System-generated works.** This is the future case and it changes everything. When the system produces a curated composition or a generated image, there is no single artist. The entire sale revenue (minus operating costs) enters the attribution pool. The relevant contributors are the training corpora that shaped the generative model's output space -- the artists who defined the region the system is generating within. If the system generates an image in the Post-Impressionist region, the artists who taught the model what Post-Impressionism looks like have a direct claim.

In this scenario, manifold-shaping credit (Layer 2) becomes the primary mechanism. The artists did not make the sold work, but they made the sold work possible. Attribution flows to them through the same Shapley computation used for traditional sales, but now it is the dominant share rather than a supplement.

### The audit trail

None of this works without a record of what happened. The system maintains a provenance ledger -- a cryptographically secured history of every contribution, every model version, every recommendation, and every sale. When attribution is computed, the ledger provides the evidence: which contributions were in the training data, which model version served the recommendation, what the buyer's path was.

The ledger is tamper-evident. Any modification to a historical entry invalidates the chain. Contributors can verify their own records. This is what makes the attribution promise credible rather than rhetorical -- it is not a claim about what the platform will do, but a verifiable record of what it did.

### What this means for you

If you are an **artist**: your work earns in two ways. When your own work sells, you receive the primary share. When works in your region of the aesthetic space sell, you receive a smaller share -- because your contribution to the training data helped the model surface those works. You get paid for the taste you teach the model, not just for your own inventory turning over.

If you are a **gallery**: your inventory and your market data are contributions with measurable effects. Sharing data with the platform is not a concession -- it is an investment with a traceable return. The attribution system tells you exactly how much value your data generated.

If you are a **collector**: the recommendations you receive are shaped by the contributions of artists, galleries, and data providers. The attribution system ensures those contributors are compensated, which incentivises more and better contributions, which improves the recommendations you receive. The flywheel works because attribution makes sharing rational.

If you are a **data provider** (an auction house, a curatorial database): your historical records improved the model's price estimates and curatorial understanding. Attribution measures that improvement and compensates it. The alternative -- contributing data to a platform that absorbs it without compensation -- is what makes data providers withhold their best assets. Attribution is the mechanism that makes sharing worth the risk.

For the full technical treatment, see [Reader Chapter 9](../study-guide/#9).
