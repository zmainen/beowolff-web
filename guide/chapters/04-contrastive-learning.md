## Teaching the Model What "Similar" Means

### Learning by comparison

The image model from the previous chapter gives the system eyes. This chapter is about how the system learns judgment — how it decides which works belong near each other and which should be far apart.

The method is called contrastive learning, and the core idea is simple. Show the model two things that should be similar — a "positive pair." Show it things that should be different — "negative pairs." Train it to pull the similar things together and push the different things apart. Repeat, millions of times, across the entire corpus.

Think of it as training a sommelier. You do not hand them a textbook on wine chemistry. You pour two glasses and ask: "Which of these is more like the one you just tasted?" Thousands of tastings later, the sommelier has developed an internal map of the wine landscape — organized not by rules they were given, but by the comparisons they were asked to make. The comparisons *are* the curriculum.

The critical question is: **who decides what counts as a positive pair?**

### The choices that shape the space

In contrastive learning, the entire geometry of the embedding — which works end up near which other works — is determined by the definition of "positive pair." Different definitions produce fundamentally different spaces.

**Same artwork, different view.** The most basic positive pair: two different crops or slight color variations of the same painting. This teaches the model that an artwork is similar to itself under minor changes. Every contrastive system starts here. It produces a space organized by raw visual features — color, texture, composition.

**Same artist.** Two different works by the same artist are treated as similar. This teaches the model to recognize an artist's visual signature across their oeuvre. A space trained this way clusters all Rothkos together, all Vermeers together, even when individual works look quite different from each other.

**Same school or movement.** Two works from the same movement — both Impressionist, both Minimalist — are treated as similar. This produces a coarser grouping. Monet and Renoir end up near each other. The space reflects art-historical categories rather than individual authorship.

**Same iconographic subject.** Two works depicting the same subject — both Annunciations, both vanitas still lifes — are treated as similar regardless of period or style. This produces a subject-organized space.

The Rasa system uses **all of these simultaneously**, with a hierarchy of weights. Same-artist pairs get the strongest pull. Same-school pairs get a medium pull. Same-subject pairs get a lighter pull. The result is a space with nested structure: within the broad region of Impressionism, Monet and Renoir have their own sub-regions, and within Monet's sub-region, his haystacks and his water lilies have their own neighborhoods.

This hierarchical approach is not standard off-the-shelf technology. Most contrastive learning systems in production use a single definition of similarity. The layered structure, where curatorial judgment about *how similar* different relationships are directly shapes the geometry, is what makes this architecture specifically suited to art.

For the full technical treatment, including the mathematical loss function and temperature parameters, see [Reader Chapter 4](../../study-guide/#4).

### Where curatorial judgment enters the model

Here is the point that matters most for art-world professionals: **the training data encodes curatorial decisions, and those decisions become the model's notion of similarity.**

If two works are classified as "same school" in the metadata, the model will learn to place them near each other. If the metadata classifies them differently, the model will push them apart. The metadata is not a neutral description of the world — it reflects the choices of whoever built the catalog. Which works belong to "Post-Impressionism"? Is Cezanne a Post-Impressionist or a proto-Cubist? Is a particular contemporary painter part of the "New Leipzig School" or not? These are curatorial judgments, and they have direct consequences for what the model learns.

This is not a flaw. Every recommendation system encodes values — the question is whether those values are visible and debatable or hidden and unexamined. In the Rasa system, the values enter through the metadata graph, which is a structured, inspectable artifact. You can ask "what categories define the positive pairs?" and get a concrete answer. You can argue about whether Iconclass subject categories or art-historical period labels are the right organizing principle, and that argument has direct consequences for the model's behavior.

The alternative — training only on behavioral data (what collectors click on) — is not value-free. It encodes the values of whatever collector population generated the data, including all their biases, blind spots, and market-driven preferences. Visual-only training encodes the implicit values of whatever image corpus was used. There is no neutral position. The question is which values you want to encode and whether you can see them.

### The hard cases

Contrastive learning has a characteristic failure mode that is especially relevant for art. When the system is looking for "hard negatives" — works that the model currently thinks are similar but should be pushed apart — it can make mistakes.

Consider two vanitas still lifes from the Dutch Golden Age, one by Pieter Claesz and one by Willem Claesz Heda. They look nearly identical: the same skulls, hourglasses, half-peeled lemons, tipped-over goblets. If the positive-pair definition is "same artist," these are negatives — the model should push them apart. But pushing them apart means teaching the model that these two intimately related works are dissimilar, which is clearly wrong from any curatorial perspective.

This is the "false negative" problem, and the Rasa system addresses it through the multi-level positive hierarchy. Because same-school and same-subject pairs are also treated as positives (with lower weight), the Claesz and Heda works are simultaneously pulled together (as same-school, same-subject) and distinguished (as different-artist). The model learns that they are close but not identical — which is, in fact, the right answer.

The risk is not eliminated, only managed. The weighting between these levels — how much to prioritize artist identity versus school membership versus subject matter — is a design choice with real consequences. Weight artist identity too heavily and the model collapses each artist's oeuvre into a single point, losing the internal variation that makes an artist's development interesting. Weight school membership too heavily and the model cannot distinguish individual voices within a movement. These weights are tuned empirically, but the choice of starting values reflects a judgment about what matters — and that judgment is debatable.

For the full technical treatment, including hard-negative mining and the false negative problem, see [Reader Chapter 4](../../study-guide/#4).

### What this means for you

The contrastive training process is where the model's aesthetic sensibility is formed. It does not arrive with opinions about art. It develops them through exposure to millions of comparisons, guided by the structure of the metadata graph.

For galleries representing artists whose work sits at the intersection of multiple traditions — an artist who is formally Minimalist but contextually part of a specific regional scene — the model's placement of their work depends on how those traditions are encoded in the metadata and weighted in the training. Understanding this is the beginning of understanding how to work with the system rather than being opaque to it.

For collectors, the model's notion of "similar to what you like" is built from these trained comparisons. If the system recommends a work you find surprising, it is because the embedding space places that work near your interests along some dimension — visual, art-historical, behavioral — that may not be immediately obvious. The recommendation is not random. It reflects a specific, learned geometry. Whether that geometry matches your actual taste is the question the system is trying to answer, and your responses (saving, ignoring, purchasing) are how it refines its answer over time.
