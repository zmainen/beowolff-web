# Implications for Rasa

The Artsy survey is not an academic exercise. It maps the closest public precedent to what Rasa is building, and the distance between their approach and ours clarifies what Rasa actually proposes.

## The genome and the embedding: top-down meets bottom-up

The Art Genome Project and the Rasa embedding solve the same problem — "how are these artworks related?" — from opposite directions.

The genome works top-down. Human experts define 1,031 named dimensions, then score every artwork along each dimension. The result is interpretable: you can say "these works are similar because they both score high on 'Bright and Vivid Colors' and 'Geometric.'" You can explain every recommendation. A gallerist can examine the gene profile and agree or disagree with specific scores. This interpretability has real commercial value — it builds trust with a market that is skeptical of algorithmic curation.

The costs of this approach are equally clear. Human scoring is slow, subjective, and expensive. The 74 Visual Qualities genes represent a decade of curatorial effort, and they still cover only a fraction of what the human eye can perceive. The taxonomy is Western-centric, static between updates, and limited by what humans can consistently name and score. It cannot capture the unnamed similarities — the thing that makes two works feel alike without either the viewer or the genomer being able to articulate why.

Rasa works bottom-up. A neural network learns 768 unnamed dimensions from the data itself — images, metadata, collector behavior, market price. The result captures finer-grained similarity than any hand-curated vocabulary can. It can represent the unnamed similarities because its dimensions are not constrained by what humans can articulate. The cost is interpretability: you cannot point to dimension 417 and say "this is the impasto dimension." The meaning is distributed across all dimensions at once.

These approaches are complementary, not competing. The genome vocabulary provides a language for interrogating what the embedding has learned. If Rasa's embedding consistently clusters artworks tagged "Impasto" near each other, that is evidence the embedding has recovered a meaningful visual feature. If it separates works that the genome groups under the same gene, that is either a failure of the embedding or a refinement beyond what the gene captures — and distinguishing those cases is where science happens.

## Validation vocabulary

Rasa's five similarity axes — Balanced, Palette, Feel/Iigaya, Subject, Style — compress into five directions what Artsy spreads across ~94 named visual-perceptual genes. This is a large compression ratio, and it raises a question: does the compression preserve the important structure?

The genome gives us a way to test this. Take any axis — say Palette — and check whether the artworks that score highest on Artsy's color genes (Bright and Vivid, Dark Colors, Earth Tones, Pastel, Primary Colors) cluster along the Palette axis in Rasa's embedding space. If they do, the axis is capturing what the gene captures. If they cluster along a *different* axis, the axes may need relabeling. If they do not cluster at all, something is missing.

This is not the only validation strategy, but it is one that uses expert human judgment (the genome scores) as ground truth without requiring us to recruit our own panel of genomers. The vocabulary is public even if the scores are not — and for validation purposes, we need the vocabulary more than we need the scores. We can score a sample of artworks ourselves, using the genome's category definitions, and check alignment.

The 74 Visual Qualities genes plus the ~20 visually-descriptive Medium and Techniques genes provide a ready-made annotation schema for this work. Each gene has a name, a description, and a family assignment. A human annotator can read the description of "Chiaroscuro" and apply it to a set of test artworks without any knowledge of Artsy's internal scoring. The resulting labels can then be projected into Rasa's embedding space to check for structure.

## The 1-to-100 gap

The genome's scored associations — which artworks carry which genes at which scores — are the single most valuable external validation dataset for Rasa, and they are not public. Each scored association is a human expert judgment about a specific perceptual dimension of a specific artwork. No other public dataset offers this combination of specificity and scale.

If Artsy were to provide access to this data — even for a subset of their corpus — it would enable direct comparison between the genome's hand-scored similarity and Rasa's learned similarity. For every pair of artworks, the genome defines a distance (in 1,031-dimensional gene space), and Rasa defines a distance (in 768-dimensional embedding space). Comparing these distances — where they agree, where they disagree, and where disagreement reflects a genuine insight rather than a failure — would be the most informative evaluation possible.

This data access question is a natural component of a partnership conversation. The gene data is not competitively sensitive in the way that user behavior or pricing algorithms are — it represents curatorial knowledge, not business intelligence. Making it available for research, even under a restricted license, would generate scientific value that benefits Artsy's positioning as a platform grounded in art expertise.

## Architectural contrast

Artsy's six overlapping similarity mechanisms — genome layers, context grids, direct related, personalized recommendations, Vortex ML, and duplicate detection — represent the state of the art in production art recommendation. Each mechanism is a specialist; the API is a switchboard that routes queries to the appropriate specialist based on context.

Rasa proposes a different architecture: a single embedding space that fuses all five input signals (image, metadata, biography, behavior, price) during training. Similarity is distance in this space. Recommendation is nearest-neighbor search. Personalization is a learned transformation of the query point. Cold start is handled by the signals that are available — image and metadata are always present, even for a work with no behavioral history.

The architectural bet is that a unified space outperforms a switchboard. The potential gains are:

**Cross-referencing is geometric, not procedural.** When genome distance and behavioral co-occurrence and visual similarity are all directions in the same space, a query automatically balances them based on the geometry. There is no need for a merging heuristic that combines candidate lists from different systems with different ranking logics.

**New signals improve everything.** Adding a sixth input signal (say, exhibition history) does not require building a seventh recommendation mechanism. It adjusts the geometry of the existing space, and all queries benefit.

**The representation is portable.** The same embedding supports recommendation, search, valuation, attribution, and deduplication. Different applications query the same space with different distance metrics or in different subspaces.

The potential costs are equally real:

**Interpretability is harder.** When a recommendation comes from genome distance, you can explain it by listing shared genes. When it comes from a unified embedding, you can only describe the aggregate similarity — "these works are close in the learned space" — unless you develop post-hoc interpretation methods.

**Training is more complex.** Fusing five input modalities during training requires careful balancing of loss functions. If the market-price signal dominates, the embedding learns to group works by price tier rather than by visual or art-historical kinship. If the behavioral signal dominates, cold-start performance degrades. Getting the balance right is an engineering challenge that Artsy's switchboard architecture avoids entirely.

**Failure is correlated.** When the genome layer produces bad recommendations, the collaborative filtering layer might compensate. When a unified embedding produces bad recommendations, every downstream application sees the same error. There is no independent fallback.

## Convergence

The quantum experiments (Chapter 7) reveal that Artsy is moving toward the same architectural destination — unified vector representations in a single space, with user profiles as centroids of liked objects. The trajectory from their phase-2 experiments (LLM re-ranking existing pipeline candidates) to phase-3 experiments (CLIP multimodal embeddings in Weaviate) traces the same insight that motivates Rasa: the switchboard is harder to maintain and worse at cross-referencing than a unified space.

But their path of arrival shapes what they build. Artsy's prototype uses off-the-shelf CLIP (512 dimensions, trained on web image-caption pairs) and ref2vec-centroid (the user is the average of their likes). Rasa proposes purpose-trained embeddings (768 dimensions, trained on art-specific contrastive data) with sequence modeling for user behavior (capturing temporal dynamics and multi-modal taste, not just a running average).

The difference is in the training data and the loss function. CLIP's notion of visual similarity is shaped by the internet — it knows that photos of cats are similar to other photos of cats. It does not know that a Rothko is similar to a Newman in ways that matter to a collector but different from a Newman in ways that matter to a curator. A purpose-trained embedding can learn these distinctions because the training data encodes them. The Art Genome scores — if accessible — would be the ideal supervision signal.

The quantum experiments also lack attribution. There is no mechanism to trace a recommendation back to the training data or content that influenced it. Artsy's prototype, like most recommendation systems, treats the model as a black box: data goes in, recommendations come out. Rasa's attribution-grade commitment is the architectural distinction that separates a recommendation system from a contribution-tracking system — and it is the feature that makes the system legible to the artists, galleries, and data partners whose work it depends on.

The Artsy schema gives us a concrete specification of what we need to match or exceed — not just in recommendation quality, but in the diversity of queries the system can serve. Any artwork, any user, any context. Cold start and warm alike. If the unified embedding can do this with one system instead of six, the engineering advantages compound. If it cannot, we will know exactly where it fails, because Artsy's architecture maps the requirements with precision.

---

*This chapter synthesizes findings from Chapters 1–7. For the Art Genome Project taxonomy, see [Chapter 1](#1). For the Visual Qualities vocabulary, see [Chapter 2](#2). For the Metaphysics data model, see [Chapter 3](#3). For the six similarity mechanisms, see [Chapter 4](#4). For the open-source landscape, see [Chapter 5](#5). For the quantum prototype, see [Chapter 7](#7).*
