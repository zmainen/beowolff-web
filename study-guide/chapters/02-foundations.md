## Embeddings, Geometry, and What "Similarity" Means

Everything in the Beowolff-Embed program rests on a single architectural bet: that it is possible to represent artworks, artists, collectors, and market segments as points in a shared continuous space, such that geometric relationships between points — distances, angles, clusters — encode meaningful semantic relationships. The entire recommendation system, the cold-start mechanism, the attribution pipeline, the visualiser — all of them are downstream of this one commitment. If the embedding space is well-constructed, every downstream task becomes a geometric query. If it is poorly constructed, no amount of downstream engineering will recover what was lost.

This chapter establishes what embeddings are, why geometry carries meaning, and how the choice of training objective determines which notion of "meaning" gets encoded. If you take one thing from this chapter, it should be this: **the loss function is the most consequential architectural decision in the program**, because it determines what "nearby" means, and "nearby" is the primitive on which everything else is built.

### From discrete objects to continuous spaces

An artwork in a database is a row: an ID, an artist name, a date, a medium, a set of tags. These are discrete attributes. You can check whether two works share an artist or a period, but you cannot ask "how similar are they?" in a graded way. Discrete representations support equality testing, not similarity.

An embedding maps each discrete object into a continuous vector — a list of real numbers, typically hundreds of them. A Rembrandt self-portrait becomes a vector in $\mathbb{R}^{768}$. A Rothko color field becomes another vector in the same space. Now similarity is not a binary predicate but a continuous quantity: the distance between them, or the angle between them, or their dot product. You can ask not just "are these the same artist?" but "how close are these in the space of visual-curatorial-behavioral meaning?"

**The neuroscience bridge is population coding.** You know this already, but let me name the parallel explicitly because it will recur throughout the guide. A hippocampal place cell fires maximally at one location. A population of place cells, taken together, encodes position as a firing-rate vector — a point in a high-dimensional space where each axis is one cell's firing rate. The animal's position in physical space is encoded as a position in the neural population's firing-rate space. Change the axes, and you have the same structure: an artwork's position in aesthetic-curatorial-market space is encoded as a position in the embedding's coordinate space.

The parallel goes deeper than analogy. In both cases, the representation was not given — it was learned. Place cells do not arrive with a map; they learn to tile the environment through experience. Embedding vectors do not arrive with meanings; they learn to tile the manifold of artworks through training. And in both cases, what the representation encodes depends on the learning signal. Place cells in environments with different reward structures develop different maps. Embeddings trained with different loss functions develop different geometries.

### Dimensionality: why 768, and what does each dimension "mean"?

The program uses 768-dimensional embeddings (the default output of a ViT-L/14 vision transformer). This raises an immediate question: what does dimension 437 represent? Is it "redness"? "Baroque-ness"? "Price tier"?

The answer is: nothing individually. This is the key insight about distributed representations, and it is the same insight that makes population coding work in neuroscience. A single place cell's firing rate does not "mean" anything about position by itself — it is the pattern across the population that carries the spatial information. Similarly, a single embedding dimension does not correspond to a human-interpretable feature. The semantic content lives in the geometry — in the relative positions of points — not in the individual coordinates.

This is both a feature and a limitation. It is a feature because it means the representation can encode far more structure than 768 independent features could. A 768-dimensional space has vastly more "directions" than 768 — every linear combination of axes is a direction, and the model can learn to use all of them. It is a limitation because the representation is opaque: you cannot look at a vector and read off what the model thinks about that artwork. Interpretability requires additional tools (projection, probing classifiers, concept-bottleneck layers), and the program's "taste portrait" proposal is precisely an attempt to recover some interpretability from the dense representation.

Why 768 specifically? It is the output dimension of a ViT-L/14 (Vision Transformer, Large variant, 14x14 patch size). Smaller models (ViT-B, 768-d; ViT-S, 384-d) produce lower-dimensional embeddings; larger models (ViT-H, 1280-d; ViT-g, 1536-d) produce higher-dimensional ones. There is a tradeoff: higher dimensionality gives the model more capacity to represent fine distinctions, but also increases compute and memory costs for everything downstream (nearest-neighbor search, UMAP, serving). The choice of 768 is not principled in the way a neuroscientist would want — it is the default for the model family the program chose. The program file notes ViT-H/14 (1280-d) as a follow-up ablation.

### Geometry as semantics

Once objects are embedded as vectors, the geometric relationships between them carry meaning. Three measures dominate, and they are not interchangeable.

**Euclidean distance** measures the straight-line separation between two points:

$$d(\mathbf{x}, \mathbf{y}) = \|\mathbf{x} - \mathbf{y}\|_2 = \sqrt{\sum_{i=1}^{d}(x_i - y_i)^2}$$

This is the most intuitive measure — it is literal distance in the embedding space. Two artworks that are close in Euclidean distance are "nearby" in a spatial sense. But Euclidean distance is sensitive to vector magnitude: a vector that has been scaled up will be "far" from one that has not, even if they point in the same direction.

**Cosine similarity** measures the angle between two vectors, ignoring magnitude:

$$\text{cos}(\mathbf{x}, \mathbf{y}) = \frac{\mathbf{x} \cdot \mathbf{y}}{\|\mathbf{x}\| \|\mathbf{y}\|} = \frac{\sum_i x_i y_i}{\sqrt{\sum_i x_i^2} \sqrt{\sum_i y_i^2}}$$

Cosine similarity is 1 when two vectors point in the same direction, 0 when they are orthogonal, and -1 when they point in opposite directions. It does not care how long the vectors are. This matters because in many embedding spaces, magnitude encodes something different from direction. A vector's direction might encode "what kind of artwork this is" while its magnitude encodes "how confident the model is" or "how prototypical this example is." Cosine similarity asks only about kind, ignoring confidence.

**Dot product** is the un-normalized version:

$$\mathbf{x} \cdot \mathbf{y} = \sum_i x_i y_i = \|\mathbf{x}\| \|\mathbf{y}\| \cos(\theta)$$

It combines direction and magnitude. In the two-tower retrieval architecture the program proposes, the ranking score for a user-item pair is their dot product. This is not an accident: the dot product lets the model use magnitude as a "general quality" signal. An item embedding with large magnitude gets a boost in the dot product with every user embedding, regardless of direction. This is a deliberate design choice — the model can learn to make "universally appealing" items have large embeddings and "niche" items have smaller ones.

**The neuroscience bridge here is population-vector decoding.** Georgopoulos's work on motor cortex showed that the direction of arm movement could be decoded by a weighted sum of preferred-direction vectors across a neural population — essentially a dot product between the population activity vector and each neuron's tuning vector. The cosine of the angle between the population vector and the movement direction predicted movement accuracy. The Beowolff-Embed system uses the same primitive: the dot product between a user vector and an item vector predicts preference.

**When to use which?** Cosine similarity for retrieval when you want to find items "of the same kind" regardless of how typical they are. Dot product for ranking when you want popularity or quality to influence the score. Euclidean distance for clustering and UMAP, where you need metric-space properties (triangle inequality). In practice, if all vectors are L2-normalized (projected to the unit hypersphere), all three are equivalent up to monotonic transformation: cosine similarity = 1 - (Euclidean distance$^2$ / 2), and the dot product equals the cosine similarity. Many systems normalize embeddings for exactly this reason — it simplifies the choice. The Beowolff-Embed program does not explicitly state whether it normalizes, but the standard practice in contrastive learning (Chapter 4) is to normalize.

### The loss function determines the geometry

Here is the insight that makes the rest of the program legible: **the training objective determines what "nearby" means.** Different loss functions produce embeddings where different artworks cluster together, and the choice of loss is a choice about what the embedding considers important.

Consider two artworks:

- **Work A**: A Rothko color-field painting — large rectangles of luminous red and orange, 1961, oil on canvas.
- **Work B**: A Barnett Newman "zip" painting — a field of deep red divided by a thin vertical band, 1953, oil on canvas.
- **Work C**: A contemporary Chinese oil painting of red lanterns against a red sky, 2019.

Now consider what "nearby" means under different loss functions:

**Image loss** (self-supervised, trained on visual similarity only): A and C might be close — both are dominated by red, both have large color fields. B is different in texture (the zip disrupts the field). The geometry reflects what things *look like*.

**Entity loss** (trained on structured metadata — artist, period, school): A and B are close — both are Abstract Expressionists, both are canonical postwar American painters working in the color-field tradition. C is far. The geometry reflects *art-historical categories*.

**Behavior loss** (trained on what collectors do — who saves, follows, purchases): A and B are very close — collectors who buy Rothko also buy Newman. C is far. The geometry reflects *market behavior*.

**Biography loss** (trained on artist text — Wikipedia, catalogs): A and B are close — their biographies mention the same movements, galleries, peers. C occupies a completely different biographical neighborhood. But within the C neighborhood, the model knows which other contemporary Chinese painters are close.

**Price loss** (auxiliary regression on sale prices): A and B are in the same astronomical tier. C is in a different tier entirely. The geometry reflects *market value*.

The Beowolff-Embed program proposes to make all five of these notions of "nearby" cohabit a single space, trained jointly. This is precisely where the engineering difficulty lives. If image-similarity and entity-similarity agree (they often do), the model converges happily. When they disagree — when two works look similar but belong to unrelated traditions, or belong to the same tradition but look nothing alike — the model must find a compromise. The compromise is not guaranteed to be good. This is the multi-head loss instability risk named in Chapter 13 of the study guide outline.

### Multi-modal embedding: one space, many signals

The program proposes what the ML literature calls a **multi-modal embedding**: a single vector space shared by inputs from different modalities (images, text, metadata, behavior, price). This means an artwork's image embedding, its entity embedding, and its biography embedding all live in the same $\mathbb{R}^{768}$. A user's behavioral history, encoded by the user tower, also lives in the same space. A text query ("luminous postwar American color-field paintings, large scale") also lives there.

Why is this useful? Because it enables **cross-modal retrieval**: you can query with an image and retrieve text, or query with an entity ("Iconclass 11H — male saints") and retrieve images, or query with a user state and retrieve items. All of these are nearest-neighbor lookups in a single space. The alternative — separate embedding spaces for each modality, joined at query time — is called **late fusion**, and it has a critical weakness: the fusion model (typically a learned combiner or a simple weighted average) must be trained separately and can only discover cross-modal relationships that the individual embeddings already represent. If the image embedding does not encode anything about art-historical period, no amount of late fusion with the entity embedding will help image queries recover period structure.

**Joint training** is the alternative. All towers (image, entity, text, behavior, price) are trained simultaneously against a shared loss that pulls cross-modal pairs together. The image tower learns to produce embeddings that are compatible with entity embeddings; the entity tower learns to produce embeddings that are compatible with image embeddings. Each modality's encoder adjusts to accommodate the others. The result is a space where cross-modal structure is baked in, not bolted on.

The price of joint training is exactly the multi-objective optimization problem mentioned above. With late fusion, each encoder can be optimized independently for its own objective. With joint training, the image encoder must satisfy image-level losses *and* cross-modal alignment losses, and these can conflict. The program bets that the cross-modal alignment gain is worth the optimization difficulty. This is a defensible bet — CLIP (Radford et al., 2021) showed that joint image-text training produces embeddings that generalize far better than either modality alone — but it is a bet, not a certainty.

### A worked example: what "nearby" looks like

To make the geometry concrete, consider a simple scenario with four artworks embedded in a (greatly simplified) 3-dimensional space. In reality this is 768-d, but the principles are identical.

Suppose after training, the embeddings are:

| Work | Vector | Description |
|:-----|:-------|:------------|
| Rothko, *No. 61* | (0.8, 0.1, 0.6) | Color field, Abstract Expressionism |
| Newman, *Vir Heroicus* | (0.7, 0.2, 0.5) | Color field, Abstract Expressionism |
| Vermeer, *Girl with Pearl* | (0.1, 0.9, 0.3) | Dutch Golden Age portrait |
| Rembrandt, *Self-Portrait* | (0.2, 0.8, 0.4) | Dutch Golden Age portrait |

The cosine similarities (computed as the dot product of normalized vectors) cluster as expected: Rothko-Newman similarity is high (~0.98), Vermeer-Rembrandt similarity is high (~0.99), and cross-group similarities are low (~0.6). A nearest-neighbor query for Rothko returns Newman first, then Rembrandt, then Vermeer.

Now imagine we add a fifth work: a 2019 painting by a living artist who paints large-scale luminous color fields but is classified as "Contemporary Asian Art" in the metadata. Under image loss alone, it lands near Rothko and Newman. Under entity loss alone, it lands in a different quadrant. Under joint training, it lands somewhere between — close enough to Rothko-Newman for visual similarity to surface it in recommendation, but offset in the entity direction so that entity-based queries can distinguish it.

This compromise is exactly what the program needs. A collector browsing Rothko should see this work because it is visually kindred. An art historian querying "Abstract Expressionism" should not get it as a top result, because it is not AbEx. The joint embedding allows both queries to produce correct results because the two directions (visual and art-historical) are not the same direction in the space.

### What this buys and what it costs

The embedding-as-foundation approach gives the program a single source of truth for "similarity" that all downstream applications consume. The visualiser projects the same embeddings that the recommender searches and the attribution pipeline traces. This is an elegant design — no reconciliation between incompatible similarity measures.

The cost is opacity and coupling. Opacity: the embedding is a dense vector; you cannot inspect it to understand what the model thinks about an artwork. The "taste portrait" concept (Chapter 7) is an attempt to address this. Coupling: if the embedding is wrong in a systematic way — say, it collapses an important distinction that matters for recommendation — the error propagates to every downstream application simultaneously. There is no independent check, because all applications share the same geometric substrate.

For the reader interrogating the program: the questions to ask are not whether embeddings are a good idea (they are; this is settled technology), but whether a single shared embedding can serve all five loss terms without catastrophic compromise, and whether the training procedure (Chapters 3-5) is sufficient to produce a geometry that is actually useful for the downstream tasks the program promises.

The next chapter addresses the first signal in the embedding: what the image tower sees, and why it starts from DINOv3 rather than from scratch.
