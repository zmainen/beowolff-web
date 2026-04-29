## Multi-Modal Alignment: CLIP, SigLIP, and Structured-Entity Towers

The Beowolff-Embed architecture trains three encoder towers — image, text, and structured entities — and aligns their output spaces so that representations from all three modalities cohabit a single vector space. A Caravaggio painting, the text "Italian Baroque chiaroscuro with dramatic tenebrism," and the entity tuple {artist: Caravaggio, period: Baroque, medium: oil-on-canvas, Iconclass: 73D} should all land near each other. A Mondrian, its description, and its entities should land somewhere else — and the relative positions should encode meaningful relationships (Caravaggio nearer to Artemisia Gentileschi than to Mondrian; Mondrian nearer to Malevich than to Caravaggio).

This chapter covers the lineage that makes this possible: CLIP established the template, SigLIP refined the training objective, and the structured-entity tower is the program's own architectural contribution. The entity tower is arguably the single most consequential design decision in the proposal. Understanding why requires understanding what the alternatives give up.

### CLIP: the architecture that launched multi-modal learning

CLIP (Contrastive Language-Image Pretraining, Radford et al., 2021) is a dual-encoder model trained on 400 million image-caption pairs scraped from the internet. The architecture is simple. Two encoder towers — one for images (a Vision Transformer or ResNet), one for text (a Transformer) — each produce a fixed-dimensional vector. Training pushes the image vector and its matched caption vector close together while pushing them apart from all mismatched pairs in the batch.

Concretely, given a batch of $N$ image-caption pairs, CLIP computes:

$$\mathcal{L} = -\frac{1}{2N}\sum_{i=1}^{N}\left[\log\frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)} + \log\frac{\exp(\text{sim}(T_i, I_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(T_j, I_j)/\tau)}\right]$$

where $\text{sim}(I, T) = \frac{I \cdot T}{\|I\|\|T\|}$ is cosine similarity and $\tau$ is a learned temperature parameter. This is the InfoNCE loss from Chapter 4 applied symmetrically: images try to find their own caption among $N$ candidates, and captions try to find their own image. The other $N-1$ items in the batch serve as negatives.

What the model learns is not image recognition or language understanding in isolation — it learns a *shared space* where visual and linguistic representations are commensurable. This produces an emergent capability that was not directly trained for: **zero-shot classification**. To classify an image into one of $K$ categories, you encode each category name as text ("a photo of a dog," "a photo of a cat"), encode the image, and pick whichever text embedding has the highest cosine similarity with the image embedding. No fine-tuning, no labeled training data for those categories. The shared space does the work.

For the neuroscience reader: this is the same principle as mixed selectivity in prefrontal cortex. PFC neurons respond to combinations of task-relevant variables (stimulus, rule, response) because the population code lives in a space where these variables are jointly represented. CLIP's shared space lets you query across modalities for the same reason — the dimensions encode features that are meaningful to both images and text.

CLIP's limitations are also instructive. First, it requires enormous batch sizes — OpenAI trained with $N = 32{,}768$ — because the softmax denominator in InfoNCE sums over all $N$ negatives. Small batches give noisy gradient estimates. Second, it learns only from image-text pairs, which flattens all information about an image into whatever the caption happens to say. A caption that reads "Still life with flowers, oil on canvas, 1660" tells the model about subject and medium but says nothing about the brushwork, palette, or the artist's relationship to the Dutch Golden Age tradition. Everything not in the caption is invisible to the text tower.

### From CLIP to SigLIP: fixing the batch-size dependency

SigLIP (Zhai et al., 2023) made a deceptively simple change: replace the softmax-based InfoNCE loss with independent sigmoid losses per pair.

In CLIP's softmax formulation, the loss for image $I_i$ involves a sum over all $N$ items in the batch:

$$p(\text{match} | I_i) = \frac{\exp(\text{sim}(I_i, T_i)/\tau)}{\sum_{j=1}^{N}\exp(\text{sim}(I_i, T_j)/\tau)}$$

This is a classification problem: "which of the $N$ captions goes with this image?" The quality of the learning signal depends heavily on $N$ because the denominator must contain enough negatives to make the classification challenging. With $N = 256$ you might never see a hard negative; with $N = 32{,}768$ you almost certainly will.

SigLIP replaces this with binary classification per pair:

$$\mathcal{L} = -\frac{1}{N^2}\sum_{i=1}^{N}\sum_{j=1}^{N}\left[y_{ij}\log\sigma(z_{ij}) + (1 - y_{ij})\log(1 - \sigma(z_{ij}))\right]$$

where $y_{ij} = 1$ if $i = j$ (matched pair) and $y_{ij} = 0$ otherwise, $z_{ij} = \text{sim}(I_i, T_j)/\tau + b$ is the logit with a learnable bias $b$, and $\sigma$ is the sigmoid function. Each pair is independently classified as matching or not matching. There is no softmax normalization across the batch.

Why this matters practically: the sigmoid formulation lets each pair contribute independently to the gradient. You still benefit from larger batches (more negatives, more informative gradients), but the improvement is graceful rather than cliff-like. SigLIP at batch size 1,024 can match CLIP at batch size 32,768 on standard benchmarks. This is not a minor engineering convenience — it determines whether you can train on a single 8-GPU node or need a multi-node setup with all the communication overhead that entails.

**SigLIP-2** (Tschannen et al., 2025) adds several refinements: improved training recipes for smaller models, distillation from larger teacher models, better data curation, and multi-resolution training. The Beowolff program references SigLIP-2 as the initialization for the text tower because it provides strong text-image alignment out of the box, and the sigmoid loss means the text tower can be jointly trained with the other modalities without requiring the massive batch sizes that CLIP demanded.

An important subtlety: the program does not use SigLIP-2 as a frozen feature extractor. It initializes the text tower from SigLIP-2 weights and then fine-tunes jointly with the image and entity towers. The SigLIP-2 initialization gives the text encoder a strong starting point for understanding language about visual content; the joint fine-tuning adapts it to the specific vocabulary and semantics of art description, curator notes, and artist biographies.

### The structured-entity tower: why text is not enough

This is the most important architectural decision in the program, and it deserves a thorough explanation of the problem it solves.

**The naive approach** would be to take all metadata about an artwork — artist, period, medium, school, museum, Iconclass classification — and serialize it into a text string: "Rembrandt van Rijn, Dutch Golden Age, oil on canvas, portraiture, Rijksmuseum, Iconclass 31A25 (old man)." Feed this to the text tower alongside the title and description. The text encoder handles everything. Simple.

This approach has three problems that compound as the corpus scales.

**Problem 1: entity identity.** Text encoders do not naturally understand that "Rembrandt van Rijn," "Rembrandt Harmenszoon van Rijn," and "Rembrandt" are the same entity. They might learn this from co-occurrence, but it requires seeing enough examples of each variant — and for obscure artists, you may see their name only a handful of times. A learned embedding table with one entry per resolved entity sidesteps this entirely. Entity resolution happens once, upstream; the model sees a canonical ID.

**Problem 2: compositionality of metadata.** An artwork's metadata is not a sentence — it is a *structured tuple*. The artist, the period, the medium, and the subject classification are independent axes that combine to characterize the work. Text serialization imposes a linear order that does not exist in the data, and the positional biases of transformer language models will assign different weight to metadata that appears early versus late in the string. A structured representation treats each field as an independent embedding to be composed, not a token to be parsed.

**Problem 3: metadata-graph-aware positives.** This is the killer advantage. When you train contrastively, you need to define what counts as a positive pair (Chapter 4). In a text-only system, positives are (image, its text description). In a structured-entity system, the metadata graph gives you a much richer positive-sampling distribution:

- Two works by the same artist are positives (weight: high)
- Two works from the same school are positives (weight: medium)
- Two works sharing an Iconclass branch are positives (weight: low-medium)
- Two works from the same period + region are positives (weight: low)

These weights are hierarchical: same-artist positives are stronger signals than same-school, because two Rembrandts share more structure than a Rembrandt and a Vermeer (even though both are Dutch Golden Age). The entity tower makes this hierarchy *native to the model's training signal*. With text-only CLIP, you cannot express "these two works are related because they share an artist" except by hoping the text encoder notices the same name appears in both captions.

**The architecture in detail.** Each metadata vocabulary — artists, periods, media, schools, museums, Iconclass codes — gets its own learned embedding table. If there are 50,000 distinct artists in the corpus, the artist embedding table has 50,000 rows, each a $d$-dimensional vector (where $d$ matches the shared embedding dimension, typically 768 or 1024). Same for the other vocabularies.

For a given artwork, the system looks up the embedding for each of its metadata entries — one artist vector, one period vector, one medium vector, potentially several Iconclass vectors — and feeds this set of vectors to a small aggregator transformer. The aggregator has 2-4 layers and uses self-attention to compose these independent embeddings into a single entity-set vector that represents the work's full metadata profile.

Why a transformer and not, say, a mean pool? Because metadata fields interact. An oil-on-canvas from 1660 means something different if the artist is Rembrandt (probable portrait or biblical scene, dark palette, dramatic lighting) than if the artist is Vermeer (probable domestic interior, luminous palette, careful geometry). The transformer's self-attention lets the aggregated representation capture these interactions.

The entity-set vector then participates in the same contrastive loss as the image and text vectors. All three — image embedding, text embedding, entity-set embedding — are trained to be mutually aligned for the same artwork and mutually repulsive for different artworks, with the hierarchical positive-sampling scheme weighting the "different" part according to metadata distance.

**What this gives at inference that text-only cannot.** At query time, you can construct an entity-set vector for a metadata query — {period: Baroque, medium: fresco, Iconclass: 73B (life of Christ)} — without any associated image or text. This vector lands in the shared space, and nearest-neighbor search returns artworks that match this structural description. You are querying the knowledge graph through the embedding. With a text-only system, you would need to phrase this as natural language ("Baroque frescos depicting the life of Christ") and hope the text encoder interprets it correctly. The entity tower gives you a precise, structured query language for free.

**Cold-start transfer through entity structure.** When a new artist enters the corpus with a handful of works, the entity tower already has learned vectors for their period, medium, school, and Iconclass categories. Even before the artist's own embedding is well-trained (it needs interactions or at least enough examples for the contrastive loss to shape it), their works inherit neighborhood structure from the other metadata axes. A new artist classified as "contemporary, oil on linen, geometric abstraction" enters the manifold near other contemporary geometric abstractionists, because those entity embeddings are already trained on the rest of the corpus. This is cold-start transfer at the entity level — and it works even when the biography tower (see below) has nothing to contribute because the artist has no Wikipedia entry.

### The biography signal: cold-start through language

The program attaches artist biographies — from Wikipedia, Artwiki, and catalogs raisonnés — not to individual artworks but to *artist entities*. The biography text is encoded by the text tower and its embedding is fused with the artist's entity embedding. This means the artist's learned representation in the entity table is shaped by two forces: the contrastive signal from their artworks (visual similarity to other works) and the semantic signal from their biography (who influenced them, what tradition they work in, what critics say about them).

The precedent here is explicit: Spotify's Discover Weekly team, led by Erik Bernhardsson, scraped music blogs and journalism to build "cultural vectors" for every track and artist. When a new track was released with no listening data, its cultural vector — derived from press coverage, reviews, and blog mentions — placed it in the right neighborhood of the embedding space. Users who liked similar-sounding, similarly-reviewed music would see the new track surface. This was specifically the mechanism that solved Spotify's cold-start problem for new releases.

Beowolff is making the same move. Consider an emerging Brazilian painter who works in geometric abstraction influenced by Lygia Clark and Helio Oiticica. Their Wikipedia stub reads: "Maria Silva is a Brazilian painter known for geometric abstraction. Influenced by the Neo-Concrete movement, particularly Lygia Clark and Helio Oiticica, her work explores the boundary between painting and sculpture through shaped canvases and participatory installations."

The text tower encodes this and attaches it to the artist entity. The biography mentions Clark and Oiticica — artists whose entity embeddings are well-established because they have extensive interaction histories, auction records, and curatorial documentation. The contrastive training process pushes Silva's artist embedding toward the region of the manifold occupied by Clark and Oiticica, because the text embedding of her biography is semantically similar to theirs. Before a single collector has interacted with Silva's work, the model knows where she belongs.

This is not a heuristic or a hand-coded rule. It is a learned representation — the biography text passes through the same SigLIP-2-initialized text encoder that processes artwork descriptions, and the resulting vector is shaped by the same contrastive training that shapes everything else. The biography is just another signal, weighted into the loss, contributing to the geometry.

The quality of this signal varies enormously, and the program's risk analysis (Chapter 13 of the study guide outline) correctly flags this. Major artists have rich, well-written Wikipedia articles. Obscure artists have stubs or nothing. The signal degrades gracefully — when the biography is absent or low-quality, the entity tower's other axes (period, medium, school, Iconclass) still provide cold-start structure. But the biography tower's marginal contribution to cold-start recall is directly measurable in ablation: train with and without biographies, measure cold-start recall@K on held-out works by artists not seen in training, and compare. If the biography tower is earning its keep, the difference should be substantial.

### The price head: market value as a learned direction

The fifth signal in the program — market price — enters the model differently from the others. Instead of a contrastive loss (which requires positive and negative pairs), price uses an auxiliary regression head: a small MLP that takes the artwork's embedding vector as input and predicts the log-price percentile of its most recent sale.

Why log-price percentile rather than raw price? Art prices span many orders of magnitude — a student work might sell for $500, a Basquiat for $110 million. Log-transformation compresses this range. Percentile-ranking within the training corpus further normalizes the distribution, making the regression target well-behaved regardless of the absolute price scale.

The regression loss is auxiliary, meaning it is added to the total training loss with a weighting coefficient $\lambda_{\text{price}}$ that is much smaller than the contrastive losses:

$$\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{contrastive}} + \lambda_{\text{price}} \cdot \|\hat{p}(x) - p(x)\|^2$$

where $\hat{p}(x)$ is the predicted log-price percentile and $p(x)$ is the ground truth. The weighting is small because price is not the primary organizing principle of the embedding — visual similarity, curatorial kinship, and behavioral co-occurrence matter more for recommendation. But the regression head has a subtle and important effect on the embedding geometry: it introduces a *market-value direction* into the learned space.

What does this mean geometrically? The regression head is a linear or near-linear function of the embedding vector. For it to predict price accurately, the embedding must encode information that correlates with price. This means there exists a direction in the embedding space along which works increase in market value — a "price axis." This axis is not orthogonal to all other axes (expensive works tend to share certain visual and curatorial properties), but it introduces an additional dimension of variation that the model would not learn from contrastive losses alone.

The downstream affordance is price-aware queries. A collector browsing works similar to a given piece can add a constraint: "like this, but in a lower price tier." Geometrically, this means searching for nearest neighbors in the subspace orthogonal to the price axis, or searching in a cone that combines similarity with a price-direction constraint. The embedding supports this because price is baked in, not layered on.

The price head also contributes to cold-start prediction for market segment. A new work by a new artist enters the embedding with no sale history — but the model can predict its approximate price tier from its embedding vector (which is shaped by image content, entity structure, and biography). This prediction is not a valuation (the model is not an appraiser), but it provides a useful prior: "based on everything else this model knows about works like this, they typically trade in the X percentile range." For a recommendation system that serves both collectors and galleries, this prior matters.

### How the towers compose

The full item tower produces a single embedding vector for each artwork by combining the outputs of three sub-towers:

1. **Image tower** (DINOv3-initialized ViT): artwork image $\to$ image embedding $I \in \mathbb{R}^d$
2. **Text tower** (SigLIP-2-initialized Transformer): title + description + biography $\to$ text embedding $T \in \mathbb{R}^d$
3. **Entity tower** (learned tables + aggregator transformer): metadata tuple $\to$ entity-set embedding $E \in \mathbb{R}^d$

The combination function could be as simple as a learned weighted mean, or as complex as another transformer layer that attends across the three vectors. The program file does not specify this — it is a hyperparameter to be determined empirically. What matters architecturally is that all three vectors live in the same $d$-dimensional space and are trained jointly under the same contrastive and regression objectives. The composition produces a single item embedding that carries information from all three modalities.

When any modality is missing — no image (rare), no biography (common for obscure artists), no Iconclass coding (occasional) — the remaining towers still produce valid embeddings. The item embedding degrades gracefully: less information, less precise placement in the manifold, but not garbage. This is the practical meaning of multi-modal alignment. The modalities are aligned precisely so that one can compensate for another's absence. The whole architecture is designed around the assumption that most items will be missing at least one signal, because that is the reality of the art corpus.
