# Quantum: Prototyping the Next-Generation Stack

While Artsy's production recommendation system runs on genome distance, collaborative filtering, and the Vortex ML service (Chapters 3–4), a separate repository called `quantum` reveals what comes next. It contains 17 numbered experiments — all local, none deployed — that prototype a fundamentally different approach: vector embeddings in Weaviate, CLIP multimodal search, LLM-powered ranking, and ref2vec user profiles. This is not a toy. It uses real Artsy data from Gravity, queries real Metaphysics endpoints, and is maintained by production engineers.

The trajectory is clear: Artsy is moving from a switchboard of specialized mechanisms toward a unified vector space. They are arriving at the same architectural destination as Rasa, from a different starting point and through a different path.

## The stack

**Vector database**: Weaviate, hosted at `weaviate.stg.artsy.systems` (staging). Three vectorizer modules used across experiments:

- `text2vec-openai` — OpenAI `text-embedding-3-small`, 1,536 dimensions. Text-only embeddings for artist bios, artwork metadata, and marketing collections.
- `multi2vec-clip` — CLIP embeddings, 512 dimensions. Joint text+image embeddings in a shared vector space. The production-track vectorizer.
- `ref2vec-centroid` — user profile vectors computed as the centroid of liked objects' vectors. No separate user embedding model; the user's position in the space is defined by what they like.

**LLM providers**: OpenAI (`gpt-4o`, `gpt-3.5-turbo`, `text-embedding-ada-002`, `text-embedding-3-small`) and Anthropic (`claude-3-opus`, `claude-3.5-sonnet`, `claude-3-haiku`). Abstracted through Vercel AI SDK.

**Data source**: Gravity REST API via staging endpoints, with export scripts that serialize artworks, artists, and collections to JSON. The export code reveals internal data structures not visible through the public API.

## The experiments

### Embedding artist bios (experiment 07)

The simplest experiment and the conceptual starting point. Three escalating scripts:

**Minimal version**: Hardcoded bios for Yayoi Kusama and Claude Monet. Embedded with `text-embedding-ada-002`, stored in Weaviate, queried with `nearVector`. A search for "interested in optical illusions" returns Kusama over Monet. Trivial, but it establishes the pattern: embed text, store vectors, query by semantic proximity.

**Artist-scale version**: ~1,000 artists fetched from Metaphysics. Only `slug` and `blurb` are embedded. Weaviate handles vectorization server-side via `text2vec-openai` with `text-embedding-3-small` at 1,536 dimensions. The `blurb` property is vectorized; the `slug` is excluded from vectorization but stored for identification. Batch inserts of 100.

**Artwork-scale version**: ~280 artworks with extensive metadata. The Weaviate class includes multiple vectorized properties — and here the "collector-facing terminology" pattern first appears:

| Artsy Internal Field | Vectorized As |
|:---------------------|:-------------|
| `attributionClass.name` | `rarity` |
| `mediumType.name` | `medium` |
| `saleMessage` | `saleMessage` |
| `dominantColors` | `colors` (comma-separated string) |

All property names are vectorized alongside their values (`vectorizePropertyName: true`), meaning the embedding captures not just "oil on canvas" but "medium: oil on canvas" — the semantic context of the field name helps the model understand what the value means.

### LLM-as-ranker (experiment 08)

Three approaches to using GPT-4o to rank artwork recommendations:

**Direct prompting**: Artwork metadata formatted as XML-like blocks:

```
<artwork>
"Day of Destiny" is a work by Wowser Ng
Wowser Ng is a male Chinese artist, born 1998
Rarity: limited edition
Medium: Print
Materials: Digital painting on fine art canvas
Price info: $1,500 - 1,900
Colors: red, pink
</artwork>
```

System prompt: *"Your task is to recommend three artworks from a larger list of recommendation candidates. ONLY include artworks that are for sale. Prefer works that display a price. Always make sure that the medium type matches the user's request. Always include a justification."*

**Position bias mitigation**: Citing [arXiv:2305.08845](https://arxiv.org/pdf/2305.08845) (the "Lost in the Middle" paper on LLMs as rankers), the second script implements a two-pass debiasing strategy. First pass: shuffle candidate order randomly, feed to GPT-4o at temperature 1, extract top picks as JSON. The code comments suggest running multiple iterations (though defaulting to 1 to save tokens: *"I wouldn't recommend going above 3"*). Second pass: merge all first-pass picks, feed to a final GPT-4o call for re-ranking with justifications. Additional rule: *"Avoid recommending multiple works by the same artist UNLESS the user has specifically asked for an artist."*

**Tool-use pipeline**: The LLM is given a tool `getArtworkRecommendations` that queries three real Metaphysics recommendation pipelines simultaneously — `me.recommendedArtworks` (collaborative filtering), `me.artworkRecommendations` (Vortex ML), and `artworksForUser` (user-level). Each source contributes 10 candidates. The LLM then re-ranks the combined pool. This is the most interesting architecture: the existing production pipelines generate candidates, and the LLM acts as a re-ranker on top.

### Personalized advisory (experiments 10–11)

**Tone and voice (experiment 10)**: Fetches the most comprehensive user profile in the repo — followed artists (with blurbs), followed genes, artwork recommendations from three pipelines, and artworks the user has inquired on — all in a single Metaphysics GraphQL query. The system prompt establishes Artsy's four brand voice principles:

1. *"Show our intelligent edge"* — discerning
2. *"Be an inspiring authority"* — fresh
3. *"Focus on the substance"* — meaningful
4. *"Strike a familiar tone"* — inviting

A critical instruction: *"You DO NOT use the words 'gene' or 'genes'. Instead, you refer to the characteristics that connect artists, artworks, architecture, and design objects across history as 'characteristics' or 'traits'."* The Art Genome Project's internal vocabulary is hidden from the collector-facing experience.

The second script adds good-example editorial prose: *"Exemplifying the artist's distinctive approach to Cubist sculpture, Alexander Archipenko's bronze cast Woman Combing Her Hair..."* These real snippets from Artsy editorial establish the target register.

**Brushy CLI (experiment 11)**: A three-step advisory pipeline. First, fetch artwork, artist, gallery, and collector data from Metaphysics and Gravity (including the partner genome endpoint: `partner/{slug}/artwork/{slug}/genome`). Second, generate an image description using Claude 3 Opus vision — the artwork image is fetched, resized to 500x500 via Gemini, converted to base64, and sent with the prompt: *"Describe the supplied artwork in 150 words or less. Consider visual qualities, colors, medium as well as any associated artistic styles or movements."* Third, synthesize everything into a "should you buy this?" narrative at temperature 0 with seed 42 for reproducibility.

The advisory prompt: *"You are an experienced art advisor who advises collectors about which works to collect and why. You offer concise and direct explanations, with a minimum of extraneous adjectives and filler words."* The output is a 150-word recommendation based on the collector's followed artists, followed genes, and existing collection.

### Medium classification fine-tuning (experiment 12)

Fine-tunes GPT-3.5-turbo to classify artwork medium type from free-text materials descriptions. Training data format: JSONL pairs like `{"category": "Painting", "medium": "Oil on canvas"}`. Datasets of 100 and 999 examples, split 80/20 for train/validation.

The test cases reveal the hard edges of the problem:

| Input | Expected |
|:------|:---------|
| Oil on canvas | Painting |
| Stainless steel, painted wood | Sculpture |
| Screenprint in colors on Cream Speckletone paper | Print |
| Pigment transfer on polylaminate | Photography |
| BASF Luran | Print |
| Archival pigment print with polymer on Washi paper | Mixed Media |

"BASF Luran" is a thermoplastic — knowing it maps to "Print" requires material science knowledge that a language model might or might not have absorbed. This is the kind of domain-specific classification that justifies fine-tuning over few-shot prompting.

### Ref2vec collaborative filtering (experiment 13)

The core innovation: instead of computing user embeddings from profile features, compute them as the **centroid of vectors from liked objects**. Weaviate's `ref2vec-centroid` module does this automatically — when a user's `likedArtworks` references are updated, their vector is recomputed as the mean of those artworks' vectors.

Three Weaviate collections: `SmallNewTrendingArtists` (~180 artists, text-embedded), `SmallNewTrendingArtworks` (~240 artworks, text-embedded), and `Users` (ref2vec-centroid from liked artworks and liked artists). Cold start requires only a few liked objects — as soon as references exist, the centroid is computable.

The demo creates a user "Percy Cat" with 3 random liked artworks and 3 random liked artists, then queries for similar artworks via `nearObject` from the user's computed vector. This is collaborative filtering without a collaborative filtering model — the geometry of the embedding space does the work.

### Curated discovery with CLIP (experiment 14)

The most architecturally complete experiment before infinite discovery. Four entity types with cross-references:

**Text-only artworks** (01): `text2vec-openai` at 1,536 dimensions. Extensive property schema with careful vectorization control. Vectorized: title, date, rarity, medium, materials, dimensions, price, location, categories (from genes), tags, additional information. Skipped from vectorization: internal IDs, slug, price amounts in minor USD, image URL. A planned but unimplemented field: `imageDescription` (set to `null` with `// TODO: add computer vision step?`).

**CLIP multimodal artworks** (03): `multi2vec-clip` at 512 dimensions. Text fields: title, medium, materials, categories, tags, additional information. Image field: the artwork image stored as a blob, resized to **512x512** via Artsy's CDN. Batch size drops from 100 to 5 due to image payload size. This is the first experiment that places text and images in a shared vector space.

**User profiles** (02): `ref2vec-centroid` from `likedArtworks` references. The schema includes `dislikedArtworks` references, but dislikes are **not included** in `referenceProperties` for the centroid — they are tracked but do not affect the vector. This is a deliberate design choice: only positive signal drives the user vector.

**Marketing collections** (06): Uses `reranker-cohere` with `rerank-multilingual-v3.0` for re-ranking search results. GPT-4o generates one-sentence summaries from each collection's long markdown description.

**Cross-references** (05): `hasArtist` links from artworks to artists, enabling graph traversal across entity types.

### Infinite Discovery — the production prototype (experiment 17)

The most important experiment, and the one that maps directly to a production feature visible in the metaphysics codebase (Chapter 4, mechanism 6).

**CLIP multimodal embeddings**: `multi2vec-clip` at **512 dimensions**, explicitly set. Text fields: title, medium, materials, categories, tags, additional information, date, artist name. Image: artwork image resized to **224x224** at quality 75 (down from 512x512 in experiment 14 — a speed/quality tradeoff). Batch size of 2 due to base64 image payloads.

**User vectors**: `ref2vec-centroid` with a schema that distinguishes between `likedArtworks` (drives the centroid) and `seenArtworks` (tracked but excluded from the centroid). The comment is explicit: seen-but-not-liked *"Should not be considered a negative signal."* This avoids the implicit-negative-feedback trap where showing a user an artwork they ignore is treated as a rejection.

**The Gravity data pipeline**: The export code reveals the full internal data model. Published, for-sale artworks are queried with field selectors that expose the four genome layers:

```ruby
.only(:id, :_slugs, :genome, :automated_genome, :partner_genome,
      :visual_genome, :title, :dates, :attribution_class, :category,
      :medium, :price_listed, :price_currency, :tags, :auto_tags,
      :additional_information, :default_image, :artist_ids, :colors)
```

The four genome layers:

- `genome` — human genomer annotations (the Art Genome Project proper)
- `automated_genome` — ML-predicted classifications
- `partner_genome` — gallery-applied genes via the partner portal
- `visual_genome` — computationally derived visual features

These merge into `total_genome` for the export. The filtering logic is precise:

```ruby
categories: w.total_genome
  .without("Art", "Career Stage Gene")
  .select{ |k,v| k !~ /(galleries based|made in)/i && v == 100 }
  .keys
```

Only genes with **score 100** (full match) survive the filter. The generic "Art" gene, "Career Stage Gene," and location-based genes ("galleries based in...," "made in...") are excluded. This is aggressive filtering — it reduces the ~1,031-gene vocabulary to only the most confident, non-trivial assignments. The result is a sparse but high-precision category vector.

The collector-facing terminology renaming is applied consistently:

| Gravity Field | Export Name |
|:-------------|:-----------|
| `attribution_class` | `rarity` |
| `category` | `medium` |
| `medium` | `materials` |
| `total_genome` (keys) | `categories` |
| `tags` + `auto_tags` | `tags` |
| `sale_message` | `price` |
| `price_listed` | `list_price_amount` |

The rationale is documented: *"we opt to use collector-facing terminology rather than Artsy internal jargon, on the assumption that this would perform better with the embedding models."* This is a practical insight — embedding models trained on web text understand "rarity" better than `attribution_class`, and "materials" better than `medium` (which in Artsy's schema means the materials string, not the art form).

## What quantum reveals about Artsy's trajectory

The 17 experiments trace a developmental arc:

**Phase 1 (experiments 01–07)**: Basic infrastructure. Function calling, streaming, OpenAI Assistants API, text embeddings. Learning the tools.

**Phase 2 (experiments 08–12)**: LLM-as-intelligence-layer. Using GPT-4o and Claude to rank, describe, advise, and classify artworks. The LLM is a reasoning engine operating on metadata, not a representation learning system.

**Phase 3 (experiments 13–17)**: Vector-first architecture. CLIP multimodal embeddings, ref2vec user profiles, cross-referenced entity types, marketing collection integration. The embedding is the representation; the LLM is optional.

The progression from phase 2 to phase 3 is significant. In phase 2, the LLM re-ranks candidates from existing recommendation pipelines — it sits on top of the switchboard. In phase 3, the vector space *is* the recommendation system — similarity is distance, personalization is centroid proximity, and the existing pipelines become unnecessary. This is the same transition that Rasa proposes, arrived at empirically through prototyping rather than architecturally from first principles.

The key differences between quantum's prototype and Rasa's proposal:

**Embedding model**: Quantum uses off-the-shelf CLIP (512 dimensions). Rasa proposes a purpose-trained foundation embedding (768 dimensions) that fuses five input modalities during training. CLIP was trained on web image-caption pairs, not on art; its representation of visual similarity reflects internet-scale photographic similarity, not curatorial or art-historical similarity.

**User modeling**: Quantum uses ref2vec-centroid — the user's position is the average of their liked artworks' positions. This is elegant but limited: it cannot represent multi-modal taste (a collector who likes both minimalist sculpture and Baroque painting), and it drifts as new likes accumulate without weighting recency. Rasa's sequence-modeling approach (Chapter 7 of the Reader) can capture temporal dynamics and multi-modal preferences.

**Training data**: Quantum embeds existing metadata and images using pre-trained models. It does not fine-tune the visual representation. Rasa proposes contrastive training on art-specific data — learning which visual features matter for art similarity, not just for general image similarity. The Art Genome scores, if available, would be the ideal supervision signal for this.

**Attribution**: Quantum does not address attribution. There is no mechanism to trace a recommendation back to the training data or content that influenced it. Rasa's attribution-grade commitment is the architectural distinction that separates a recommendation system from a contribution-tracking system.

## Engineering quality

The experiments are well-structured TypeScript with consistent patterns: Weaviate client initialization, data loading, schema creation, batch insertion, and query. Error handling is minimal (this is experimental code), but the data pipeline code is careful — the Gravity export scripts handle edge cases, the terminology renaming is systematic, and the filtering logic is precise.

The evaluation infrastructure is thin. Experiment 08 tracks recommendation position statistics. Experiment 12 reports accuracy on 8 test cases. No precision/recall metrics, no A/B test results, no stored evaluations. The experiments are exploratory probes, not validated systems. This is appropriate for their stated purpose — local experimentation, not production deployment — but it means the transition from prototype to production will require substantial evaluation work.

---

*Source: [github.com/artsy/quantum](https://github.com/artsy/quantum). All code analysis based on repository contents as of May 2026. Experiments numbered per the repo's directory structure.*
