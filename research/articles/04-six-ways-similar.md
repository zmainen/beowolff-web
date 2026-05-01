# Six Ways to Be Similar

The most revealing thing about Artsy's architecture is not how they compute similarity but how many different ways they compute it. There is no single "similar artworks" endpoint. Instead, six distinct mechanisms coexist in the same API, each with its own notion of distance, each addressing failure modes that the others cannot.

This is not accidental. It is the architecture that emerges when a platform discovers that no single similarity metric works for all contexts, and builds pragmatic solutions for each.

## 1. Artwork layers (genome distance)

The oldest mechanism. Each artwork has "layers" — similarity shells computed server-side within Gravity. The primary layer, called "main" (type: "synthetic"), returns artworks that are close in the Art Genome gene space. If a Rothko scores high on "Color Field Painting," "Abstract Expressionism," "Bright and Vivid Colors," and "Large Scale," then its main layer contains other artworks with similar gene profiles.

The resolver code reveals the mechanics: `relatedLayersLoader` calls `related/layers` on Gravity, then `relatedLayerArtworksLoader` calls `related/layer/{type}/{id}/artworks` with the source artwork ID. The code enhances each layer with the source artwork reference, and moves "fair" layers to the front of the list when applicable — contextual relevance takes priority over content similarity.

This is the simplest form of content-based recommendation: represent each artwork as a vector of gene scores, compute distance between vectors, return the nearest neighbors. The geometry is explicit and interpretable — you can explain why two works are neighbors by listing the genes they share. The weakness is coverage: every artwork must be scored by a genomer before it can participate in this system. Un-genomed artworks are invisible to layer-based similarity.

Pagination on layers is not truly supported — the schema includes cursors, but `hasNextPage` always returns false. This confirms the system returns fixed-size precomputed result sets rather than dynamic paginated queries.

## 2. Context grids (co-occurrence)

The "you might also like" section on an artwork page uses a priority system with five tiers. The resolver code in `artworkContextGrids/index.ts` implements strict fallthrough:

1. **AuctionArtworkGrid** — if the artwork is in an active auction, show other works from the same sale (via `saleArtworksLoader`). If the auction has closed, fall through.
2. **ShowArtworkGrid** — if the artwork is in a show at a fair with `has_full_feature`, show other works from that show. For regular shows, also include this grid.
3. **ArtistArtworkGrid** — other works by the same artist, sorted by `-merchandisability` (Gravity's internal merchandising score).
4. **PartnerArtworkGrid** — other works from the same gallery.
5. **RelatedArtworkGrid** — algorithmically similar works, falling back to the "main" layer from mechanism 1.

Two works shown at the same gallery are "similar" not because they look alike or share genes but because a gallerist — a human expert in aesthetic coherence — chose to represent them together. Gallery programs have aesthetic identities; works shown together tend to share sensibility even when they differ in medium, period, and style.

The priority ordering reveals a commercial logic: auction context first (highest conversion intent), then show context (curated relevance), then artist context (brand loyalty), then gallery context (trust network), then content similarity (last resort). The algorithmic recommendation is the fallback, not the primary signal. This inversion — human curation first, algorithms second — is characteristic of the art market, where institutional endorsement carries more weight than computed similarity.

## 3. Direct related artworks

The `artwork.related` field returns a flat list of related works via `similarArtworksLoader`, which calls `related/artworks` on Gravity with `for_sale: true`. This is also the mechanism behind `similarToRecentlyViewed` on the `Me` type — take the user's `recently_viewed_artwork_ids` and call the same Gravity endpoint to find similar works that are available for purchase.

## 4. Personalized recommendations (authenticated)

When a user is logged in, the `Me` type exposes several recommendation streams that incorporate behavioral data:

**`me.artworkRecommendations`** — fetches up to 50 artwork recommendations from Vortex's GraphQL endpoint. Each result is an `{artworkId, score: Float}` pair. The resolver hydrates the full artwork objects from Gravity.

**`me.basedOnUserSaves`** — collaborative filtering from the user's saved artworks. If you save three Rothkos, this returns works saved by other users who also saved those Rothkos.

**`me.similarToRecentlyViewedConnection`** — artworks similar to recently viewed works, using Gravity's `related/artworks` endpoint. Captures short-term browsing intent rather than long-term taste.

**`me.artistRecommendations`** — recommended artists via two switchable sources:
- `HYBRID` (default) — described in the code as *"Hybrid machine learning-based recommendations from Vortex."* The actual ML approach is not documented.
- `SIMILAR_TO_FOLLOWED` — graph-based similarity from Gravity's `UserSuggestedSimilarArtistsIndex`. If you follow artist A, this finds artists geometrically close in the followed-artist graph.

The quiz system (`quiz.ts`) reveals how cold-start personalization works: after the user completes an art-preference quiz, the resolver takes each "saved" artwork and calls `relatedLayerArtworksLoader` with `{id: "main", type: "synthetic"}` — the same synthetic-main layer used by the context grid fallback. Results are merged proportionally based on how many quiz artworks the user engaged with. The quiz bootstraps personalization by routing through the genome-distance layer.

## 5. Vortex analytics (ML backend)

Vortex is the dedicated ML service, stitched into Metaphysics as a remote GraphQL schema. Its capabilities, visible through the 1,777-line schema file:

- **`newForYouRecommendations`** — the engine behind `artworksForUser`. Returns `{artworkId, score, version, publishedAt}`. The `version` field suggests simultaneous A/B testing of model variants. When results are insufficient, the resolver backfills from curated marketing collections sorted by `-decayed_merch` or from a hand-curated "artwork-backfill" OrderedSet in Gravity.
- **`artistAffinities`** — per-user artist affinity scores, sorted by score. Used to personalize artist recommendations and discovery surfaces.
- **`artistSparklines`** — year-over-year activity sparklines for artists, with tier classifications. This powers the "trending" signals in the collector interface.
- **`marketPriceInsights`** — batch queries for demand rank, liquidity rank, sell-through rate, and median prices segmented by artwork size.
- **AI-generated captions** — `artwork_captions` endpoint produces alt text for artwork images, cached for 24 hours.

For auction-specific recommendations, a separate `auctionLotRecommendationsLoader` calls a Vortex REST endpoint (not GraphQL), suggesting the auction recommendation pipeline is architecturally distinct from the general-purpose one.

## 6. Infinite Discovery (KNN + MLT hybrid)

The newest and most ML-explicit mechanism, implemented in `infiniteDiscovery/discoverArtworks.ts`. Unlike the other mechanisms, this one exposes its internal parameters:

```
mltFields: ["genes", "materials", "tags", "medium"]
osWeights: [0.6, 0.4]
```

It calls `artworks_discovery` on Gravity, which runs a hybrid search: 60% k-nearest-neighbor (KNN) vector similarity and 40% More Like This (MLT) text similarity. The KNN component operates on some form of embedding vector stored in the search index — the representation is not specified in the metaphysics code, but the `quantum` experiments (Chapter 7) show Artsy prototyping CLIP-based 512-dimensional embeddings as a replacement.

The MLT fields — genes, materials, tags, medium — are the text features that OpenSearch uses for fuzzy matching. This is a classic hybrid retrieval architecture: dense vector similarity for semantic matching, sparse text similarity for keyword matching, blended with fixed weights.

The resolver takes `likedArtworkIds` as a "taste profile" input and blends curated picks with algorithmic results via `curatedPicksSize`. This is Artsy's production implementation of the swipe-to-discover experience that the `quantum` experiments are building a next-generation version of.

## Duplicate detection

A distinct use of similarity: `artworkDuplicatePair` exposes a `similarityScore: Float` and `detectionVersion: String` for identifying duplicate listings. When multiple galleries list the same physical artwork, the system detects potential duplicates and presents them for review. Duplicates can be `dismissed` or `merged` with field-level overrides (title, date, dimensions, medium, price). The `matchMetadata` field carries raw debugging data from the detection algorithm.

The detection trigger is a mutation: `detectArtworkDuplicatesMutation` sends a `partner_id` and optional `detection_version` to `artwork_duplicate_pair/detect` on Gravity. All computation happens server-side — metaphysics passes through the score.

## The architecture as a whole

| Mechanism | Data source | Cold-start? | Interpretable? | Personalized? |
|:----------|:-----------|:------------|:---------------|:-------------|
| Genome layers | Gene scores (Gravity) | Yes | Yes | No |
| Context grids | Co-occurrence (Gravity) | Yes | Yes | No |
| Direct related | Gravity `related/artworks` | Yes | No | No |
| Personal recs | Behavioral trace (Vortex) | No | No | Yes |
| Infinite Discovery | KNN + MLT (Gravity) | Partial | No | Yes |
| Duplicate detection | Image/metadata (Gravity) | Yes | No | No |

The cold-start problem is solved by falling back from personalized to unpersonalized mechanisms. A new artwork with no interaction history is still discoverable through genome similarity, gallery co-occurrence, or text-based MLT matching. A new user with no behavioral trace sees genome-based and context-based recommendations until enough interactions accumulate to power the ML systems. The quiz acts as a cold-start accelerator — a few swipes produce enough signal to activate the genome-distance layer.

What this architecture does not do is learn a single representation that captures all similarity signals at once. Genome distance and behavioral co-occurrence and visual similarity are separate computations, producing separate candidate sets, merged at the API layer by priority rules and backfill logic. The `quantum` experiments show Artsy exploring a transition toward unified vector representations (CLIP embeddings in Weaviate with ref2vec-centroid user profiles), but the production system remains a switchboard.

---

*Source: [github.com/artsy/metaphysics](https://github.com/artsy/metaphysics) (MIT). Resolver analysis from `src/schema/v2/artwork/`, `src/schema/v2/infiniteDiscovery/`, `src/schema/v2/me/`, and `src/lib/stitching/vortex/`, surveyed May 2026.*
