# The Metaphysics Schema: How a Marketplace Models Art

Metaphysics is Artsy's central GraphQL API — the single gateway through which every Artsy client (web, mobile, internal tools) queries art data. It aggregates eleven backend services into a unified schema of roughly 30,000 lines of GraphQL type definitions. The full schema is committed to the public repository, making it possible to study how the world's largest art marketplace structures its data — what it considers worth modeling, what relationships it maintains, and where it draws the boundary between public and private.

The name is apt. Metaphysics is the schema that defines what exists in Artsy's world and how things relate to each other. It is an ontology implemented as an API.

## Architecture

Metaphysics is built with Express, express-graphql, and TypeScript. It does not store data itself. Instead, it stitches together schemas from multiple specialized backends using `mergeSchemas`:

| Service | Domain |
|:--------|:-------|
| **Gravity** | The monolith: artworks, artists, genes, partners, users, search, similarity, duplicate detection |
| **Vortex** | ML analytics: recommendations, market insights, artist affinities, AI-generated captions |
| **Diffusion** | Auction results, comparable lots, predictive pricing |
| **Exchange** | Commerce: orders, offers, transactions |
| **Convection** | Consignment: artwork submissions from collectors |
| **Impulse** | Messaging and conversations |
| **Positron** | Editorial articles and content |
| **Delta** | Popular and trending data |
| **Causality** | Live auction bidding state |
| **Gemini** | Image CDN and transformation proxy |
| **Geodata** | Geolocation services |

A single GraphQL query can traverse data from multiple services in one request. You can ask for an artwork's title (Gravity), its market demand rank (Vortex), its current auction bid (Causality), and whether the logged-in user has saved it (Gravity) — all in a single query. The API layer handles the fan-out to each backend and assembles the response.

The Vortex integration is the most architecturally interesting. It has its own GraphQL API at `VORTEX_API_BASE/graphql`, stitched in as a remote executable schema. The stitching transform prefixes all Vortex types with `Analytics` (so `ArtistRecommendation` becomes `AnalyticsArtistRecommendation`) and then extends local types to attach Vortex data — `Artwork.pricingContext` resolves by delegating to Vortex with the artwork's `artistId`, `category`, and `sizeScore`. Authentication flows through Gravity's `me/token` endpoint with a dedicated `VORTEX_APP_ID`.

Public endpoints exist at `metaphysics-cdn.artsy.net/v2` (CDN-cached) and `metaphysics-production.artsy.net/v2` (uncached). Introspection is disabled, but the full schema is readable in the repository as `_schemaV2.graphql`. Gravity supports blue/green deployment via `GRAVITY_API_PERCENT_REDIRECT`, allowing traffic splitting between instances.

## The critical architectural insight

Metaphysics contains **zero ML compute**. It is purely an orchestration layer. All scoring, embedding, similarity computation, and recommendation generation happens in two backend services:

- **Gravity**: similarity via `related/artworks`, `related/layers`, `gene/{id}/similar`, `artworks_discovery` (KNN+MLT), `artwork_duplicate_pair/detect`
- **Vortex**: affinity scores, artist/artwork recommendations, new-for-you, price insights, demand/liquidity ranks, AI captions

This separation is clean and deliberate. The GraphQL schema defines what questions you can ask. The backends define how those questions are answered. Replacing the intelligence layer — substituting a different recommendation engine, a different similarity metric, a different pricing model — requires no changes to the schema. The types and fields remain the same; only the resolvers' upstream calls change. This is the advantage of an orchestration architecture over a monolithic one.

## The Artwork type

The Artwork type has over 150 fields, organized into domains:

### Identity and description

`title`, `date`, `medium`, `category`, `description`, `slug`. The `medium` field is a free-text string — "oil on canvas," "bronze," "chromogenic print" — not a structured decomposition. There is no formal medium taxonomy in the schema itself; the Art Genome Project's Medium and Techniques family serves that role through gene associations.

### Physical properties

`height`, `width`, `depth`, and `diameter` in both inches and centimeters, with derived fields: `aspectRatio`, `sizeScore` (a computed numeric rank), `sizeBucket` (categorical: SMALL, MEDIUM, LARGE), and `orientation` (square, landscape, or portrait). Size matters for collectors in ways that most recommendation systems ignore — a 6-foot canvas and a 6-inch study may be by the same artist in the same style but serve entirely different purposes.

### Genome data

The `partnerGenome` field bridges the Artwork type and the Art Genome Project. It exposes a connection of genes with integer values from 0 to 100:

```graphql
type PartnerGenome {
  genesConnection: PartnerGenomeGenesConnectionConnection
}
type PartnerGenomeGene {
  geneValue: Int!  # 0–100
}
```

The Gravity data model (visible through the `quantum` experiments' export code) reveals that artworks actually carry **four separate genome layers**:

- `genome` — the core Art Genome Project annotations, applied by human genomers
- `automated_genome` — ML-predicted classifications
- `partner_genome` — genes applied by gallery partners through the partner portal
- `visual_genome` — computationally derived visual features

These four layers merge into `total_genome` for querying. The GraphQL schema exposes only `partnerGenome`, but the internal data model distinguishes the provenance of each classification — human expert, automated model, gallery partner, or computer vision pipeline.

### Market data

`price`, `priceCurrency`, `listPrice`, `saleMessage`, `priceMin`, `priceMax` (as structured Money objects). Beyond basic pricing: `marketPriceInsights` (demand rank, liquidity, annual lots sold, estimated value), `pricingContext` (histogram distribution from Vortex analytics), `comparableAuctionResults`, and `realizedPrice`.

The demand and liquidity fields reveal Artsy's scoring approach. `demandRank` is a float from Vortex, thresholded as "high demand" when `demandRank * 10 >= 9`. `liquidityRank` is classified into four tiers: <0.25 Low, 0.25–0.70 Medium, 0.70–0.85 High, 0.85+ Very High. These are market-position metrics, not content metrics, but they participate in the same recommendation ecosystem.

### Collector engagement

`isSaved`, `isSavedToList`, `isDisliked`, `recentSavesCount`. The `collectorSignals` field bundles behavioral indicators: `curatorsPick`, `increasedInterest`, and auction-specific signals. These are the raw inputs to collaborative filtering — they encode not what art looks like but how people respond to it.

### ML-derived fields

Several fields pass through ML scores from Gravity and Vortex:

- `caption` — AI-generated alt text via Vortex's `artwork_captions` endpoint (cached for 24 hours)
- `completeness_tier` — data quality score ("Incomplete," "Good," "Excellent")
- `sizeScore` — numeric size ranking used to bucket artworks for price comparison

Sorting parameters on artwork queries reveal Gravity's internal scoring: `-merchandisability` and `-decayed_merch` are opaque but load-bearing. Artworks sorted by merchandisability appear in artist grids and gallery pages — it is the default ranking, and it is not publicly documented.

## The Gene type

Genes are first-class entities in the schema. The resolver code in `gene.ts` reveals how gene mode is determined — by matching the gene family name against patterns like "content," "medium," "technique," "appearance genes." Subject-matter genes show artworks; non-subject-matter genes show artists. The `similar` field calls `gene/{id}/similar` on Gravity, which returns related genes — the similarity computation happens entirely within Gravity and is not exposed.

## Vortex: the ML service

The Vortex GraphQL schema is 1,777 lines, committed as `src/data/vortex.graphql`. It exposes:

| Query | Returns |
|:------|:--------|
| `artistRecommendations` | `{artistId, score: Float}` — ML-based artist recommendations |
| `artworkRecommendations` | `{artworkId, score: Float}` — ML-based artwork recommendations |
| `newForYouRecommendations` | `{artworkId, score, version, publishedAt}` — personalized feed |
| `artistAffinities` | `{artistId, score}` — per-user artist affinity scores |
| `marketPriceInsightsBatch` | Demand rank, liquidity, sell-through rate, price ranges by size |
| `pricingContext` | Histogram bins of comparable artwork prices |
| `partnerStats` | Gallery analytics: pageviews, sales, inquiry counts |

The `version` field on `newForYouRecommendations` suggests Artsy runs multiple recommendation model versions simultaneously — likely for A/B testing different algorithms. The `score` fields are bare floats with no documented range or meaning, indicating they are relative rankings rather than calibrated probabilities.

Vortex is the name: it appears in config vars (`VORTEX_API_BASE`, `VORTEX_APP_ID`, `VORTEX_TOKEN`), in the schema stitching code, and in loader factories. It is absent from any public blog post, conference talk, or README. The service is genuinely internal — not just proprietary code but a proprietary name that has barely leaked into public artifacts.

## The Infinite Discovery resolver

The most ML-explicit code in metaphysics is `src/schema/v2/infiniteDiscovery/discoverArtworks.ts`. It reveals a hybrid search architecture:

```
mltFields: ["genes", "materials", "tags", "medium"]
osWeights: [0.6, 0.4]
```

This calls `artworks_discovery` on Gravity, which runs a blended search: 60% k-nearest-neighbor (KNN) vector similarity and 40% More Like This (MLT) text similarity. The MLT fields — genes, materials, tags, medium — are the text features that Elasticsearch/OpenSearch uses for fuzzy matching. The KNN component presumably operates on some form of embedding vector stored in the search index, though the vector representation is not specified in the metaphysics code.

The resolver takes `likedArtworkIds` as a "taste profile" input and blends curated picks with algorithmic results via `curatedPicksSize`. This is the same infinite-scroll discovery experience that the `quantum` experiments (Chapter 7) are prototyping a replacement for.

## Image infrastructure

Images flow through a sophisticated CDN with no computational intelligence at the schema level:

**CDN**: CloudFront at `d32dm0rphc51dk.cloudfront.net/{token}/:version.jpg`. Versions: large, medium, small, square, tall, normalized.

**Transformations**: On-demand crop and resize via the Gemini image service. Both `cropped()` and `resized()` return structured types with `url`, `src`, `srcSet` (1x and 2x for retina), and dimensions. Default quality: 85.

**Deep Zoom**: Microsoft DZI format for high-resolution viewing — tile-based pan-and-zoom on artwork images.

**Placeholder**: `blurhash` — a compact perceptual hash of color distribution for instant placeholder rendering.

No embedding fields, no feature vectors, no visual similarity data. The images are served for display. The `quantum` experiments' use of Artsy's CDN for CLIP embedding (resizing to 224x224 at quality 75 for vector computation) happens entirely outside the metaphysics schema.

## Search

Text-based, powered by OpenSearch. Seventeen searchable entity types. Three modes: `SITE`, `AUTOSUGGEST`, `INTERNAL_AUTOSUGGEST`. Server-side highlights mark matched terms with `<em>` tags. No visual search exists in the codebase — no "find artworks that look like this image" endpoint.

## What the schema reveals

The Metaphysics schema is a map of institutional priorities:

**Market data is richly structured.** Pricing, auction results, demand rank, liquidity — first-class fields with dedicated types and computed aggregations.

**Art-historical data is semi-structured.** Genes provide structured classification, but medium, provenance, and exhibition history remain free text.

**Visual data is served, not analyzed.** The image CDN is sophisticated; the schema contains no visual intelligence.

**Behavioral data is captured but opaque.** Saves, dislikes, and views flow in; recommendations flow out. The transformation between them happens entirely within Vortex and Gravity, invisible to the schema.

**The intelligence layer is cleanly separated.** Metaphysics defines the questions; Gravity and Vortex answer them. This separation means the intelligence can be upgraded, replaced, or augmented without changing the API contract — which is exactly what the `quantum` experiments are prototyping.

---

*Source: [github.com/artsy/metaphysics](https://github.com/artsy/metaphysics) (MIT). Schema analysis based on `_schemaV2.graphql`, resolver source code, and `src/data/vortex.graphql`, surveyed May 2026.*
