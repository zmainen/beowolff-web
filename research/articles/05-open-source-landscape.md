# The Open-Source Landscape

Artsy has 298 public repositories on GitHub. This is an unusually large open-source footprint for a company in the art world — or, for that matter, for a marketplace of any kind. Their decision to open-source their production mobile app (a rarity even in tech) reflects a specific philosophy about competitive advantage: the code is not the moat; the data, the network, and the institutional relationships are.

Surveying the full corpus reveals both the breadth of what they have published and the precision of what they have not.

## What is published

### Frontend and web

**force** (633 stars) is the artsy.net website itself — React, Relay, Express SSR. One of the earliest successful isomorphic JavaScript applications when it was open-sourced in 2014, with over 41,000 commits to date. It is the most actively developed repository in the org: 165 commits in the last four weeks, with continuous deployment — multiple deploy PRs merged per day via the `artsyit` bot.

**fresnel** (1,296 stars) is an SSR-compatible responsive layout library built on CSS media queries. Unlike most responsive solutions, it renders all breakpoints server-side and hydrates client-side with `matchMedia`. General-purpose, usable by anyone. The most-starred Artsy library by a wide margin.

**palette** (221 stars) is their design system — primitive UI elements on styled-system, shared across React and React Native. **reaction** (356 stars) is a shared React component library. **positron** (86 stars) is their editorial CMS.

### Mobile

**eigen** (3,771 stars) is the crown jewel: Artsy's flagship mobile app, fully open-sourced, React Native + Expo, with 28,000+ commits and biweekly releases like clockwork (9.4.0 → 9.5.0 → 9.6.0 → 9.7.0, roughly every two weeks through April 2026). Eighty percent of recent merged PRs are automated — schema syncs from metaphysics, Expo runtime version bumps, release preparation. The humans write features; the machines keep the plumbing running.

**eidolon** (2,704 stars, archived) was an auction bidding kiosk in Swift — historically significant as an early large-scale Swift application. **Emergence** (351 stars, archived) was an Apple TV app for art shows.

### Infrastructure and DevOps

**hokusai** (98 stars) is a Docker + Kubernetes deployment CLI — Artsy's primary deployment tool. Actively maintained through March 2026.

**docker-preoomkiller** (18 stars) watches container memory and sends SIGTERM before Docker's OOM killer strikes. **orbs** packages their CircleCI configuration. **artsy-local** runs the full Artsy stack locally.

### Backend and APIs

**metaphysics** (366 stars) is the central GraphQL gateway, covered in detail in Chapters 3 and 4. Sixty-five commits in the last four weeks, with ~4–5 active contributors.

**garner** (342 stars) is a Rack-based HTTP cache with 304 support. **cohesion** (16 stars) defines the analytics event schema — every tracking event flowing to Redshift/Segment. With 526 releases, it is one of the most actively iterated repos despite its low star count.

### Search and data structures

A cluster of mathematically interesting Ruby libraries for probabilistic data structures on Redis:

- **forgetsy** — trending analysis using exponentially decaying counters. Artsy used this to detect real-time trends at art fairs, tracking artist popularity over hours rather than weeks. The engineering blog describes the math: two counter sets with decay rates λ and 2λ, normalized against each other to produce a dimensionless 0–1 trending score. Handles ~10^6 categories at hundreds of writes per second.
- **hyperloglog-redis** — cardinality estimation with HyperLogLog.
- **space-saver-redis** — top-K estimation using the Space-Saver algorithm.

### Art data and classification tooling

**the-art-genome-project** (39 stars) is covered in Chapters 1 and 2.

**rosalind** (6 stars) is the batch genome operations tool — Rails + React, connected to Gravity and OpenSearch, used by genomers for large-scale classification work. Named after Rosalind Franklin.

An internal tool called **Helix** (not open-sourced, but described in a 2015 blog post) is the individual-artwork classification interface where genomers add, remove, and modify gene assignments with autosaving across hundreds of editable fields per artwork.

### iOS ecosystem contributions

Artsy was historically a major contributor to the iOS ecosystem. **Moya** (network abstraction layer in Swift) originated at Artsy and grew into an independent project with 15,000+ stars. **ARAnalytics**, **ARTiledImageView**, and **ARCollectionViewMasonryLayout** were all Artsy contributions that predated their shift from native iOS to React Native.

## What is not published — and what is

The original assessment was straightforward: Artsy has published no ML work. The deeper investigation complicates this. Their production ML remains private, but their prototype next-generation system is public.

### The production intelligence layer (private)

The systems that power similarity, recommendation, and market analytics in production are entirely proprietary:

- **Gravity** — the Rails monolith that stores all core data and computes artwork similarity via `related/artworks`, gene similarity via `gene/{id}/similar`, and the KNN+MLT hybrid search for discovery. Gravity is not open-source.
- **Vortex** — the ML analytics service that generates artist/artwork recommendations, affinity scores, demand/liquidity rankings, and AI-generated captions. Its 1,777-line GraphQL schema is visible in the metaphysics repo, but its implementation is private. The name "Vortex" does not appear in any public blog post or conference talk.
- **Iconicity scoring** — a Spark-based system described in a 2017 blog post that ranks artworks by importance. Features: artist follower counts, museum affiliations, genome classifications, engagement metrics. Museum affiliation receives 2× the weight of raw popularity — curatorial significance matters more than traffic. The computation runs across ~1 million artworks in under five minutes. The `iconicity` field still appears in metaphysics sort options.
- **Automated genome classification** — 283 of the 1,031 genes are marked as automated. The Gravity data model reveals a `visual_genome` field alongside `genome`, `automated_genome`, and `partner_genome`, strongly suggesting a computer vision pipeline for automated classification. No public code exists for any of these.

### The prototype intelligence layer (public)

The `quantum` repository (3 stars, covered in Chapter 7) contains 17 experiments prototyping a replacement stack: CLIP multimodal embeddings in Weaviate, LLM-powered re-ranking, ref2vec-centroid user profiles. This is real engineering against real data — staging endpoints, actual Gravity exports, production-grade metadata. It is not deployed, but it reveals Artsy's strategic direction clearly.

### Historical ML artifacts (public, archived)

**match** (archived October 2023) — reverse image search using the Goldberg perceptual hashing algorithm. Kubernetes + Elasticsearch. Not deep learning — DCT coefficients and spatial frequency analysis for binary hash codes. Useful for duplicate detection, not stylistic similarity.

**pixmatch** — Ruby client for TinEye's commercial reverse image search. **barium-ion** (archived) — experiments with Google Image Search. **custom-elasticsearch-similarity** (archived) — custom Elasticsearch scoring plugin.

These archived repos trace Artsy's journey through visual similarity approaches: TinEye's commercial API, Google's visual search, custom Elasticsearch scoring, perceptual hashing, and eventually whatever internal capability superseded all of them.

### Published ML knowledge (blog posts)

Artsy's engineering blog contains several technically substantive posts about their ML and data work:

**Iconicity scoring** (2017): Spark + Hive on a Hadoop cluster (EC2, Cloudera Manager). Features weighted via `ElementwiseProduct`, normalized via `StandardScaler`. The infrastructure processes ~1M artworks in under 5 minutes.

**Trend detection** (2014): The Forgetsy system. Linear, weighted combinations of decay-rate deltas produce trending scores. Used at The New York Armory Show on physical terminals displaying real-time trends.

**Automatic frame cropping** (2014): Classical computer vision (not deep learning) with OpenCV. Canny edge detection, contour search, rectangular filtering. 85% success rate on 5,000 museum images. Dilation was the key preprocessing step: it thickened frame borders while flooding out internal detail edges.

**Technology stack** posts (2012, 2015, 2017) reference a "similarity graph" and an "artwork similarity graph" as core infrastructure. The 2015 post mentions Amazon Elastic MapReduce for processing; the 2017 post mentions Elasticsearch powering "real-time artwork similarity features." The analytics team uses Jupyter, Redshift, pandas, scikit-learn, and pyplot.

No conference talks about Artsy's ML work are publicly discoverable. Their engineering team has been prolific on React Native, open source culture, and GraphQL, but ML/data science work has been kept internal.

## Engineering culture

The codebase quality tells a story about the organization:

**Team size**: ~12–18 engineers, with some working across repos. ~10 active public committers, 12–13 unique human authors on force, 6–7 on eigen, 4–5 on metaphysics. This is lean for a platform of Artsy's complexity.

**Automation**: Excellent. The `artsyit` bot handles deploys, schema syncs, and release preparation. Danger (via dangerfile.ts) enforces PR hygiene — changelog entries, migration enforcement, test-only file checks. Claude AI reviews PRs on metaphysics via a reusable workflow from `artsy/duchamp`. Maestro handles mobile E2E testing.

**TypeScript discipline**: Pragmatic. Metaphysics runs `strict: true` with `noImplicitAny: false` — strict overall but honest about where they punt. Force uses `strictNullChecks: true` selectively and skips the rest. The ESLint config disables `@typescript-eslint/no-explicit-any` and `ban-ts-comment` while keeping `no-unused-vars: error`. This is the configuration of engineers who actually run their linter.

**Testing**: 1,370 test files in eigen, 563 in metaphysics. Serious coverage for both a mobile app and an API gateway.

**Code review**: Mixed depth. Feature PRs from senior engineers get substantive review — one reviewer caught a security hole and a component duplication in a single review. Routine PRs merge in 8–14 minutes with one approval and no comments. This works for a trusted team but creates bus-factor risk.

**Dependency health**: Force and eigen are current (TypeScript 5.x, React 18, Expo 54). Metaphysics carries debt: TypeScript 4.6 and graphql 15, both significantly stale. The API layer — the most architecturally critical service — has the oldest dependencies. This is the classic pattern: the service that is hardest to upgrade is the one that falls furthest behind.

**Release discipline**: Eigen ships biweekly with automated release preparation. Force deploys continuously. Metaphysics ships as needed. The cadence is appropriate to each codebase's role.

## The strategic logic

Artsy's open-source strategy follows a clear principle: **publish the infrastructure, keep the intelligence.** The web app, mobile app, API gateway, deployment tools, design system, and analytics schema are all open. The gene scoring data, recommendation algorithms, visual similarity models, and ML infrastructure are all closed.

A competitor who cloned every public Artsy repository would have a competent art marketplace platform with no art data, no scored genome, no trained models, and no user behavior. They would have the pipes but no water.

The `quantum` repository is the one breach in this wall. Whether it is a deliberate signal (we are open to collaboration on ML) or an accidental exposure (the staging URLs and data export scripts are public) is unclear. But it provides a window into the intelligence layer that no other public artifact offers.

---

*Source: [github.com/artsy](https://github.com/artsy) (298 public repositories surveyed May 2026). Engineering metrics from GitHub API, blog posts from artsy.github.io. Star counts and archive status as of survey date.*
