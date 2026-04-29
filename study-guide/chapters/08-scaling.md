## 8. Scaling Realities: UMAP, FAISS, and the Geometry of Large Corpora

The program operates at scales where the difference between an algorithm that works and an algorithm that ships is not cleverness but data structure. At 690k artworks, vanilla UMAP runs but takes an hour. At 3M, it may not finish. At 10M, you need a different algorithm entirely — or, more precisely, you need the same algorithm with a different engine under the hood. This section covers the three scaling problems the program faces and the engineering that solves each: dimensionality reduction for visualization, approximate nearest-neighbor search for real-time serving, and tile-pyramid rendering for putting hundreds of thousands of images in a browser.

### 8.1 UMAP at scale

You already know dimensionality reduction. PCA finds the linear subspace that captures maximum variance. t-SNE preserves local neighborhood structure by matching probability distributions in high and low dimensions. UMAP is the successor to t-SNE, and the Phase 0 visualizer uses it to project the 13-dimensional Iigaya feature space down to the 2D coordinates that become the "art landscape" in the browser.

What UMAP actually does — at the level needed to see where it breaks — is manifold learning via fuzzy simplicial sets. The core idea in three steps:

1. **Build a local neighborhood graph.** For each point, find its $k$ nearest neighbors in the high-dimensional space. Weight the edges by distance, normalized so that each point's local notion of "nearby" is calibrated relative to its own local density. This is a fuzzy simplicial set — a topological structure that says "these points are connected, with this degree of confidence."

2. **Construct the same kind of graph in low dimensions.** Initialize 2D coordinates (usually via spectral embedding of the neighbor graph, then refined by stochastic gradient descent). Define a low-dimensional fuzzy simplicial set using the same local-distance weighting.

3. **Minimize the cross-entropy between the two graphs.** Adjust the low-dimensional coordinates so that the low-dimensional neighborhood structure matches the high-dimensional one as closely as possible. Points that are neighbors in high-D should be neighbors in 2D; points that are distant in high-D should be distant in 2D.

The improvement over t-SNE is twofold. First, global structure: t-SNE's cost function (KL divergence on pairwise probabilities) is dominated by local relationships, so clusters that should be far apart can end up adjacent in the projection. UMAP's cross-entropy cost penalizes both "neighbors that got separated" and "non-neighbors that got pushed together," which preserves more of the large-scale geometry. The clusters in the art landscape are not just internally coherent — their relative positions are meaningful. Impressionist landscapes being near Post-Impressionist landscapes, and both being far from Japanese woodblocks, is a global-structure claim that t-SNE would not reliably preserve. Second, speed: UMAP is substantially faster, partly from algorithmic improvements and partly from avoiding the $O(N^2)$ pairwise probability computation that t-SNE requires.

**The k-NN bottleneck.** Step 1 is where scaling hurts. Finding the $k$ nearest neighbors of each point requires, naively, computing all $\binom{N}{2}$ pairwise distances. At $N = 690{,}000$ and $d = 13$, that is about $2.4 \times 10^{11}$ distance computations. UMAP's default implementation (via the `pynndescent` library) uses random-projection forests and neighbor-of-neighbor exploration to approximate this, and at $N \sim 500$K it works in reasonable time — roughly 30-60 minutes on a modern CPU.

But the cost grows superlinearly. At 3M points, `pynndescent` takes hours or may exhaust memory. At 10M, it is not practical. This is the scaling wall, and it is specifically a neighbor-search wall, not a UMAP wall — once you have the neighbor graph, the SGD layout phase scales roughly linearly with $N$.

**Using FAISS to precompute the neighbor graph.** The solution is to replace UMAP's internal neighbor search with FAISS, which is built for exactly this problem (more on FAISS in Section 8.2). The pattern:

1. Build a FAISS HNSW index over the high-dimensional embeddings.
2. Query the index for $k$ neighbors of each point.
3. Pass the precomputed neighbor graph to UMAP via the `precomputed_knn` parameter.

This is what the Phase 0 visualizer does. The FAISS neighbor search at 690k points with 13-dimensional features takes under a minute. The UMAP layout step then takes roughly 10-20 minutes. Total: maybe 30 minutes on CPU, versus an hour or more with UMAP's native neighbor search.

At 3M points, the same pattern holds: FAISS HNSW neighbor search scales well past 10M vectors (that is what it was built for), and the UMAP layout phase on 3M points takes perhaps an hour on a GPU (via cuML, NVIDIA's GPU-accelerated UMAP implementation). At 10M, the UMAP layout itself becomes the bottleneck. This is where parametric UMAP enters.

**Parametric UMAP.** Standard UMAP is non-parametric: it produces coordinates for a fixed set of points but has no function that can map a new point into the existing layout. If you add 10,000 new artworks next week, you recompute UMAP on the full 700k. This is the stability problem that matters for the "living surface" requirement (Section 8.3 covers this in detail).

Parametric UMAP replaces the coordinate-optimization step with a neural network. Instead of directly optimizing 2D coordinates, you train a small MLP (or a deeper network) to predict 2D coordinates from the high-dimensional input. The loss function is the same UMAP cross-entropy — the network learns to produce a layout that preserves the neighborhood structure. Once trained, the network is a function: hand it a new 768-dimensional embedding, and it returns a 2D coordinate in the existing layout, in milliseconds, with no recomputation.

The tradeoff: parametric UMAP produces slightly worse layouts than non-parametric UMAP (a learned function is a weaker optimizer than direct coordinate SGD). The gain: stability, speed at inference, and the ability to handle the 5M-to-20M scale by training on a representative sample (say, 500k landmarks chosen by stratified sampling) and then projecting the rest through the learned function. This is the production path for Phase 1 onward, when the corpus grows past what non-parametric UMAP can handle gracefully.

When to use which:

| Scale | Approach |
|:------|:---------|
| < 500K | Vanilla UMAP on CPU. Precompute neighbors with FAISS if you want speed. |
| 500K -- 3M | FAISS neighbor graph + GPU UMAP (cuML). Or parametric UMAP trained on a 500K sample, rest projected. |
| 3M -- 20M+ | Parametric UMAP on a landmark sample. All non-landmark points projected via the learned function. Possibly hierarchical: cluster first (HDBSCAN or k-means), UMAP within each cluster, stitch together. |

### 8.2 FAISS and approximate nearest neighbors

FAISS (Facebook AI Similarity Search) is the library that makes two things possible in this program: the neighbor graph that UMAP needs (Section 8.1), and the real-time retrieval that the two-tower recommendation system needs (Chapter 6). The enabling algorithm inside FAISS for both tasks is HNSW — Hierarchical Navigable Small World graphs.

**Why exact nearest-neighbor search is too slow.** The recommendation system computes a user embedding at query time and finds the $K$ nearest item embeddings. At 1M items with 768-dimensional embeddings, exact search means 1M dot products over 768 floats — roughly 30-50 ms on CPU with optimized BLAS. Tolerable in isolation, but the production system serves many queries per second, each must return in under 100 ms end-to-end, and the corpus will grow to 3-10M items. At 10M on CPU: 300-500 ms per query, far too slow.

**The HNSW algorithm.** HNSW builds a multi-layer graph over the embedding space that allows navigating from any starting point to the neighborhood of any query point in $O(\log N)$ hops. The intuition — and this is the best bridge to something you already know — is a skip list in high-dimensional space.

A skip list is a data structure for 1D ordered search: you maintain multiple "express lanes" of decreasing density over the same sorted data. To find an element, you start at the top (sparsest) layer, take big jumps until you overshoot, drop down one layer, take smaller jumps, and so on until you reach the bottom layer where every element is present. The total search time is $O(\log N)$ instead of $O(N)$.

HNSW does the same thing in high-dimensional metric spaces. The construction:

1. **Layer 0 (bottom):** contains every vector. Each vector is connected to its $M$ nearest neighbors (where $M$ is a tunable parameter, typically 16-64). This is a navigable small-world graph — you can reach any node from any other node by greedily following the edge to the neighbor closest to your target, in a small number of hops.

2. **Layer 1:** contains a random subset of the vectors (probability of inclusion decays exponentially with layer). Same navigable-small-world structure, but sparser — longer jumps.

3. **Layer $L$ (top):** contains just a few vectors. Entry point for search.

To search for the $K$ nearest neighbors of a query vector $q$:

- Start at the top layer. Greedily navigate to the node closest to $q$ (at each step, move to the neighbor that is closest to $q$, stop when no neighbor is closer than the current node).
- Drop to the next layer. Use the arrival node as the entry point. Greedily navigate again, but now keep a priority queue of the best $K$ candidates seen so far, and explore more broadly (checking each candidate's neighbors).
- Continue dropping layers until Layer 0. The priority queue at the end contains the approximate $K$ nearest neighbors.

The "approximate" is key. HNSW does not guarantee finding the true nearest neighbors — a greedy path through the graph might miss a closer point that is not reachable via the explored edges. The parameter `ef_search` (size of the dynamic candidate list during search) controls the recall-speed tradeoff: larger `ef_search` explores more of the graph, finding more true neighbors but taking longer.

**Concrete numbers.** These are realistic benchmarks for the scales the program will encounter, on 768-dimensional embeddings (the SigLIP output dimension already in use):

| Corpus size | Exact search (CPU) | HNSW (CPU) | HNSW recall@10 | Index memory |
|:------------|:-------------------|:-----------|:----------------|:-------------|
| 690K | ~25 ms | ~1 ms | 95-98% | ~2.5 GB |
| 1M | ~40 ms | ~1-2 ms | 95-98% | ~3.5 GB |
| 3M | ~120 ms | ~2-3 ms | 93-97% | ~10 GB |
| 10M | ~400 ms | ~3-5 ms | 92-96% | ~35 GB |

These assume `M=32` (edges per node), `ef_construction=200` (exploration breadth during index building), and `ef_search=128` (exploration breadth during query). The recall numbers are recall@10 — the fraction of the true 10 nearest neighbors that HNSW returns. At 95% recall, you miss about 1 of the 10 nearest neighbors on average, replaced by a point that is nearly as close. For recommendation (where you are retrieving candidates for a downstream ranker, not an exact answer), this is more than sufficient.

The 25x-to-100x speedup is what makes two-tower retrieval servable in real time. Without approximate nearest-neighbor search, you cannot serve personalized recommendations at this scale without either enormous GPU budgets or unacceptable latency. This is not a minor optimization — it is the enabling technology for the production architecture described in Chapter 6.

**Building the index.** Index construction is more expensive than search — you are building the multi-layer graph offline. At 1M vectors, HNSW index construction takes roughly 5-10 minutes on CPU. At 10M, roughly an hour. This is a batch operation, run after each model retrain, not on the serving path. The `batch-embed.py` script already builds a FAISS index as part of its pipeline (using `IndexFlatIP` for exact search — the move to HNSW is a Phase 2+ upgrade when corpus size demands it).

**IVF+PQ as a memory-constrained alternative.** FAISS also offers inverted-file indexing with product quantization (IVF+PQ), which compresses vectors from ~3 KB to 64-128 bytes, enabling 10M vectors to fit in ~1 GB of RAM. The tradeoff is lower recall (85-92%) and a training step for the quantization codebook. At the program's current scale, HNSW's memory footprint is manageable and the higher recall is worth it. IVF+PQ becomes relevant past 10M items.

### 8.3 Parametric vs. non-parametric dimensionality reduction

This distinction matters specifically because of the visualizer's "living surface" requirement. The Phase 0 visualizer is not a one-time rendering — it is an interactive surface that must accommodate new artworks as they are ingested without tearing up the existing layout. The choice between parametric and non-parametric dimensionality reduction determines whether that is cheap or expensive.

**The stability problem.** Standard (non-parametric) UMAP optimizes coordinates directly. Adding 1,000 new artworks to a 690K-point layout means re-running UMAP on 691K points. The new layout will be topologically similar — the same clusters, the same relative positions — but the absolute coordinates will be different. Every point moves. This means every tile in the rendering pyramid is invalidated, every cached thumbnail position is wrong, and the user's mental model of "Dutch portraits are in the upper left" may break because the upper left is now something else.

You can mitigate this with initialization tricks (initialize the new UMAP run with the previous coordinates, fix the old points and only optimize the new ones), but these are heuristics. UMAP's loss function is non-convex and its solutions are not unique; even warm-starting from a good initialization, the optimizer may drift enough to rearrange parts of the layout.

**Parametric UMAP solves this.** A parametric model is a function $f: \mathbb{R}^{d} \to \mathbb{R}^{2}$ (where $d$ is the embedding dimension — 13 for Iigaya features, 768 for the learned embedding in Phase 1+). Once trained, $f$ maps any input vector to a 2D coordinate deterministically. New artworks get mapped by a forward pass through $f$ — no recomputation, no coordinate drift for existing points.

The function $f$ is typically a small MLP: 3-4 hidden layers, 256-512 units each, with batch normalization and ReLU activations. Training uses the same UMAP loss (fuzzy-simplicial-set cross-entropy) as the non-parametric version, on a training set of embedding vectors. The training set should be representative of the manifold — stratified sampling across clusters, not a random subsample that could oversample dense regions and undersample the tails.

**Training cost.** On a single GPU, training parametric UMAP on 500K points with 768 dimensions takes roughly 30-60 minutes. Once trained, inference is nearly free: mapping 10,000 new points takes under a second. This is the asymmetry that makes parametric UMAP the production path — you pay once (at each model retrain, since the embedding space changes) and get stable, instantaneous projection for all subsequent ingestion.

**When non-parametric is still the right choice.** For exploratory analysis, non-parametric UMAP is simpler and produces slightly better layouts. Phase 0, with a fixed corpus, correctly uses non-parametric. The switch to parametric happens when the corpus starts growing continuously.

**What happens when the embedding model retrains.** Parametric UMAP learns a function from a specific embedding space. When the embedding model retrains, the space changes and the parametric UMAP must be retrained. This is unavoidable: any layout is stable only within a model version. The practical cadence is: retrain embeddings (days), rebuild FAISS index (minutes), retrain parametric UMAP (an hour), regenerate tiles (hours). A batch pipeline, not a real-time concern.

### 8.4 Tile-pyramid rendering

How do you put 690,000 artwork thumbnails in a browser?

You do not. You put a small number of tiles in the browser, and you swap tiles as the user pans and zooms. This is the same approach that Google Maps, OpenStreetMap, and every other web mapping application uses — and it is applied here to embedding-space coordinates instead of geographic coordinates. The engineering is well understood. The application to UMAP layouts is a matter of wiring, not invention.

**The quadtree tiling approach.** The 2D space of UMAP coordinates is recursively divided into four quadrants, then each quadrant into four sub-quadrants, down to a maximum zoom depth. Each subdivision is a "tile" — a square region of the 2D space. At zoom level $z$, the space is divided into $4^z$ tiles (though most are empty and not generated).

At each zoom level, a tile contains:
- A data file (Apache Arrow feather format) with the coordinates, metadata, and texture-atlas references for every artwork whose 2D position falls within that tile's bounds.
- One or more texture atlas sprite sheets — PNG images containing a grid of artwork thumbnails packed side-by-side.

**Resolution levels tied to zoom.** The progressive-reveal UX works because different zoom levels carry different data:

| Zoom level | What the user sees | Data loaded per tile |
|:-----------|:-------------------|:---------------------|
| z=0 to z=2 (landscape) | Colored dots. Each dot is one artwork. | Arrow file with coordinates, corpus label, cluster ID. No images. ~50 KB per tile. |
| z=3 to z=5 (neighborhood) | Thumbnail sprites at ~64px. | Arrow file + texture atlas sprite sheet (256 thumbnails in a 16x16 grid, ~200 KB PNG). |
| z=6 to z=7 (gallery wall) | Individual thumbnails at ~256px with metadata overlay. | Arrow file + high-res sprite sheet + full metadata columns. |

The browser loads only the tiles visible in the current viewport, at the current zoom level. Pan left: load the tiles to the left, discard the tiles that scrolled off-screen. Zoom in: replace the current tile with its four children at the next zoom level. This is exactly how slippy-map tile servers work.

**Apache Arrow feather files.** The data layer uses Apache Arrow's feather format rather than JSON or CSV because feather files are columnar, typed, and designed for zero-copy memory mapping. The browser reads them with `apache-arrow` for JavaScript, and the columns (float32 coordinates, string metadata, uint16 atlas indices) are immediately available for WebGL rendering without parsing. At 690K artworks, the total Arrow file size is roughly 50-100 MB across all tiles — easily cacheable, fast to serve from a CDN.

**Texture atlas sprite sheets.** The image layer packs thumbnails into sprite sheets at build time. This is the PixPlot pattern: instead of loading 256 individual 64px JPEG files (256 HTTP requests, significant overhead), you load a single 1024x1024 PNG containing all 256 thumbnails in a grid. The Arrow file for each tile includes UV coordinates (which sub-rectangle of the sprite sheet corresponds to which artwork). The WebGL renderer draws a textured quad for each artwork, sampling from the sprite sheet.

At 690K artworks with ~64px thumbnails in 16x16 sprite sheets, you need about 2,700 sprite sheets. At ~200 KB each, that is roughly 540 MB total. Not all are loaded at once — only the tiles in the viewport are live. Typically 20-40 tiles are visible at any moment at the neighborhood zoom level, corresponding to 5,000-10,000 thumbnails in 20-40 sprite sheets — well within a modern browser's texture memory.

**What about 3M or 10M artworks?** The tiling approach scales linearly: more artworks means more tiles at the deepest zoom levels, but the number of tiles visible at any moment stays constant (it is determined by the viewport size, not the corpus size). The build-time cost of generating sprite sheets grows linearly — at 10M artworks, you would generate roughly 40,000 sprite sheets, taking perhaps a day of CPU time. The serving cost does not change: the CDN serves exactly the tiles the user requests, and the user requests roughly the same number regardless of total corpus size.

The deep-zoom tile generation is embarrassingly parallel (each tile is independent) and can be distributed across as many workers as you have. There is no algorithmic barrier to 20M artworks — only storage and build-time costs that grow linearly.

**The implementation stack.** The Phase 0 prototype uses a Canvas 2D renderer that iterates over all 690K points — workable on desktop, insufficient on mobile and at larger scales. The production path is either Deepscatter (by Nomic/Ben Schmidt, purpose-built for quadtree-tiled Arrow data at 20M+ points) or deck.gl (Uber's general-purpose WebGL layer, more flexible, more engineering). Both are proven at scale. The choice is engineering pragmatics, not research.

### 8.5 Memory and compute budgets

The numbers that determine what fits where. These are practical estimates — they should be checked against actual hardware before committing to an architecture, but they are grounded in published benchmarks and the program's known parameters.

**Embedding storage.** A single embedding vector at 768 dimensions in float32 is $768 \times 4 = 3{,}072$ bytes, or about 3 KB.

| Corpus size | Raw embeddings | FAISS HNSW index | Iigaya features (13-d) | UMAP coordinates (2-d) |
|:------------|:---------------|:-----------------|:-----------------------|:-----------------------|
| 690K | 2.0 GB | 2.5 GB | 34 MB | 5.3 MB |
| 3M | 8.8 GB | 11 GB | 149 MB | 23 MB |
| 10M | 29 GB | 35 GB | 497 MB | 76 MB |

**What fits in RAM.** A standard cloud VM with 64 GB RAM can hold the embeddings and FAISS index for up to about 3M artworks. At 10M, you need either a high-memory instance (128-256 GB) or a strategy that avoids loading everything at once (IVF+PQ compression, sharded indices, or GPU-based search which streams from host memory). The current bot VM is memory-constrained — the Phase 0 job file notes this as a reason to run UMAP on a GCP instance rather than the bot server.

**What fits in GPU memory.** A single A100 has 80 GB of HBM. During training, the model (ViT-L/14 is about 1.2 GB in float32, 600 MB in float16), the optimizer states (roughly 3x model size for Adam), the batch data, and the activations compete for that memory. A batch of 8K images at 224x224 with a ViT-L backbone fills most of an A100. This is why the program specifies 8x A100 — each GPU handles a fraction of the batch, and the contrastive loss requires large batches (8K-32K) to provide enough in-batch negatives.

For inference (embedding computation, not training), a single GPU can process the full corpus: 690K images at a throughput of roughly 200-500 images/second on a ViT-L, meaning the full corpus embeds in 20-60 minutes. The `batch-embed.py` script distributes across 8 GPUs, reducing this to under 10 minutes.

**FAISS index construction.** Building an HNSW index is CPU-bound and single-threaded in FAISS's current implementation (the graph construction is inherently sequential per insertion, though FAISS parallelizes some internal operations). At 1M vectors, expect 5-10 minutes. At 10M, roughly an hour. Memory during construction is approximately 2x the final index size (the graph is built incrementally, with working buffers). This runs as a batch job after embedding computation, not on the serving path.

**Tile pyramid generation.** The build pipeline for the visualizer — UMAP projection, HDBSCAN clustering, quadtree partitioning, sprite sheet generation — is CPU-bound and I/O-bound (fetching images, writing PNGs). At 690K artworks:

| Step | Time estimate | Bottleneck |
|:-----|:-------------|:-----------|
| FAISS k-NN graph | 1-2 min | CPU compute |
| UMAP layout | 10-30 min (CPU) or 2-5 min (GPU via cuML) | CPU/GPU compute |
| HDBSCAN clustering | 2-5 min | CPU compute |
| Quadtree partitioning + Arrow files | 5-10 min | CPU + disk I/O |
| Sprite sheet generation | 1-3 hours | Image fetching + PNG encoding |

Total: 2-4 hours, dominated by sprite sheet generation (which is embarrassingly parallel and could be reduced to 30 minutes with 8 workers). At 3M artworks, multiply sprite sheet time by roughly 4x. At 10M, by roughly 15x — but by that point the sprite sheets are generated in a distributed pipeline and the wall-clock time depends on how many workers you throw at it, not on the corpus size.

**Serving costs.** The tile pyramid is a static file tree on object storage or a CDN. No server-side computation at browse time. Storage: ~500 MB at 690K, ~2 GB at 3M, ~7 GB at 10M. CDN bandwidth is negligible. The real cost is the batch pipeline that builds the tiles, not serving them.

**The cost that matters.** At the program's current and near-term scales, none of these resource requirements are bottlenecks. The embeddings fit in RAM, the FAISS index fits in RAM, the tile pyramid fits on cheap storage, and the batch pipeline runs in hours on a single machine. The first real architectural decision point arrives around 10M items, when the FAISS index exceeds commodity RAM and you must choose between high-memory instances, GPU-based search, or IVF+PQ compression. That decision can be deferred until the corpus approaches that scale.
