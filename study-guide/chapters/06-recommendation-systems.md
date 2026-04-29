## Recommendation Systems: Collaborative Filtering, Two-Tower Retrieval, and the Cold-Start Problem

The Beowolff-Embed model exists to power a recommendation system: given a collector's history of interactions — what they browsed, saved, collected, inquired about, purchased — surface artworks they are likely to value. This chapter covers the recommendation-system concepts the program file assumes, from the classical foundations through the production architecture the program proposes.

The core abstraction is straightforward: collaborative filtering learns a value function over items from observed choices. A user interacts with a corpus, generating signals about which interactions were rewarding (saves, purchases), and the system learns to predict which future interactions will be rewarding. The mathematical machinery — matrix factorization, embedding learning, temporal credit assignment — is shared with many other fields that learn value functions from sequential decisions. What makes recommendation distinctive is the sparsity of the signal: most users interact with a tiny fraction of available items, and the system must generalize from these sparse observations to the full user-item matrix.

### Collaborative filtering: users who agreed will agree again

The oldest and most intuitive recommendation principle: if two users agreed in the past, they will probably agree in the future. If Alice and Bob both loved works by Gerhard Richter and Anselm Kiefer, and Alice also loves Georg Baselitz, then Bob might too — even if Bob has never seen a Baselitz.

**Matrix factorization** makes this precise. Arrange all users as rows and all items as columns of a matrix $R$, where $R_{ui}$ is user $u$'s rating of item $i$ (or simply 1/0 for interacted/not-interacted). This matrix is enormous and almost entirely empty — most users have interacted with a tiny fraction of available items. The key insight: the underlying preference structure is low-dimensional. Users do not have independent preferences for every item; they have preferences along a small number of taste dimensions (love of color vs. restraint, figurative vs. abstract, contemporary vs. historical), and these dimensions explain most of the variation in the full matrix.

Matrix factorization decomposes $R \approx U \cdot V^T$, where $U \in \mathbb{R}^{|users| \times k}$ and $V \in \mathbb{R}^{|items| \times k}$ for some small $k$ (typically 50-300). Each user gets a $k$-dimensional vector (their position in taste space), and each item gets a $k$-dimensional vector (its position in the same space). The predicted preference is the dot product: $\hat{R}_{ui} = u_i \cdot v_j$. Training minimizes the reconstruction error on observed entries of $R$.

**Implicit vs. explicit feedback.** Netflix has star ratings (explicit). Art platforms mostly don't. What they have is implicit feedback: clicks, page views, dwell time, saves, follows, and the crown jewel — purchase. Implicit feedback is noisier (a click might mean interest or might mean misclick), always positive (you observe what people did, not what they rejected), and much more abundant. The shift from explicit to implicit feedback was the pivotal development in industrial recommendation, and the Beowolff program operates entirely in the implicit regime. The treatment is to model implicit feedback as noisy observations of a latent preference: each interaction provides evidence for preference, weighted by the signal strength (purchase > save > dwell > click).

**What collaborative filtering gives.** When it works, CF discovers structure that no content-based system could find. Two artworks that look nothing alike and share no metadata can be CF-linked because the same collectors consistently value both — revealing a taste dimension that is invisible to visual or curatorial analysis. This is serendipity: recommendations that surprise because they reveal something about the user's own taste that they did not know how to articulate.

**Where collaborative filtering fails.** When a user or item has no interaction history. This is the cold-start problem, and for art it is not an edge case — it is the dominant case. Most artworks in Artsy's catalog have been seen by a handful of users or none. Most users have interacted with a handful of works. The preference matrix is not just sparse; it is almost entirely empty. CF on a nearly empty matrix produces nearly empty recommendations.

### Content-based recommendation: features instead of co-occurrence

Content-based methods sidestep the cold-start problem by recommending items that are *similar in features* to items the user has liked, rather than items that *co-occur* with liked items in other users' histories. If a collector has saved three Impressionist landscapes with blue-dominant palettes, the system can recommend other Impressionist landscapes with blue palettes — even if no other collector has interacted with those works.

The strength is obvious: every item that has features (an image, metadata, a description) can be recommended, regardless of interaction history. The weakness is equally obvious: content-based systems cannot discover the kind of non-obvious preferences that CF excels at. They recommend more of the same — similar styles, similar periods, similar subjects. The collector who loves both Caravaggio and Dan Flavin (dramatic chiaroscuro and fluorescent light installations — connected by their mastery of light, not by any surface visual similarity) would never get one from the other through content features alone.

**Why the program combines both.** The two-tower architecture in Beowolff is designed to be a hybrid: the item tower is content-based (image + entity + biography describe the item intrinsically), and the training signal is collaborative (behavioral co-occurrence determines what gets pushed together). The item tower's content features are learned under a contrastive objective where positives come from behavioral data (users who saved both works) as well as metadata (same artist, same Iconclass branch). This means the learned content features are not generic visual features — they are content features *shaped by how people actually interact with art*. The image tower does not just see that two works are visually similar; it learns that certain kinds of visual similarity predict co-preference and others do not.

### Two-tower retrieval: the production architecture

Every major recommendation system at scale — YouTube, Pinterest, Spotify, Amazon — uses a variant of the same architecture. It is simple, and understanding it concretely is essential for evaluating the program.

**The setup.** Two encoder networks (towers): one maps items to embedding vectors, one maps users to embedding vectors. Both produce vectors in the same $d$-dimensional space. The predicted relevance of item $v$ for user $u$ is the dot product $\hat{r}_{uv} = u \cdot v$.

**Why two towers instead of one?** This is about serving, not training. At query time — when a user opens the app and needs recommendations *now* — the system must score millions of candidate items in milliseconds. A single model that takes (user, item) as input and produces a relevance score would need to run once per candidate item — millions of forward passes. Infeasible.

Two towers decouple the problem. **Item embeddings are pre-computed.** Every item in the catalog is run through the item tower once (offline, in a nightly batch job), and its embedding vector is stored in an index. When a user queries the system, the user tower computes a single user embedding from their interaction history, and then an approximate nearest-neighbor search (using FAISS with HNSW — Chapter 8 of the study guide outline) finds the $K$ item embeddings closest to the user embedding. This is sub-linear in the catalog size. One forward pass through the user tower plus one ANN lookup, regardless of whether the catalog has 100,000 or 100,000,000 items.

**A concrete example of the serving pipeline.** A collector opens the Beowolff app. Here is what happens in the next 200 milliseconds:

1. **Retrieve interaction history.** The system fetches the collector's last $N$ interactions from a feature store — their recent clicks, saves, follows, and purchases, each represented as the pre-computed item embedding of the artwork they interacted with, tagged with the action type.

2. **Compute user embedding.** This sequence of interaction tokens is fed through the user tower (a transformer — Chapter 7). The output is a single $d$-dimensional vector representing the user's current taste state.

3. **Approximate nearest-neighbor search.** The user vector is sent to a FAISS index containing pre-computed embeddings for all items in the catalog (say, 3 million artworks). HNSW search returns the top 1,000 candidates in under 10ms.

4. **Re-ranking.** A smaller, more expensive model (or heuristic rules) re-ranks the 1,000 candidates considering business logic: diversity (don't show 10 Monets in a row), freshness (surface newly listed works), price range (match the collector's historical range), and perhaps the user's explicitly stated preferences.

5. **Return results.** The top 20-50 items are sent to the client.

The critical point: the item tower runs offline. It never runs at query time. This is why multi-modal fusion (image + entity + biography + price) can be computationally expensive without affecting serving latency. The item tower can take 100ms per artwork to compute an embedding — that is fine when you run it in a nightly batch on GPUs. What matters for latency is the user tower (one forward pass per query) and the ANN search (one lookup per query).

### The cold-start problem: why art is especially hard

"Cold-start" refers to the inability of a recommendation system to produce good recommendations when it lacks interaction history — for a new user (user cold-start), a new item (item cold-start), or both. This is THE defining challenge for art recommendation, and the program is explicitly architected around it. Worth understanding why.

**What "warm" and "cold" mean.** A warm item is one with enough interaction history that collaborative filtering alone can place it accurately in the embedding space. A warm user is one whose interaction history is long enough that the user tower can produce a reliable taste vector. Cold items and cold users lack this history.

**Why art is colder than other domains.** Compare to music recommendation: an average Spotify user streams 30-50 tracks per day. After a week, the system has hundreds of interactions. Most popular tracks have millions of listens. The preference matrix is dense enough for CF to work well. Now consider art: an average Artsy user browses perhaps a few dozen works per session, visits irregularly, and most artworks in the catalog have been viewed by fewer than 100 users total. Many have been viewed by zero. The preference matrix is orders of magnitude sparser than in music.

The long tail exacerbates the problem. In music, the top 10,000 tracks account for a large share of all listening. In the art market, the top 500 artists account for a large share of sales by dollar volume, but the next 50,000 artists — the ones where a recommendation-driven discovery would actually produce a new sale rather than confirming an existing intention — have very few interactions. These are the artists for whom the recommendation system's value is highest, and they are the coldest items in the corpus.

**How the content tower carries cold items.** This is the core design rationale. A new work by an emerging artist enters the corpus with no interaction history. CF has nothing. But the item tower can still produce an embedding:

- The image tower processes the artwork's visual content and places it near visually similar works.
- The entity tower looks up embeddings for the artist's period, medium, school, and Iconclass categories — all pre-trained on the rest of the corpus — and composes them into a metadata vector that places the work in the right neighborhood.
- If the artist has a biography, the text tower encodes it and pushes the artist's entity embedding toward related artists mentioned in the text.

The resulting embedding is less precise than one informed by thousands of interactions, but it is not random — it is an informed guess based on everything the model knows about works like this one. And that guess is testable: the cold-start evaluation metric (Recall@K on items with no behavioral history) directly measures how well this informed guess works.

**Stratified evaluation.** The program specifies evaluating warm-item and cold-item performance separately, and this is not a methodological nicety — it is essential. A single aggregate Recall@K metric can look great while hiding terrible cold-start performance, because warm items are easy and dominate the average. Stratified evaluation forces the model to demonstrate that it actually handles the hard case. The program goes further: it designates the cold-start metric as "the differentiator" — the metric that justifies the entity tower, the biography signal, and the overall multi-modal architecture. If cold-start recall is no better than a vanilla image-only embedding, the structured-entity tower and biography signal are not earning their complexity.

### Evaluation methodology

**Temporal splits.** The standard approach to evaluating a recommendation system is a temporal split: train on all interactions before time $T$, predict interactions at time $T+1$, and measure how well the predictions match reality.

Why not a random split? Because random splits leak information from the future. If you randomly assign interactions to train and test sets, the model might train on a user's purchase in March and be tested on their browse in January. The model "knows" what the user will buy and can use that to predict earlier behavior. This is not cheating in the usual sense — there is no train/test contamination — but it produces inflated metrics that do not reflect real-world performance, where the system must predict future behavior from past behavior.

Temporal splits respect the arrow of time. The model sees the world as it existed before $T$ and must predict what happens after $T$. This is a harder, more honest evaluation.

**Recall@K.** The most intuitive recommendation metric. Given a user, the system produces a ranked list of $K$ recommended items. Recall@K asks: of all items the user actually interacted with in the test set, what fraction appeared in the top-$K$ recommendations?

$$\text{Recall@}K = \frac{|\{\text{relevant items}\} \cap \{\text{top-}K \text{ recommendations}\}|}{|\{\text{relevant items}\}|}$$

**Worked example.** Suppose a collector interacted with 5 artworks in the test period: works A, B, C, D, E. The system's top-10 recommendations are [X, B, Y, A, Z, W, D, Q, R, S]. Three of the 5 relevant items (B, A, D) appear in the top 10. Recall@10 = 3/5 = 0.6.

Recall@K does not care about *where* in the top K the relevant items appear — B at position 2 and D at position 7 are treated equally. This is a limitation.

**NDCG@K** (Normalized Discounted Cumulative Gain) fixes this by giving more credit for relevant items appearing earlier in the ranked list. The discounting function is logarithmic: a relevant item at position 1 gets full credit, at position 2 gets credit/log2(3), at position 3 gets credit/log2(4), and so on.

$$\text{DCG@}K = \sum_{i=1}^{K}\frac{\text{rel}_i}{\log_2(i+1)}$$

where $\text{rel}_i = 1$ if the item at rank $i$ is relevant and 0 otherwise. NDCG normalizes DCG by the ideal DCG (what you would get if all relevant items were ranked at the top):

$$\text{NDCG@}K = \frac{\text{DCG@}K}{\text{IDCG@}K}$$

**Worked example of NDCG@10.** Same setup: 5 relevant items (A, B, C, D, E), system returns [X, B, Y, A, Z, W, D, Q, R, S].

Positions of relevant items: B at rank 2, A at rank 4, D at rank 7.

$$\text{DCG@10} = \frac{0}{\log_2 2} + \frac{1}{\log_2 3} + \frac{0}{\log_2 4} + \frac{1}{\log_2 5} + \frac{0}{\log_2 6} + \frac{0}{\log_2 7} + \frac{1}{\log_2 8} + \frac{0}{\log_2 9} + \frac{0}{\log_2 10} + \frac{0}{\log_2 11}$$

$$= 0 + 0.631 + 0 + 0.431 + 0 + 0 + 0.333 + 0 + 0 + 0 = 1.395$$

The ideal DCG (all 5 relevant items at the top):

$$\text{IDCG@10} = \frac{1}{\log_2 2} + \frac{1}{\log_2 3} + \frac{1}{\log_2 4} + \frac{1}{\log_2 5} + \frac{1}{\log_2 6} = 1.0 + 0.631 + 0.5 + 0.431 + 0.387 = 2.949$$

$$\text{NDCG@10} = \frac{1.395}{2.949} = 0.473$$

Compare to Recall@10 = 0.6. NDCG is lower because the relevant items are not at the top of the list — B at position 2 is good, but A at position 4 and D at position 7 are penalized by the logarithmic discount. NDCG rewards a system that places its best recommendations first.

**Why the cold-start metric is "the differentiator."** Any competent two-tower model trained on behavioral data will achieve reasonable warm-item metrics — that is table stakes. What distinguishes the Beowolff architecture from a vanilla CF model is its performance on cold items. If the entity tower, biography signal, and multi-modal alignment are doing real work, cold-start Recall@K should substantially exceed what an image-only or text-only baseline achieves. The program targets 2x Recall@10 improvement on cold-start items over the Iigaya baseline as a Phase 1 success criterion. This is a concrete, testable claim. If the architecture meets it, the complexity is justified. If it does not, the structured-entity tower and biography signal need rethinking.

### The hybrid recommender as the end state

The program's full architecture — item tower for content, user tower for behavior, contrastive training that uses both content and behavioral signals — is a hybrid recommender that gets the best of both worlds. Content features carry cold items (the CF weakness). Behavioral signals discover non-obvious preferences (the content-based weakness). The shared embedding space means both kinds of information cohabit and reinforce each other: a content-based match between two artworks becomes stronger evidence if behavioral data also shows co-preference, and weaker evidence if it does not.

This is the same strategy that every major recommendation platform converged on — not because it is theoretically elegant, but because it is the only architecture that handles the full range of real-world conditions. Warm items with rich behavioral data? CF signals dominate and the recommendations are precise. Cold items with no behavioral data? Content signals carry them into approximately the right region of the embedding space. New users with sparse histories? Content-based features of the few items they have interacted with are enough to produce a coarse taste vector that improves with every additional interaction.

The art-specific twist is that the cold case is not the exception — it is the default. Most artworks are cold. Most users are cold. The entire architecture is, in a sense, a cold-start architecture that happens to also work when things are warm.
