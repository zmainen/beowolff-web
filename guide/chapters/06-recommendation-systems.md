## How Recommendations Work

### Two traditions, one system

Recommendation systems broadly fall into two camps, and the Rasa system combines both.

**Collaborative filtering** is the older tradition: people who agreed in the past will agree in the future. If Alice and Bob both collected works by Gerhard Richter and Anselm Kiefer, and Alice also collected Georg Baselitz, then Bob might want to see Baselitz too — even if Bob has never encountered Baselitz on the platform. The system does not need to know *why* Alice and Bob agree. It only needs to observe *that* they agree, and extrapolate.

When collaborative filtering works, it produces discoveries that no other method can. Two artworks that share nothing visible — different periods, different media, different subjects — can be linked because the same collectors consistently value both. This is the system finding a dimension of taste that exists in human behavior but has never been named in art criticism. It is the recommendation equivalent of serendipity.

**Content-based recommendation** is the other tradition: recommend works that are similar *in their features* to works the collector has liked. If you have saved three Impressionist landscapes with blue-dominant palettes, the system finds other Impressionist landscapes with blue palettes. No other collectors need to be involved — the system works entirely from the properties of the works themselves.

Content-based recommendation never surprises. It gives you more of what you already like, along dimensions the system can measure. It is reliable, predictable, and limited.

### Why art needs both

The art market has a structural problem that makes pure collaborative filtering fail: most artworks have very little behavioral data.

Compare art to music. An average Spotify user streams 30 to 50 tracks per day. After a week, the system has hundreds of interactions. Popular tracks have millions of listens. The data matrix is dense enough for collaborative filtering to work well.

Now consider art. A typical collector on Artsy browses a few dozen works per session and visits irregularly. Most artworks in the catalog have been viewed by fewer than a hundred people. Many have been viewed by nobody at all. The data matrix is almost entirely empty. Collaborative filtering on an empty matrix produces empty recommendations.

This is the cold-start problem, and in art it is not an edge case — it is the dominant case. The long tail of the art market — the 50,000 artists beyond the top 500 household names — is where a recommendation system could produce the most value (a genuine discovery, a sale that would not otherwise have happened). These are precisely the artists with the least behavioral data.

The Rasa architecture is designed specifically to handle this. The content signals from the item tower — image, metadata, biography, price — carry cold items into the right neighborhood of the embedding space even when no collector has ever interacted with them. Collaborative filtering takes over as behavioral data accumulates. The transition is smooth, not a hard switch, because both types of signal live in the same space.

For the full technical treatment, including the mathematics of collaborative filtering and the evaluation methodology, see [Reader Chapter 6](../../study-guide/#6).

### How the system actually serves a recommendation

Here is what happens, concretely, when a collector opens the app and the system needs to produce recommendations in under a second.

**Step 1: Pre-compute item embeddings.** Every artwork in the catalog is processed through the item tower (image + metadata + biography + price) once, offline, in a nightly batch job. The result — a 768-dimensional point for each work — is stored in an index. This is the expensive part, but it happens in the background. The item tower can take as long as it needs per work because it never runs while a collector is waiting.

**Step 2: Compute the collector's position.** When the collector opens the app, the system retrieves their recent interaction history — what they browsed, saved, followed, and inquired about. This history is processed through a separate model (the "user tower," covered in Chapter 7) that compresses it into a single 768-dimensional point — the collector's current position in the same space as all the artworks.

**Step 3: Find what is nearby.** The system searches the pre-computed index for the artworks closest to the collector's position. This is a fast geometric lookup — finding nearest neighbors in a high-dimensional space. Modern indexing technology (called FAISS) can search through millions of points in under ten milliseconds.

**Step 4: Re-rank and diversify.** The raw nearest-neighbor results go through a second pass that applies business logic: do not show ten works by the same artist in a row; surface newly listed works; match the collector's typical price range; ensure variety across media and periods. This step turns a geometric proximity list into a curated feed.

**Step 5: Return results.** The top 20 to 50 works are sent to the collector's screen.

The entire process takes under 200 milliseconds. The key architectural insight is that the item tower (the expensive, multi-modal computation) runs offline. It never runs at query time. What runs at query time is the user tower (one forward pass) and the nearest-neighbor search (one index lookup). This is why the system can be as complex as it needs to be in its understanding of each artwork without sacrificing response time.

This architecture — called "two-tower retrieval" because it has one tower for items and one for users — is not novel to Rasa. It is the standard architecture used by YouTube, Pinterest, Spotify, and Amazon. What is specific to Rasa is the richness of the item tower (five signals instead of the typical two or three) and the focus on cold-start performance as the primary success metric.

### How success is measured

The system is evaluated by a deceptively simple question: of the works a collector actually interacted with in a test period, how many appeared in the system's top recommendations?

This metric is called Recall@K — if K is 10, it asks: what fraction of the collector's actual interests appeared in the top 10 recommendations? The system never gets to peek at the test period during training; it must predict future behavior from past behavior. The evaluation uses a temporal split: train on everything before a cutoff date, predict what happens after.

The critical evaluation design in Rasa is **stratified measurement.** Warm-item performance (how well the system recommends works with lots of behavioral data) and cold-item performance (how well it recommends works with no behavioral data) are measured separately. A single aggregate metric can look impressive while hiding terrible cold-start performance, because warm items are easy and dominate the average.

The cold-start metric is what the Rasa team calls "the differentiator." Any competent system can recommend popular works to active collectors — that is table stakes. The question that justifies the entire multi-modal architecture — the five signals, the entity tower, the biography component — is whether cold-start recall substantially exceeds what a simpler, image-only system achieves. The Phase 1 target is twice the cold-start recall of the existing baseline. If the system hits that target, the complexity has earned its keep. If it does not, the architecture needs rethinking.

For the full technical treatment, including NDCG metrics and the evaluation methodology, see [Reader Chapter 6](../../study-guide/#6).

### What this means for you

For gallerists representing emerging artists: the cold-start architecture means your artists do not need to already be popular to appear in recommendations. If their metadata is accurate, their biography is written, and their images are in the system, the content tower will place them in the right neighborhood. The system is explicitly designed and measured on its ability to surface works that have no behavioral history — which is exactly the condition your newest artists are in.

For collectors: the two-tower architecture means the system is learning from you in real time. Your interactions — what you browse, what you save, what you inquire about — shift your position in the space. The recommendations you see are a reflection of where you currently are in the landscape of all art. They will change as you change.

For artists: your work's position in the system is not fixed. It is shaped by five signals, and as those signals accumulate — more collectors interact with your work, your biography grows, your auction history develops — your position becomes more precise. The system's understanding of your work deepens with time, which means that early inaccuracies (being placed near the wrong neighborhood because the biography signal was thin) are self-correcting as more data arrives.
