## Five Signals in One Space

### Beyond what the eye can see

The previous chapters described a model that learns to see images and learns which images should be near each other. That model alone would produce a visual similarity engine — useful, but limited. Two works could look nothing alike and still be deeply connected: a Caravaggio oil painting and a Gentileschi oil painting share a chiaroscuro language, a historical moment, a set of patrons, a market tier, and a constellation of biographical facts that pure visual analysis might not capture.

The Rasa system fuses five distinct types of information into a single space. Each type enters through its own processing component (called a "tower"), and all five towers are trained together so that their outputs are compatible — they all produce points in the same 768-dimensional space.

### Signal 1: The image

This is the visual tower described in Chapters 3 and 4. It processes the artwork's image and produces a point in the embedding space based on what the work looks like — palette, composition, texture, brushwork, spatial structure. It starts from DINOv3 and is fine-tuned on art.

### Signal 2: Structured metadata

Every artwork has metadata: the artist's name, the period, the medium, the school or movement, the museum or collection it belongs to, and (often) an Iconclass subject classification — a standardized system of over 44,000 iconographic concepts used by museums worldwide.

The Rasa system does not treat this metadata as text. Instead, each type of metadata gets its own lookup table of learned points. There is a table for artists (one point per artist), a table for periods, a table for media, a table for Iconclass codes. For any given artwork, the system looks up the relevant points from each table and combines them using a small neural network that understands how they interact.

Why not just convert metadata to text and feed it to a language model? Because text flattens structure. The string "Rembrandt van Rijn" is just a sequence of characters to a language model — it has to learn from context that this string refers to the same entity as "Rembrandt" and "Rembrandt Harmenszoon van Rijn." The structured approach gives each artist a single, unambiguous identity from the start.

More importantly, the structured approach enables the hierarchical training described in Chapter 4. The system knows, precisely, that two works share an artist, or share a school, or share an Iconclass branch. These structured relationships define the positive pairs that shape the embedding. With text alone, this hierarchy would need to be inferred from string matching — fragile and imprecise.

For the full technical treatment of the entity tower, see [Reader Chapter 5](../../study-guide/#5).

### Signal 3: Artist biography

This is the cold-start killer — the single most commercially significant architectural decision in the system.

The model reads artist biographies from Wikipedia, Artwiki, and published catalogs. These texts are processed by a language model and the resulting representation is attached to the artist's entry in the metadata system. The biography does not describe individual works — it describes the artist: their influences, their training, their place in art history, what critics have said about them.

Why does this matter? Because of an emerging Brazilian painter whose Wikipedia entry describes them as "influenced by the Neo-Concrete movement, particularly Lygia Clark and Helio Oiticica." Clark and Oiticica are established artists with extensive interaction histories, auction records, and rich embeddings. The biography text contains the names and concepts that connect the new artist to these established figures. When the model processes the biography, it places the new artist's embedding near Clark and Oiticica — before a single collector has viewed their work.

This is not a heuristic or a hand-coded rule. It is a learned representation, trained jointly with all the other signals. The biography passes through the same training process as everything else, and its contribution to the embedding is shaped by the same contrastive objectives. If the biographical connection turns out to be misleading — if the new artist's work is actually nothing like Clark or Oiticica despite the textual connection — the visual signal will pull them away once enough images have been processed. The biography provides an initial placement, not a permanent one.

**The Spotify precedent.** This is exactly the move Spotify made to solve their cold-start problem for new music releases. Erik Bernhardsson's team at Spotify scraped music blogs, press coverage, and journalism to build "cultural vectors" for every track and artist. When a new album dropped with zero listening data, its cultural vector — derived from what journalists and bloggers had written about the artist — placed it in the right neighborhood of the recommendation space. Listeners who liked music from that neighborhood would see the new release surface. The cultural vector carried the new release until enough listening data accumulated to take over. Rasa is making the same move with artist biographies. For the full technical treatment, see [Reader Chapter 5](../../study-guide/#5).

**The quality problem.** Major artists have rich, well-written Wikipedia articles. Emerging artists may have stubs or nothing at all. The biography signal degrades gracefully — when the biography is absent or thin, the other signals (image, metadata, Iconclass) still provide positioning. But the marginal benefit of the biography tower is directly tied to biography quality, and this creates an uneven landscape: artists who are well documented in English-language sources benefit more than those who are not. This is a real limitation, not a theoretical one.

### Signal 4: Collector behavior

When collectors interact with the platform — browsing, saving, following artists, adding works to collections, inquiring about purchases — these actions generate behavioral data. Two works that are consistently saved by the same collectors, or that appear in the same personal collections, carry a strong signal of similarity that no amount of visual or metadata analysis could produce.

Behavioral data is what makes the system a recommendation engine rather than just a similarity search tool. It captures the dimensions of taste that are invisible to image analysis and resist art-historical categorization — the collector who loves both Caravaggio and Dan Flavin, connected by mastery of light rather than by any surface resemblance or historical relationship.

Behavioral data enters the model through the "user tower," covered in Chapters 6 and 7. For now, the important point is that behavior is one of the five signals that shape the embedding space. Works that are behaviorally linked — consistently co-preferred by collectors — are pulled together in the space even if they have nothing in common visually or categorically. This is the signal that produces serendipitous recommendations: discoveries the collector could not have articulated as a search query but recognizes as right when they see them.

### Signal 5: Market price

The fifth signal is sale history from the auction market. The model does not use raw prices (which span from hundreds to hundreds of millions of dollars). It uses a normalized, compressed version: the log-price percentile, which places each work on a smooth scale relative to the rest of the corpus.

Price enters differently from the other signals. It is not trained contrastively (there is no notion of "positive and negative price pairs"). Instead, a small additional component of the model learns to predict an artwork's price tier from its embedding. The effect is that the embedding learns a "market direction" — a direction in the space along which works increase in price tier.

This enables price-aware queries. A collector browsing works similar to a given piece can add a price constraint: "like this, but in a more accessible range." The system can answer this because price is baked into the space, not layered on as a filter after the fact.

The price component also contributes to cold-start prediction. A new work by an unknown artist, with no sale history, can be placed in an approximate price tier based on its embedding — based on what the model knows about works that look like this, by artists like this, in traditions like this. This is not an appraisal. It is a rough prior. But for a recommendation system that serves both collectors shopping in specific ranges and galleries pricing new work, even a rough prior is useful.

### How missing signals are handled

In practice, most artworks are missing at least one signal. Many have no behavioral data (they are cold-start items). Many artists have no biography. Some works have incomplete Iconclass coding. Not all works have sale histories.

This is not a failure condition — it is the normal case, and the system is designed for it. When a signal is missing, the remaining towers still produce valid embeddings. The work's position in the space is less precise than it would be with all five signals, but it is not random. An artwork with only an image and metadata still lands in a meaningful neighborhood. As more signals accumulate — a biography is written, a collector saves the work, a sale occurs — the embedding becomes more precise and the work's position shifts to reflect the new information.

This graceful degradation is the practical payoff of training all signals jointly. Because the towers learn to be compatible with each other, each one can partially compensate for another's absence. A strong visual signal can carry a work that has no behavioral data. A rich biography can carry an artist whose images have not yet been processed. The architecture is built around the assumption that incompleteness is the norm, not the exception.

For the full technical treatment, including the mathematical details of how the towers compose and the price regression, see [Reader Chapter 5](../../study-guide/#5).
