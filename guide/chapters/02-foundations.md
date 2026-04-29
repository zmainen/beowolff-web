## What Is an Embedding?

### The core idea

An artwork in a database is a row: an artist name, a date, a medium, a set of tags. You can check whether two works share an artist. You cannot ask "how similar are they?" in any graded way — not from the database entry alone.

An embedding changes this. It turns every artwork into a point in a mathematical space. A Rembrandt self-portrait becomes a point. A Rothko color field becomes another point, somewhere else in the same space. The distance between them is a measure of similarity. Not binary (same or different) but continuous — a Rothko is this far from a Newman, that far from a Vermeer, some other distance from a Basquiat. The whole collection becomes a landscape you can navigate by proximity.

The space has many dimensions — Beowolff-Embed uses 768. You cannot visualize 768 dimensions, but the principle is the same as two or three: points that are nearby are similar, points that are far apart are different. No single dimension corresponds to "redness" or "Impressionism" or "price tier." The meaning lives in the pattern across all dimensions at once, the way a chord's character lives in the combination of notes, not in any single one.

### What "similar" means depends on how the model was trained

This is the most important idea in the entire system: **the embedding does not discover some objective, eternal truth about which artworks are similar.** It learns a specific notion of similarity from its training, and different training choices produce different notions.

Consider three works:

- **Work A**: A Rothko color-field painting — large rectangles of luminous red and orange. 1961, oil on canvas.
- **Work B**: A Barnett Newman "zip" painting — a field of deep red divided by a thin vertical line. 1953, oil on canvas.
- **Work C**: A contemporary Chinese oil painting of red lanterns against a red sky. 2019.

Are A and C similar? Visually, yes — both are dominated by red, both have large expanses of color. A model trained only on visual features might place them near each other.

Are A and B similar? Art-historically, yes — both are canonical Abstract Expressionists working in the color-field tradition. A model trained on metadata (artist, period, school) would place them together and push C far away.

What about from a market perspective? A and B trade in the same astronomical price tier. C does not. A model trained on sale prices would group A and B and separate C.

None of these groupings is wrong. Each reflects a real dimension of similarity. The Beowolff-Embed system tries to capture all of them at once — visual, curatorial, behavioral, biographical, and market — in a single space. When it works, the space encodes a richer, more layered notion of "similar" than any single signal could. A collector browsing works near Rothko sees Newman (because they share art-historical context) but also sees that contemporary Chinese painter (because the visual kinship is real). The system does not collapse these into one dimension. It preserves them as different directions you can move in.

For the full technical treatment of embedding geometry, see [Reader Chapter 2](../../study-guide/#2).

### Why one space, not five

You might ask: why not build five separate systems — one for visual similarity, one for art-historical context, one for price, and so on — and combine the results at the end?

The answer is that combining at the end is much weaker than learning together from the start. If the visual system and the metadata system are trained separately, neither one knows about the other. The visual system cannot learn that certain kinds of visual similarity matter more than others because collectors respond to them. The metadata system cannot learn that certain art-historical categories contain visually diverse works that should not all be treated as identical.

When all five signals are fused during training, each component adjusts to accommodate the others. The image component learns not just what things look like, but which visual features predict curatorial kinship. The metadata component learns not just categorical membership, but which categories contain meaningful internal structure. The result is a single space where cross-referencing happens automatically — you can query with an image and find works that are similar in ways that go beyond what any eye can see, because the space has absorbed curatorial knowledge, collector behavior, and market structure into its geometry.

### What this means for you

If you are a gallerist, it means that works by your emerging artists can be positioned in this space based on what the model knows about their visual language, their biographical context, and their place in art history — before a single collector has interacted with them. The embedding does not require popularity to function.

If you are a collector, it means the system's notion of "works you might like" is not just "works that look like what you have saved." It can surface works that share a deeper kinship — same tradition, same formal concerns, same market segment — even if they look quite different on the surface.

If you are an artist, it means that the model is learning a position for your work in a landscape that includes every other work in the corpus. That position is shaped by your visual language, your biography, your exhibition history, and (eventually) how collectors respond to your work. Understanding that this position exists — and that it influences what collectors see — is the beginning of understanding what algorithmic recommendation means for artistic practice.

For the full technical treatment, including the mathematics of distance and similarity, see [Reader Chapter 2](../../study-guide/#2).
