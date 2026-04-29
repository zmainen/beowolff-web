## 8. 690,000 Works in Your Browser

You have seen maps of the world that start as a blue-and-green marble and let you zoom in until you can read the name of a street cafe. Google Maps does not load every cafe sign at once. It loads what you are looking at, at the resolution you are looking at it, and swaps in more detail as you zoom.

The Beowolff visualiser does the same thing, except the territory is not the earth. It is the space of aesthetic relationships between artworks. Every work in the collection has a position in this space, and that position is determined by what the model has learned about it -- its visual qualities, its curatorial context, its neighborhood of related works. The visualiser lets you see that space, pan across it, and zoom into any region to discover what the model thinks belongs together and why.

This chapter explains how hundreds of thousands of artworks end up in your browser at once, what the layout means, and what you can learn from exploring it.

### From 768 dimensions to two

The model represents every artwork as a point in a space with hundreds of dimensions. You cannot visualise hundreds of dimensions. You can visualise two.

UMAP is an algorithm that projects the high-dimensional space down to a flat, two-dimensional map while preserving the most important thing: which works are near each other. Two paintings that are neighbors in the model's full space should be neighbors on the screen. Two paintings that are distant should be distant. UMAP does this by building a graph of neighborhood relationships in the high-dimensional space, then arranging points on a 2D plane so that those relationships are preserved as faithfully as possible.

The result is a landscape. Regions form. You will see a cluster of Dutch Golden Age portraits, another of Abstract Expressionist canvases, another of Japanese woodblock prints. These clusters are not hand-labeled -- they emerge from the model's learned sense of similarity. When you see Impressionist landscapes next to Post-Impressionist landscapes, and both far from Ukiyo-e, that is the model telling you something about how it organises visual and contextual kinship.

Not every aspect of the full space survives the projection. Compressing hundreds of dimensions into two necessarily loses information, the way any map projection distorts some feature of the globe. But UMAP is specifically designed to preserve neighborhood structure -- the local relationships that tell you "these works are related" -- and it does this better than older methods. The clusters you see are real features of the model's understanding. Their relative positions are meaningful, though not perfectly precise.

For the full technical treatment, see [Reader Chapter 8](../study-guide/#8).

### The Google Maps trick

Even with the layout computed, you cannot load 690,000 artwork images into a browser at once. The images alone would be gigabytes. The solution is the tile pyramid.

The 2D layout is divided into a grid of tiles, like a mosaic. At the widest zoom level, each tile covers a large region and shows artworks as small colored dots -- no images, just positions and cluster memberships. Zoom in a level, and the tile subdivides into four smaller tiles, each showing its region at higher resolution. Now the dots become tiny thumbnails, packed into sprite sheets (a single image file containing a grid of many thumbnails, so the browser loads one file instead of hundreds). Zoom in further, and the thumbnails grow larger, metadata appears, and you can see individual works at gallery-wall scale.

At any moment, the browser loads only the tiles visible in your viewport -- perhaps twenty or thirty. Pan left, and it loads the tiles to the left and discards the ones that scrolled off screen. This is why the system works regardless of collection size. Whether the total collection is 690,000 works or ten million, you are always looking at roughly the same number of tiles. The rest sit on a server, waiting to be requested.

The tiles are generated once, offline, and served as static files from a content delivery network. There is no server-side computation when you browse. The cost of building the tiles grows with the collection; the cost of serving them does not.

### What the neighborhoods mean

When you explore the visualiser, you will notice that some regions are dense and others are sparse. Dense regions mean many works that the model considers similar. A thick cluster of Baroque portraits tells you the model has a confident, well-populated concept of what a Baroque portrait is. A sparse frontier between Surrealism and Abstract Expressionism might mean the model sees these traditions as related but distinct, with relatively few works bridging the gap.

The boundaries between regions are as interesting as the regions themselves. Where do Impressionist landscapes shade into Post-Impressionist ones? Where does geometric abstraction meet color field painting? The model's answer may not match any art historian's -- or it may surface relationships that a human would recognise but not have articulated. An unexpected neighbor is not necessarily a mistake. It is an invitation to ask what the model is seeing.

Some things to watch for:

**Familiar clusters that confirm the model works.** If Dutch Golden Age still lifes cluster together, the model has learned something real about that tradition. This is not surprising, but it is necessary. A model that scatters Vermeer across the map has failed at a basic task.

**Surprising proximities.** A contemporary photographer appearing near the Nabis. A Chinese ink painter adjacent to Abstract Expressionism. These are the discoveries -- the model has found a structural similarity (visual, contextual, or both) that the usual art-historical categories do not capture.

**Missing clusters.** A tradition you know should be distinct but that the model has merged with something else. This tells you the model's training data did not include enough examples, or the features it learned do not capture the relevant distinction. It is diagnostic: it tells you where the model needs more data or a different signal.

### As the collection grows

The visualiser is designed for a living collection, not a static snapshot. When new works are added to the platform, they need positions on the map. Two approaches handle this.

For the current prototype, the layout is recomputed from scratch when the collection changes significantly. This works at 690,000 works -- recomputation takes about an hour. At three million works, it would take longer but remain feasible.

For production at scale, the system will use a trained projection function -- a small neural network that has learned the mapping from the model's full space to 2D coordinates. New works are fed through this function and placed on the existing map in milliseconds, without disturbing the positions of existing works. Your mental map of "Dutch portraits are in the upper right" remains stable even as the collection grows. The projection function is retrained periodically, when the underlying model is updated, but between those updates, the landscape holds still.

This stability matters. A map that rearranges itself every time a new work arrives is not a useful tool for navigation. The goal is a surface you can learn, return to, and build intuition about over time.

### What you are actually seeing

The visualiser is not a neutral presentation of facts. It is a rendering of the model's learned understanding -- shaped by its training data, its architecture, and its objectives. A region that appears empty may reflect a gap in the training corpus, not the absence of art in that tradition. A cluster that seems too large may mean the model has not yet learned to distinguish subtraditions within it.

Think of it as a map drawn by a knowledgeable but fallible cartographer. The major continents are in the right places. The coastlines are approximately correct. But the details reward scrutiny, and the most interesting thing about any map is what it reveals about the mapmaker's assumptions.
