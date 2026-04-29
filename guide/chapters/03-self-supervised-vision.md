## How the Model Learns to See

### Starting from a model that already knows how to look

The Rasa image system does not start from scratch. It begins with a model called DINOv3, built by Meta in 2024, that has already learned to see by looking at over a billion images — photographs, illustrations, screenshots, everything the web contains. This pre-trained model arrives with a sophisticated understanding of visual structure: edges, textures, color relationships, spatial composition, object shapes.

Why start here instead of building a vision system trained only on art? Because much of what the model needs to know is universal. Edges are edges whether they appear in a photograph or a painting. Texture is texture. The difference between warm and cool light, the way a composition balances mass across a frame, the distinction between smooth gradients and sharp boundaries — these are visual facts that transfer from one domain to another. Relearning them from scratch on art alone would cost weeks of computation and produce results no better than what DINOv3 already provides.

What DINOv3 does *not* know is what matters specifically in art. It can see that two paintings have similar color palettes, but it does not know that palette is an art-historically significant feature. It can detect brushwork texture, but it does not know the difference between Impressionist broken color and Expressionist impasto. It sees compositional structure but does not know the conventions of devotional painting or the rules of ukiyo-e spatial arrangement.

This is where fine-tuning comes in. The Rasa system takes DINOv3's universal visual knowledge and adjusts it — trains it further on art, with art-specific objectives — so that the model learns to see what matters for art, not just what matters for photographs of dogs and kitchens. The universal foundation stays; art-specific refinement builds on top.

### What the model sees at different levels

A useful way to think about what the image model learns: it processes visual information in layers, from simple to complex, much like the human visual system.

**Low-level features** are the basics. Edges, corners, color patches, local textures. These are the building blocks that every image is made of. A model that has seen a billion images has already learned these thoroughly. They transfer perfectly to art — the edge of a brushstroke is an edge, whether it appears in a Velazquez or a photograph.

**Mid-level features** are where things get interesting for art. Spatial composition — where masses sit in the frame, how the eye moves through the work. Material properties — the difference between the sheen of oil paint and the transparency of watercolor. Larger-scale texture — the rhythmic quality of Bridget Riley's op-art versus the stillness of a Morandi. These features partially transfer from photographs, but art has its own conventions. A painting's composition follows different rules than a snapshot. The model needs to learn these art-specific patterns, and fine-tuning targets this level.

**High-level features** are the most abstract. Semantic meaning — is this a portrait? A landscape? A still life? Art-historical identity — is this Mannerist? Neoclassical? Conceptual? The pre-trained model can distinguish a portrait from a landscape (it learned that from photographs), but it cannot distinguish a Mannerist portrait from a Baroque one without being trained on art-historical examples. This is the level where the most art-specific learning happens.

The Rasa system's training strategy follows from this analysis: keep the low-level features as they are (they are already good), and focus the training effort on mid-level and high-level features where art-specific knowledge matters. For the full technical treatment of this layer-by-layer strategy, see [Reader Chapter 3](../../study-guide/#3).

### Self-supervised learning: teaching itself

Here is the part that surprises most people. DINOv3 was not trained by humans labeling millions of images ("this is a dog," "this is a landscape"). It taught itself, by a method called self-supervised learning.

The principle is elegant. Take an image. Create two different crops of it — say, a large crop showing most of the painting and a small crop showing just a corner. Show the large crop to one copy of the model (the "teacher") and the small crop to another copy (the "student"). Train the student to produce the same internal representation as the teacher, even though it is seeing less of the image.

Why does this work? Because to match the teacher's understanding of the whole painting from just a corner, the student must learn what the corner *implies* about the rest. A hand holding a brush, painted in a particular style, against a particular background, implies something about the full composition. Over millions of images, the model learns the regularities of visual structure — the features that make it possible to infer the whole from the part. These features turn out to be exactly the kind of rich, hierarchical visual representations that are useful for everything downstream.

No one told the model what to look for. It figured it out by the task of predicting what it could not see from what it could.

### Why augmentation matters for art

There is a subtle but important design issue here. During training, the model creates those different views of each image using random transformations — cropping, flipping, changing colors slightly, adding blur. The model learns to treat all of these transformations as irrelevant: a slightly bluer version of a painting should produce the same representation as the original.

For photographs, this is fine. A dog is a dog whether the photo is slightly warmer or slightly cropped.

For art, some of these transformations destroy signal. Color is not a nuisance variable in art — Matisse's palette is a primary signature of his work. Aggressive color shifts during training tell the model that palette does not matter. But palette matters enormously. Similarly, cropping away the edges of a Mondrian can remove all compositional structure. Flipping a painting left-to-right reverses the directional logic of a history painting or the position of a hand in religious iconography.

The Rasa system must customize these transformations for art — reduce color shifting, use larger minimum crops, disable left-right flipping for categories where mirror symmetry is not meaningful. This is not a minor technical detail. It is a statement about what the model should learn to care about and what it should learn to ignore. Getting it wrong means training a model that is blind to features that make art recommendation interesting.

For the full technical treatment of augmentation design for art, see [Reader Chapter 3](../../study-guide/#3).

### What this means for you

The image model is the foundation of visual discovery in the system. When it works well, it captures not just what a work depicts but how it is made — the palette, the handling of paint, the spatial logic, the visual temperature. Two works that share a formal sensibility will be near each other in the embedding even if they depict entirely different subjects, come from different periods, or are made by artists who never heard of each other.

This is visual kinship at a scale no human could survey. A curator might recognize the formal connection between a Rothko and a Sugimoto seascape — both pursue a particular quality of luminous stillness at large scale. The model can find that connection across three million works in milliseconds, not because it understands "luminous stillness" as a concept, but because it has learned that certain combinations of visual features — color temperature, tonal range, compositional simplicity, scale — co-occur in ways that make those works geometrically close.

Whether this kind of purely visual kinship is sufficient for good recommendation is the question that the remaining chapters address. The short answer is no — which is why the model does not stop at images.
