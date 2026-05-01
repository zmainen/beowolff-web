# Visual Qualities: A Perceptual Vocabulary

Of the Art Genome Project's 17 gene families, one stands apart for a perceptual similarity system: Visual Qualities. Its 74 genes attempt to name the dimensions along which artworks differ in appearance — not what they depict, not when they were made, not what movement they belong to, but what they look like. Almost all of them (71 of 74) are manually applied by human genomers. They represent the closest thing the art world has produced to a systematic vocabulary for visual experience.

## The genes, by perceptual domain

The 74 Visual Qualities genes cluster into natural groups. The clustering is ours, not Artsy's — the genome itself is flat within each family, with no sub-hierarchy.

### Color (11 genes)

Black and White, Bright and Vivid Colors, Dark Colors, Earth Tones, Iridescence/Opalescence, Metallic, Pastel Colors, Primary Colors, Shiny/Glossy, Glittery, Color Gradient.

These describe the chromatic character of a work. They are coarse — "Bright and Vivid Colors" encompasses everything from a Matisse to a Jeff Koons — but they capture distinctions that matter for browsing. A collector who responds to muted palettes can navigate away from "Bright and Vivid" toward "Earth Tones" and "Pastel Colors." The vocabulary does not attempt to specify hue (there is no "Blue" gene in Visual Qualities), only chromatic quality. This is a deliberate design choice: specific hues are too granular for a hand-applied taxonomy, while chromatic character (bright vs. muted, warm vs. cool) is reliably perceivable and consistently applicable across works.

Three of the 11 are about surface reflectance rather than color per se: Shiny/Glossy, Glittery, and Metallic. These cross the boundary between color and texture — a glossy surface changes the perceived color of a work depending on viewing angle and lighting. Including them in the color group rather than the texture group reflects a pragmatic judgment about how people actually talk about these qualities.

### Composition and spatial structure (14 genes)

Allover Composition, Asymmetrical, Symmetrical, Balance, Dense Composition, Scattered Composition, Sparse, Blown-Off-Roof Perspective, Emphasis on Linear Perspective, Extreme Angle, Columns and Totems, Piles/Stacks, Single Object, Layered Images.

This is the richest sub-group and the one most directly connected to formal art analysis. "Allover Composition" describes a Pollock or a Kusama — no focal point, no hierarchy, the entire surface treated equally. "Dense" and "Sparse" are opposites along a continuum that a neural network should be able to recover from pixel statistics alone. "Blown-Off-Roof Perspective" is a delightfully specific gene — it describes the aerial cutaway view used in Japanese Heian-period narrative scrolls and some contemporary illustration, where a building's roof is removed to reveal the interior. It is the kind of category that only a human expert would think to name.

The perspective genes (Blown-Off-Roof, Linear Perspective, Extreme Angle) encode spatial reasoning about the implied three-dimensional structure of a two-dimensional image. These are hard for purely visual models to capture — they require understanding what the image is *of*, not just what it looks like as a pattern of pixels.

### Form and geometry (14 genes)

Angular, Biomorphic, Bulbous, Curvilinear Forms, Irregular Curvilinear Forms, Crystalline and Geological Forms, Fragmented Geometry, Geometric, Linear Forms, Irregular Linear Forms, Molecular, Slender, Viscous Forms, Tangled Forms.

These genes describe the shapes that dominate a work. "Biomorphic" (Arp, Miró, early Kandinsky) is the organic counterpart to "Geometric" (Mondrian, Albers, Judd). The vocabulary captures gradations that a binary classification would miss: "Curvilinear Forms" vs. "Irregular Curvilinear Forms" distinguishes controlled curves (Art Nouveau) from wild ones (de Kooning). "Viscous Forms" names a quality shared by Dalí's melting clocks and Lynda Benglis's poured latex — a quality that has no standard art-historical term but is instantly recognizable.

### Line and contour (5 genes)

Arabesque/Scroll, Calligraphic, Contour Line, Radiating Lines, Striped.

A small but distinctive group. "Calligraphic" bridges East Asian painting traditions and Western gestural abstraction — it names a quality of line that is recognizable across cultures. "Radiating Lines" describes both Bridget Riley's Op Art and Baroque ceiling paintings. The gene names are vocabulary for formal analysis of line, the kind of thing a drawing professor teaches but that rarely appears in marketplace taxonomy.

### Surface and texture (12 genes)

Cracked, Creased/Crinkled/Wrinkled, Marbleized, Patinated and Oxidized, Rough, Smooth Surface, Sharp/Prickly, Woven/Perforated, Mirrored, Transparent/Translucent Medium, Wrapped, Raw versus Finished.

Texture is the most medium-dependent of the perceptual domains. "Cracked" applies to aged oil paintings, raku ceramics, and intentionally crackled glaze — the same visual quality arising from entirely different causes. "Smooth Surface" is meaningful for sculpture and ceramics in ways it is not for painting. "Mirrored" describes Anish Kapoor's *Cloud Gate* and Jeff Koons's *Balloon Dog* but also certain photographic techniques. The genome treats these as visual qualities regardless of medium, which is the right abstraction for a similarity system but creates noise for a classification system.

"Patinated and Oxidized" is one of only three automated genes in the Visual Qualities family, presumably because it correlates strongly with material (bronze, copper) and age, making algorithmic detection feasible.

### Pattern (5 genes)

Dotted, Grid, Patterns, Fractal-like/Kaleidoscopic, Pixelated.

"Patterns" is the catch-all; the others are specific pattern types. "Fractal-like/Kaleidoscopic" names a quality shared by Islamic geometric art, Escher, and certain strains of digital art — recursive self-similarity at multiple scales. "Pixelated" is a distinctly contemporary gene that would have been meaningless before digital imaging. The genome is a product of its time.

### Optical effects and rendering style (8 genes)

Abstract Illusionism, Blurred, Divisionist, Flatness, Hard-Edged, Highly Detailed, Highly Ornamented, Stained Glass Effect.

These genes describe how a work handles the relationship between representation and surface. "Flatness" is Clement Greenberg's central critical category — the quality that distinguishes modernist painting from illusionistic painting. "Hard-Edged" is its complement in geometric abstraction. "Blurred" describes Richter's photo-paintings but also Impressionist atmospheric effects and out-of-focus photography. "Divisionist" (one of the three automated VQ genes) refers to the systematic color-mixing technique of Seurat and Signac — small enough in scope to be algorithmically detectable.

### Dynamic qualities (2 genes)

Dynamism, Psychedelic.

The smallest sub-group. "Dynamism" is broad — it covers Futurist painting, kinetic sculpture, and any work that conveys motion or energy. "Psychedelic" is narrow — it names a specific visual tradition rooted in 1960s counterculture but visible in contemporary digital art and certain strains of Op Art.

## Beyond Visual Qualities: the perceptual genes hiding in other families

The boundary between "Visual Qualities" and "Medium and Techniques" is porous. Several genes classified under Medium and Techniques describe what art looks like rather than how it was made:

- **Chiaroscuro** — dramatic light-dark contrast (Caravaggio, Rembrandt)
- **Sfumato** — soft, smoky transitions between tones (Leonardo)
- **Impasto** — thick, textured paint application (Van Gogh, de Kooning)
- **Gestural** — visible evidence of the artist's physical movement (Pollock, Kline)
- **Large Brushstrokes/Loose Brushwork** — visible brush marks as compositional element
- **Monochrome Painting** — single-color works (Yves Klein, Ad Reinhardt)
- **Hatching** — parallel lines used for shading or texture
- **Splattered/Dripped** — paint applied through gravity or force
- **Stains/Washes** — thin, translucent paint layers (Frankenthaler, Louis)
- **Trompe l'oeil** — illusionistic rendering intended to deceive the eye
- **Close-Up** — extreme proximity as compositional strategy
- **Shallow Depth of Field** — photographic bokeh effect

And from Subject Matter:

- **Line, Form, and Color** — art about its own formal elements
- **Visual Perception** — art about the act of seeing (Albers, Riley)
- **Light as Subject** — Turrell, Eliasson, Flavin
- **Shadows, Reflections, Silhouettes** — light phenomena as subject
- **Atmospheric Landscapes** — landscape dominated by atmosphere rather than geography

Counting these cross-family additions, the total vocabulary for visual-perceptual description reaches roughly 94 genes. This is the full set of named dimensions available for describing what art looks like, as curated by Artsy's classification specialists over a decade of practice.

## What the vocabulary reveals

Several things stand out about this vocabulary as a whole:

**It is human-scale.** Ninety-four named dimensions is a number that a trained specialist can hold in working memory and apply consistently. It is far too few to capture the full richness of visual experience — a DINOv2 vision transformer works in 768 or 1,024 dimensions, and even that may be too few for some distinctions. The genome vocabulary is a compression, designed for the throughput constraints of human judgment rather than the capacity constraints of machine perception.

**It is Western-centric with patches.** The vocabulary was developed primarily by English-speaking art professionals working with Western art-historical categories. Genes like "Chiaroscuro," "Sfumato," and "Divisionist" are deeply rooted in European painting tradition. The genome compensates with genes like "Calligraphic" (bridging East Asian and Western gestural traditions) and "Blown-Off-Roof Perspective" (specifically naming a Japanese compositional device), but the overall framework assumes Western formal analysis as the default language.

**It privileges vision over the other senses.** There is no gene for "loud" or "quiet" — for the visual qualities that create a sense of volume or stillness. There is no gene for haptic invitation — the quality that makes you want to touch a sculpture. "Rough" and "Smooth Surface" gesture toward tactility, but they describe visual texture, not the embodied experience of encounter. This is appropriate for a system that operates through screens, but it is a real limitation for art that was made to be experienced in person.

**It mixes levels of abstraction.** "Geometric" and "Biomorphic" are high-level categories that subsume many specific forms. "Dotted" and "Striped" are low-level features that a simple image filter could detect. "Psychedelic" is a cultural category as much as a visual one. The vocabulary does not maintain a consistent level of abstraction, because the art it describes does not occupy a consistent level of abstraction. This is a feature, not a bug — but it makes the vocabulary harder to use as a formal feature space.

**It is a vocabulary, not a theory.** The genes name perceptual qualities. They do not explain how those qualities relate to each other, how they interact, or why they matter aesthetically. "Dense Composition" and "Allover Composition" are related but distinct. "Bright and Vivid Colors" and "Primary Colors" overlap but are not nested. The genome provides the words but not the grammar. A learned embedding provides the grammar but not the words. Together, they might provide both.

---

*Source: [github.com/artsy/the-art-genome-project](https://github.com/artsy/the-art-genome-project) (CC-BY-4.0). Perceptual groupings are ours; Artsy's taxonomy is flat within each family.*
