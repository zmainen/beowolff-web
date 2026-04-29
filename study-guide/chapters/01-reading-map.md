## Reading Map

### How to use this guide

This guide exists so you can read the Beowolff-Embed program file (`platform/jobs/beowolff-embed.md`) and the attribution whitepaper ([Appendix A](#A)) and interrogate them rather than approve them on trust. Every ML concept those documents invoke — from contrastive losses to Shapley approximations to machine unlearning — is explained here at the depth needed to ask whether it was the right choice and what the alternatives would have been.

The guide is written for a computational neuroscientist who commands representation learning, Bayesian inference, RL, population coding, and behavioral statistics but has not built production ML systems. Where your existing intuitions provide a bridge, the guide names them explicitly: embedding geometry maps to population coding, contrastive learning maps to predictive coding, user-tower sequence models map to evidence accumulation, attribution maps to credit assignment. Where no bridge exists — production pipeline tooling, approximate nearest-neighbor search, model distillation attacks — the guide builds the concept from scratch.

The structure follows the program file's logic, not a textbook's. [Chapters 2](#2)–[4](#4) lay foundations. [Chapters 5](#5)–[7](#7) cover the model architecture. [Chapter 8](#8) covers scaling. [Chapters 9](#9)–[11](#11) cover attribution, privacy, and operations. [Chapters 12](#12)–[13](#13) cover security and risk. Three appendices provide the source whitepapers and literature surveys.

Read linearly on first pass. After that, use the index below to jump from any term in the program file to its explanation.

### Term index

| Term in program file or whitepaper | Where it's explained |
|:---|:---|
| Embedding, vector representation, continuous manifold | [From discrete objects to continuous spaces](#2/from-discrete-objects-to-continuous-spaces) |
| Distance, cosine similarity, dot product | [Geometry as semantics](#2/geometry-as-semantics) |
| Multi-modal, five signals, joint training | [Multi-modal embedding: one space, many signals](#2/multi-modal-embedding-one-space-many-signals) |
| Dimensionality, 768 dimensions | [Dimensionality: why 768?](#2/dimensionality-why-768-and-what-does-each-dimension-mean) |
| ViT, Vision Transformer, ViT-L/14, ViT-H/14 | [The DINO family](#3/the-dino-family-teacher-student-distillation-with-momentum) |
| DINOv3, DINOv1, DINOv2, self-supervised | [Self-supervised learning](#3/self-supervised-learning-the-general-idea), [DINOv1 to DINOv3](#3/dinov1-to-dinov2-to-dinov3-what-each-generation-added) |
| Teacher-student, distillation, momentum encoder | [The DINO family](#3/the-dino-family-teacher-student-distillation-with-momentum) |
| Scaling laws (self-supervised vision) | [DINOv1 to DINOv3](#3/dinov1-to-dinov2-to-dinov3-what-each-generation-added) |
| Fine-tune vs. from-scratch | [Fine-tune vs. from-scratch: the design decision](#3/fine-tune-vs-from-scratch-the-design-decision) |
| Augmentation suite | [Augmentation: why it matters more than you think](#3/augmentation-why-it-matters-more-than-you-think) |
| Contrastive, InfoNCE | [The contrastive objective: deriving InfoNCE](#4/the-contrastive-objective-deriving-infonce) |
| Temperature (in loss) | [Temperature: the sharpness dial](#4/temperature-the-sharpness-dial) |
| In-batch negatives, batch size (8K-32K) | [In-batch negatives and why batch size matters](#4/in-batch-negatives-and-why-batch-size-matters) |
| Hard-negative mining, FAISS-based mining | [Hard-negative mining](#4/hard-negative-mining-where-the-real-lift-comes-from) |
| Multi-positive contrastive, hierarchically weighted positives | [Multi-positive contrastive: the program's variant](#4/multi-positive-contrastive-the-programs-specific-variant) |
| CLIP, SigLIP, SigLIP-2 | [CLIP](#5/clip-the-architecture-that-launched-multi-modal-learning), [From CLIP to SigLIP](#5/from-clip-to-siglip-fixing-the-batch-size-dependency) |
| Entity tower, learned embedding table, aggregator transformer | [The structured-entity tower](#5/the-structured-entity-tower-why-text-is-not-enough) |
| Structured entities, Iconclass, LCSH | [The structured-entity tower](#5/the-structured-entity-tower-why-text-is-not-enough) |
| Biography text, Wikipedia, Artwiki, cold-start via biography | [The biography signal: cold-start through language](#5/the-biography-signal-cold-start-through-language) |
| Price head, auxiliary regression, log-price percentile | [The price head: market value as a learned direction](#5/the-price-head-market-value-as-a-learned-direction) |
| Collaborative filtering, content-based | [Collaborative filtering](#6/collaborative-filtering-users-who-agreed-will-agree-again), [Content-based recommendation](#6/content-based-recommendation-features-instead-of-co-occurrence) |
| Two-tower retrieval, item tower, user tower | [Two-tower retrieval: the production architecture](#6/two-tower-retrieval-the-production-architecture) |
| Cold-start, warm items, cold items | [The cold-start problem: why art is especially hard](#6/the-cold-start-problem-why-art-is-especially-hard) |
| Temporal eval split, Recall@K, NDCG | [Evaluation methodology](#6/evaluation-methodology) |
| Stratified by item warmth | [The cold-start problem](#6/the-cold-start-problem-why-art-is-especially-hard), [Evaluation methodology](#6/evaluation-methodology) |
| Transformer over interaction sequences | [Transformers for interaction sequences](#7/transformers-for-interaction-sequences) |
| PinnerSage, YouTube session-based recommenders | [PinnerSage and YouTube session-based models](#7/pinnersage-and-youtube-session-based-models) |
| User state vector | [The user state vector: a point in motion](#7/the-user-state-vector-a-point-in-motion) |
| Multi-head prediction (click, save, follow, ...) | [Multi-head prediction](#7/multi-head-prediction-different-readouts-of-the-same-state) |
| UMAP, parametric UMAP | [UMAP at scale](#8/81-umap-at-scale) |
| FAISS, HNSW, approximate nearest neighbours | [FAISS and approximate nearest neighbors](#8/82-faiss-and-approximate-nearest-neighbors) |
| Tile-pyramid mosaics, deck.gl | [Tile-pyramid rendering](#8/84-tile-pyramid-rendering) |
| Shapley value, four axioms, cooperative game theory | [Cooperative game theory and the Shapley value](#9/cooperative-game-theory-and-the-shapley-value) |
| Leave-one-out vs. Shapley | [Why leave-one-out might actually be enough](#9/why-leave-one-out-might-actually-be-enough) |
| Layered attribution architecture | [The layered attribution architecture](#9/the-layered-attribution-architecture) |
| Manifold-shaping credit | [The layered attribution architecture](#9/the-layered-attribution-architecture) (Layer 2) |
| Representational effect ($\Delta$) and meaning function ($\omega$) | [Factoring difference and meaning](#9/factoring-difference-and-meaning) |
| Droit de suite | [The layered attribution architecture](#9/the-layered-attribution-architecture) (Layer 2) |
| System-generated works, generative attribution | [System-generated works](#9/system-generated-works) |
| Data Shapley, Ghorbani & Zou | [Data Shapley and its computational descendants](#9/data-shapley-and-its-computational-descendants) |
| Beta-Shapley, Kwon & Zou | [Data Shapley and its computational descendants](#9/data-shapley-and-its-computational-descendants) |
| Influence functions, Koh & Liang | [Data Shapley and its computational descendants](#9/data-shapley-and-its-computational-descendants) |
| TRAK, DataInf, KNN-Shapley | [Data Shapley and its computational descendants](#9/data-shapley-and-its-computational-descendants) |
| Filix ledger, Merkle-CRDT, content-addressed | [The Filix ledger as audit trail](#9/the-filix-ledger-as-audit-trail) |
| Reproducible training, deterministic data ordering | [The Filix ledger as audit trail](#9/the-filix-ledger-as-audit-trail), [Reproducibility in practice](#11/112-reproducibility-in-practice) |
| Per-recommendation provenance logging | [The Filix ledger as audit trail](#9/the-filix-ledger-as-audit-trail) |
| A/B testing (for attribution) | [The layered attribution architecture](#9/the-layered-attribution-architecture) (Layer 4) |
| CKA, RSA (representation comparison) | [Factoring difference and meaning](#9/factoring-difference-and-meaning) |
| Differential privacy | [Differential privacy](#10/101-differential-privacy) |
| Federated learning, federated user tower | [Federated learning](#10/102-federated-learning) |
| Machine unlearning, SISA training | [Machine unlearning and SISA training](#10/103-machine-unlearning-and-sisa-training) |
| Training pipeline | [Training pipelines](#11/111-training-pipelines) |
| Model versioning, model cards | [Model versioning and model cards](#11/113-model-versioning-and-model-cards) |
| A/B testing (for recommendation quality) | [A/B testing for recommenders](#11/114-ab-testing-for-recommenders) |
| Distillation attacks, model stealing | [Model distillation attacks](#12/121-model-distillation-attacks) |
| Membership inference | [Membership inference](#12/122-membership-inference) |
| Open/closed boundary, private ledger with public anchors | [The open/closed boundary under adversarial pressure](#12/123-the-openclosed-boundary-under-adversarial-pressure) |
| Watermarking, output perturbation | [Watermarking and output perturbation](#12/124-watermarking-and-output-perturbation) |
| Multi-head loss instability | [Multi-head loss instability](#13/131-multi-head-loss-instability) |
| Cold-start metric noise | [Cold-start metric noise](#13/132-cold-start-metric-noise) |
| Biography text quality | [Biography text quality](#13/133-biography-text-quality) |
| Behaviour data sparsity | [Behaviour data sparsity](#13/134-behaviour-data-sparsity) |
| Attribution noise at scale | [Attribution noise at scale](#13/135-attribution-noise-at-scale) |
| Reproducibility gap | [Reproducibility as constraint vs. aspiration](#13/136-reproducibility-as-constraint-vs-aspiration) |
| Meaning function as political choice | [The meaning function is a political choice](#13/137-the-meaning-function-is-a-political-choice) |
| Pinterest ItemSage | [CLIP](#5/clip-the-architecture-that-launched-multi-modal-learning), [Two-tower retrieval](#6/two-tower-retrieval-the-production-architecture) |
| Spotify Discover Weekly | [The biography signal](#5/the-biography-signal-cold-start-through-language), [Collaborative filtering](#6/collaborative-filtering-users-who-agreed-will-agree-again) |
| Ocean Protocol | [The layered attribution architecture](#9/the-layered-attribution-architecture), [Appendix B](#B) |

### Appendices

Three reference documents are included as appendices:

- **[Appendix A: Attribution Whitepaper](#A)** — the original theoretical framework extending Shapley values to heterogeneous contribution types via the Filix provenance ledger. [Chapter 9](#9) revises this framework into a layered architecture.
- **[Appendix B: Practical Value Attribution Systems](#B)** — survey of how music streaming, advertising, affiliate marketing, stock photography, and AI-generated content systems actually distribute value. The mechanisms that work are simpler than you'd expect.
- **[Appendix C: Shapley Literature Survey](#C)** — academic literature on Data Shapley, influence functions, and data valuation for ML. Identifies where the Beowolff proposal extends existing work and where it enters novel territory.

### Dependency graph

Some chapters build on others. Here is what requires what:

```
Chapter 2 (Foundations)
  +-- Chapter 3 (Self-Supervised Vision) -- needs: what embeddings are
  +-- Chapter 4 (Contrastive Learning) -- needs: geometry, loss functions
  |     +-- Chapter 5 (Multi-Modal) -- needs: contrastive objectives
  |           +-- Chapter 6 (Recommendation) -- needs: two-tower architecture
  |                 +-- Chapter 7 (Sequence Modelling) -- needs: user tower concept
  +-- Chapter 8 (Scaling) -- needs: what embeddings are, what FAISS does
  +-- Chapter 9 (Attribution) -- needs: what the embedding is and how it's trained

Chapters 10-13 are largely independent of each other
and can be read in any order after Chapter 9.

Appendices A-C are reference documents, not sequential reading.
```

### Gaps and assumptions

Places where the program file compresses reasoning that the guide expands:

- **"Multi-modal contrastive pretraining"** (program, Architecture) compresses the entire content of [Chapters 4](#4) and [5](#5) into three words. The guide spends ~8,000 words on what this means.
- **"Attribution-grade"** (program, Attribution) is used throughout as if it is a well-defined standard. It is not — [Chapter 9](#9) develops what it would need to mean.
- **"Shapley-on-Filix"** (program, Attribution) implies a unified computation. [Chapter 9](#9) argues this is better understood as a layered architecture with Shapley as one component. [Appendix B](#B) surveys how real systems solve the same problem with simpler mechanisms.
- **"Deterministic data ordering, fixed seeds, pinned environments"** (program, Attribution) makes reproducibility sound like a checklist. [Chapter 11](#11) explains why it is much harder than this in practice and what "reproducible within tolerance" actually requires.
- **"Right-to-delete that actually propagates"** (program, implicit) invokes machine unlearning as if it is a solved problem. [Chapter 10](#10) is honest about what is and is not tractable.
- **The program assumes millions of users with dense interaction histories.** [Chapter 13](#13) examines what happens if this assumption is false.

### Adding to this guide

New chapters or appendices follow this process:

1. **Write the chapter** as a markdown file in `chapters/` (or `chapters/appendices/` for reference documents). Use `##` for the chapter title, `###` for sections. LaTeX math via `$...$` (inline) and `$$...$$` (display).

2. **Register it** in `index.html` by adding an entry to the `entries` array. Chapters get numeric keys (`'14'`, `'15'`, ...). Appendices get letter keys (`'D'`, `'E'`, ...).

3. **Update this reading map.** Add relevant terms to the term index table with deep links: `[Section title](#key/heading-slug)`. The slug is the heading text, lowercased, punctuation removed, spaces replaced with hyphens.

4. **Cross-reference from other chapters.** Link to chapters with `[Chapter N](#N)` syntax. Link to sections with `[text](#N/slug)`. The frontend intercepts all `#key` and `#key/slug` links automatically.

Source whitepapers in `chapters/appendices/` are copies of the canonical documents at their original paths. To update an appendix, update the source and re-copy.
