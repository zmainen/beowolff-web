## How to Use This Guide

### Who this is for

You work in the art world. You are an artist, a gallerist, a collector, a curator, an art advisor. You have read about AI in art. You know that machine learning systems learn patterns from data, that recommendation algorithms power Spotify and Netflix, that "embeddings" and "neural networks" are real things and not just marketing language. But you have never trained a model, read a loss function, or written code.

This guide is for you. It explains the Beowolff-Embed program — a recommendation system designed for the art market — in plain language, one piece at a time.

### What this guide does not do

It does not teach you machine learning. It does not prepare you to build this system yourself. It gives you enough understanding to evaluate what the system does, how it makes decisions, where the risks are, and what it means for your practice.

If you want more technical depth on any topic, each chapter links to the corresponding chapter in the [Reader](../../study-guide/), which was written for researchers and engineers. The Reader has the math. This guide has the meaning.

### How the chapters work

Each chapter covers one component of the Beowolff-Embed system. They are numbered to match the Reader, so you can move between the two documents without getting lost.

Read them in order the first time. After that, use the table below to jump to whatever you need.

### Where to find what

| You want to understand... | Go to |
|:---|:---|
| What an embedding is and why it matters | [Chapter 2 — What Is an Embedding?](02-foundations.md) |
| How the model learns to see visual features in art | [Chapter 3 — How the Model Learns to See](03-self-supervised-vision.md) |
| How the model learns what counts as "similar" | [Chapter 4 — Teaching the Model What "Similar" Means](04-contrastive-learning.md) |
| Why the model reads metadata, biographies, and prices | [Chapter 5 — Five Signals in One Space](05-multi-modal-alignment.md) |
| How recommendations actually work in practice | [Chapter 6 — How Recommendations Work](06-recommendation-systems.md) |
| How the system learns your taste over time | [Chapter 7 — Learning What You Like](07-sequence-modelling.md) |

### Key terms, briefly

A few terms appear throughout the guide. Here they are in one place, so you do not have to hunt for definitions.

| Term | What it means |
|:---|:---|
| **Embedding** | A list of numbers that represents an artwork (or an artist, or a collector) as a point in a mathematical space. Similar things are nearby. |
| **Model** | The trained system that produces embeddings. It learned its behavior from data. |
| **Tower** | One component of the model that processes one type of information — images, text, metadata, or behavior. |
| **Cold-start** | The problem of recommending something that has no interaction history. Most artworks are cold-start items. |
| **Contrastive learning** | The training method: show the model pairs of things that should be similar and pairs that should not. It learns by contrast. |
| **Collaborative filtering** | The classic recommendation approach: people who agreed in the past will agree in the future. |
| **Two-tower retrieval** | The production architecture: one tower for items, one tower for users, both producing points in the same space. Recommendation is finding what is nearby. |
| **Attribution** | Tracing a recommendation back to the data that shaped it, so contributors can be compensated. |

### What the later chapters cover

The Reader has thirteen chapters. This guide covers the first seven — the parts that explain what the system does and how it works. Chapters 8 through 13 of the Reader cover scaling infrastructure, attribution mechanics, privacy, production operations, security, and risk analysis. Those chapters matter, but they are primarily relevant to engineers and governance teams. If you want to understand the attribution mechanism in particular (how artists get compensated), the [attribution whitepaper](../../attribution-whitepaper.md) is the right starting point, and [Reader Chapter 9](../../study-guide/#9) provides the technical treatment.
