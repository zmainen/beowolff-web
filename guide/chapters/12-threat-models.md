## 12. What Could Go Wrong (Adversaries)

The system's architecture draws a deliberate boundary: the protocol and tools are open and auditable, the model and data are private and proprietary. This chapter asks what happens when someone tries to cross that boundary.

The adversary is not a teenage hacker. It is a well-funded competitor -- an existing art platform or a technology company entering the art market -- with engineering talent, money, and patience. The question is not whether a breach is theoretically possible but whether the defenses make it more expensive than doing the work honestly.

### Model stealing

The most direct attack: query the system's API enough times to approximate the model. The competitor submits thousands of artworks, collects the system's responses (similarity scores, recommendations, or embedding vectors), and trains their own model to mimic those responses. No access to the original model's architecture, training data, or code is needed. Just inputs and outputs.

How much querying is required depends on what the API returns. If the API returns full embedding vectors (a rich, detailed representation of each artwork), a competitor could collect vectors for a large sample of works and train their own model to reproduce them. If the API returns only ranked lists (these five works are similar to the one you asked about), the attack is harder but not impossible.

**Defenses.**

**Rate limiting.** Cap how many queries any single user, account, or network address can make. This forces the attacker to either work slowly (months instead of days) or use many accounts (expensive and detectable). Rate limiting is simple to implement and hard to circumvent at scale.

**Output noise.** Add a small amount of random perturbation to every response. Legitimate users do not notice the difference -- the recommendations are still excellent. But a competitor training a surrogate model on thousands of noisy responses gets a degraded copy. The noise is calibrated so that it is invisible for normal use but meaningful when accumulated across the thousands of queries an attack requires.

A subtle point: if the noise is random each time, the attacker can query the same artwork multiple times and average the responses to cancel the noise. The defense is to make the noise deterministic -- the same artwork always gets the same perturbation. This prevents averaging without introducing inconsistency.

**Watermarking.** Embed a detectable statistical signature in the model's outputs. If the competitor's surrogate model reproduces the watermark, this provides evidence of copying for legal enforcement. Watermarking does not prevent the attack, but it makes it provable after the fact.

**Tiered access.** Return richer information (full embedding vectors, detailed scores) only to authenticated partners with contractual relationships. Public users see only ranked lists, which contain less information for an attacker to learn from.

**Terms of service.** Contractually prohibit systematic querying for model extraction. Legal rather than technical, but it provides an enforcement mechanism.

**Is this enough?** Against a casual competitor, yes. Against a major technology company willing to invest months of systematic querying -- probably not, if they are sufficiently motivated. No single defense is impenetrable, but the combination raises the cost substantially. The honest assessment is that if a large company decides to clone the recommendation model, they probably can, given enough time and resources. The question is whether it is worth their effort.

### Membership inference

A different kind of attack: can someone determine whether a specific artwork was in the training data? Or whether a specific user's behavioral data was included?

The basic idea: a model tends to be more "confident" about data it was trained on than about similar data it has not seen. An artwork in the training set will have a more coherent embedding -- closer to its expected neighbors, more consistent with its metadata -- than a similar artwork the model has never encountered. An attacker who knows what to look for can sometimes distinguish the two cases.

**Why this matters.** For artworks, the concern is commercial rather than personal -- confirming that a publicly available artwork was in the training data reveals something about the training corpus composition, which is a competitive secret. For user data, the concern is sharper: confirming that a specific user's behavioral data was in the training set reveals that the platform has data on that user, which may itself be sensitive.

**Defenses.** The formal defense against membership inference is the differential privacy described in the previous chapter -- which, as noted, is currently too costly for production recommendation quality. The practical defense relies on the model being well-regularised (not memorizing individual training examples) and on the output perturbation described above. These provide empirical protection without formal guarantees.

### Why the data and relationships matter more than the algorithm

The system's competitive advantage does not rest on the model architecture. The architecture is built from published components -- vision transformers, contrastive learning, two-tower retrieval. The study guide you may have read explains every piece. A competent ML team could build the same architecture in months.

What a competitor cannot replicate:

**The data corpus.** The combination of artwork images and metadata from Artsy, auction records from Artnet, and behavioral data from millions of users. Each source is individually available or licensable. The union -- under a single training pipeline, with cross-source entity resolution and quality control -- represents years of engineering and relationship-building.

**The relationships.** Gallery onboarding, artist agreements, auction-house integrations, collector accounts. These are commercial network effects that compound over time. A new entrant can build them, but it takes years and trust.

**The attribution track record.** Once Shapley-based attribution is running and contributors are receiving payments, the platform has a demonstrated history of fair compensation. This is a trust asset that a new entrant cannot replicate on day one. They can promise attribution. They cannot prove a track record of it.

An important implication: even a successful model-stealing attack gives the competitor only a snapshot. The stolen model does not improve as the original is retrained on new data. The competitor must continuously steal, which means continuously querying at volume, which rate limiting and monitoring can detect.

### The combined defense posture

No single defense is sufficient against a determined, well-resourced adversary. The realistic defense is layered:

1. Rate limiting raises the time cost.
2. Output perturbation raises the quality cost (the copy is measurably worse).
3. Watermarking provides evidence for legal enforcement.
4. Tiered access limits what anonymous users can learn.
5. Terms of service provide contractual enforcement.
6. The data and relationship moat makes even a successful model theft insufficient to compete, because the model is only one component of the product.

This is the standard defense posture for production ML systems -- not impenetrable, but raising the cost of attack above the value of success. For an art intelligence platform, where the market is specialised and the value of the model is intertwined with the data and relationships behind it, this posture is likely adequate.

The most important investment is not in technical defenses but in building the data corpus and relationship network that make the model valuable in the first place -- and that no amount of API querying can replicate.

For the full technical treatment, see [Reader Chapter 12](../study-guide/#12).
