## Sequence Modelling for User Behaviour

The user tower in Beowolff-Embed is a transformer that takes a collector's chronological interaction history — clicks, saves, follows, collection-adds, purchase inquiries — and compresses it into a single vector: the user's current taste state. This vector lives in the same embedding space as all the item embeddings. Recommendation reduces to finding the item embeddings nearest to the user vector. The chapter explains why sequence order matters, how transformers process interaction sequences, what prior systems the program draws on, and what the user state vector means geometrically.

### Why sequences, not bags

The simplest representation of a user's history is a bag: the unordered set of items they have interacted with. Average the item embeddings, and you have a user vector. This works surprisingly well for a first approximation, and many production systems used exactly this approach (or slight variants like TF-IDF-weighted averaging) for years.

But bags lose two signals that matter.

**Recency.** A collector who spent last month browsing Minimalism and yesterday started exploring German Expressionism has different near-term preferences than one who did the reverse. The bag sees both histories as {Minimalism, German Expressionism} and produces the same user vector. A sequence model naturally down-weights distant history and up-weights recent interactions — not by discarding old data, but by letting the model learn that recent actions are more predictive of the next action.

**Session structure.** Interactions are not uniformly distributed in time. They cluster into sessions — a 20-minute browse on a Tuesday evening, a focused deep-dive into a single artist on a Saturday afternoon. Within a session, interactions are coherent: they reflect a single thread of curiosity or intent. Across sessions, they may reflect different facets of the user's taste. A bag flattens this structure. A sequence model can learn to segment sessions (long time gaps between interactions signal session boundaries) and weight within-session coherence differently from cross-session diversity.

The user tower is, in effect, an evidence accumulator. It integrates noisy behavioral signals — a click is weak evidence of preference, a purchase is strong — over time to form a taste representation. Recent signals matter more not because old signals are discarded, but because the accumulated state has already absorbed earlier evidence and each new interaction shifts the state from where it currently sits. A new interaction that confirms existing taste produces a small update (the representation is already in the right region). A new interaction that contradicts the pattern — a collector who has browsed fifty Impressionist landscapes suddenly saving a Brutalist photograph — produces a larger update, because it carries more information relative to the current state.

The "decision variable" here is not scalar but a $d$-dimensional vector moving through the full embedding space. The user's taste is not a single threshold to cross but a point in a high-dimensional landscape, and its trajectory traces the evolution of their aesthetic interests.

### Transformers for interaction sequences

The user tower is a transformer operating over a sequence of interaction tokens. Each token represents one interaction event and encodes three things: what artwork was interacted with (the pre-computed item embedding), what kind of interaction it was (click, save, follow, add-to-collection, inquire-to-purchase), and when it happened (temporal position).

**Token construction.** For each interaction at position $t$ in the user's history:

$$x_t = \text{item\_emb}(a_t) + \text{action\_emb}(\text{type}_t) + \text{pos\_enc}(t)$$

where $\text{item\_emb}(a_t)$ is the pre-computed $d$-dimensional embedding of the artwork from the item tower, $\text{action\_emb}$ is a learned embedding for the action type (a small lookup table with 5-6 entries), and $\text{pos\_enc}(t)$ is a positional encoding.

**Positional encoding** deserves attention because it works differently here than in language models. In NLP, positional encoding marks the word's position in a sentence (position 1, 2, 3...). For interaction sequences, what matters is not ordinal position but *temporal distance*. An interaction yesterday and an interaction a month ago should have different positional signals even if they are adjacent in the sequence (because the user may have had no activity in between). The program would likely use a time-aware positional encoding — for example, sinusoidal encodings applied to the log of the time delta between consecutive interactions, or learned time-bucket embeddings (0-1 hours, 1-24 hours, 1-7 days, 1-4 weeks, 1+ months). This lets the model learn that two interactions in the same session (minutes apart) are related differently than two interactions in different months.

**Self-attention: a worked example.** Self-attention is the mechanism that lets each token in the sequence "look at" every other token to decide what information to extract. Let me walk through this concretely with a sequence of 4 interaction tokens.

Suppose a collector's recent history is:

| Position | Artwork | Action |
|---|---|---|
| $t=1$ | Rothko, *No. 61* | browse |
| $t=2$ | Rothko, *Orange and Yellow* | save |
| $t=3$ | Judd, *Untitled (Stack)* | browse |
| $t=4$ | Flavin, *monument for V. Tatlin* | browse |

Each token $x_t$ is a $d$-dimensional vector (combining item embedding, action embedding, and positional encoding). Self-attention computes, for each token, a weighted average of all tokens in the sequence, where the weights are determined by learned compatibility functions.

Three linear projections produce **queries**, **keys**, and **values**:

$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

where $X \in \mathbb{R}^{4 \times d}$ is the matrix of input tokens and $W_Q, W_K, W_V \in \mathbb{R}^{d \times d_k}$ are learned weight matrices. The attention weights are:

$$A = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)$$

This produces a $4 \times 4$ attention matrix. Each entry $A_{ij}$ is the weight that token $i$ assigns to token $j$. The output for each token is the weighted sum of all value vectors:

$$\text{output} = A \cdot V$$

What happens in our example? Token $t=4$ (Flavin) computes attention weights over all four tokens. If the model has learned that Minimalist/Light-and-Space art is a coherent cluster, the Flavin token will assign high attention weights to the Judd token ($t=3$, because both are Minimalist) and moderate attention to the Rothko tokens ($t=1,2$, because Abstract Expressionism is adjacent but distinct). The browse-only Rothko ($t=1$) might get less weight than the saved Rothko ($t=2$), because the save signal is encoded in the token and the model has learned that saves are more indicative of preference.

Critically, self-attention is permutation-equivariant without positional encoding — it treats the sequence as a set. The positional encoding breaks this symmetry and lets the model attend more to recent tokens. In our example, the fact that Judd and Flavin were browsed recently (tokens 3 and 4) while Rothko was earlier (tokens 1 and 2) affects the attention pattern because the time-aware positional encoding encodes this temporal structure.

**Causal masking.** During training, the user tower must predict the *next* interaction, so it should not see the future. A causal mask sets $A_{ij} = 0$ for $j > i$ — token $t$ can attend only to tokens at positions $\leq t$. This is identical to the autoregressive masking in GPT-style language models. At inference time, when computing the user's current state, the model has access to the full history and the mask is not needed.

The connection to language modeling is not a coincidence. The user tower is doing exactly what a language model does — predicting the next token given the history — except the "tokens" are interaction events and the "vocabulary" is the set of all artworks. The loss function during training is: given the user's first $t$ interactions, predict which artwork they interact with at step $t+1$. This is next-token prediction over a behavioral sequence.

### PinnerSage and YouTube session-based models

The program references two architectural templates for the user tower. Understanding what each contributes clarifies the design space.

**PinnerSage (Pinterest, 2020).** Pinterest's user representation system. The key idea: a single embedding vector is insufficient for users with diverse interests. A collector who loves both Renaissance portraiture and contemporary installation art occupies two distinct regions of the embedding space, and averaging them produces a vector in the middle that represents neither interest well.

PinnerSage addresses this with medoid-based clustering. It clusters a user's interaction history into groups (using the item embeddings), finds the medoid (most representative item) of each cluster, and represents the user as a small set of cluster embeddings rather than a single vector. At retrieval time, each cluster embedding is used to query the ANN index separately, and results are merged.

For Beowolff, the PinnerSage insight is relevant because art collectors are notoriously multi-faceted. A single collector might have a deep collection of Dutch Golden Age painting, a separate interest in contemporary photography, and a casual curiosity about African masks. These are three distinct regions of the embedding space. The question for the architecture is whether a single transformer output vector can represent this diversity or whether a multi-cluster approach is needed. In practice, most systems start with a single-vector user tower (simpler to train and serve) and add multi-cluster representations as a later refinement when evaluation shows that single-vector underperforms for users with high taste entropy.

**YouTube's deep neural network recommender (Covington et al., 2016).** YouTube's system established the two-stage architecture that is now standard: a **candidate generation** stage that retrieves a few hundred candidates from billions of videos using a lightweight model (the "recall" stage), followed by a **ranking** stage that scores each candidate with a heavier model (the "precision" stage).

The candidate generation model is the two-tower architecture from Chapter 6: user tower produces a user embedding, ANN search against pre-computed video embeddings retrieves top-K candidates. The ranking model is a deeper network that takes the (user, candidate) pair as input — along with richer features like video freshness, channel authority, and user-channel affinity — and produces a precise relevance score.

For Beowolff, the two-stage architecture means the user tower does not need to be perfect — it needs to be good enough to put the right 500 items into the candidate set out of 3 million. The ranking model handles the fine-grained discrimination. This relaxes the engineering pressure on the user tower: a slightly noisy user embedding that captures the rough taste direction is sufficient if the ranker compensates downstream.

YouTube's system also demonstrated the importance of training on implicit feedback with proper calibration. Watch time (not clicks) turned out to be the right training signal because clicks are noisy and gameable. For art, the analogous insight is that purchase inquiries and collection-adds are stronger signals than page views, and the loss should weight them accordingly.

### Multi-head prediction: different readouts of the same state

The user tower in the Beowolff program has multiple prediction heads, each corresponding to a different interaction type:

| Head | Signal | Preference strength | Loss weight |
|---|---|---|---|
| Click/browse | Saw it, maybe liked it | Weak | Low |
| Save | Deliberately bookmarked for later | Moderate | Medium |
| Follow (artist) | Ongoing interest in an artist's trajectory | Moderate-high | Medium-high |
| Add to collection | Curated into a personal exhibition | High | High |
| Inquire to purchase | Initiated a buying conversation | Very high | Very high |

Each head is a small MLP that takes the user tower's output vector and predicts the probability that the user's next interaction will be of that specific type with a given item. The heads share the same underlying user representation but specialize in predicting different signals.

**Why weight them differently in the loss.** The training loss is a weighted sum of the per-head losses:

$$\mathcal{L}_{\text{user}} = \sum_{h \in \text{heads}} w_h \cdot \mathcal{L}_h$$

where $w_h$ is the weight for head $h$. Purchase inquiry gets the highest weight because it is the strongest signal of genuine preference (the user is willing to spend money) and because it aligns most directly with the system's commercial purpose (generating sales). A click gets the lowest weight because it is the noisiest signal — people click for many reasons that have nothing to do with preference.

This weighting has a direct effect on the embedding geometry. The user vector moves more in response to purchase-inquiry training examples than in response to click training examples, because the gradient is amplified by the loss weight. Geometrically, the embedding space is shaped so that "purchase-similarity" matters more than "click-similarity" — two items that tend to be purchased by the same users will be closer together than two items that are merely browsed by the same users.

The multi-head architecture embodies a clean separation: one shared internal state, multiple specialized readouts that tap into it at different levels of commitment. A click is fast and cheap — low commitment, noisy signal. A save is more deliberate — higher commitment, cleaner signal. A purchase inquiry is the most costly action available — very high commitment, very informative. Each head reads out the same underlying taste representation but applies its own learned threshold and weighting. The shared representation benefits from all of them during training (each head's gradient flows back into the transformer), while at inference each head can drive a different product surface.

**What each head reveals independently.** Beyond their role in training the shared representation, the per-head predictions are individually useful at inference time:

- The **click head** predicts what will catch the user's eye — useful for ordering results in a browsing interface.
- The **save head** predicts what the user wants to return to — useful for "works you might want to save" suggestions.
- The **follow head** predicts which artists the user might want to track — useful for artist recommendation, distinct from artwork recommendation.
- The **collection-add head** predicts what would fit into the user's existing collections — useful for "add to your collection" prompts.
- The **purchase-inquiry head** predicts what the user might actually buy — the most commercially valuable prediction, but also the most sensitive to get wrong (a bad purchase recommendation wastes the user's time and the gallery's).

Each head can drive a different product surface. The embedding and the user tower are shared infrastructure; the heads are the specialized interfaces.

### The user state vector: a point in motion

The user tower's final output — after the transformer has processed the interaction sequence and (during training) the relevant prediction head has made its prediction — is a $d$-dimensional vector: the user state. This vector lives in exactly the same space as all item embeddings. What it represents is the user's current position in taste space.

**Geometric interpretation.** Imagine the embedding space as a landscape where every artwork is a fixed point and the user is a movable point. The user's position determines which artworks are "close" — those are the recommendations. When the user interacts with a new artwork, the user tower reprocesses their updated history and produces a new user vector, which may have shifted.

How it shifts depends on what the interaction was:

- **A click on a Minimalist sculpture** nudges the user vector toward the Minimalism region. Small nudge — clicks are weak evidence.
- **Saving a particular Flavin light installation** nudges the user vector more substantially toward Flavin's neighborhood. The save signal is stronger, and the direction is more specific (not just Minimalism but Flavin's particular corner of it).
- **A purchase inquiry on a James Turrell** produces the largest shift, and the direction is precise: toward Turrell's location in the embedding, weighted by the high loss weight on purchase inquiries.

Over many interactions, the user vector traces a trajectory through the embedding space. This trajectory IS the operational definition of "learning what the user likes." There is no separate taste model, no explicit preference profile, no handcrafted rules. The user's taste is their position in the space, and the system's understanding of their taste is the vector that position corresponds to.

**Why this geometric picture matters for the program.** The Beowolff program proposes that user profiles should be interpretable — the "taste portrait" concept from the design conversation. In principle, the user vector's position in the embedding space can be decomposed into human-readable components: "40% affinity toward Color Field painting, 25% toward contemporary photography, 20% toward pre-Columbian sculpture, 15% diffuse." This decomposition is possible because the item embeddings are anchored to known artworks and artists — you can project the user vector onto interpretable axes defined by artist or movement clusters.

In practice, whether this decomposition is faithful depends on the embedding geometry. If the space is well-organized (clusters correspond to recognizable categories), the projection is meaningful. If the space has significant entanglement (dimensions encode mixtures of unrelated properties), the projection is misleading. The structured-entity tower (Chapter 5) is designed to push toward the former: by training with entity-level objectives (same-artist positives, same-period positives), the embedding space is encouraged to organize along axes that correspond to curatorial categories. This is not guaranteed but it is architecturally incentivized.

**What happens when the user is cold.** A new user with zero or few interactions has a user vector that is essentially the transformer's default output — a point near the center of the space (or wherever the model's initialization places users with empty histories). This is a bad representation: it is far from every specific artwork and near nothing in particular.

As the user makes their first few interactions, the user vector moves rapidly — each new interaction carries high marginal information because the prior is so diffuse. After 20-30 interactions, the vector stabilizes into a region that reflects the user's emerging taste. After hundreds of interactions, each new interaction produces only a small perturbation unless it signals a genuine taste shift (the collector who has bought 50 Abstract Expressionist paintings suddenly inquires about an Ai Weiwei — that is a large, informative perturbation).

This cold-to-warm trajectory is the behavioral mirror of the item-side cold-start story from Chapter 6. The system handles it similarly: for cold users (few interactions), fall back to content-based features of the few items they have interacted with. As the user warms up, behavioral signals increasingly dominate. The transition is smooth because both signals live in the same embedding space — there is no hard switch between "content mode" and "CF mode," just a gradual shift in which signal carries more weight.

**Movement as learning.** It is worth pausing on the claim that the user vector's movement IS "learning what the user likes." This is not a metaphor. The recommendation system produces results by finding item embeddings near the user vector. If the user vector moves toward Minimalism, the system starts recommending more Minimalist works. If the user then clicks on those recommendations (positive feedback), the vector moves further toward Minimalism. If they do not (negative feedback — or rather, absence of positive feedback), the vector drifts back or toward whatever the next interaction points at.

This is a feedback loop, and like all feedback loops, it has pathologies. Filter bubbles are the best known: the user vector converges to a local region and the system only shows items from that region, preventing the user from discovering interests they do not yet know they have. Diversity injection at the ranking stage (showing some items from outside the nearest-neighbor set) is the standard mitigation. The program's multi-head architecture also helps: the different heads have different loss weights, so the user vector is not driven solely by clicks (which would converge fastest and narrowest) but also by saves and collection-adds (which reflect more deliberate, diverse preferences).

The user state vector is the most compact summary of the program's ambition: one vector, in one space, encoding everything the system knows about one person's relationship to art. Its movement through that space over time is the system learning. Its position at any moment is the system's best guess about what the user wants to see next. And because the space also contains every artwork, every artist entity, every metadata category, the user's position is inherently interpretable — not as a list of preferences, but as a location in a landscape whose landmarks are the artworks and artists and traditions of the corpus.
