## Learning What You Like

### Order matters

The simplest way to represent a collector's taste is to take every work they have interacted with, average their embeddings, and call the average "the collector's position." This actually works surprisingly well as a first approximation. But it throws away something important: the sequence.

A collector who spent last month browsing Impressionism and yesterday began exploring Brutalist architecture has different near-term interests than one who did the reverse. The average of {Impressionism, Brutalist architecture} is the same in both cases, but the trajectory is not. Yesterday's interest is more likely to predict what the collector wants to see today than last month's. Averaging flattens this temporal signal.

Beyond recency, interactions cluster into sessions — a focused half-hour of browsing on a weekday evening, a deep dive into a single artist's oeuvre on a Saturday afternoon. Within a session, interactions are coherent: they reflect a single thread of curiosity. Across sessions, they may reflect completely different facets of the collector's taste. A bag of interactions loses this structure. A sequence model preserves it.

The Beowolff user tower is a sequence model. It takes a collector's interactions in chronological order and processes them through a neural network that can attend to the pattern of the sequence — what came before, what came after, how much time elapsed between events — to produce a single point representing the collector's current state.

For the full technical treatment of the transformer architecture used in the user tower, see [Reader Chapter 7](../../study-guide/#7).

### Different actions mean different things

Not all interactions carry the same weight. The system distinguishes five types of action, each reflecting a different level of commitment:

| Action | What it signals | How much the system learns from it |
|:---|:---|:---|
| **Browse / click** | You saw it, maybe you were curious | A small nudge |
| **Save** | You deliberately bookmarked it for later | A moderate signal |
| **Follow an artist** | Ongoing interest in their trajectory | A strong signal |
| **Add to a collection** | You curated it into a personal exhibition | A strong signal |
| **Inquire to purchase** | You initiated a buying conversation | The strongest signal |

These are weighted differently in the model's training. A purchase inquiry shifts the collector's position substantially — the system takes it as serious evidence of preference. A casual browse barely moves anything. This weighting has a direct effect on what the system learns to prioritize: it pays more attention to the patterns behind saves and purchases than to the patterns behind clicks, because saves and purchases are cleaner indicators of genuine interest.

Each action type can also drive different product surfaces. The browse-prediction component tells the system what will catch your eye — useful for ordering search results. The save-prediction component identifies works you might want to return to. The follow-prediction component recommends artists rather than individual works. The purchase-prediction component identifies what you might actually buy — the most commercially valuable prediction, and the most sensitive to get wrong.

These are all different readouts of the same underlying understanding of your taste. The system does not maintain five separate profiles. It maintains one representation of you, and different components of the model read it at different thresholds.

### Your taste as a moving point

Here is the geometric picture. The embedding space is a landscape where every artwork is a fixed point. You, the collector, are a movable point in the same landscape. Your position determines which artworks are "near you" — those are your recommendations. When you interact with a work, your point shifts.

How it shifts depends on what you did:

- **You click on a Minimalist sculpture.** Your point nudges toward the Minimalism region. Small nudge — a click is weak evidence.
- **You save a Dan Flavin light installation.** Your point moves more substantially, toward Flavin's particular neighborhood — not just Minimalism in general, but Flavin's specific corner of it.
- **You inquire about purchasing a James Turrell.** Your point makes its largest move, precisely toward Turrell's location in the space.

Over many interactions, your point traces a path through the landscape. This path *is* the system's understanding of your taste. There is no separate taste profile, no handwritten rules, no explicit preference questionnaire. Your taste is your position, and the system's understanding is the trajectory that brought you there.

### The cold-to-warm transition

A new collector with no interaction history starts at a default position — roughly the center of the space, near nothing in particular. The recommendations at this point are generic: popular works, editorially curated selections, a broad sample.

As the collector makes their first few interactions, the point moves rapidly. Each new action carries high information because the system knows so little. After 20 or 30 interactions, the point stabilizes into a region that reflects emerging taste. After hundreds of interactions, each new action produces only a small perturbation — unless it signals a genuine shift. The collector who has saved 50 Abstract Expressionist paintings and then inquires about an Ai Weiwei installation — that is a large, informative move, because it says something new.

This mirrors the cold-start story for items from Chapter 6, but on the collector side. The system handles it the same way: content features of the few works you have interacted with are enough to produce a rough position, and behavioral signals refine it over time. The transition is smooth, not a hard switch.

### The filter bubble risk

Any system that learns from your behavior and then shows you more of what you seem to like creates a feedback loop. You click on Impressionism. The system shows more Impressionism. You click on what is shown. The system becomes more confident you like Impressionism. Your point converges to one region and stays there.

This is the filter bubble, and every major recommendation platform struggles with it. The mitigation in Beowolff is twofold.

First, the re-ranking step (Step 4 in Chapter 6) deliberately injects diversity. Some fraction of the recommendations come from outside your nearest neighbors — works from adjacent regions of the space, works from different traditions, works by emerging artists the system has not yet placed with confidence. These are exploration recommendations, and they exist to prevent the system from converging prematurely.

Second, the multi-action weighting helps. Because the system distinguishes between casual browsing and deliberate saving, a collector's point is not driven solely by clicks (which are fast, easy, and narrow). Saves and collection-adds reflect more deliberate, often more diverse preferences. A collector who clicks only on Impressionism but saves works from five different traditions will have a more broadly positioned point than the click pattern alone would suggest.

### What the system cannot see

The user tower learns from behavioral sequences. It does not read minds. Several important aspects of collector experience are invisible to the system:

**Private looking.** Works you studied carefully but did not click on, save, or follow. The system sees only actions, not attention.

**Offline encounters.** A gallery visit, an art fair, a studio visit, a conversation with a friend — these shape your taste but leave no trace in the system.

**Negative preferences.** The system sees what you did, not what you rejected. If you scrolled past fifty works before saving one, the system knows about the save but not the fifty passes. It infers negative preference from absence, which is much weaker than an explicit signal.

**Context.** You might browse high-end contemporary art during a focused research session and affordable prints during a casual scroll. The system tries to segment these sessions by the time gaps between interactions, but it has no access to your intent.

These limitations are inherent to any behavioral recommendation system, not specific to Beowolff. They are worth naming because they define the boundary between what the system can learn about you and what remains yours alone.

For the full technical treatment, including the mathematics of attention mechanisms and multi-head prediction, see [Reader Chapter 7](../../study-guide/#7).

### What this means for you

The system is learning from the order and type of your interactions, not just the set. Browsing Impressionism last week and Brutalist architecture today is a different signal from the reverse. Your taste, as the system understands it, is not a fixed category — it is a trajectory through a landscape, and where you are heading matters as much as where you have been.

This means the system rewards engagement. Not in the manipulative sense of maximizing screen time, but in the structural sense that more interactions produce a more precise understanding of your taste, which produces better recommendations, which makes the system more useful. The first thirty interactions teach the system the broad outlines. The next three hundred teach it the nuances.

For gallerists: your collectors' interaction patterns on the platform are shaping what the system shows them. Understanding that saves and collection-adds carry far more weight than casual browsing means that encouraging your collectors to use these features is not just good UX — it directly improves the quality of what the system surfaces for them. And if the system learns your collector's taste well, it becomes a tool for surfacing your artists' work to the right audience.
