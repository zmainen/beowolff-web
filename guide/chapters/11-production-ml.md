## 11. From Research to Reality

A trained model is not a product. It is an ingredient. The gap between "the model works in the lab" and "users see recommendations" is where most ML projects stall, and understanding that gap matters if you are evaluating whether this system's promises are engineering-feasible or aspirational.

This chapter covers what "production" means: the infrastructure that turns a trained model into a living system that serves real users, improves over time, and produces the verifiable records that attribution requires.

### The model runs continuously

A recommendation model is not trained once and deployed forever. The art market changes. New artists emerge. Collectors' tastes evolve. Works sell and leave the available inventory. A model trained on last year's data will give increasingly stale recommendations as the world moves on.

Production means the model is retrained periodically -- incorporating new works, new behavioral data, new market signals. Each retraining produces a new model version. The old version continues serving recommendations while the new one is evaluated. If the new version is better, it replaces the old one. If it is worse (it happens), the old version stays.

This creates a lineage of model versions, each with its own training data, its own performance measurements, and its own record in the provenance ledger. The lineage matters for attribution: when a sale occurs, the system needs to know which model version produced the recommendation, and what data that version was trained on.

### Model versions

Think of model versions the way a printmaker thinks of edition states. Each state is a complete, functional artifact. It differs from the previous state because something changed -- new data, an improved algorithm, a corrected error. The states are numbered, documented, and preserved.

Each model version comes with a structured record -- called a model card -- that describes what went into it and what came out. What data was it trained on? What architecture was used? How well did it perform on held-out test sets? Where does it fail? What is it not designed to do?

The model card is not just documentation. It is the link between the model and the attribution system. Every Shapley computation references a specific model version: "What was the model's performance when trained on this subset of data?" If model versions are not tracked with their corresponding data and evaluation results, attribution is impossible.

### A/B testing: the controlled experiment

When a new model version is ready, the question is whether it actually improves things. "Improves" is not obvious -- better for whom, by what measure? The standard approach is A/B testing: a controlled experiment where some users see recommendations from the old model and others see recommendations from the new one.

The logic is the same as a randomised controlled trial. Users are randomly assigned to groups. The only difference between groups is which model generates their recommendations. After a predetermined period (typically two to four weeks), the outcomes are compared.

What gets measured matters. The tempting metric is engagement -- clicks, time spent browsing. But a recommender optimised for engagement can learn to show provocative or sensational content that keeps people clicking without serving their actual interests. For an art platform, the metrics that matter are:

**Conversion.** Did the recommendation lead to a purchase inquiry or sale? This is what the platform's revenue model depends on.

**Discovery.** Did users encounter artists or traditions they had not previously engaged with? This measures whether the system is broadening taste rather than reinforcing existing preferences -- essential for the long-tail artists who are invisible to conventional sales channels.

**Retention.** Did users come back? Short-term engagement is easy to game. Return visits over weeks signal genuine satisfaction.

**Cold-start performance.** Did the new model successfully recommend works with no prior interaction history? This is the system's stated differentiator -- where the biography text and entity structure are supposed to earn their keep.

A/B testing for art has a complication that clinical trials do not face: the goods are unique. If the treatment model recommends a painting to User A and they buy it, that painting is gone. User B in the control group can no longer encounter it. Every sale in one group changes the available inventory for the other. In practice, this effect is negligible when the collection is large relative to the number of sales during the test -- which, for a platform with millions of works, it typically is.

### Reproducibility: harder than it sounds

The attribution system requires that the same training data produces the same model. This sounds like a minimum standard. It is actually a demanding engineering constraint.

Modern models are trained on GPUs -- specialised processors that execute thousands of operations in parallel. The order in which those operations complete is not guaranteed, and because of the way computers handle decimal numbers, adding the same numbers in a different order can produce slightly different results. The differences are tiny -- at the level of the fifteenth decimal place -- but they accumulate across millions of operations and thousands of training steps. Two runs on the same data, with the same configuration, on the same hardware, can produce models that agree on most predictions but disagree on boundary cases.

For most applications, this does not matter. For attribution, it does. Attribution computes the marginal effect of including or excluding a contributor's data. If the model itself varies between runs, the marginal effect inherits that variation. If the variation is larger than the effect you are trying to measure, the attribution is noise, not signal.

The practical solution is not to demand perfect reproducibility -- which is extremely expensive to achieve -- but to define a tolerance and verify empirically that the system stays within it. "The model's performance varies by less than 0.1% across repeated training runs" is a testable, achievable standard. The attribution system then needs to detect contributions whose effects are larger than this tolerance -- which, at the corpus level (an artist's entire body of work, not a single painting), they typically are.

For the full technical treatment, see [Reader Chapter 11](../study-guide/#11).

### The serving stack

Between a trained model and a user seeing a recommendation, several engineering layers intervene. The important ones for an art-world reader:

**Candidate generation.** The model does not score every work in the collection against every user in real time. Instead, it first retrieves a shortlist of a few hundred candidates that are close to the user's profile in the model's space. This uses the approximate nearest-neighbor search described in the scaling chapter -- the same technology that lets you search 690,000 works in milliseconds.

**Ranking.** A more sophisticated model re-scores the shortlist, incorporating factors the initial retrieval could not: inventory status, business rules (promote featured artists), diversity constraints (do not show five Monets in a row), and recency. This produces the final ordered list the user sees.

**Provenance logging.** Every recommendation served is logged: which model version produced it, what the user's state was, what the candidate set was, and why each work was ranked where it was. This is the raw material for attribution. If a sale follows, the log connects the sale to the model version and the training data that produced the recommendation.

This logging adds cost -- storage and processing for every recommendation, potentially gigabytes per day at scale. The system treats it as non-negotiable infrastructure. Without it, attribution is a promise. With it, attribution is a computation grounded in evidence.

### What "production" means for you

If you are evaluating this system as a potential contributor -- an artist depositing works, a gallery sharing inventory data, a collector sharing browsing history -- "production" means three things:

First, the model is alive. It retrains, improves, and adapts. Your contribution's effect on the model is not a one-time event but an ongoing relationship.

Second, every version is documented and auditable. You can ask: which version of the model was trained on my data? How did it perform? What was my contribution's measured effect? The provenance ledger and model cards make these questions answerable.

Third, improvements are tested before deployment. A new model version does not go live until a controlled experiment confirms it is better. This protects you as a contributor: your data is not fed into a system that degrades without anyone noticing.
