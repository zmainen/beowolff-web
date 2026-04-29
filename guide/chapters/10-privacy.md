## 10. Your Data, Your Choice

When you use the platform -- browsing works, saving favorites, making purchases -- you generate data. That data improves the model: the system learns your preferences and uses them (along with thousands of other users' preferences) to train better recommendations for everyone. This chapter explains what the platform knows about you, what "delete my data" actually means in a machine learning system, and where the honest limits of current technology lie.

### What the platform knows

Your activity on the platform generates several kinds of data:

**Browsing history.** Which works you looked at, how long you spent on each, what you scrolled past. This is the most voluminous signal and also the noisiest -- looking at a work for ten seconds might mean you loved it, or that you were reading the wall text, or that you set your phone down.

**Explicit actions.** Saves, follows, collection groupings, inquiries, purchases. These are higher-quality signals because they reflect deliberate choices. A collector who adds a work to a named collection ("German Expressionism -- to consider") has told the model something precise about their taste.

**Purchase history.** What you bought, when, at what price. This is the highest-value signal and also the most sensitive.

The platform uses this data in two distinct ways. First, it builds your personal profile -- a representation of your taste that drives the recommendations you see. Second, it aggregates your data with other users' data to train the model itself. Your browsing patterns contribute, in a small way, to the model's understanding of how art preference works in general.

### What "delete my data" means -- and does not mean

Under privacy regulations like GDPR, you have the right to request deletion of your personal data. The platform can comply in a straightforward sense: remove your profile, your browsing history, your saved collections, your purchase records from the database. After deletion, the platform no longer holds your data in any identifiable form.

But here is the problem. Before your data was deleted, it was used to train the model. The model learned from your behavior -- along with thousands of other users' behavior -- and adjusted its internal parameters accordingly. Your data shaped the model, and deleting the data from the database does not automatically un-shape the model. The model still carries traces of your preferences in its parameters, mixed inseparably with traces of every other user's preferences.

This is the fundamental privacy challenge of machine learning. The model is not a database where your row can be cleanly removed. It is more like a painting that was influenced by hundreds of reference images: destroying one of the reference images does not change the painting that already exists.

### Machine unlearning: the hard problem

Researchers call this challenge "machine unlearning" -- making a model forget what it learned from specific data. The theoretically perfect solution is simple: delete the data and retrain the model from scratch on everything that remains. The retrained model provably does not contain any trace of your data, because it never saw it.

The practical problem is that retraining is expensive. The model is trained on millions of data points over days or weeks of computation. Retraining from scratch for every deletion request is not feasible.

Several approximate approaches exist. You can fine-tune the existing model to "forget" specific data, but there is no guarantee that subtle traces do not remain. You can partition the training data into separate segments and train independent sub-models on each, so that deleting data from one segment only requires retraining that segment's sub-model. This reduces the cost but adds engineering complexity and can reduce model quality.

### What actually works today

The honest answer is that the most practical approach -- and the one most production ML systems use -- is scheduled clean retrains. The platform maintains a queue of deletion requests. At regular intervals (say, monthly), it applies all pending deletions to the training data and retrains the model from scratch. Between retrains, the model still carries traces of deleted users' data. After the retrain, those traces are gone.

This is not perfect. There is a gap between the moment you request deletion and the moment the model forgets you. That gap is bounded and disclosed -- you know it will be at most one retraining cycle. But it is real.

### Differential privacy: strong guarantees, real costs

There is a technique called differential privacy that provides a mathematical guarantee: an adversary looking at the model's output cannot reliably determine whether your data was included in the training set. The model's behavior is essentially the same whether you were in the data or not.

The way it works is conceptually simple: during training, carefully calibrated noise is added to the learning process. The noise blurs individual contributions so that no single user's data has a detectable effect on the model's output. The more noise, the stronger the privacy guarantee.

The problem is that noise degrades the model. Recommendation systems depend on learning fine-grained distinctions between user preferences. Adding enough noise for strong privacy guarantees can reduce recommendation quality by 10-30%. For a platform whose value proposition is precisely the quality of its recommendations, this is a significant cost.

The current state of practice reflects this trade-off. Large technology companies use differential privacy for aggregate statistics (how many people clicked on a feature? what is the average session length?) where moderate noise is tolerable. No major recommendation platform has deployed differential privacy for personalised recommendations at the quality level users expect. The technique works in principle; the cost is currently too high for this application.

The system is designed so that differential privacy can be adopted as the technology matures. The research is improving yearly. But it would be dishonest to promise it today.

### What the platform commits to

Given these technical realities, here is what the platform actually offers:

**Database deletion.** Request it, and your identifiable data is removed from the database promptly. Your profile, your history, your records -- gone.

**Model deletion via scheduled retrains.** Your data's influence on the model is removed at the next scheduled retraining cycle. The maximum latency is disclosed. This is a real commitment, not a paper one -- the retraining infrastructure exists because the attribution system also requires periodic retraining.

**Transparency about the gap.** The platform does not pretend that database deletion equals model deletion. It tells you what each one means, what the timeline is, and what the residual risk looks like during the gap.

**Architecture for future improvement.** The training pipeline is designed so that more sophisticated unlearning techniques -- segment-based training, differential privacy -- can be introduced as they mature, without redesigning the entire system. The honest position is: the baseline is scheduled clean retrains, the architecture supports upgrades, and the platform will adopt stronger guarantees as they become practical.

### What you can do

You control what the platform knows about you by controlling what you share. Browsing without an account generates no identifiable data. Creating an account and browsing generates a profile. Saving, following, and purchasing generate richer data that improves the recommendations you receive but also increases your footprint in the system.

The trade-off is direct: more data shared means better recommendations for you and a more useful contribution to the model, but also a larger presence that takes longer to fully erase. This is a genuine trade-off, not one that can be designed away. What the platform can do -- and does -- is make the trade-off visible, give you control over it, and honor your choices when you make them.

For the full technical treatment, see [Reader Chapter 10](../study-guide/#10).
