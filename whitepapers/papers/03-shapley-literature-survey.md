# Shapley Values for Data Valuation and Attribution in Machine Learning

**A Literature Survey for the Beowolff-Embed Attribution Program**

Zachary F. Mainen, April 2026

---

## 1. Foundations: Data Shapley and Its Descendants

The application of Shapley values to machine learning data valuation begins with Ghorbani and Zou (2019), who formalized the idea that each training example's worth can be measured as its average marginal contribution to model performance across all possible subsets of the training set. Their "Data Shapley" framework proved that the Shapley value is the unique valuation satisfying four axioms -- efficiency (all value is distributed), symmetry (equal contributions receive equal credit), the null-player property (zero-contribution data gets zero credit), and linearity (values decompose across additive performance measures). The practical contribution was a truncated Monte Carlo estimator that made the computation feasible for models up to moderate scale, including logistic regression and small neural networks. Their experiments demonstrated that Data Shapley could identify mislabeled examples, detect outliers, and guide data acquisition -- all more effectively than leave-one-out scores, which lack the subset-averaging that gives Shapley its theoretical guarantees.

The central limitation of Data Shapley is noise. Because the estimator averages over random permutations of the training set, and each permutation requires retraining (or approximating retraining), the variance of the estimate is high relative to the signal, especially when individual data points have small marginal effects -- which is exactly the regime that matters for large training sets. Kwon and Zou (2022) addressed this with Beta Shapley, which relaxes the efficiency axiom by introducing a Beta-distribution weighting over coalition sizes. The insight is that not all coalition sizes are equally informative: in data valuation, small coalitions (where each addition has large marginal effect) and very large coalitions (where the model is nearly fully trained) carry most of the signal, while mid-sized coalitions are noisy. By concentrating weight on informative coalition sizes, Beta Shapley reduces variance without sacrificing the other three axioms. Empirically, it outperforms Data Shapley on mislabeled-data detection and subset selection across several benchmarks.

Jia et al. (2019) took a different route to scalability. Their KNN-Shapley replaces the expensive model-retraining step with a K-nearest-neighbor surrogate computed over pre-trained feature embeddings. Because KNN classifiers have a closed-form Shapley solution (the marginal contribution of each training point depends only on its rank in the distance ordering to the test point), the computation drops from exponential to polynomial. The paper demonstrated comparable utility to full Data Shapley on tasks including noisy label detection, data summarization, and domain adaptation, while achieving orders-of-magnitude speedup. For the Beowolff program, KNN-Shapley is particularly relevant because it operates in embedding space: the "model" being valued is the embedding plus a KNN probe, which is close to how we would naturally evaluate whether a data contribution improved the embedding's neighborhood structure.

A related line of work, Datamodels (Ilyas et al., 2022), sidesteps the Shapley formalism entirely by training linear models that predict a test point's output as a function of which training points were included. This requires training thousands of models on random subsets and fitting a regression from subset indicators to outputs. Despite its brute-force character, the approach revealed that simple linear functions can predict model behavior surprisingly well, and the learned coefficients provide a data attribution that is complementary to Shapley -- less principled axiomatically but potentially more accurate empirically when enough retraining runs are available.

## 2. Influence Functions and Gradient-Based Approximations

The computational bottleneck in Shapley-based methods is retraining. Influence functions (Koh and Liang, 2017) offer an alternative: instead of retraining the model without a data point, estimate the effect of removing it using the model's curvature at convergence. Specifically, the influence of training point $z$ on a test prediction is approximated by the product of the test gradient, the inverse Hessian, and the training gradient. This is a first-order Taylor expansion of the leave-one-out effect, computable without any retraining. The original paper demonstrated that influence functions could identify training examples responsible for specific predictions, detect dataset errors, and even construct adversarial training-set attacks.

The trouble is that influence functions are fragile in deep learning. Basu, Pope, and Feizi (2020) showed through extensive experiments that influence function estimates diverge from actual leave-one-out effects as network depth and non-convexity increase. The Hessian inversion is ill-conditioned, the first-order approximation breaks down when the loss surface is highly curved, and the estimates are sensitive to regularization and training details. Bae et al. (2022) sharpened this critique by showing that even when influence functions correlate with leave-one-out on average, the rank ordering of influential examples can be substantially wrong -- which is what matters for data valuation, where you care about which points are most and least valuable, not the average effect.

Despite these limitations, the gradient-based approach has been refined into practically useful tools. TracIn (Pruthi et al., 2020) simplifies the computation by tracing the loss change on a test point whenever a training point is used in a gradient step, summed across training checkpoints. This avoids the Hessian entirely, requiring only per-example gradients at saved checkpoints. The approximation is coarser than influence functions but more stable, and it scales to large models because checkpoint saving is standard practice.

Schioppa et al. (2022) attacked the Hessian bottleneck directly with Arnoldi iteration, achieving the first successful application of influence functions to full-size Transformers with hundreds of millions of parameters, on datasets with tens to hundreds of millions of examples. This was a genuine engineering breakthrough: it demonstrated that with the right numerical linear algebra, influence functions are not inherently limited to small models. However, the approximation quality still depends on the loss landscape's local convexity, which is model- and task-dependent.

The most significant recent development is TRAK (Park et al., 2023), which combines ideas from random projections with the gradient-based framework. TRAK projects per-example gradients into a low-dimensional random subspace, then fits a linear model from projected training gradients to test outputs. This achieves the accuracy of Datamodels (which requires thousands of retraining runs) using only a handful of trained models. TRAK demonstrated effectiveness on ImageNet classifiers, CLIP vision-language models, and language models including BERT and mT5. For the Beowolff program, TRAK's applicability to CLIP is directly relevant: it has been validated on a multimodal contrastive model whose architecture resembles what we plan to build.

DataInf (Kwon et al., 2023) provides a closed-form influence approximation specifically designed for parameter-efficient fine-tuning (LoRA). Because LoRA constrains updates to a low-rank subspace, the Hessian within that subspace is tractable. DataInf demonstrated orders-of-magnitude speedup over standard influence computation on RoBERTa-large, Llama-2-13B, and Stable Diffusion, while maintaining accuracy in identifying mislabeled examples. This is relevant to any deployment scenario where the embedding model is fine-tuned rather than trained from scratch -- which is likely for Beowolff-Embed, where the vision tower is initialized from DINOv3 and fine-tuned on art data.

## 3. Data Markets and the Economics of Valuation

Agarwal, Dahleh, and Sarkar (2019) formalized the data marketplace problem: how should a platform set prices for data when the value of each dataset depends on what other datasets the buyer already has? They showed that a randomized Shapley mechanism is "robust to replication" -- a data seller cannot inflate their payment by submitting duplicates -- and proposed it as the basis for a marketplace pricing rule. The theoretical result is clean: Shapley-based pricing is the unique mechanism satisfying fairness, no-free-rider, and replication-robustness simultaneously. The practical limitation is that computing the prices requires evaluating the buyer's utility function on exponentially many data subsets, which circles back to the approximation problem.

Ocean Protocol (Trentmann et al., 2022) took the market infrastructure approach: tokenize data as ERC-20 assets, let prices emerge from trading, and use compute-to-data architecture so data never leaves the provider's control. Ocean solves the trust and privacy problems elegantly but does not solve the valuation problem -- tokens trade at market-clearing prices that reflect supply and demand, not marginal contribution to any particular model. A dataset that is redundant with freely available alternatives will trade at zero regardless of its intrinsic quality. This is economically correct but does not help a contributor understand whether their data actually improved a model.

The gap between Shapley-based valuation and market-based pricing is real. Shapley measures contribution to a specified utility function (typically model performance on a test set). Markets aggregate information from multiple buyers with different utility functions, different existing data holdings, and different willingness to pay. These are not the same thing, and conflating them leads to confusion. For the Beowolff program, the relevant question is contribution to a specific model's embedding quality, not market price of the data in the abstract. Shapley is the right tool; market pricing is a separate mechanism for when the data is traded externally.

Zheng et al. (2025) recently proposed Asymmetric Data Shapley, which modifies the Shapley framework to account for structure in data markets -- the observation that data contributions are not symmetric (early entrants may have structurally different value than late ones, and complementary data has different value than redundant data). This connects to the temporal Shapley extension discussed in the Beowolff attribution whitepaper, where early contributions to a small training set are weighted differently from late contributions to a large one.

## 4. Attribution for Embeddings and Representation Learning

This is where the literature thins most sharply. The vast majority of data valuation work measures contribution to a downstream classifier's accuracy on held-out test data. The value function $v(S)$ is almost always "train a model on subset $S$, evaluate accuracy on test set $T$." For an embedding model, the question is different: how did this training example shape the geometry of the learned representation space? Accuracy on a downstream probe is one way to measure this, but it collapses the richness of the embedding into a single scalar and depends on the choice of probe task.

No published work directly computes Shapley values where the value function is a measure of embedding geometry rather than classifier accuracy. This is a genuine gap.

Several adjacent results are relevant, however. TRAK's application to CLIP (Park et al., 2023) is the closest existing work to what Beowolff needs. CLIP is a contrastive model that learns aligned image-text embeddings, and TRAK attributes CLIP's behavior -- specifically, its image-text alignment scores -- to individual training examples. The limitation is that TRAK's value function is still a scalar output (the alignment score for a specific query), not a geometric property of the embedding space as a whole. It answers "which training examples made this particular image-text pair align well?" rather than "which training examples shaped this region of the embedding manifold?"

Deng, Tang, and Ma (2024) extended influence functions to models trained with non-decomposable losses, including contrastive losses. Standard influence functions assume the training loss decomposes as a sum over individual examples, which contrastive losses violate (each loss term involves pairs or batches of examples). Their method handles this by marginalizing over the batch structure, producing per-example influence estimates even when the loss couples examples. This is directly relevant to Beowolff-Embed, which will be trained with multi-task contrastive losses where each term involves positive and negative pairs.

Lin et al. (2024) developed an efficient Shapley framework specifically for diffusion models, with experiments on DDPM, Latent Diffusion Models, and Stable Diffusion fine-tuned on Post-Impressionist art. Their value function measures image quality (FID) and demographic diversity rather than embedding geometry per se, but the computational infrastructure -- model pruning and efficient fine-tuning to reduce the cost of subset retraining -- transfers directly to embedding models. The art-domain experiments are particularly relevant: they demonstrated that Shapley attribution can identify which artists' contributions most improved a model trained on Post-Impressionist paintings.

Sun et al. (2025) proposed bridging gradient-based and representation-based attribution methods, explicitly addressing the gap between "what changed in the gradients" and "what changed in the representations." This is the closest work to the factorization we want: separating the structural effect of a data point on the embedding from its downstream task value.

For measuring embedding similarity itself -- the "difference" side of the attribution -- three tools from the representation learning literature are directly applicable. Representational Similarity Analysis (RSA; Kriegeskorte, Mur, and Bandettini, 2008) compares representations by correlating their pairwise distance matrices: given two representations of the same set of stimuli, RSA measures whether they impose similar similarity structures. This was developed in computational neuroscience for comparing brain activity patterns to model predictions, but it applies without modification to comparing two versions of a learned embedding -- before and after a data contribution.

Centered Kernel Alignment (CKA; Kornblith et al., 2019) improves on RSA by centering the kernel matrices before computing their alignment, making the measure invariant to isotropic scaling and more robust to the dimensionality difference between compared representations. CKA has become the standard tool for comparing neural network layers across architectures and training runs. For embedding attribution, CKA provides a natural value function: $v(S)$ could be the CKA similarity between the embedding trained on subset $S$ and some reference embedding (either the full model or an oracle).

Williams, Kunz, and Kornblith (2021) generalized these shape metrics further, establishing axiomatic foundations for representation comparison and proving uniqueness results analogous to Shapley's axioms for the attribution problem. Their framework distinguishes between metrics that are sensitive to the overall scale of representations and those that are purely shape-based -- a distinction that matters for embedding attribution, where we might want to credit data that changes the topology of the embedding (its neighborhood structure) differently from data that merely rescales distances.

## 5. Practical Deployment and Known Limitations

The gap between theory and practice in data attribution is substantial, and several recent papers confront it directly.

Nguyen, Seo, and Oh (2023) conducted a Bayesian analysis of training data attribution and found that "the influence of an individual training sample is often overshadowed by the noise stemming from model initialization and SGD batch composition." Their conclusion is sobering: TDA should only be trusted for predictions that are consistently influenced by specific training data across different random seeds. Such noise-independent pairs exist but are rare. For an attribution system that needs to assign Shapley values to individual data contributions, this raises the question of whether the signal is detectable above the stochastic training noise.

Cheng et al. (2025) examined the adoption of training data attribution in industry practice and found that while the theory is well-developed, practical deployment remains limited by computational cost, the difficulty of validating attribution in the absence of ground truth, and the mismatch between what TDA measures (influence on a specific model checkpoint) and what stakeholders want to know (influence on the deployed system's behavior over time).

Wang et al. (2025) addressed hyperparameter sensitivity, showing that existing attribution methods are brittle to choices like learning rate, batch size, and the number of training epochs used in the subset retraining. This is a practical concern for any production system: if Shapley values change substantially when the training pipeline is tweaked, their use as a basis for compensation is undermined.

Ilyas and Engstrom (2025) recently proposed MAGIC, described as "near-optimal data attribution for deep learning," which appears to push the accuracy-efficiency frontier further than TRAK, though details are still emerging. The trajectory of the field suggests that computational barriers are being steadily reduced, but the fundamental challenge remains: any method that requires retraining (even efficient retraining) scales at best linearly with the number of training points to be attributed, and sublinearly only with strong structural assumptions.

For Beowolff-Embed specifically, the practical constraints are: (a) the embedding model will be large (ViT-L/14 scale), (b) the training set will grow over time as new data partners contribute, (c) attribution must be computed incrementally rather than from scratch at each update, and (d) the attribution must be credible enough to serve as a basis for revenue distribution. Constraint (d) is the hardest: it requires not just that the Shapley estimates be approximately correct, but that they be demonstrably so -- auditable, reproducible, and robust to the choice of evaluation methodology.

## 6. Factoring Attribution into Effect and Value

The Beowolff program requires distinguishing two components of a data contribution's worth: how much it changed the model (representational difference) and whether that change was valuable (alignment with the task objective). No published work makes this decomposition explicitly in the Shapley framework, but the ingredients exist in adjacent literatures.

In causal inference, the distinction between treatment effect and outcome specification is standard. A treatment may have a large effect that is irrelevant to the outcome of interest, or a small effect that is precisely targeted. Pearl's do-calculus and Rubin's potential outcomes framework both separate the causal mechanism (what the intervention does) from the evaluation criterion (what we care about). Translating this to data attribution: a training example that dramatically reshapes the embedding geometry (large representational effect) might do so in a direction that is orthogonal to task performance (zero value), while a subtle adjustment that shifts a few neighborhoods (small representational effect) might fix a systematic bias (high value).

The factorization $\phi_i = \Delta_i \cdot \omega_i$ -- where $\Delta_i$ is the representational effect of data point $i$ and $\omega_i$ is the value weight of that effect -- is not standard in the literature. But it can be constructed from existing tools. Let $\Delta_i$ be measured by CKA dissimilarity between the embedding trained with and without point $i$. Let $\omega_i$ be the change in a task metric (recommendation quality, sales conversion, user satisfaction) per unit of CKA dissimilarity. Then $\phi_i = \Delta_i \cdot \omega_i$ decomposes the Shapley value into effect and value components.

This decomposition has practical consequences. An artist whose corpus defines a distinctive region of the embedding -- a school of Post-Internet figuration, say -- may have a large $\Delta$ (the embedding geometry is measurably different with their work included) even when $\omega$ is initially small (nobody has yet bought from that region). If the attribution system only tracks $\phi = \Delta \cdot \omega$, the artist gets no credit until sales materialize. If it tracks $\Delta$ and $\omega$ separately, the system can recognize the representational contribution independently of whether it has yet been monetized. This matters for incentive design: artists should be credited for teaching the model to see, not only for generating revenue.

The "manifold-shaping credit" described in the Beowolff pitch -- where an artist receives attribution when any work in their region of the embedding sells, because the model's ability to surface that region was shaped by their contribution -- is a special case of this decomposition. It requires measuring $\Delta$ at the regional level (how much did this artist's data shape this neighborhood?) and $\omega$ at the transaction level (did a sale in this neighborhood occur?). The Shapley framework handles the first part naturally; the second requires connecting the attribution pipeline to the revenue event stream.

## 7. Synthesis: What the Literature Supports and Where It Falls Short

The literature provides strong foundations for what Beowolff needs, but the specific combination -- Shapley attribution for an embedding model where the value function measures geometric properties of the representation space -- has not been built before.

**What is well-established:** Shapley values are the uniquely fair attribution mechanism (Shapley 1953). They can be applied to ML data valuation (Ghorbani and Zou 2019). Efficient approximations exist for large-scale models (TRAK, DataInf, Beta Shapley). Influence functions work for Transformers when implemented carefully (Schioppa et al. 2022). Contrastive losses can be handled (Deng et al. 2024). Representation similarity measures (CKA, RSA) provide principled ways to quantify embedding change.

**What has been demonstrated but not for embeddings specifically:** Shapley attribution for diffusion models (Lin et al. 2024). TRAK applied to CLIP (Park et al. 2023). Data valuation in fine-tuned generative models (DataInf, Kwon et al. 2023). These show the machinery works for complex models and non-standard losses, but they all use scalar value functions rather than geometric ones.

**What is genuinely novel in the Beowolff proposal:** Using embedding geometry (CKA dissimilarity, neighborhood structure change) as the Shapley value function. Decomposing attribution into representational effect and task value. Temporal weighting that credits early contributions to a growing embedding. Extending attribution from data to code, policy, and method contributions. None of these have been done in the published literature. They are feasible constructions from existing components, but they require engineering and validation that has not been reported.

**What remains risky:** The signal-to-noise ratio of individual data point attribution in large embedding models (Nguyen et al. 2023 suggests it may be low). The sensitivity of attribution to training hyperparameters (Wang et al. 2025). The computational cost of incremental Shapley updates as the data corpus grows. The gap between attribution that is approximately correct and attribution that is credible enough to underpin revenue distribution. These are not objections to the approach -- they are engineering challenges that the program must solve and validate empirically.

---

## References

Agarwal, A., Dahleh, M., & Sarkar, T. (2019). A marketplace for data: An algorithmic solution. *Proceedings of the 2019 ACM Conference on Economics and Computation*, 701--726.

Bae, J., Ng, N., Lo, A., & Ghassemi, M. (2022). If influence functions are the answer, then what is the question? *Advances in Neural Information Processing Systems (NeurIPS)*, 35.

Basu, S., Pope, P., & Feizi, S. (2020). Influence functions in deep learning are fragile. *arXiv preprint arXiv:2006.14651*.

Cheng, D., Bae, J., Bullock, J., & Kristofferson, D. (2025). Training data attribution: Examining its adoption and use cases. *arXiv preprint*.

Deng, J., Tang, W., & Ma, J. W. (2024). A versatile influence function for data attribution with non-decomposable loss. *arXiv preprint*.

Ghorbani, A., & Zou, J. (2019). Data Shapley: Equitable valuation of data for machine learning. *Proceedings of the 36th International Conference on Machine Learning (ICML)*, 97, 2242--2251.

Ilyas, A., Park, S. M., Engstrom, L., Leclerc, G., & Madry, A. (2022). Datamodels: Predicting predictions from training data. *Proceedings of the 39th International Conference on Machine Learning (ICML)*.

Ilyas, A., & Engstrom, L. (2025). MAGIC: Near-optimal data attribution for deep learning. *arXiv preprint*.

Jia, R., Wu, F., Sun, X., Xu, J., Dao, D., Kailkhura, B., Zhang, C., Li, B., & Song, D. (2019). Scalability vs. utility: Do we have to sacrifice one for the other in data importance quantification? *arXiv preprint arXiv:1911.07128*.

Koh, P. W., & Liang, P. (2017). Understanding black-box predictions via influence functions. *Proceedings of the 34th International Conference on Machine Learning (ICML)*, 70, 1885--1894.

Kornblith, S., Norouzi, M., Lee, H., & Hinton, G. (2019). Similarity of neural network representations revisited. *Proceedings of the 36th International Conference on Machine Learning (ICML)*.

Kriegeskorte, N., Mur, M., & Bandettini, P. A. (2008). Representational similarity analysis -- connecting the branches of systems neuroscience. *Frontiers in Systems Neuroscience*, 2, 4.

Kwon, Y., & Zou, J. (2022). Beta Shapley: A unified and noise-reduced data valuation framework for machine learning. *Proceedings of the 25th International Conference on Artificial Intelligence and Statistics (AISTATS)*, 151, 8780--8802.

Kwon, Y., Wu, E., Wu, K., & Zou, J. (2023). DataInf: Efficiently estimating data influence in LoRA-tuned LLMs and diffusion models. *Proceedings of the International Conference on Learning Representations (ICLR)*, 2024.

Lin, C., Lu, M., Kim, C., & Lee, S.-I. (2024). An efficient framework for crediting data contributors of diffusion models. *arXiv preprint arXiv:2407.03153*.

Nguyen, E., Seo, M., & Oh, S. J. (2023). A Bayesian approach to analysing training data attribution in deep learning. *arXiv preprint arXiv:2305.19765*.

Park, S. M., Georgiev, K., Ilyas, A., Leclerc, G., & Madry, A. (2023). TRAK: Attributing model behavior at scale. *Proceedings of the 40th International Conference on Machine Learning (ICML)*.

Pruthi, G., Liu, F., Sundararajan, M., & Kale, S. (2020). Estimating training data influence by tracing gradient descent. *Advances in Neural Information Processing Systems (NeurIPS)*, 33.

Schioppa, A., Zablotskaia, P., Vilar, D., & Sokolov, A. (2022). Scaling up influence functions. *Proceedings of the AAAI Conference on Artificial Intelligence*, 36.

Shapley, L. S. (1953). A value for n-person games. In H. W. Kuhn & A. W. Tucker (Eds.), *Contributions to the Theory of Games, Volume II* (pp. 307--317). Princeton University Press.

Sun, W., Liu, H., Kandpal, N., Raffel, C., & Yang, Y. (2025). Enhancing training data attribution with representational optimization. *arXiv preprint*.

Trentmann, N., Trautman, P., Koppel, M., et al. (2022). Ocean Protocol: Tools for the Web3 data economy. *arXiv preprint arXiv:2203.16063*.

Wang, W., Deng, J., Hu, Y., Zhang, S., & Jiang, X. (2025). Taming hyperparameter sensitivity in data attribution. *arXiv preprint*.

Williams, A. H., Kunz, E., Kornblith, S., & Hinton, G. (2021). Generalized shape metrics on neural representations. *Advances in Neural Information Processing Systems (NeurIPS)*, 34.

Zheng, X., Huang, Y., Chang, X., Jia, R., & Tan, Y. (2025). Rethinking data value: Asymmetric Data Shapley for structure-aware valuation in data markets and machine learning pipelines. *arXiv preprint*.
