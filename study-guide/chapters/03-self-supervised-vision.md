## Self-Supervised Vision: What DINOv3 Does and Why It Matters

The program initializes its image tower from DINOv3, Meta's 2024 self-supervised vision model. This is not a throwaway detail — it is one of the most consequential choices in the architecture, because it determines what the image encoder already knows before art-specific training begins. Understanding the choice requires understanding what self-supervised learning is, what DINO specifically does, why the DINO lineage is the right one for art (as opposed to, say, CLIP or a masked autoencoder), and what fine-tuning from a pretrained checkpoint actually buys.

### Self-supervised learning: the general idea

Supervised learning requires labels: every training image paired with a human-provided annotation ("this is a cat," "this painting is Baroque"). Self-supervised learning eliminates the labeling bottleneck by defining a training objective that can be computed from the data itself — no human annotation needed. The model learns by predicting parts of its own input.

**The neuroscience bridge is predictive coding.** The brain learns representations not by being told what things are but by predicting its own sensory input and updating when predictions fail. Rao and Ballard's 1999 model proposed that each level of the visual hierarchy generates top-down predictions of activity at the level below, and the residual (prediction error) is what propagates upward. Learning happens by adjusting the generative model to minimize prediction error. Self-supervised learning in ML is a computational parallel: the model generates a prediction (of a masked patch, of another augmented view, of the next token in a sequence) and updates its parameters to minimize the prediction error. In both cases, the representation emerges as a byproduct of getting better at prediction.

Early self-supervised methods in vision used hand-designed **pretext tasks**: predict the rotation applied to an image (0, 90, 180, or 270 degrees), solve a jigsaw puzzle of image patches, colorize a grayscale image. These worked — the model had to learn something about visual structure to solve the task — but the representations were limited by the specificity of the pretext task. A model trained to predict rotation learns about canonical orientations but not about texture or color.

The breakthrough came from methods that replaced hand-designed pretext tasks with objectives derived from the structure of representation learning itself. The two major families are:

1. **Contrastive methods** — learn by pulling together different views of the same image and pushing apart views of different images. SimCLR, MoCo, and DINO belong here.
2. **Masked prediction methods** — learn by masking parts of the input and predicting the masked parts. Masked autoencoders (MAE) and BEiT belong here.

DINO sits in the contrastive family, but with a specific twist: it uses **self-distillation** rather than explicit contrastive negatives. This matters, and the next section explains why.

### The DINO family: teacher-student distillation with momentum

DINO's core idea is deceptively simple. Take an image. Create two augmented views of it — say, a large crop covering 60% of the image and a small crop covering 20%. Feed the large crop to a **teacher** network and the small crop to a **student** network. Train the student to match the teacher's output.

If this were the whole story, the model would collapse: both networks would learn to output the same constant vector for every image, achieving perfect match with zero effort. DINO prevents collapse through three mechanisms:

1. **The teacher is not trained by gradient descent.** Instead, the teacher's weights are an exponential moving average (EMA) of the student's weights. After each training step, the teacher is updated as:

$$\theta_\text{teacher} \leftarrow m \cdot \theta_\text{teacher} + (1 - m) \cdot \theta_\text{student}$$

where $m$ is the momentum coefficient, typically 0.996-0.999. The teacher changes slowly; the student changes quickly. This asymmetry prevents the trivial solution — the student cannot "cheat" by simply becoming the teacher, because the teacher is always a smoothed, slightly stale version of the student.

2. **Centering.** The teacher's output is centered by subtracting a running mean of its recent outputs. This prevents the teacher from collapsing to a single point.

3. **Sharpening.** The teacher's output distribution is sharpened with a low temperature before the student tries to match it, encouraging the teacher to commit to confident, discriminative representations.

The training objective is cross-entropy between the teacher's (sharpened, centered) output distribution and the student's output distribution, summed over all pairs of local-crop (student) and global-crop (teacher) views. In pseudocode, if $p_t$ is the teacher's output and $p_s$ is the student's output, the loss is:

$$\mathcal{L} = -\sum_{\text{views}} p_t \log p_s$$

Why does this work for learning visual representations? The student sees a small crop — maybe just a corner of a painting showing a hand holding a brush. The teacher sees a large crop — maybe half the painting including the figure, the background, and the hand. The student must learn to produce the same representation from partial information that the teacher produces from fuller information. To do this, the student must learn what the small crop *implies* about the rest of the image. Over millions of images, this forces the network to learn hierarchical visual features — edges, textures, object parts, object identities, scene structure — because these are the regularities that make prediction from partial views possible.

**The backbone: Vision Transformers (ViT).** DINO uses a Vision Transformer as its encoder. A ViT works by:

1. Splitting the image into non-overlapping patches (e.g., 14x14 pixels each for a ViT-*/14).
2. Linearly projecting each patch into a vector (the "patch embedding").
3. Adding positional encodings so the model knows where each patch came from.
4. Processing the sequence of patch embeddings through transformer layers (self-attention + feed-forward).
5. Using the output of a special [CLS] token (prepended to the sequence) as the image-level representation.

Model size is denoted by a letter: **S**mall (22M parameters), **B**ase (86M), **L**arge (307M), **H**uge (632M), and **g**iant (1.1B). The program specifies ViT-L/14 as the default, with ViT-H/14 as an ablation. The "/14" means 14x14 pixel patches — a 224x224 image becomes a sequence of 16x16 = 256 patches. Larger patches (e.g., /16) mean fewer tokens and faster processing; smaller patches mean more spatial detail but quadratic cost in the self-attention layers.

### DINOv1 to DINOv2 to DINOv3: what each generation added

**DINOv1 (Caron et al., 2021).** The original paper. Trained on ImageNet (1.2M images, 1000 classes — but the class labels were not used). Demonstrated that self-distillation with ViT produces features that are remarkably good for downstream tasks without any fine-tuning. The famous result: DINO's attention maps spontaneously learn to segment objects, without ever being trained on segmentation. The [CLS] token's attention in the last layer highlights the main object in the scene — an emergent property, not a trained one.

For art, the relevant observation is that DINOv1 features already separate semantic categories (animals from landscapes from portraits) even though the model never saw a label. But it was trained only on ImageNet — photographs of objects — and the features were not strong enough for fine-grained discrimination within domains like art.

**DINOv2 (Oquab et al., 2023).** Three major improvements. First, **curated pretraining data**: a new dataset called LVD-142M (142 million images), automatically curated from web crawls using DINO features themselves to de-duplicate and select informative images. This is a bootstrap: use a DINO model to curate data to train a better DINO model. Second, **scaling to ViT-g** (1.1B parameters). Third, **stabilized training** with improved augmentation, regularization, and distillation protocols. The result was features that matched or exceeded supervised features on a wide range of benchmarks — classification, segmentation, depth estimation, retrieval — without any labels at all.

DINOv2 features are substantially better than DINOv1 for art-related tasks. The larger model and richer data give it better texture discrimination (important for medium classification — oil vs. watercolor vs. fresco), better compositional understanding (important for art where spatial arrangement carries meaning), and better fine-grained object recognition (important for Iconclass subject classification).

**DINOv3 (Caron et al., 2024).** Further scaling, improved training stability, and an expanded model zoo from ViT-S/16 up to ViT-7B/16. Currently represents the state of the art in self-supervised visual features. The improvements from DINOv2 to DINOv3 are less dramatic than from v1 to v2 — they are primarily about scaling and robustness rather than new ideas. But for a system that will be fine-tuned on art, starting from the strongest possible initialization matters, because the quality of low-level and mid-level features transfers directly.

### What DINO features encode at different layers

A key fact for understanding the program: the information encoded by a ViT changes across layers, and which layers matter depends on the task.

**Low layers (1-6 in a 24-layer ViT-L)** encode low-level visual features: edges, textures, color distributions, local spatial patterns. These are largely universal — they look the same whether the model was trained on photographs, satellite images, or paintings. Low-level features transfer essentially for free. The program does not need to train these; DINOv3 has already learned them from 1.7 billion images.

**Mid layers (7-16)** encode mid-level features: object parts, spatial layout, texture at a larger scale, material properties. This is where art starts to diverge from natural images. The spatial composition of a painting — the placement of figures, the balance of masses, the leading lines — is encoded here. Medium effects (the difference between an oil painting's impasto and a watercolor's washes) are encoded here. These features partially transfer from DINOv3's natural-image training, but the specifics of art-making (brush traces, compositional conventions, gilding, patina) are underrepresented in the pretraining data. Fine-tuning updates these layers.

**High layers (17-24)** encode semantic features: object identity, scene type, conceptual categories. In DINOv3, these layers cluster natural images by semantic category — all dogs together, all kitchens together, all beaches together. For art, these layers should cluster by art-historical category, Iconclass subject, and stylistic tradition. But DINOv3 has never seen an Iconclass label or an art-historical category. Its high-level features will separate "painting of a person" from "painting of a landscape," but they will not separate "Mannerist portrait" from "Neoclassical portrait" without further training.

The program's strategy follows from this layer-by-layer analysis: **keep the low layers frozen or lightly fine-tuned, and concentrate training budget on mid and high layers where art-specific signal lives.** This is the standard transfer-learning playbook, and it works because low-level visual statistics (edges, textures, spatial frequencies) are shared across domains. A neuroscientist might recognize this as analogous to the stability of V1 receptive fields across tasks, with plasticity concentrated in higher visual areas.

### Fine-tune vs. from-scratch: the design decision

The design conversation (Phase 3 of the design record) considered two paths:

**From scratch.** Train a DINO model entirely on art data (~20M images). The argument: art has different generative processes than natural images. Brush traces, period-specific palettes, compositional conventions, medium effects — these create statistical regularities that DINOv3, trained on web photographs, may not have learned to represent well. A model trained from scratch on art would learn axes that natively separate Caravaggio from Vermeer, rather than treating both as "indoor scene with figure, low key." The data is sufficient — 20M is between DINOv1's 1.2M and DINOv2's 142M, well within the scaling regime where self-supervised learning works.

**Fine-tune from DINOv3.** Start from DINOv3-L (or H), which has already paid the computational cost of learning universal visual features, and update the weights on art data with the multi-modal contrastive objectives.

**The decision reversed during the design conversation.** Initially the from-scratch path seemed appealing because of the domain mismatch. But the reversal happened when the user moved from "we want a vision model for art" to "we want a multi-modal model that fuses vision with structured entities." Once the primary novelty is in the cross-modal alignment — teaching the image tower to be compatible with entity embeddings and biography text — the from-scratch argument weakens. Here is why:

1. **Low-level features transfer.** Edges are edges. Texture is texture. These do not need to be relearned for art. Training from scratch wastes 2-4 weeks of 8xA100 compute relearning features that DINOv3 already has.

2. **The art-specific signal lives where fine-tuning operates.** Mid- and high-level representations — the features that should distinguish Caravaggio from Vermeer — are exactly what fine-tuning updates. You do not need a from-scratch model to learn art-specific high-level features; you need to update high-level features in a pretrained model.

3. **The multi-modal objectives will reshape the feature space anyway.** When the image tower is trained jointly with entity and text towers, the contrastive loss will pull image representations toward their corresponding entity representations. This is a strong training signal that will reshape mid- and high-level features. The starting point (DINOv3 vs. random initialization) determines how fast this convergence happens and how much low-level quality is preserved during training, not whether art-specific features emerge.

4. **Compute is better spent on the novel parts.** Joint multi-modal training on five signal types is where the research risk lives. Spending compute budget on re-deriving universal visual features is not the highest-value use.

**What would change this decision?** If DINOv3 features proved to be fundamentally wrong for art — if the pretrained feature space actively conflated art-historically important distinctions in ways that fine-tuning could not overcome — then from-scratch training would be necessary. The program parks this as a separate research question to be revisited based on Phase 1 ablation results: compare DINOv3 fine-tuned features against randomly initialized features on the cold-start recall metric. If the fine-tuned model's cold-start recall plateaus well below the from-scratch model, the decision reverses.

### Augmentation: why it matters more than you think

Data augmentation is the process of transforming training images to create artificial diversity — random crops, color jitter, blur, horizontal flips. In self-supervised learning, augmentation is not a nice-to-have; it is constitutive of the training objective. DINO's loss function requires two different views of the same image. The augmentation pipeline defines what counts as "the same image seen differently" versus "a different image." Everything the model learns to be invariant to is determined by the augmentations.

Standard DINO augmentations include:

- **Random resized crop** — crop a random rectangle from the image and resize it to the standard input size. Creates scale invariance.
- **Horizontal flip** — flip left-to-right with 50% probability. Creates mirror invariance.
- **Color jitter** — randomly perturb brightness, contrast, saturation, and hue. Creates color invariance.
- **Gaussian blur** — apply random blur. Creates sharpness invariance.
- **Solarization** — invert pixels above a threshold. A more aggressive color perturbation.

For natural images, these are all defensible. A photograph of a dog should produce the same representation whether the dog is slightly more blue, slightly blurred, or seen from a slightly different crop. The model should learn "this is a dog" regardless of these nuisance variations.

**For art, some of these augmentations are wrong.** This is one of the more subtle design issues in the program, and the design conversation flagged it explicitly.

**Hue jitter destroys palette signal.** Matisse's palette is not a nuisance variable — it is a primary stylistic signature. Aggressive hue jitter (the standard DINO setting rotates hue by up to 20 degrees) tells the model that a Matisse with green shifted toward blue is "the same" as the original. But palette is exactly what should distinguish a Matisse from a Derain. For art, hue jitter must be reduced or eliminated, or applied only within a narrow range that preserves the warm/cool character of the palette.

**Heavy cropping loses composition.** In natural images, a crop of 20% of a photograph usually still contains recognizable objects. In art, a 20% crop of a Piet Mondrian might contain a single colored rectangle — losing all compositional structure. More subtly, many paintings encode meaning at the edges: the relationship between figure and frame, the way a composition breathes at the margins, the vignetting of a Baroque painting. Aggressive cropping with small minimum crop ratios will destroy this signal. The augmentation suite for art should use larger minimum crop sizes than the DINO default (typically 8% of image area for the smallest crops).

**Horizontal flip is wrong for some art.** Left-right composition carries meaning in art with written language (calligraphy, illuminated manuscripts, Japanese prints), in religious iconography (Christ's right hand is always his right), and in compositions with directional narratives (history paintings, comic-like sequences). The safe default is to disable horizontal flips, or to enable them only for categories where mirror invariance is defensible (abstract art, landscapes without text).

**Gaussian blur is mostly fine.** Sharpness differences between a photograph of a painting and the painting itself are nuisance variation that the model should ignore. Blur augmentation helps with this. An exception: for very fine-grained medium classification (distinguishing a stipple engraving from a mezzotint), blur augmentation would destroy the discriminative signal. But this is a niche concern.

The lesson: **augmentation is not a hyperparameter to be tuned; it is a statement about what the model should ignore.** Getting augmentation wrong for art means training the model to be invariant to features that carry signal. The program file does not specify the augmentation suite in detail — it flags this as requiring domain-specific work — but the study guide should make clear that this is a first-order design decision, not a detail to be left to default settings.

### What this chapter means for the program

The choice to start from DINOv3 and fine-tune is a bet on three propositions: (1) low-level visual features are universal enough to transfer from natural images to art; (2) art-specific signal concentrates in mid-to-high layers that fine-tuning can reach; (3) the multi-modal contrastive objectives will reshape the representation more than the starting initialization will constrain it. All three propositions are defensible but empirical. Phase 1 ablations should test them.

The augmentation question is a design risk that deserves explicit Phase 1 attention. A good augmentation suite for art does not exist off the shelf. The program should budget engineer time for designing, testing, and ablating art-specific augmentations alongside the main training runs. If augmentation is treated as a default setting, the model will learn to discard exactly the signals that make art recommendation interesting.

The next chapter covers how the embedding space learns its geometry — the contrastive training objective that teaches the model what "similar" means.
