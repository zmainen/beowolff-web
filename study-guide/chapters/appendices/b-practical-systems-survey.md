# Value Attribution and Revenue-Sharing Systems in Practice

*A survey of how platforms, marketplaces, and creative systems actually track and distribute value among contributors.*

*Compiled April 2026 for the Rasa art recommendation program.*

---

## Why This Survey Exists

Rasa faces a value attribution problem with five distinct layers: property rights over original artworks, corpus-level contribution to recommendation models, recommendation-referral tracking, infrastructure contribution measurement, and (eventually) revenue-sharing for co-created works. A design conversation concluded that a unified Shapley framework is overengineered for this — each layer should use the simplest mechanism that captures causal contribution at its scale. This survey examines what real systems actually do, not what they claim to do, so we can learn from their engineering tradeoffs and institutional failures.

---

## 1. Music Streaming Royalties: The Pro-Rata Default and Its Discontents

Music streaming is the closest analogue to an art recommendation platform: many creators contribute to a pool, consumption is tracked at fine granularity, and the platform must decide how to divide a fixed revenue pot among claimants.

### The Pro-Rata Model

Spotify, Apple Music, and most streaming services use a **pro-rata** (or "streamshare") model. Spotify allocates roughly 70% of its total revenue to rights holders — 58.5% to sound recording owners and 12% to musical composition owners. The platform tallies total streams in a given month, determines each rights holder's proportional share of those streams, and distributes accordingly. There is no fixed per-stream rate; payments vary by country, subscription type, and total streams that month. Historical per-stream values have ranged from $0.003 to $0.008, though Spotify emphasizes this is an emergent outcome, not a set price.

The pro-rata model has a structural problem: it overweights heavy listeners. A subscriber who plays background music 18 hours a day generates the same $10.99/month in revenue as someone who plays 30 minutes, but the heavy listener's thousands of streams dilute the per-stream payout for every artist. This creates a transfer from casual-listener revenue to the artists that heavy listeners favor — overwhelmingly, major-label pop and ambient playlists. Starting in late 2023, Spotify imposed a minimum threshold of 1,000 streams within 12 months plus a minimum number of unique listeners, explicitly to filter out streaming fraud and micro-catalog noise, redirecting those fractions back toward established artists.

### User-Centric Payment

The alternative — **user-centric payment** — allocates each subscriber's fee only to the artists they personally streamed. If you listen to three artists all month, your $10.99 is split three ways (after the platform cut). Deezer adopted a modified version in 2023, weighting streams from "professional" artists (those exceeding 1,000 monthly listens) at double the rate of non-professional streams. Tidal has promoted artist-friendly royalties but the specifics remain vague; one artist reported per-track royalties three times Spotify's rate in 2015, though Tidal cautioned this might not be sustainable.

User-centric payment is more intuitive and arguably fairer, but it creates its own complexity: the marginal stream of a casual listener is worth much more than that of a power listener, so artists with devoted but moderate-volume fanbases gain at the expense of those who thrive on passive listening. The reason most platforms haven't switched: it doesn't materially change outcomes for the head of the distribution, and the tail-end artists it helps don't have the bargaining power to force the change.

### The Mechanical/Performance Rights Stack

Behind the streaming platforms sits a layered institutional apparatus. In the US, SoundExchange collects digital performance royalties for sound recordings from non-interactive services (satellite radio, internet radio), distributing over $1 billion annually split between featured artists, copyright owners (labels), and non-featured artists (session musicians, through SAG-AFTRA and AFM funds). Rates are set through Copyright Royalty Board proceedings — webcasting pays roughly $0.0018-$0.0023 per non-subscription/subscription performance. ASCAP and BMI collect performance royalties for musical compositions, with ASCAP collecting approximately $1.84 billion in 2024 and distributing $1.7 billion. Their distribution formulas are famously opaque — ASCAP has been criticized for refusing to release the weighting formulas that determine how much a composition earns per television or radio use.

The Mechanical Licensing Collective, established by the Music Modernization Act (2018), issues blanket mechanical licenses to streaming services, maintaining a public database for matching compositions to recordings. By October 2022, the MLC had paid nearly $700 million in royalties. It partnered with five companies (Blokur, Jaxsta, Pex, Salt, SX Works) to improve data matching, because the persistent operational problem is identification: connecting a stream event to the right composition, then to the right publisher and songwriter.

**Rasa lesson:** The music industry demonstrates that the hard problem is not computing shares — it's identification and matching. Knowing who contributed what to which outcome is the bottleneck, not the division algorithm. Invest in clean metadata and contribution graphs before worrying about the allocation formula.

---

## 2. Advertising Attribution: From Rules to Incrementality

Digital advertising has spent two decades refining attribution, and the trajectory is instructive: they started with simple rules, tried to compute precise multi-touch models, and are now largely retreating to causal experiments.

### Rule-Based Attribution Models

The basic models assign credit for a conversion (purchase, signup) to one or more advertising touchpoints:

- **Last-click:** 100% credit to the final ad clicked before conversion. Simple, understandable, and systematically wrong — it credits the last touch while ignoring everything that built awareness.
- **First-click:** 100% credit to the first ad interaction. Equally wrong in the opposite direction.
- **Linear:** Equal credit to every touchpoint in the path. Democratically wrong — a display ad impression three weeks before purchase doesn't contribute equally to a search ad clicked minutes before.
- **Time-decay:** More credit to touchpoints closer to conversion. A reasonable heuristic but still arbitrary in its decay curve.
- **Position-based (U-shaped):** 40% to first touch, 40% to last touch, 20% distributed among middle touches. A structured compromise with no empirical basis.

All of these are making up a causal story from observational data, and research consistently shows they diverge from experimental measurements of actual impact.

### Data-Driven Attribution

Google's data-driven attribution (the default in Google Ads since 2021) uses machine learning to analyze both converting and non-converting user paths. It employs a counterfactual approach: comparing users exposed to a specific ad against similar users in a control group to isolate the ad's actual contribution. Credit is assigned based on how adding each ad interaction to the path changes the estimated conversion probability. The system considers temporal proximity, ad format, device type, and query signals, and may reassign credit for up to 7 days post-conversion as more data becomes available.

This is substantively better than rule-based models, but it remains observational — it estimates what would have happened without each touchpoint, rather than measuring it directly.

### Incrementality Testing

The advertising industry's version of A/B testing for causal contribution is **incrementality testing**. The core idea: run a randomized experiment where a treatment group sees your ad and a control group sees a "ghost ad" (a public service announcement or blank) in the same slot. The difference in conversion rates between groups is the ad's true incremental impact — the purchases it actually caused, beyond what would have happened anyway.

Incrementality testing is the gold standard because it establishes causality, but it's expensive: you're deliberately not showing your ad to a fraction of your audience. Most companies run incrementality tests periodically to calibrate their data-driven models, not as a continuous measurement system.

**Rasa lesson:** The advertising industry's trajectory — from simple heuristics through algorithmic attribution to causal experiments — maps directly onto our recommendation attribution layer. Start with last-touch referral tracking (which recommendation led to purchase), but plan for periodic A/B testing to measure whether the recommendation system is actually driving sales or just correlating with them. The data-driven attribution models are tempting but misleading without ground truth from experiments.

---

## 3. Affiliate and Referral Systems: The Cookie Window

Affiliate marketing is the simplest form of value attribution: someone links to a product, you track the click, and if a purchase follows within a defined window, the referrer gets a commission.

### Amazon Associates

Amazon Associates, the largest affiliate program, pays fixed commission rates by product category: 10% for luxury beauty, 5% for digital music and handmade items, 4.5% for physical books and kitchen products, 4% for most electronics, 2% for televisions and digital video games, and 1% for groceries. Some categories pay 0% (gift cards, wireless plans, alcohol). Flat-fee "bounties" supplement percentage commissions: $3 for Prime signups, $25 for Audible annual membership signups, $15 for Amazon Business account signups, $5 for a user's first Prime Video stream.

The attribution mechanism is a **24-hour cookie window**: if a user clicks your affiliate link and purchases within 24 hours, you earn the commission. If they add an item to their cart within that window but purchase within 89 days, you still earn. This is architecturally simple — the cookie encodes the referrer ID, and Amazon's purchase system checks for it at checkout.

### What Works and What Doesn't

Affiliate tracking works well for single-step attribution: one referrer, one purchase. It breaks down when multiple referrers are involved (the user sees a blog review, then clicks a YouTube link, then searches directly — who gets credit?). Amazon's answer is last-click within the cookie window, which means the last affiliate link clicked before purchase gets everything. This is unfair to upstream influence but operationally clean.

The deeper problem is fraud: click injection, cookie stuffing, and manufactured referrals. Amazon's response has been aggressive program enforcement and rate cuts (commission rates have declined substantially since the program's early days), treating the affiliate channel as a cost center to be minimized rather than a partnership to be optimized.

**Rasa lesson:** For recommendation-to-purchase tracking, the affiliate model provides the right template: tag each recommendation interaction with a session/referral ID, apply a reasonable attribution window (probably longer than 24 hours for art purchases, which involve consideration), and credit the last recommendation touchpoint. This is layer 3 (recommendation referral tracking), and it should be simple — the complexity should live in layers 2 and 4.

---

## 4. Open Source Contribution Valuation: The Unsolved Problem

Several systems have attempted to measure and compensate open source contributions, with varying degrees of success.

### Gitcoin and Quadratic Funding

Gitcoin Grants has distributed over $60 million to more than 3,000 open source projects using **quadratic funding** — a mechanism where matching funds are allocated based on the breadth of community support rather than raw donation size. The square root of each contribution is summed, then squared to determine matching. This means a project with 100 donors giving $1 each receives far more matching than a project with one donor giving $100, because the mechanism rewards demonstrated community value over individual patron depth.

Quadratic funding doesn't measure contribution value computationally — it's a revealed-preference mechanism. The community votes with money, and the matching formula amplifies broad consensus. This works for deciding which projects to fund, but it says nothing about how to distribute rewards within a project (who wrote the critical code vs. who updated the README).

### tea.xyz

tea.xyz attempts the harder problem: measuring individual contribution value within the open source ecosystem. Their **teaRank** algorithm assigns dynamic scores based on a project's "orientation within and utilization by the OSS ecosystem over time" — essentially, PageRank applied to the dependency graph. Projects that are widely depended upon by other widely-used projects score higher. Contributors earn rewards proportional to their project's teaRank through a token-based staking system (TEA token), where community members can also stake tokens to projects they value.

The system is plausible for measuring ecosystem-level importance (npm packages that half the internet depends on deserve more than rarely-used utilities), but it doesn't solve within-project attribution and is vulnerable to dependency manipulation — creating artificial dependency chains to inflate rank.

### SourceCred (Defunct)

SourceCred used a PageRank-like algorithm applied to contribution graphs (issues, pull requests, reviews, discussions) to assign "cred" scores to contributors. It was adopted by several DAOs and crypto projects but failed to achieve mainstream traction and has largely gone dormant. The core problem: any system that tries to reduce diverse contribution types (code, reviews, community management, documentation) to a single scalar is making implicit value judgments that will always feel arbitrary to someone.

**Rasa lesson:** For corpus-level contribution measurement (layer 2 — which training data shaped the model), the dependency-graph approach from tea.xyz is more relevant than the social-contribution approach from SourceCred. In Rasa's case, "dependency" means: which training examples most influenced the recommendation model's neighborhood structure? This is measurable through leave-one-out or small-coalition Shapley on the training set, and it's a narrower, more tractable problem than general "contribution value."

---

## 5. Stock Photography and Content Matching: The Getty/Shutterstock Model

Stock photography platforms are perhaps the closest existing analogue to an art marketplace with algorithmic curation.

### How Contributors Are Paid

Getty Images pays contributors between 15% and 45% per license, depending on exclusivity. Exclusive contributors get higher rates and priority placement; non-exclusive contributors accept lower rates in exchange for the ability to sell the same images elsewhere. License prices range from under $1 to several hundred dollars per image, depending on usage rights and resolution.

Shutterstock moved in 2020 from a flat per-download rate to a percentage-based model, cutting minimum payments from $0.25 to $0.10 per download (15% of revenue) at the entry level, with higher rates for higher-volume contributors. By 2015, Shutterstock had paid approximately $250 million total to contributors, with roughly $83 million paid in 2014 alone across about 80,000 contributors — averaging roughly $1,000/year per contributor, though the distribution is extremely skewed.

### YouTube Content ID: Automated Attribution at Scale

YouTube's Content ID is the most sophisticated automated content-matching system in operation. It maintains a database of digital fingerprints for copyrighted audio and video, against which every upload is automatically checked. When a match is found, the copyright holder chooses to block the video, track it, or monetize it (placing ads with revenue going to the rights holder). By 2016, the system had paid approximately $2 billion to copyright holders; by 2021 it was processing nearly 1.5 billion Content ID claims annually. Google has invested over $100 million in the system's development and claims it detects over 98% of known copyright infringement.

The dispute process is asymmetric: users can dispute claims, but the copyright claimant makes the final decision unless the user takes legal action. Since 2016, disputed videos continue earning during the dispute period, with revenue held in escrow and released to the winner.

**Rasa lesson:** Content ID demonstrates that automated matching can work at enormous scale, but only with massive investment and a willingness to accept false positives (music industry groups dispute Google's 98% effectiveness claim). For Rasa, the analogous system is provenance tracking for artworks — ensuring that when art sells, the attribution chain to the original creator is unambiguous. This is layer 1 (property rights/contract) and should be the simplest, most reliable layer in the system.

---

## 6. Data Marketplaces: The Promise vs. Reality Gap

Data marketplaces have attempted to create transparent attribution and compensation for shared data, with limited success.

### Snowflake Marketplace and AWS Data Exchange

Snowflake Marketplace allows organizations to share and access "live, query-ready datasets." The platform handles access control and billing, but attribution is coarse: a data provider lists a product at a price, subscribers pay for access, and Snowflake takes a cut. There is no downstream usage tracking — once data is accessed, the provider has no visibility into how it's used or what value it generates. AWS Data Exchange operates similarly: subscription-based pricing for curated datasets, with no mechanism for tracking the downstream value created by the data.

### Ocean Protocol

Ocean Protocol attempted something more ambitious: a decentralized data marketplace with "compute-to-data" — the idea that you can run computations against someone's data without the data leaving the provider's infrastructure, enabling attribution at the query level. The OCEAN token was meant to create a market for data services. In practice, adoption has been minimal. The protocol is used primarily within the crypto-native community for prediction market data ("Predictoor" and "Data Farming" programs), not for the enterprise data marketplaces originally envisioned. The gap between the whitepaper vision (transparent, fair data markets) and the operational reality (niche DeFi tooling) is instructive.

**Rasa lesson:** Data marketplaces fail when they try to track downstream value at fine granularity. The operational overhead exceeds the value of precise attribution. Snowflake's approach — coarse pricing, no downstream tracking — works because it's simple. For Rasa's training data attribution (layer 2), this means: measure contribution at training time (what did each data source contribute to model quality?), not at inference time (which training example influenced this specific recommendation?). The former is tractable; the latter is computationally intractable and operationally unnecessary.

---

## 7. AI-Generated Content Attribution: The Legal and Technical Frontier

The AI-generated content space is where attribution is most contested and least resolved, making it the most instructive for Rasa's eventual generative features.

### The Current State: No Attribution

None of the major generative AI systems — Midjourney, Stable Diffusion (Stability AI), or DALL-E (OpenAI) — provide attribution to the creators whose work was used in training. Stable Diffusion was trained on LAION-5B, a dataset derived from Common Crawl web scraping. Midjourney's training data sources are less documented but are known to include scraped web images without artist consent.

The legal consequences are mounting. In 2023, artists Sarah Andersen, Kelly McKernan, and Karla Ortiz sued Stability AI, Midjourney, and DeviantArt, alleging infringement of millions of artists' rights. A subsequent lawsuit in November 2023 represented over 4,700 artists. By 2025, the stakes escalated: Universal Pictures and Disney sued Midjourney, calling it "a bottomless pit of plagiarism," and Warner Bros. Discovery followed. Getty Images sued Stability AI, though Getty largely lost the UK case in late 2025.

### Consent-Based Approaches

**Fairly Trained** offers certification for AI companies that obtain proper licenses before training on copyrighted work. This is a market signal rather than a technical mechanism — it certifies behavior, not attribution. The certification criteria is binary: the company either obtained consent for all training data or it didn't.

**Spawning.ai** built "Have I Been Trained," a tool that lets artists check whether their work appears in training datasets, and "Kudurru," a tool for detecting and blocking AI scrapers. Their approach is opt-out/consent infrastructure rather than compensation infrastructure. (Their site was under maintenance at time of writing, which may itself be a data point about the viability of these tools.)

### Adobe: The Compensation Experiment

Adobe is the notable outlier. Firefly was trained primarily on Adobe Stock images, Creative Commons content, and public domain material — sources Adobe could plausibly claim licensing rights over. (This claim was undermined when it was revealed that Firefly was also trained on images from Midjourney and competitors.) Adobe has announced a bonus compensation program for Adobe Stock contributors whose images were used in Firefly training, though specific dollar amounts and mechanisms have not been fully disclosed. This is the only major AI company that has attempted any form of training-data compensation, however imperfect.

### Content Credentials (C2PA)

The Content Authenticity Initiative (CAI), founded by Adobe, the New York Times, and Twitter in 2019, developed the C2PA standard for embedding provenance metadata in digital content. Content Credentials encode the publisher, device, location, time, and editing history of a file, secured with cryptographic signatures. The standard has over 6,000 member organizations. However, C2PA tracks provenance (who created/edited this file), not contribution (who deserves credit for the training data that made generation possible). It also faces adoption problems — critics note that metadata can be stripped, signers don't necessarily verify accuracy, and as of 2025 real-world usage remains minimal.

**Rasa lesson:** The AI attribution landscape demonstrates two things. First, the absence of attribution is not a stable equilibrium — the lawsuits and regulatory pressure will force some resolution. Second, the consent-based approach (Fairly Trained, Spawning) is architecturally simpler and legally cleaner than the computation-based approach (trying to compute each training example's contribution to each output). Rasa should build consent and licensing into the platform from the start, using the recommendation model's contribution analysis (layer 2) as a supplementary signal for fair compensation rather than the primary legal basis for it.

---

## 8. The General Pattern: What Works and What Doesn't

Across all these systems, several patterns emerge.

### Successful attribution systems simplify aggressively

Every system that actually works in production has made peace with approximation. Spotify doesn't compute each stream's true value — it divides a pool proportionally. Amazon Associates uses last-click with a 24-hour window. Content ID uses fingerprint matching with binary outcomes (match or no match). The systems that tried for precise multi-contributor attribution (SourceCred's contribution scoring, Ocean Protocol's compute-to-data) have either stalled or pivoted.

The lesson is not that precision doesn't matter — it's that the precision-complexity tradeoff has a steep curve, and most of the value comes from the first 20% of attribution accuracy. Getting the right person paid the right order of magnitude matters much more than computing the fifth decimal place.

### Identification is harder than allocation

The music industry's biggest problem isn't the royalty formula — it's matching streams to compositions to publishers to songwriters. The MLC spent $62 million in startup and first-year costs, largely on data infrastructure for matching. Content ID required $100+ million in fingerprinting technology. In every domain, knowing who contributed what is the engineering bottleneck, not computing how much they should be paid.

### Different layers need different mechanisms

No successful system uses a single attribution mechanism for all contribution types. Music royalties separate mechanical from performance from synchronization rights, each with different collection societies and rates. Advertising separates impression attribution (data-driven models) from causal measurement (incrementality tests). Stock photography separates the license transaction (simple percentage) from the discovery mechanism (algorithmic ranking with no explicit revenue sharing).

This validates Rasa's layered approach: property rights for direct sales, corpus contribution analysis for training data, referral tracking for recommendations, A/B testing for infrastructure, and separate terms for generative/co-created works. Each layer uses the mechanism appropriate to its causal structure.

### Disputes are resolved institutionally, not computationally

Content ID doesn't compute the "right" answer — it flags matches and lets the parties negotiate (with an asymmetric power structure favoring claimants). ASCAP doesn't publish its weighting formulas. Spotify's minimum-stream threshold was a policy decision, not an algorithmic one. When attribution is genuinely ambiguous, every successful system falls back to institutional authority — a governing body, a contract term, a dispute resolution process — rather than trying to compute the answer more precisely.

### The consent layer is foundational

Adobe's approach (train on licensed data, compensate contributors) is legally defensible in a way that Stability AI's approach (train on everything, litigate later) is not. For Rasa, this means the first layer — clear property rights and artist consent — is not just legally necessary but architecturally foundational. Everything else (recommendation attribution, corpus contribution, infrastructure measurement) sits on top of a clear consent and licensing agreement with every contributing artist.

---

## Summary Table: Mechanisms by Layer

| Rasa Layer | Analogous Systems | Mechanism | Precision | Complexity |
|---|---|---|---|---|
| 1. Property rights | Stock photography licensing, music mechanical rights | Contract + per-transaction royalty | Exact | Low |
| 2. Corpus contribution | tea.xyz dependency graphs, (no good analogue) | Leave-one-out / small-coalition Shapley at training time | Approximate | Medium |
| 3. Recommendation referral | Amazon Associates, affiliate tracking | Session-tagged last-touch with attribution window | Coarse | Low |
| 4. Infrastructure/algorithm | Ad incrementality testing | Periodic A/B tests measuring lift | Statistical | Medium |
| 5. Generative co-creation | Adobe Firefly contributor compensation, music sampling credits | Negotiated revenue share + consent | Contractual | High |

The overarching principle: use the cheapest mechanism that captures the causal structure of each layer. Don't compute what you can contract. Don't model what you can measure. And don't measure what doesn't change the payment by more than the measurement costs.
