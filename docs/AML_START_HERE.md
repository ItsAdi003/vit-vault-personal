# AML: start here (40-minute read, assumes you know nothing)

**How to use this page.** Read top to bottom in order. Part 1 tells you the truth about who built what, so you never say anything you cannot defend. Parts 2-3 teach the ideas from zero. Part 4 is the talk, word for word. Part 5 is the questions. Part 6 is the clock.

---

## Part 1. The project, and who built what (read this first, 6 minutes)

### The project in plain English

**VIT Media Vault** is a photo website for college events. Students upload photos into folders. The website then, automatically:

1. **Tags each photo** with words like `stage`, `crowd`, `coding`.
2. **Rejects duplicates** (the same photo uploaded twice).
3. **Guesses the kind of event**: INSTITUTIONAL (seminar, graduation) or COMMITTEE (hackathon, fest).
4. **Lets you search** by describing a photo in words.

For Advanced Machine Learning, **the part that matters is step 3** (a probability model that decides the event type) **plus a clustering model that groups photos by theme without any labels**.

### Where the "AI" is, and where it is not

| Part of the project | Is it AI / machine learning? | Who made it |
|---|---|---|
| Website pages, login, folders, upload, storage | **No.** Ordinary software engineering | Original project (Raunak and OM, the earlier team) |
| **CLIP**: looks at a photo and produces tags and a list of 512 numbers | **Yes, deep learning**, but a *ready-made* model from OpenAI. Nobody here trained it | Used by the original project |
| Duplicate detection (perceptual hash) | **No.** A classical image-fingerprint algorithm, not learned | Original project |
| **Naive Bayes** with a hand-written probability table (used by the live app) | **Yes**, a probabilistic model, but its numbers are *typed in by hand*, not learned | Original project (file `services/classifier.py`) |
| **Naive Bayes that learns its numbers from data** | **Yes, machine learning** | Added in this repository with an AI assistant |
| **EM clustering (Gaussian mixture) + PCA** | **Yes, machine learning** | Added in this repository with an AI assistant |
| Search that mixes keywords (BM25) with meaning (CLIP) | BM25 is classical; CLIP part is deep learning | Added in this repository with an AI assistant |
| **AI Lab page** (the live demo screen) | The *page* is not AI; it shows the models above | Added in this repository with an AI assistant |
| Tests, experiments, notes, bug fixes | Not AI; they check the AI parts | Added in this repository with an AI assistant |

You can verify the "original project" rows yourself: `git log -- ai-service/services/classifier.py` shows commits by **Raunak**; `GalleryPage.jsx` and `LoginPage.jsx` show **Raunak, RAUNAK2005 and OM**. One commit in this repository (`38acfd0`, the port/CORS fix) carries an AI-assistant co-author line. Everything else added this semester is uncommitted until you commit it.

### What you can truthfully say you did

* Took over an existing project, got it running on your machine, and diagnosed real problems (the stopped database, ports blocking uploads).
* Chose the AML direction and had the probabilistic parts built, tested and measured.
* Ran the tests and experiments and can show the results live.
* **You are being judged on whether you understand it.** Parts 2-5 are written so that you will.

### The honest answer to "Who wrote this?"

Say what is true. A safe wording:

> "The base application, with login, gallery and the first Naive Bayes classifier, comes from an earlier version of this project. For AML we extended it: a Naive Bayes that learns its probabilities from data, an EM clustering model, tests and experiments. I built those extensions with an AI coding assistant, and I studied them so I can explain every step. Ask me about any part."

Do not claim you wrote code you did not write, and do not hide the use of an AI assistant: the commit history shows it, and a mentor who finds out afterwards is far more damaging than one who hears it from you. **Check your college's rule on AI assistance and follow it.**

---

## Part 2. The ideas, from zero (20 minutes)

Read these in order; each one uses the previous.

### 2.1 What a "tag" is here

CLIP (the ready-made model) looks at a photo and scores it against 31 words (`sports, stage, coding, hackathon, seminar, ...`). The five best scoring words above 5 % become the photo's **tags**. Example: a photo of a person at a podium might get `[portrait, faculty, presentation]`.

CLIP also gives each photo an **embedding**: a list of 512 numbers that describes the photo. Similar photos have similar lists. You will need this for clustering.

### 2.2 Probability and Bayes' theorem

`P(A | B)` means "the probability of A **given that** B is true".

Bayes' theorem turns a question you *can't* answer into one you can:

```
P(category | tags)  =  P(tags | category) × P(category)  /  P(tags)
   posterior              likelihood            prior         evidence
```

* **Prior** `P(category)`: how likely each event type is *before* looking at the photo. We use 50 % / 50 %.
* **Likelihood** `P(tag | category)`: "at a COMMITTEE event, how often does the tag `coding` appear?"
* **Posterior** `P(category | tags)`: the answer we want, after seeing the tags.
* The **evidence** `P(tags)` is the same for both categories, so we can ignore it and just compare the top parts.

### 2.3 "Naive" Bayes

Multiplying likelihoods for several tags is only legal if the tags are **independent given the category**. That is the "naive" assumption:

```
P(category | t1, t2, t3)  ∝  P(category) × P(t1|category) × P(t2|category) × P(t3|category)
```

It is not perfectly true (tags like `stage` and `auditorium` go together), but it often still classifies well. Its weak point: the percentages it prints are over-confident.

The picture a mentor expects (a **Bayesian network**, the "graphical model" in AML):

```
          Category
         /   |   \
      tag1  tag2  tag3        arrows from the cause to the evidence
```

### 2.4 One full worked example (do this on paper before the viva)

Tags: **portrait, faculty, presentation**. The table in `classifier.py` says how likely each tag is for each category, and each value is smoothed with `(p + 0.1) / 1.2`:

| Tag | INSTITUTIONAL | COMMITTEE |
|---|---:|---:|
| portrait | 0.30 → 0.333 | 0.40 → 0.417 |
| faculty | 0.65 → 0.625 | 0.15 → 0.208 |
| presentation | 0.70 → 0.667 | 0.30 → 0.333 |
| **multiply** (prior 0.5 cancels) | **0.1389** | **0.0289** |

```
P(INSTITUTIONAL | tags) = 0.1389 / (0.1389 + 0.0289) = 0.8276   → "82.8 %"
```

Second example: tags **hackathon, coding** → COMMITTEE 0.9387 (93.9 %).

### 2.5 Why logarithms

Multiplying many numbers below 1 gives tiny values the computer rounds to 0. So the code adds **logarithms** instead of multiplying. The winner is unchanged. You only need to say: *"We work in log-space to avoid numerical underflow."*

### 2.6 Smoothing

If a tag was never seen with a category, its probability would be 0 and wipe out the whole product. **Smoothing** adds a small pseudo-count so nothing is ever exactly zero.

### 2.7 Hand-set versus learned probabilities

* The live app's table is **written by hand** from common sense ("graduation is more likely at an institutional event").
* **Learning it**: collect photos whose event type is known, then count. If 40 of 100 COMMITTEE photos have `coding`, then `P(coding | COMMITTEE) = 0.4`. Counting like this is **maximum likelihood estimation (MLE)**. With smoothing: `(count + α) / (n + 2α)`.
* That learning code exists and is tested (it agrees with the scikit-learn library to 9 decimal places). It has **not** been trained on real gallery photos yet, because that needs the database.

### 2.8 How to tell whether a classifier is good

* **Accuracy**: share of correct predictions. Misleading if one class is much bigger.
* **Confusion matrix**: a small table of what was predicted versus what was true.
* **Precision / recall / F1**: how many of the flagged items are right, and how many of the true ones were found.
* **Cross-validation**: split the data into 5 parts, train on 4, test on 1, repeat; the average is more honest than one split.

### 2.9 Clustering without labels, and EM

Imagine the 512-number embeddings as points. Photos of similar things form clouds. A **Gaussian mixture model** says: "the points come from several bell-shaped clouds; each point belongs to one cloud, but we don't know which."

**EM (Expectation-Maximisation)** finds the clouds by repeating two steps:

1. **E-step**: for every point, work out how likely it belongs to each cloud (soft percentages).
2. **M-step**: move each cloud's centre and width to fit the points that are mostly its own.

Repeat until nothing changes. A key fact: **the fit score (log-likelihood) can only go up or stay the same each round**. We verified that in a test.

### 2.10 PCA and BIC

* **PCA** squashes 512 numbers per photo to 10 while keeping as much of the variation as possible. Needed because fitting clouds in 512 dimensions with few photos is unreliable.
* **BIC** scores a model by *how well it fits* minus *a penalty for complexity*. Lower is better. We use it to pick the number of clouds.

---

## Part 3. What happens to one photo (5 minutes)

```
Photo uploaded
  → CLIP (ready-made)  → tags [portrait, faculty, presentation]  + 512-number embedding
  → Naive Bayes        → P(INSTITUTIONAL)=0.83, P(COMMITTEE)=0.17  → "INSTITUTIONAL"
  → perceptual hash    → fingerprint; if almost equal to an old one → rejected as duplicate
  → saved in the database with tags, category, embedding

Many photos' embeddings
  → PCA (512 → 10 numbers) → EM Gaussian mixture → groups of similar photos, number of groups chosen by BIC
```

Your demo screen (**AI Lab**, `http://localhost:5173/lab`) shows exactly this, live, and **does not need the database**.

---

## Part 4. The 7-minute talk, word for word

**[0:00 · show the AI Lab page]**
"Good morning. Our project is VIT Media Vault, a photo vault for college events. Each uploaded photo is tagged by a pretrained model called CLIP. For Advanced Machine Learning I will focus on the reasoning that sits on top of those tags: a Naive Bayes network that infers the event category, and an EM Gaussian mixture that finds themes without labels."

**[0:45 · pick a photo, point at the tags]**
"I upload a photo. CLIP, which we did not train, gives these tags. They are the observed evidence."

**[1:15 · AML tab, point at the two bars]**
"The category is the parent node of a Bayesian network and the tags are its children, assumed independent given the category. Bayes' rule says the posterior is the prior times the product of the tag likelihoods. For these three tags that gives 0.83 for INSTITUTIONAL."
*(If asked to show it: 0.1389 / (0.1389 + 0.0289) = 0.8276.)*

**[2:15 · click `hackathon`, `coding`; then untick everything]**
"Each tag adds a weight of evidence. Adding hackathon and coding pushes it to about 94 % COMMITTEE. With no tags we are back to the 50/50 prior."

**[3:15 · Add photos → the cluster map]**
"Now a second model, with no labels at all. These photos' embeddings are reduced with PCA and a Gaussian mixture is fitted by EM, which we implemented ourselves. It alternates soft assignments and parameter updates, and the likelihood never decreases, as this curve shows. BIC chooses the number of clusters."

**[4:45 · Experiments panel]**
"Two measurements. First, maximum-likelihood estimation is consistent: with more data the learned probabilities converge to the truth and accuracy approaches the best possible, 98.3 %. Second, on 7,000 real CLIP embeddings EM reaches NMI 0.65 at ten clusters, slightly below k-means, and BIC keeps decreasing past ten, so I would not trust BIC alone."

**[6:00 · back to the top]**
"Limits, honestly: the live probabilities are hand-set. The learned model is implemented and tested but not yet trained on gallery labels, because the database is currently unavailable. The tags are not truly independent, so the percentages are over-confident. Next steps are training on labelled events, calibrating the probabilities, and trying a model that allows dependencies between tags."

**[6:45]** "I'm happy to go deeper on any part."

---

## Part 5. Questions you will probably get

**Q1. In one sentence, what does your project do?**

It tags event photos with a pretrained vision model, infers whether the event is institutional or committee-run using a Naive Bayes network, rejects duplicates, and groups photos by theme with an EM mixture model.

**Q2. Which parts are AI and which are not?**

CLIP is deep learning but ready-made. Naive Bayes and EM are the machine learning we study here. Perceptual hashing and BM25 are classical algorithms, and the website, login and storage are ordinary software.

**Q3. What did you build and what came from the earlier project?**

The website, login, folder upload, CLIP tagging, duplicate detection and the first hand-written classifier came from the earlier version. We added a learnable Naive Bayes, an EM clustering model with PCA, an AI Lab demo page, hybrid search, unit tests, experiments and fixes.

**Q4. Did you use AI tools?**

Yes: the extensions were written with an AI coding assistant. I ran the tests and experiments, studied the algorithms, and can derive and explain each one. *(Say this only because it is true; follow your college's policy.)*

**Q5. What is Naive Bayes? Why "naive"?**

A classifier that applies Bayes' rule and assumes features are independent given the class. That assumption is the "naive" part; it is rarely exactly true but often works for classification.

**Q6. Where do the probabilities come from?**

In the live app they are written by hand. We also implemented learning them by counting with smoothing (maximum likelihood), tested it against scikit-learn, and showed on simulated data that it converges to the truth. Training it on real gallery photos is the next step.

**Q7. Walk me through one prediction.**

Use the worked example in 2.4: three tags, multiply likelihoods per category, divide by the sum: 0.8276.

**Q8. What does EM do? Why does the likelihood not decrease?**

It fits a mixture model with hidden cluster membership by alternating an E-step (soft assignments) and an M-step (re-fit parameters). Each step either improves or keeps the likelihood, so it cannot go down; we confirmed this in a test and on real data. It can stop at a local optimum, so initialisation matters.

**Q9. How do you pick the number of clusters?**

By BIC, which balances fit against complexity. In our test BIC kept falling past ten, so on its own it over-splits.

**Q10. How do you know it works?**

Unit tests (21 pass), agreement with scikit-learn, a simulation showing convergence to the optimal 98.3 %, and EM scoring NMI 0.65 on 7,000 real embeddings. We also state what we have not measured: accuracy on real gallery labels.

**Q11. What are the limitations?**

Hand-set probabilities in production, tags are not independent so confidences are over-confident, only two classes, and the clustering test used a public image set rather than campus photos.

**Q12. Why did EM not beat k-means?**

Clusters in CLIP space are roughly round and similar in size, so the extra flexibility does not help; the GMM's advantage is soft assignments and a likelihood.

**Q13. Did you train CLIP?**

No. It is a pretrained OpenAI model used as a feature extractor.

**Q14. Why is the database not working?**

The free Supabase project on a friend's account is paused after inactivity. The AI demo does not depend on it; the gallery and login return when the project is restored or replaced.

---

## Part 6. Your 40 minutes

| Minutes | Do |
|---|---|
| 0-6 | Part 1. Memorise the table and the "Who wrote this?" wording. |
| 6-26 | Part 2. Redo the worked example in 2.4 on paper twice. |
| 26-31 | Part 3. Draw the pipeline from memory. |
| 31-36 | Part 4. Say the talk aloud once with the page open. |
| 36-40 | Part 5, questions Q2, Q3, Q4, Q6, Q8, Q11. |

**Numbers to remember:** 31 tags · prior 50/50 · **0.8276** · **0.9387** · **98.3 %** · **NMI 0.65** · 21 tests · BIC kept falling past 10.

**Before you present:** start Flask and the web app (commands in `docs/NOTES_AML.md`, section 5), open `/lab`, keep a folder of 8-12 photos ready.
