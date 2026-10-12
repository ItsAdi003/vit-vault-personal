# Advanced Machine Learning: notes and viva guide

**One-line pitch.** *CLIP turns a photo into tags; a Naive Bayes network turns the tags into a probabilistic decision (INSTITUTIONAL vs COMMITTEE); an EM-fitted Gaussian mixture discovers themes without labels.* The "machine learning" in AML is the probabilistic reasoning on top of the deep features.

---

## 0. What to study, in this order (about 3.5 hours)

| # | Topic | Time | You are ready when you can... |
|---|---|---:|---|
| 1 | Bayes' theorem, joint factorisation, conditional independence | 30 min | derive `P(C|t) ∝ P(C) Π P(tj|C)` from the definition of conditional probability, and say why `P(t)` can be dropped |
| 2 | Naive Bayes: Bernoulli model, log-space, Laplace smoothing, MLE vs MAP | 45 min | reproduce the 0.8276 worked example (section 3.1) on paper with a calculator |
| 3 | Evaluation: confusion matrix, precision/recall/F1, log-loss, cross-validation | 30 min | explain why accuracy alone is misleading and what stratified k-fold fixes |
| 4 | EM and Gaussian mixtures, BIC | 45 min | write the E-step and M-step, and sketch why the likelihood cannot decrease |
| 5 | PCA via SVD, why reduce dimension before a GMM | 15 min | state what the first principal component is and what "explained variance" means |
| 6 | The project: CLIP → tags → Naive Bayes; EM on embeddings | 20 min | draw the pipeline and say which parts are *yours* |
| 7 | Limitations and the numbers in section 4 | 15 min | quote the three results with their caveats |

**Concepts used in this project (one line each):** Bayesian network · conditional independence / d-separation · Naive Bayes · Bernoulli likelihood · prior and posterior · MAP decision · log-sum-exp · Laplace (add-α) smoothing · maximum-likelihood estimation · consistency of an estimator · Bayes-optimal classifier · confusion matrix, precision, recall, F1 · log-loss and Brier score · stratified k-fold cross-validation · weight of evidence (log-likelihood ratio) · latent-variable model · Gaussian mixture · EM algorithm · responsibilities · local optima and k-means++ initialisation · BIC model selection · PCA / SVD · NMI and purity.

**What is CLIP, in AML terms?** A pretrained feature extractor treated as a black box that produces tags and a 512-D vector. Do not claim you trained or designed it; say what you built *on top* of it.

**Know these three numbers cold:** 0.8276 (worked-example posterior), 0.983 (Bayes-optimal accuracy, section 4.1), NMI 0.652 ± 0.014 (EM at k = 10, section 4.2).

---

## 1. Where it lives in the code

| Concept | File | Live endpoint / page |
|---|---|---|
| Hand-crafted Naive Bayes used by the app | `ai-service/services/classifier.py` | `POST /api/process` (every upload) |
| Naive Bayes **learned by MLE**, from scratch | `ai-service/ml/nb_model.py` | `POST /api/lab/nb`, AI Lab → AML |
| EM for a Gaussian mixture + PCA, from scratch | `ai-service/ml/em_gmm.py` | `GET /api/lab/embedding-map`, AI Lab → AML |
| Learn-from-gallery script + evaluation | `ai-service/experiments/train_nb.py` | saves `ml/artifacts/nb_learned.json` |
| MLE-consistency experiment | `ai-service/experiments/aml_nb_recovery.py` | AI Lab → AML → Experiments |
| EM on real embeddings | `ai-service/experiments/aml_em_gmm.py` | AI Lab → AML → Experiments |
| Tests (parity with scikit-learn, EM monotonicity) | `ai-service/tests/test_ml.py` | `python -m unittest discover -s tests` |

---

## 2. Theory you must be able to write on paper

### 2.1 Bayesian network and Naive Bayes

A Bayesian network is a DAG whose nodes are random variables; each node has a conditional probability table given its parents. The joint factorises as `P(X1..Xn) = Π P(Xi | parents(Xi))`.

Our network has one parent and V children:

```
            ┌──────────────┐
            │  C (category)│        C ∈ {INSTITUTIONAL, COMMITTEE}
            └──┬───┬───┬───┘
               ▼   ▼   ▼
              T1  T2 ... TV           Tj ∈ {0,1}: "CLIP produced tag j"   (V = 31 labels)
```

`P(C, T1..TV) = P(C) · Π_j P(Tj | C)`: the *naive* assumption is that tags are **conditionally independent given the category**.

Bayes' rule gives the posterior used for the decision:

```
P(C | t) = P(C) Π_j P(tj | C)  /  Σ_c' P(c') Π_j P(tj | c')
prediction = argmax_c P(C=c | t)            (MAP estimate)
```

### 2.2 Why log-space

A product of 31 probabilities of 0.3 is ~1e-16; with hundreds of features it underflows to 0. We add logs instead and normalise with the log-sum-exp trick:

```
log P(c, t) = log P(c) + Σ_j log P(tj | c)
P(c | t)    = exp( s_c − max_k s_k ) / Σ_k exp( s_k − max_k s_k )
```

### 2.3 Parameter estimation

| Estimator | Formula for θ_cj = P(Tj=1 | C=c) | Comment |
|---|---|---|
| MLE | `n_cj / n_c` | zero if a tag never appeared with class c, which makes the whole posterior 0 |
| Laplace (add-α) | `(n_cj + α) / (n_c + 2α)` | posterior mean under a Beta(α, α) prior (α = 1: Laplace's rule of succession) |
| MAP | the mode of the Beta posterior | for Beta(α+1, α+1) it equals the add-α formula above |

Prior: `P(c) = n_c / n`.

### 2.4 Naive Bayes is a linear classifier

For two classes the log-odds is linear in the features:

```
log P(c1|t)/P(c0|t) = log P(c1)/P(c0)  +  Σ_j [ tj · w_j^present + (1−tj) · w_j^absent ]
w_j^present = log θ_1j/θ_0j          ← "weight of evidence" of seeing tag j
```

The AI Lab plots exactly these per-tag weights.

### 2.5 Generative vs discriminative

Naive Bayes models `P(x | c)` and `P(c)` (generative); logistic regression models `P(c | x)` directly (discriminative). They have the same decision form, but NB reaches its (higher) asymptotic error faster with little data and logistic regression wins with lots of data (Ng and Jordan, 2002). With only a few hundred labelled photos NB is a sensible choice.

### 2.6 EM and Gaussian mixtures

Model: `p(x) = Σ_k π_k N(x | μ_k, diag σ²_k)` with hidden component `z`.

```
E-step:  r_nk = π_k N(x_n | μ_k, σ²_k) / Σ_j π_j N(x_n | μ_j, σ²_j)      (responsibilities)
M-step:  N_k = Σ_n r_nk ;  π_k = N_k / N
         μ_k = Σ_n r_nk x_n / N_k
         σ²_k = Σ_n r_nk (x_n − μ_k)² / N_k
```

Why the likelihood never decreases: the E-step makes a lower bound (ELBO, via Jensen's inequality) tight at the current parameters, the M-step maximises that bound, so `log p(X | θ_new) ≥ ELBO(θ_new) ≥ ELBO(θ_old) = log p(X | θ_old)`. EM converges to a **local** optimum, so initialisation matters (we use k-means++ seeding).

Model selection: `BIC = −2 ln L̂ + p ln n` where a diagonal GMM in d dimensions has `p = 2kd + (k−1)` parameters.

---

## 3. How the project applies it

### 3.1 The production classifier (hand-crafted)

`services/classifier.py` stores `P(tag | category)` for 31 tags (for example `coding: 0.15 | 0.75`, `graduation: 0.80 | 0.10`), priors 0.5/0.5, and smooths each value with `(p + 0.1) / 1.2`.

**Worked example, tags = [portrait, faculty, presentation]** (this is the real output from the test upload):

```
              portrait   faculty   presentation    product
INSTITUTIONAL  0.30→0.333  0.65→0.625  0.70→0.667    0.1389
COMMITTEE      0.40→0.417  0.15→0.208  0.30→0.333    0.0289
posterior(INSTITUTIONAL) = 0.1389 / (0.1389 + 0.0289) = 0.8276   ← "82.8 %" in the logs
```

**Second example, [hackathon, coding]:** INSTITUTIONAL `0.167 × 0.208 = 0.0347`, COMMITTEE `0.75 × 0.708 = 0.531`, so P(COMMITTEE) = 0.9387. In log-odds: `ln 4.5 + ln 3.4 = 1.504 + 1.224 = 2.73`, `e^2.73 = 15.3`, `15.3/16.3 = 0.9387`.

Be precise about what this model is: it multiplies likelihoods of the tags that are **present** and ignores absent tags (it is a "bag of observed tags" model, not a full Bernoulli NB). The learned model in `ml/nb_model.py` is the full Bernoulli version (absent tags contribute `log(1 − θ)`).

### 3.2 The learned classifier

`BernoulliNaiveBayes.fit()` counts tag occurrences per class and applies the formulas of 2.3. It was verified to agree with scikit-learn's `BernoulliNB` to 1e-9 (`tests/test_ml.py::test_matches_sklearn_bernoulli_nb`).

Ground-truth labels come for free: the uploader picks INSTITUTIONAL or COMMITTEE when creating a folder (`events.type`). `experiments/train_nb.py` does 5-fold stratified cross-validation, scores the hand-crafted table on the same photos, and saves the final model; the AI Lab then shows both posteriors side by side.

### 3.3 Unsupervised themes with EM

`ml/em_gmm.py` takes the 512-D CLIP embeddings of the corpus, projects them to 10-D with PCA (SVD), and fits a diagonal GMM for k = 1..6. The page shows the BIC for each k, the chosen k, the clusters with their most frequent tags, and the log-likelihood per iteration.

---

## 4. Experiments and results

### 4.1 Is MLE consistent? (`aml_nb_recovery.py`, run, seconds)

Data are sampled from the hand-crafted table (so the true parameters are known); the learned model is fitted on n examples; 20 seeds per size; 5 000 held-out test points. The Bayes-optimal classifier (the true model) scores **0.9830**.

| n_train | mean abs. parameter error | test accuracy | gap to Bayes-optimal |
|---:|---:|---:|---:|
| 10 | 0.1246 | 0.9380 | 0.0450 |
| 50 | 0.0692 | 0.9741 | 0.0089 |
| 200 | 0.0373 | 0.9805 | 0.0025 |
| 1 000 | 0.0164 | 0.9824 | 0.0006 |
| 5 000 | 0.0071 | 0.9828 | 0.0002 |

Reading: error falls roughly like `1/√n` (500× more data → ~17× lower error) and accuracy converges to the Bayes-optimal value. **Caveat to say out loud:** the data are synthetic and satisfy the independence assumption exactly.

### 4.2 EM on real CLIP embeddings (`aml_em_gmm.py`, run)

7 000 CIFAR-10 images are embedded with CLIP, reduced to 10 dimensions by PCA (keeps **45.5 %** of the variance) and clustered by our own diagonal-covariance EM. The class labels are **not** used for fitting; they are used afterwards only to measure agreement (NMI = normalised mutual information, purity = share of each cluster taken by its majority class).

| k | BIC (lower is better) | EM iterations | NMI vs classes | purity |
|---:|---:|---:|---:|---:|
| 2 | −135 558 | 15 | 0.176 | 0.182 |
| 4 | −149 463 | 12 | 0.526 | 0.378 |
| 6 | −159 963 | 43 | 0.628 | 0.527 |
| 8 | −168 190 | 32 | 0.652 | 0.666 |
| **10** | −172 345 | 32 | **0.665** | 0.683 |
| 12 | −175 215 | 35 | 0.643 | 0.699 |
| 14 | −176 876 | 71 | 0.627 | 0.697 |

At k = 10 (the true number of classes) over 5 random initialisations: **NMI 0.652 ± 0.014**; k-means gets NMI 0.671 and purity 0.706. The EM log-likelihood was verified to be non-decreasing at every iteration.

How to read it honestly:

* Agreement with the hidden classes rises until k ≈ 10 and then falls: the structure really is about ten groups.
* **BIC kept decreasing up to the largest k tried (14).** BIC is therefore not a reliable way to recover "10" here; it rewards splitting classes into sub-clusters. Say this before the examiner asks.
* **EM-GMM did not beat k-means** (0.652 vs 0.671 NMI). In CLIP space clusters are roughly spherical, so the extra flexibility of per-cluster variances buys nothing, and 10 PCA dimensions keep under half of the variance.
* Some classes are semantically close (cat/dog, automobile/truck, deer/horse), so purity near 0.7 is expected.

### 4.3 Learned vs hand-crafted on the gallery (`train_nb.py`)

Not yet run: it needs the Supabase project to be reachable (labels live in `events.type`). The script and a dry run on synthetic data are in place; when the database is back it prints accuracy, macro-F1 and log-loss for *majority class*, *hand-crafted table* and *learned NB (CV)* and saves a confusion matrix. **Do not quote numbers for this experiment until you have run it.**

---

## 5. How to present this to your mentor (7 minutes)

**Before you walk in:** both servers running (see the commands at the end of this section), `http://localhost:5173/lab` open, and a folder with 8-12 photos of mixed content (stage, people with laptops, group photo, food, outdoors) ready for the corpus. Do one dry run.

| Time | Say | Show on screen |
|---|---|---|
| 0:00-0:45 | **Problem and pipeline.** "A campus photo vault must decide, for every upload, what it shows and what kind of event it belongs to. A deep model, CLIP, turns the photo into tags; our contribution here is the probabilistic reasoning on top: a Naive Bayes network that infers the event category, and an EM mixture model that discovers themes without labels." | The pipeline diagram from `docs/README.md`, or the AI Lab header |
| 0:45-2:15 | **The Bayesian network.** "Category is the parent, tags are the children, conditionally independent given the category. The posterior is prior times the product of tag likelihoods, computed in log-space. Here it is by hand for three tags..." Then do the worked example from section 3.1 (0.8276). | AI Lab → *Choose a photo* → AML tab; point at the tags, then at the posterior bars |
| 2:15-3:30 | **Evidence is additive.** "Because it is linear in log-odds, every tag contributes a weight of evidence. Watch the posterior move when I add `hackathon` and `coding`." Click both tags: COMMITTEE jumps to ~94 %. Untick everything: back to the 50/50 prior. | The tag toggles and the weight-of-evidence bars |
| 3:30-5:00 | **Latent-variable model.** "Without any labels, EM fits a Gaussian mixture to the CLIP embeddings. E-step: responsibilities. M-step: weighted means and variances. The likelihood never decreases. BIC picks k." | *Photo corpus → Add photos* (select your folder) → the cluster map, BIC chips and the log-likelihood curve |
| 5:00-6:15 | **Evidence it works, and where it does not.** "MLE is consistent: parameter error falls like 1/√n and accuracy approaches the Bayes-optimal 98.3 %. On 7 000 real embeddings EM reaches NMI 0.65 at k = 10, slightly below k-means; BIC keeps decreasing past 10, so I would not trust it alone." | AML tab → *Experiments (saved results)* |
| 6:15-7:00 | **Limits and next step.** "Probabilities in the live app are hand-set; the learned model is implemented and tested but not yet trained on gallery labels. Tags are not truly independent, so posteriors are over-confident. Next: train on labelled events, calibrate, and try Tree-Augmented NB." | Back to the page header |

Commands to start the demo (two terminals, from the project folder):

```powershell
cd ai-service
.\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
python app.py
```

```powershell
cd client
npm run dev
```

**If something fails on the day:** `/lab` needs no database or login. If the page shows "AI service not reachable", Flask is not running. If no tags appear, check the Flask terminal for a Traceback and restart it.

---

## 6. Viva questions and model answers

### Basics

**Q1. What is a probabilistic graphical model? Why is Naive Bayes one?**

A PGM encodes a joint distribution as a graph: nodes are random variables, edges are dependencies, and the graph structure gives a factorisation. Naive Bayes is the simplest Bayesian network: a class node with an arrow to each feature, which gives `P(C) Π P(Xj|C)`.

**Q2. State Bayes' theorem and name the terms.**

`P(C|x) = P(x|C) P(C) / P(x)`: posterior = likelihood × prior / evidence. The evidence is the same for every class so for classification we only compare the numerators.

**Q3. What does "naive" mean? Is the assumption true for our tags?**

That features are conditionally independent given the class. It is not exactly true: CLIP's tags are the top five of one softmax, so they compete with each other, and tags like `stage` and `auditorium` co-occur. It still works for classification because the decision depends on the *ranking* of the class scores, which tolerates moderate violations; what suffers is calibration, so the posteriors are over-confident.

**Q4. Why do you work with logarithms?**

Products of many small probabilities underflow. Logs turn the product into a sum and are monotone, so the argmax is unchanged; log-sum-exp normalises without overflow.

**Q5. What is Laplace smoothing and why do you need it?**

Without it a tag never seen with a class gives probability 0 and wipes out the whole product. Adding α pseudo-counts gives every tag a non-zero probability. Bayesian reading: the posterior mean of a Bernoulli parameter under a Beta(α, α) prior.

**Q6. MLE vs MAP vs full Bayesian?**

MLE maximises `P(data | θ)`; MAP maximises `P(data | θ) P(θ)` so the prior acts as a regulariser; full Bayesian keeps the whole posterior over θ. Add-α smoothing is MAP/posterior-mean with a Beta prior.

### About this project

**Q7. Where do your probabilities come from?**

Honest answer: the live classifier uses a **hand-written table** based on domain intuition (for example "graduation" is much more likely at institutional events). We also implemented the learned version: counts from photos whose folder has a known type, with Laplace smoothing, evaluated by cross-validation. On synthetic data from the known table it provably recovers the parameters (section 4.1).

**Q8. Draw your Bayesian network and list its conditional independences.**

C → T1…T31. Given C, every pair (Ti, Tj) is independent (they are d-separated by the common parent). Marginally they are dependent through C. The Markov blanket of C is all the tags (they are its children, there are no co-parents).

**Q9. Show that your classifier is linear.**

Take the log-odds of the two posteriors: the evidence term cancels, the prior gives a bias, and each tag adds `log θ_1j/θ_0j` if present (or `log (1−θ_1j)/(1−θ_0j)` if absent). So the decision is `sign(b + w·t)`; the AI Lab draws w.

**Q10. How do you evaluate it? Why not just accuracy?**

Confusion matrix, precision, recall, F1 per class, macro-F1, and probabilistic metrics: log-loss and Brier score for calibration. Accuracy hides class imbalance (a model that always says INSTITUTIONAL scores the majority rate). We use stratified k-fold so every fold has both classes. Caveat: photos from the same folder are correlated, so fold-wise accuracy is optimistic; grouping folds by folder is the fix.

**Q11. Is "82.8 %" really 82.8 % sure?**

No. NB posteriors are typically over-confident because it multiplies evidence from dependent features as if independent. Calibration would be checked with a reliability diagram and fixed with Platt scaling or isotonic regression.

**Q12. What happens with an unseen tag or an empty tag list?**

Unseen tags get a default likelihood of 0.3 in the live code (and are ignored by the learned model's vocabulary). An empty list returns the prior, 50/50 (visible in the AI Lab when you untick every tag).

**Q13. What if the classes were imbalanced?**

The prior would be estimated as `n_c / n` and would shift decisions towards the majority class. For a deployment decision you can change the threshold on the posterior or the prior (cost-sensitive learning).

### EM / GMM

**Q14. Explain EM in two sentences.**

When the data have hidden variables (which component generated a point) we alternate between estimating the posterior over those variables with the current parameters (E-step) and maximising the expected complete-data log-likelihood (M-step); each round cannot decrease the data likelihood.

**Q15. Prove (sketch) that EM never decreases the likelihood.**

`log p(X|θ) = L(q, θ) + KL(q || p(Z|X,θ))` with `L` the ELBO. E-step sets `q = p(Z|X,θ_old)` so KL = 0 and `L = log p(X|θ_old)`. M-step maximises `L` over θ, so `log p(X|θ_new) ≥ L(q, θ_new) ≥ L(q, θ_old) = log p(X|θ_old)`.

**Q16. Difference between k-means and a GMM?**

k-means makes hard assignments and implicitly uses equal spherical clusters; GMM gives soft responsibilities and learns weights and (diagonal or full) covariances. k-means is the limit of a GMM with shared isotropic covariance as the variance → 0.

**Q17. Why PCA before the GMM? Why diagonal covariance?**

512-D with a few dozen photos would make full covariances singular and diagonal ones overly flexible; PCA to 10-D keeps the main variation and makes `2kd + k − 1` parameters affordable. Diagonal covariance keeps the model small and the M-step closed-form.

**Q18. How do you choose k?**

By BIC (penalises parameters by `p ln n`); AIC and cross-validated likelihood are alternatives. BIC picks the number of Gaussian blobs that fit the 10-D projection, which need not equal the number of semantic classes. In our 7 000-point experiment BIC kept decreasing up to the largest k we tried (14) while agreement with the true classes peaked near k = 10, so we treat BIC as a guide and confirm with a second criterion.

**Q19. What can go wrong in EM?**

Local optima (use several restarts or k-means++), singularities when a component collapses on one point (we add a variance floor `reg = 1e-6`), slow convergence near flat regions.

### Advanced / trap questions

**Q20. Could you replace the hard tags by CLIP's probabilities?**

Yes: that is *soft (virtual) evidence*. Instead of thresholding at 5 %, weight each tag's log-likelihood by its probability, or feed the 31 probabilities to a logistic regression. Thresholding discards information, which we accept for interpretability.

**Q21. Naive Bayes vs logistic regression on this task?**

Same functional form; logistic regression optimises conditional likelihood and usually wins with enough data and correlated features; NB needs no iterative optimisation, is trivially explainable and learns from a handful of samples. With a few hundred photos both are reasonable.

**Q22. How would you capture dependence between tags?**

Tree-Augmented Naive Bayes (each tag may have one extra tag parent, chosen by the Chow-Liu maximum spanning tree of conditional mutual information), or a general Bayesian network learned with score-based search; exact inference by variable elimination (for example with `pgmpy`).

**Q23. Complexity?**

NB training `O(nV)`, prediction `O(V)`. One EM iteration `O(nKd)`.

**Q24. Is this "advanced"? Where is the machine learning if CLIP does the vision?**

The pipeline is a two-stage model: a *deep representation* (CLIP) and a *probabilistic reasoner* (Bayesian network). The AML content is the reasoning layer: structure, parameter learning (MLE/MAP), inference (MAP, log-sum-exp), model selection (BIC), latent-variable learning (EM), and the evaluation methodology. We also quantify when it works (consistency experiment) and when assumptions break (correlated tags, grouped folds).

### Questions about your results

**Q25. What are NMI and purity? Why not just accuracy for clustering?**

Cluster indices are arbitrary, so you cannot compare them with class labels directly. Purity assigns each cluster its majority class and computes the resulting accuracy; it increases trivially with more clusters. NMI is the mutual information between cluster and class labels, normalised to [0, 1], and is invariant to relabelling and less biased by the number of clusters.

**Q26. Why did BIC keep decreasing past k = 10?**

Real classes are not single Gaussians: each class spreads over several sub-modes (different breeds, poses, backgrounds), so splitting a class into sub-clusters improves the likelihood by more than the penalty `p ln n`. With 7 000 points the penalty is small relative to the likelihood gain. BIC measures density-fit quality, not semantic class count.

**Q27. EM-GMM did not beat k-means. Is something wrong?**

No. k-means is the hard-assignment, equal-spherical-covariance limit of a GMM. If clusters in the embedding are roughly spherical and similar in size, the extra parameters of the GMM do not help and can overfit; 10 PCA dimensions also keep only 45 % of the variance. The GMM's advantages are soft assignments (a photo can be 60/40 between two themes), cluster weights and a likelihood you can use for model selection.

**Q28. How do you know your EM implementation is correct?**

(1) The average log-likelihood is non-decreasing at every iteration (asserted in a unit test and checked on the real run). (2) It recovers three well-separated synthetic clusters with NMI above 0.95. (3) Its final average log-likelihood matches scikit-learn's `GaussianMixture(covariance_type="diag")` within 0.05, and BIC picks the true k on the synthetic data.

**Q29. How do you know your Naive Bayes implementation is correct?**

Its posteriors equal scikit-learn's `BernoulliNB` to 1e-9 on the same data (unit test), and in the consistency experiment its parameter error falls like `1/√n` towards the Bayes-optimal classifier's accuracy of 0.983.

**Q30. Your consistency experiment samples from your own model. Isn't that circular?**

It is deliberately a *sanity check of the estimator*, not evidence that real photos obey the model: it shows that if the naive-Bayes assumption holds, the code recovers the truth and converges to the best possible accuracy. Whether real tags satisfy the assumption is a separate question, and the learned-vs-hand-crafted comparison on gallery labels (`train_nb.py`) is the experiment that answers it; it needs the database and has not been run, which we state openly.

**Q31. What would change if INSTITUTIONAL events were 90 % of the data?**

The estimated prior `n_c/n` would move to 0.9, shifting the decision boundary towards INSTITUTIONAL; accuracy would look high even for a trivial classifier, so we would report macro-F1, per-class recall and a confusion matrix, and consider class-weighted decisions.

**Q32. What is the single biggest weakness of your approach?**

That the category probabilities in production are hand-set and the independence assumption is violated, so reported confidences are not calibrated. The mitigation is built (learned model, cross-validation, explanation of each tag's contribution) but must be validated on real labelled events before trusting the numbers.

---

## 7. Limitations (say them before the examiner does)

* The production probabilities are hand-set; the learned model has not been validated on real gallery labels yet.
* Independence is violated (top-5 competition, co-occurring scenes); posteriors are not calibrated.
* Labels are folder types chosen by uploaders, which are noisy and coarse (only two classes).
* The EM experiment measures agreement with CIFAR-10 classes, not with campus-event themes.

## 8. Cheat sheet

```
Posterior          P(c|t) ∝ P(c) Π P(tj|c)         MAP: argmax
Log-odds           b + Σ wj,  wj = log θ1j/θ0j
Laplace            θ = (n_cj + α) / (n_c + 2α)
GMM                p(x) = Σ πk N(x|μk,σ²k)
EM                 E: r_nk ∝ πk N(xn|θk)   M: weighted mean/var
BIC                −2 ln L + p ln n,  p = 2kd + k − 1
Numbers            31 tags · 0.5/0.5 prior · α = 0.1 (hand table) · 5-fold CV · PCA 10-D
```
