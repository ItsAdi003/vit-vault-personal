# Natural Language Processing: notes and viva guide

**One-line pitch.** *A search query is language; photos are not. We make them comparable by encoding both into one vector space, and we compare that dense retrieval with classic sparse retrieval (keyword, TF-IDF, BM25) and with a rank-fusion hybrid, measuring each.*

---

## 1. Where it lives in the code

| Concept | File | Where you see it |
|---|---|---|
| Tokenisation, stop-words, stemming, TF-IDF, BM25 | `ai-service/ml/retrieval.py` | AI Lab → NLP, `/api/lab/search` |
| CLIP text encoder, prompt templates | `ai-service/ml/zeroshot.py`, `models/clip_model.py` | `/api/lab/tokenize`, search |
| Reciprocal Rank Fusion + ranking metrics | `ai-service/ml/retrieval.py` | hybrid column, experiments |
| Gallery search over the database | `ai-service/services/search.py`, `POST /api/search` | the AI toggle in the gallery search box |
| Paraphrase retrieval experiment | `ai-service/experiments/nlp_paraphrase.py` | AI Lab → NLP → Experiments |
| Text-to-image retrieval experiment | `ai-service/experiments/nlp_crossmodal.py` | AI Lab → NLP → Experiments |

---

## 2. Theory you must be able to explain

### 2.1 Text to tokens

Normalisation (lower-case, strip punctuation) → tokenisation → optional stop-word removal and stemming.

* **Stemming** chops suffixes (`dancing`, `dances`, `dance` → `danc`); it is crude but cheap. **Lemmatisation** uses a dictionary and returns real words (`better` → `good`).
* **Byte-pair encoding (BPE)** is what CLIP uses: start from characters and repeatedly merge the most frequent adjacent pair until the vocabulary has 49 408 entries. Rare words split into pieces, so there are no out-of-vocabulary words. `</w>` marks the end of a word. The sequence is wrapped in `<|startoftext|>` and `<|endoftext|>` and cut at **77 tokens**.

Real output from the AI Lab for *"people dancing on stage"*:

```
<|startoftext|>  people</w>  dancing</w>  on</w>  stage</w>  <|endoftext|>      (6 tokens)
```

### 2.2 Sparse retrieval

**Vector space model.** Each document is a sparse vector over the vocabulary; relevance is overlap between query and document words.

```
TF-IDF   w(t,d) = (1 + ln tf) · idf(t),   idf(t) = ln((1+N)/(1+df)) + 1
         score  = cosine(query vector, document vector)
```

**BM25** (Okapi), the usual strong baseline:

```
score(D,Q) = Σ_{t∈Q} idf(t) · f(t,D)·(k1+1) / ( f(t,D) + k1·(1 − b + b·|D|/avgdl) )
idf(t) = ln(1 + (N − n_t + 0.5)/(n_t + 0.5))        k1 = 1.5, b = 0.75
```

`k1` makes term frequency saturate (the 10th occurrence adds little); `b` normalises for document length. Both are *exact-term* methods: they cannot match "garden" with "nature".

### 2.3 The vocabulary-mismatch problem

Users and documents use different words for the same thing: synonymy (`car`/`automobile`), paraphrase (`a talk` / `a lecture`), hypernymy (`dance` / `cultural event`). Sparse methods fail here by construction. In our paraphrase test set, **83 %** of paraphrase pairs share no word at all.

### 2.4 Dense retrieval

A **bi-encoder** maps query and documents independently to vectors; relevance is cosine similarity. Documents are encoded once and indexed; a query costs one forward pass plus a nearest-neighbour search (HNSW in pgvector). A **cross-encoder** feeds the pair jointly through one transformer: more accurate, too slow to score a whole collection, so it is used for re-ranking.

### 2.5 Cross-modal retrieval

CLIP is a bi-encoder whose two towers are different modalities. Because they were trained contrastively, a sentence and the photo it describes are near each other. Therefore a *text* query can retrieve *images* without any caption or tag, which is exactly what sparse text methods cannot do (they need text on the document side).

### 2.6 Prompt engineering

The text side of a zero-shot classifier or a search query is a prompt. "a photo of a dog" is closer to training captions than "dog". Averaging the embeddings of several templates (*prompt ensembling*) reduces wording noise. The AI Lab exposes four template sets.

### 2.7 Hybrid retrieval and Reciprocal Rank Fusion

Scores from different systems are on different scales (BM25 is unbounded, cosine is in [−1, 1]), so we fuse **ranks**:

```
RRF(d) = Σ_r 1 / (k + rank_r(d))          k = 60 (Cormack et al., 2009)
```

### 2.8 Evaluation metrics

With a ranked list and a set of relevant items:

```
Precision@k  = relevant in top k / k           Recall@k = relevant in top k / |relevant|
RR           = 1 / rank of first relevant      MRR = mean RR over queries
AP           = mean of Precision@i at each relevant hit     MAP = mean AP
nDCG@k       = DCG / ideal DCG,  DCG = Σ rel_i / log2(i+1)
```

Tiny example (also a unit test): ranking `[3,1,4,0]`, relevant `{1,0}` → P@2 = 0.5, RR = 0.5, AP = (1/2 + 2/4)/2 = 0.5.

---

## 3. How the project applies it

1. **Offline.** Every uploaded photo gets (a) auto-tags (text), (b) a 512-D embedding (vector).
2. **Query time** (`POST /api/search`):
   * *sparse*: tokenise + stem the query, BM25 over `tags + caption + event title`;
   * *dense*: CLIP text embedding, cosine against all stored image embeddings;
   * *fuse*: RRF of the two rankings; keep a result if it has a sparse hit or its cosine is within 0.06 of the best (a relative cut-off, because CLIP cosines for matches are only ~0.2-0.35).
3. If the database is unreachable the endpoint returns `results: null` and the gallery falls back to the old substring filter.
4. The **AI Lab** shows four rankers side by side on a corpus you load: *keyword* (what the old search bar did), *BM25*, *CLIP dense*, *hybrid*.

Observed in the browser test: query *"colourful flowers in a garden"* → keyword: no match; BM25: one hit (the file name contains "flower_garden"); CLIP dense: all five photos ranked with the flower photo first (cosine 0.195 vs 0.158 for the next).

---

## 4. Experiments and results

### 4.1 Paraphrase retrieval, sparse vs dense (`nlp_paraphrase.py`, run)

48 campus-event phrases = 12 intents × 4 paraphrases written by us (for example "students performing a dance on stage" / "cultural fest dance competition" / "choreography show with performers"). Each phrase is a query against the other 47; the three siblings are relevant. The corpus order is shuffled to avoid tie-break leakage.

| Method | MAP | MRR | P@3 | nDCG@3 |
|---|---:|---:|---:|---:|
| TF-IDF (sparse) | 0.230 | 0.392 | 0.181 | 0.195 |
| BM25 (sparse) | 0.234 | 0.403 | 0.181 | 0.198 |
| **CLIP text tower (dense)** | **0.686** | **0.872** | **0.639** | **0.673** |
| Hybrid RRF (BM25 + CLIP) | 0.527 | 0.637 | 0.438 | 0.439 |

How to read it honestly:

* Dense is about 3× better than sparse because 83 % of relevant pairs share no word. **The test set was written to be paraphrase-heavy**, which favours dense; on queries that reuse tag words, sparse would do much better.
* **Hybrid is worse than dense alone** here. Equal-weight RRF lets a mostly-uninformative ranker drag down a good one. A weighted or gated fusion, or fusing only when BM25 has strong hits, is the fix; we report the result rather than tune it away.
* CLIP's text tower is trained to align with images, not as a sentence encoder; a dedicated model such as MiniLM would likely beat it on pure text-text similarity (the script has a `--minilm` flag; not run).

### 4.2 Text-to-image retrieval and query wording (`nlp_crossmodal.py`)

**Not run yet, so there are no numbers to quote.** `python experiments/nlp_crossmodal.py` takes seconds now that the CIFAR-10 test embeddings are cached; fill this section from `ml/artifacts/nlp_crossmodal.json` afterwards.

---

## 5. Live demo script (NLP angle)

1. *NLP tab → tokenizer.* Type "students dancing on stage". "On the sparse side it becomes stemmed words; on CLIP's side BPE tokens and a 512-D vector with norm 1."
2. Load a few photos into the corpus. Search for a phrase that shares no word with any tag (for example "colourful flowers in a garden"). "Keyword and BM25 return nothing or one lucky filename hit; dense retrieval ranks the right photo first because meaning is compared in embedding space."
3. Explain the hybrid column and RRF (`1/(60 + rank)`).
4. *Experiments panel.* "MAP 0.69 for dense vs 0.23 for BM25, and honestly: hybrid with equal weights is worse than dense on this paraphrase-heavy set."
5. Mention what is missing: multilingual queries, negation, and a labelled evaluation set from the real gallery.

---

## 6. Viva questions and model answers

### Text processing

**Q1. What is tokenisation? Why subwords?**

Splitting text into units a model can index. Subword (BPE/WordPiece) keeps the vocabulary small and removes out-of-vocabulary words because unknown words decompose into known pieces.

**Q2. Stemming vs lemmatisation vs stop-word removal?**

Stemming is rule-based suffix chopping (fast, may produce non-words); lemmatisation uses a lexicon and POS to return a dictionary form; stop-word removal drops very frequent function words that carry little discriminative information in sparse models. Neural encoders usually keep them.

**Q3. How does BPE work?**

Count symbol pairs in a corpus, merge the most frequent pair into a new symbol, repeat for a fixed number of merges. CLIP uses a byte-level variant with 49 408 symbols and lower-cases text.

**Q4. What is the 77-token limit and what happens above it?**

The text transformer's context length; longer text is truncated (the tokenizer panel shows `truncated`). Search queries are far shorter; long captions would lose their tail.

### Sparse retrieval

**Q5. Define TF-IDF and explain its intuition.**

Term frequency rewards words that occur often in a document; inverse document frequency down-weights words that occur in many documents. The product highlights terms that are frequent here and rare elsewhere.

**Q6. Why is BM25 usually better than TF-IDF?**

Saturating term frequency (k1) and length normalisation (b), plus a well-behaved idf. A word repeated 20 times does not score 20× more.

**Q7. What are k1 and b?**

`k1` controls saturation (1.2-2.0); `b ∈ [0,1]` controls how strongly long documents are penalised.

**Q8. What is the vocabulary-mismatch problem?**

Query and relevant document use different words for the same concept; exact-match methods score 0. Remedies: stemming, query expansion, synonyms, dense embeddings.

### Dense retrieval

**Q9. How does dense retrieval work?**

Encode query and documents into vectors with a neural encoder; score by cosine or dot product; retrieve the nearest neighbours, usually with an ANN index.

**Q10. Bi-encoder vs cross-encoder?**

Bi-encoder: independent encoding, precomputable document vectors, fast, slightly less accurate. Cross-encoder: joint encoding of the pair, more accurate, cost grows with the number of candidates, so use it to re-rank a short list.

**Q11. Why cosine similarity instead of Euclidean distance?**

Magnitude is mostly irrelevant for semantic similarity; on L2-normalised vectors cosine = dot product and squared Euclidean distance = `2 − 2·cos`, so rankings are identical. We normalise so pgvector's cosine operator can be used with an index.

**Q12. Is CLIP's text encoder like BERT?**

No, and this is a common slip. It is a transformer with **causal** attention (GPT-2 style) pooled at the end token, trained contrastively against images. BERT is bidirectional and trained with masked-language-modelling. What they share: a transformer producing dense vectors that you compare with cosine.

**Q13. Can I embed the query with BERT and the images with CLIP?**

No: the two spaces were not trained to align, so distances between them are meaningless. That is why joint contrastive training is needed.

**Q14. How can text retrieve images with no caption?**

Both towers map into one 512-D space where matching text-image pairs are close. The image embedding is stored; only the query needs encoding.

**Q15. Why does "a photo of a ..." help?**

It matches the style of training captions and reduces ambiguity (is "bank" a river bank?). Ensembling templates averages out phrasing noise.

### Fusion and evaluation

**Q16. What is RRF and why use ranks instead of scores?**

Reciprocal Rank Fusion sums `1/(k+rank)` over systems. Scores from different systems are not comparable; ranks are, and RRF needs no tuning (k = 60 is conventional).

**Q17. Why did the hybrid lose to dense in your experiment?**

Because 83 % of the relevant pairs have no word overlap, BM25's ranking is mostly noise, and equal weights let it dilute the dense ranking. Hybrid helps when exact tokens matter (ids, rare names, tag words). Fix: weight the rankers or fuse only above a BM25 confidence.

**Q18. Define MAP, MRR and nDCG and when you would choose each.**

MRR when only the first relevant result matters (known-item search); MAP when there are many relevant items and order matters; nDCG when relevance is graded. Precision@k when users look at one screen.

**Q19. How would you evaluate the real gallery search without labels?**

Use the metadata that CLIP never sees: folder title and caption written by the uploader. Query = folder title, relevant = photos in that folder. Or collect click logs, or have two annotators label 30 query-photo pairs.

**Q20. How do you know the improvement is not noise?**

48 queries is small; report mean and a bootstrap confidence interval, and use a paired test across queries.

### Limits and extensions

**Q21. Does it work for Hindi or Marathi queries?**

Poorly: the CLIP text tower is English-centric. Options: translate the query first (machine translation), or use a multilingual CLIP.

**Q22. Negation and counting ("no people", "three students")?**

CLIP embeddings mostly encode which concepts are present; negation and counts are unreliable. A re-ranker or an explicit filter on tags is needed.

**Q23. How is this scalable?**

Exact cosine over all vectors is O(N·d). pgvector with an HNSW index gives approximate nearest-neighbour search in sub-linear time; we create that index in `supabase/schema.sql`.

**Q24. How would you improve the search?**

Learn fusion weights on labelled queries, add a cross-encoder or caption model for re-ranking, query expansion, multilingual encoder, and collect relevance feedback.

**Q25. Where exactly is the NLP in a vision project?**

Query understanding (tokenisation, normalisation), the text encoder, prompt design for zero-shot labels, sparse vs dense retrieval, fusion, and retrieval evaluation. The documents happen to be images; the problem (matching a query to a collection under vocabulary mismatch) is the classic IR/NLP problem.

---

## 7. Limitations (say them first)

* Our paraphrase set is small (48) and was written by us to be hard for sparse methods.
* The gallery's hybrid ranking has not been evaluated on real labelled queries; the experiment that would do it needs the database.
* English only; the text encoder has a 77-token limit; no spelling correction or query expansion.

## 8. Cheat sheet

```
BM25   Σ idf · f(k1+1) / (f + k1(1−b+b|D|/avgdl))     k1=1.5 b=0.75
TF-IDF (1+ln tf)·(ln((1+N)/(1+df))+1), cosine
RRF    Σ 1/(60 + rank)
Metrics P@k, Recall@k, RR/MRR, AP/MAP, nDCG@k
CLIP text: 12 layers · width 512 · 8 heads · causal · 77 tokens · vocab 49 408 · out 512-D unit vector
Results (paraphrase): MAP  BM25 0.234 · CLIP 0.686 · hybrid 0.527
```
