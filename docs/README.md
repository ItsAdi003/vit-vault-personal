# VIT Media Vault: lab notes (AML · DL · NLP)

One project, three subjects. The same pipeline is explained from three angles:

| Subject | The part of the project it owns | Notes |
|---|---|---|
| **Advanced Machine Learning** | Probabilistic graphical model (Naive Bayes), MLE/MAP, EM and Gaussian mixtures, model evaluation | [NOTES_AML.md](NOTES_AML.md) |
| **Deep Learning** | CLIP's Vision Transformer, contrastive pre-training, zero-shot vs linear probe vs CNN from scratch, attention | [NOTES_DL.md](NOTES_DL.md) |
| **Natural Language Processing** | Tokenisation, sparse (BM25) vs dense retrieval, cross-modal search, prompts, evaluation metrics | [NOTES_NLP.md](NOTES_NLP.md) |

## 1. The system in one picture

```
 photo ──► CLIP image encoder (ViT-B/32) ──► 512-D embedding ──► stored in Postgres (pgvector)
   │                    │
   │                    └─ cosine to 31 text prompts ─► softmax ─► top-5 tags (zero-shot)
   │                                                        │
   │                                                        └─► Naive Bayes ─► INSTITUTIONAL | COMMITTEE
   └─► perceptual hash (pHash, 64 bit) ─► Hamming distance < 10 ─► duplicate → REJECTED

 query text ─► CLIP text encoder ─► 512-D vector ─► cosine vs all image embeddings ─┐
            └► BM25 over tags + captions ───────────────────────────────────────────┴► RRF fusion ─► ranked photos
```

## 2. What is in the repository

| Path | What it is |
|---|---|
| `client/` | React + Vite app: landing, login, gallery, **AI Lab** (`/lab`) |
| `ai-service/app.py`, `routes/process.py` | Flask service: `/api/process` (tag, dedup, classify, embed), `/api/search` |
| `ai-service/services/` | Production pipeline: `tagger.py`, `deduplicator.py`, `classifier.py`, `embedder.py`, `search.py` |
| `ai-service/ml/` | From-scratch ML code: `nb_model.py`, `em_gmm.py`, `retrieval.py`, `zeroshot.py`, `explain.py` |
| `ai-service/routes/lab.py` | Database-free API behind the AI Lab page |
| `ai-service/experiments/` | Reproducible experiment scripts (results land in `ml/artifacts/*.json`) |
| `ai-service/tests/` | 21 unit tests (`python -m unittest discover -s tests`) |
| `supabase/schema.sql` | Schema for a fresh Supabase project (reconstructed from the code, see caveat below) |

## 3. Running it

Terminal 1, AI service:

```powershell
cd ai-service
.\venv\Scripts\Activate.ps1
$env:PYTHONIOENCODING="utf-8"
python app.py
```

Terminal 2, web app:

```powershell
cd client
npm run dev
```

* `http://localhost:5173/lab` works with **no database and no login**. This is the page to demo.
* `http://localhost:5173/gallery` needs the Supabase project (login, storage, tables).

Run the experiments (from `ai-service/`, venv active):

```powershell
python experiments/aml_nb_recovery.py      # AML  - MLE consistency          (seconds)
python experiments/nlp_paraphrase.py       # NLP  - sparse vs dense text     (seconds)
python experiments/dl_cifar10_benchmark.py # DL   - zero-shot vs probe vs CNN (downloads CIFAR-10, ~10-20 min)
python experiments/aml_em_gmm.py           # AML  - EM clustering             (downloads/encodes CIFAR-10 once, then cached)
python experiments/nlp_crossmodal.py       # NLP  - text-to-image retrieval   (uses the cached CIFAR-10 embeddings)
python experiments/train_nb.py             # AML  - learn NB from gallery labels (needs the database)
```

Then refresh the AI Lab page; every "Experiments" panel reads the saved JSON.

## 4. Demo flow for any of the three vivas (5 minutes)

1. `/lab` → **Choose a photo** (a clear event photo works best). Say what happened: encoder → tags → category.
2. **AML tab:** toggle tags and watch the posterior move; point at the weight-of-evidence bars. Add 5+ photos to the corpus, show the EM clusters, BIC and the monotone log-likelihood curve.
3. **DL tab:** zero-shot scores, switch the prompt template, show the attention overlay and the parameter counts.
4. **NLP tab:** type a sentence into the tokenizer; then run a query in the four-way search. Use a phrase that shares no word with any tag.
5. Close with the *Experiments* panel of the subject you are being examined in.

## 5. Things you must be honest about

* **Provenance.** The gallery, login and upload flow come from the original VIT Media Vault codebase (GitHub: RAUNAK2005/vit-vault). The AI Lab, the `ml/` package, `experiments/`, the hybrid search, the tests and the fixes to the AI pipeline were added afterwards. Say clearly which parts you wrote and be ready to explain every file you present.
* **Same project in three courses.** Check each instructor's policy on re-using one project, and tell each examiner which angle this submission covers.
* **Hand-crafted probabilities.** The live Naive Bayes uses a table written by hand, not learned from data. The *learned* model exists (`train_nb.py`) but needs labelled events from the database.
* **`supabase/schema.sql` is reconstructed**, not exported from the original project, and has not been run against a fresh instance. Treat the first run as a test.
* **Free Supabase projects pause** after about 7 idle days. If the gallery is down, `/lab` still works.
* Numbers in the notes are from the scripts in this repo, run on CPU. Re-run them before the viva so you can reproduce them live.
