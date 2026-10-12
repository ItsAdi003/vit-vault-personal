# VIT Media Vault

An AI-assisted photo vault for campus events at Vidyalankar Institute of Technology. Photos are tagged, de-duplicated, classified and made searchable by natural language, using models that run locally on CPU.

| Capability | How it works |
|---|---|
| Auto-tagging | CLIP ViT-B/32 zero-shot scores against 31 text prompts, top-5 above 5 % |
| Duplicate rejection | 64-bit perceptual hash (pHash), Hamming distance < 10 |
| Event category | Naive Bayes over the tags: INSTITUTIONAL vs COMMITTEE (hand-crafted table + a learnable Bernoulli NB) |
| Semantic search | CLIP text embedding vs stored image embeddings (pgvector), fused with BM25 by Reciprocal Rank Fusion |
| Theme discovery | Gaussian mixture fitted by our own EM on PCA-reduced embeddings, k chosen by BIC |
| **AI Lab** (`/lab`) | Every model above, live, with no database needed |

## Quick start

```powershell
# terminal 1: AI service
cd ai-service
python -m venv venv; .\venv\Scripts\Activate.ps1
pip install -r requirements.txt
copy .env.example .env            # then fill in the Supabase values
$env:PYTHONIOENCODING="utf-8"; python app.py

# terminal 2: web app
cd client
npm install
copy .env.example .env.local      # then fill in the Supabase values
npm run dev                       # http://localhost:5173  ·  AI Lab: http://localhost:5173/lab
```

The AI Lab works without Supabase. The gallery, login and uploads need a Supabase project: use the existing one, or create your own and run [`supabase/schema.sql`](supabase/schema.sql).

## Tests and experiments

```powershell
cd ai-service
python -m unittest discover -s tests          # 21 tests
python experiments/aml_nb_recovery.py         # more in docs/README.md
```

## Documentation

Study notes with viva questions for three courses: [docs/README.md](docs/README.md)
([AML](docs/NOTES_AML.md) · [Deep Learning](docs/NOTES_DL.md) · [NLP](docs/NOTES_NLP.md)).

## Credits

The gallery, authentication and upload flow originate from the first version of this project by RAUNAK2005 (github.com/RAUNAK2005/vit-vault). The AI Lab, the `ml/` package, the experiments, hybrid search and tests were added in this repository.
