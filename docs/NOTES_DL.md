# Deep Learning: notes and viva guide

**One-line pitch.** *The project's perception is a pretrained Vision Transformer (CLIP ViT-B/32) whose image and text encoders were trained contrastively into one 512-D space. We use it for zero-shot tagging, embeddings and search, and we measure what that pretraining is worth against a CNN trained from scratch.*

---

## 1. Where it lives in the code

| Concept | File | Where you see it |
|---|---|---|
| Model loading, image/text encoders | `ai-service/models/clip_model.py` | every request |
| Zero-shot tagging (softmax over 31 prompts) | `ai-service/services/tagger.py` | `POST /api/process` |
| Batch encoders, prompt templates, class-weight matrix | `ai-service/ml/zeroshot.py` | AI Lab → DL |
| Attention rollout (explainability) | `ai-service/ml/explain.py` | AI Lab → DL |
| Perceptual hash (classical, non-learned) | `ai-service/services/deduplicator.py` | duplicate rejection |
| Benchmark: zero-shot vs linear probe vs CNN | `ai-service/experiments/dl_cifar10_benchmark.py` | AI Lab → DL → Experiments |

---

## 2. Theory you must be able to explain

### 2.1 From pixels to tokens (Vision Transformer)

A 224×224×3 image is cut into non-overlapping 32×32 patches: `(224/32)² = 49` patches. Each patch is flattened (32·32·3 = 3 072 numbers) and linearly projected to 768 dimensions (implemented as a convolution with kernel = stride = 32). A learnable `[CLS]` token is prepended (50 tokens) and learnable position embeddings are added, because attention itself is permutation-invariant.

### 2.2 Self-attention and the transformer block

```
Q = X Wq,  K = X Wk,  V = X Wv
Attention(Q,K,V) = softmax( Q Kᵀ / √d_k ) V          d_k = 768/12 = 64 per head
block:  X ← X + MHA(LN(X));   X ← X + MLP(LN(X))     MLP: 768 → 3072 → 768, GELU
```

* `√d_k` keeps the dot products from growing with dimension, which would saturate the softmax and kill gradients.
* Multi-head attention lets different heads look for different relations; heads are concatenated and projected.
* Residual connections and LayerNorm make 12 stacked layers trainable.
* The output of `[CLS]` after the last layer, linearly projected 768 → 512, is the image embedding.

**Parameter arithmetic (check it, examiners like this).** Per layer: attention `4·d²` + MLP `2·d·4d = 8·d²` = `12·768² ≈ 7.08 M`; ×12 layers ≈ 85 M; plus patch projection (3 072×768 ≈ 2.4 M), position embeddings and the output projection ≈ **87.5 M** (the model reports 87 456 000). The text tower is **63.2 M**; the total is **151.3 M**.

### 2.3 The text tower

A 12-layer transformer, width 512, 8 heads, **causal** (masked) self-attention, a byte-pair-encoding vocabulary of 49 408 tokens and a context of 77 tokens. The feature of the final `<|endoftext|>` token, projected 512 → 512, is the text embedding. It is GPT-2-like, not BERT-like: attention is masked so each token only sees the previous ones.

### 2.4 Contrastive pre-training (the core idea of CLIP)

For a batch of N (image, caption) pairs, embeddings are L2-normalised and all N² cosine similarities are computed. With a learned scale `s` (temperature `1/s`):

```
logits_ij = s · cos(I_i, T_j)
L = ½ [ CE(rows of logits, labels = i) + CE(columns of logits, labels = j) ]     (symmetric InfoNCE)
```

Each image must pick its own caption among N−1 distractors and vice versa. The original used 400 M pairs and a batch of 32 768. `s = exp(logit_scale)` is learned, initialised to 1/0.07 and clipped at 100 (our model reports **100.0**).

### 2.5 Zero-shot classification

Write each class as text (`"a photo of a dog."`), encode it, and use the normalised text embeddings as the rows of a linear classifier:

```
p(c | image) = softmax_c( s · <f(image), g(text_c)> )
```

No weight was trained for our 31 labels: the text encoder *generates* the classifier. Prompt templates matter because captions on the web look like "a photo of ...", not like a bare word; averaging the embeddings of several templates (*prompt ensembling*) usually helps.

### 2.6 Transfer learning options

| Option | What is trained | Needs labels? | Typical use |
|---|---|---|---|
| Zero-shot | nothing | no (class names only) | no data at all |
| k-NN on features | nothing | yes (a gallery) | quick baseline |
| Linear probe | one linear layer (512×C + C) | yes, few | strong, cheap |
| Fine-tune | all or last layers | yes, more | best accuracy, risk of forgetting |
| Train from scratch | everything | many | needs lots of data |

### 2.7 Attention rollout

Raw attention of the last layer only shows one step of information flow. Rollout multiplies the head-averaged attention matrices of all layers after adding the identity (the residual path) and row-normalising: `R = Π_l (0.5·A_l + 0.5·I)`. The `[CLS]` row of `R`, reshaped to 7×7 and up-sampled, is the heat-map in the AI Lab. It is a *plausible* explanation, not a proof of causal importance.

### 2.8 Perceptual hash (classical vision, for contrast)

pHash: grayscale → 32×32 → 2-D DCT → keep the 8×8 low-frequency block → threshold at the median → 64 bits. Similar-looking images have a small Hamming distance. It is fast and robust to re-compression but not to flips, rotations or heavy crops, and it has no semantics. The CLIP embedding is the semantic counterpart.

---

## 3. How the project applies it

1. **Upload** → Flask downloads the image, converts to RGB.
2. **Tag**: one forward pass of the image encoder and a text-encoder pass over the 31 label strings; `logits_per_image` is `s·cos`; softmax; keep the top five labels above 5 %.
3. **Embed**: the same image feature, L2-normalised, 512 floats → pgvector column `image_embedding`.
4. **Dedup**: pHash vs every stored hash; Hamming distance < 10 → REJECTED.
5. **Search**: the query text is encoded by the text tower and compared with stored embeddings by cosine similarity.

CPU latency is about 1 s for the image encoder on this laptop (first call includes warm-up); the text matrix for 31 labels is cached after the first call.

---

## 4. Experiments and results

### 4.1 Pretrained features vs training from scratch (`dl_cifar10_benchmark.py`)

**Not run to completion yet, so there are no numbers to quote.** The script is ready and the CLIP embeddings of the 7 000 CIFAR-10 images are cached on this machine, but the run was interrupted during the CNN-training step. Run `python experiments/dl_cifar10_benchmark.py` (about 10 minutes on CPU, no download needed now) and then fill this table from `ml/artifacts/dl_cifar10.json` before the viva.

### 4.2 What the AI Lab shows live

* Zero-shot probabilities and cosine similarities for all 31 labels, with a prompt-template switch (`bare`, `photo`, `event`, `ensemble`).
* Attention rollout overlay.
* Parameter counts straight from the loaded model (151.3 M).
* The 64-bit pHash drawn as a grid and, with a corpus loaded, the Hamming distance to the nearest photo.

---

## 5. Live demo script (DL angle)

1. "This is CLIP ViT-B/32: 151 M parameters, an image transformer and a text transformer sharing a 512-D space." (architecture panel)
2. Analyse a photo. "Zero-shot: these bars are `softmax(100 · cosine)`; nothing here was trained on our labels." Switch the prompt template and show the scores move.
3. "This heat-map is attention rollout through all 12 layers: where the `[CLS]` token pulled information from." Say it is an explanation aid, not proof.
4. Experiments panel: "Same CIFAR-10 test images: zero-shot with no labels, a linear probe on frozen features, and a CNN trained from scratch on the same images."

---

## 6. Viva questions and model answers

### Architecture

**Q1. What is a Vision Transformer and how does an image become a sequence?**

Split into patches, flatten and linearly embed each patch, add position embeddings and a `[CLS]` token, then run standard transformer encoder layers. For 224×224 with 32×32 patches that is 49 patches + `[CLS]` = 50 tokens.

**Q2. Write the self-attention equation and explain every term.**

`softmax(QKᵀ/√d_k)V`. `Q` queries, `K` keys, `V` values are linear projections of the tokens. `QKᵀ` scores how relevant each token is to each other token; division by `√d_k` stabilises the softmax; the softmax gives weights; weights times `V` mix information.

**Q3. Why multi-head attention?**

Several lower-dimensional attention maps in parallel can capture different relations (position, texture, object parts) at the same cost as one big head.

**Q4. Why position embeddings?**

Self-attention treats tokens as an unordered set; without positional information a shuffled image gives the same output.

**Q5. CNN vs ViT: inductive bias?**

CNNs bake in locality and translation equivariance, which helps when data are scarce. ViTs have almost no such prior and need large-scale pre-training, but then model long-range relations better and scale well.

**Q6. Why is attention cost a concern and how does patch size change it?**

`O(T² d)` in the number of tokens `T`. ViT-B/32 has 50 tokens; ViT-B/16 would have 197 (≈15× more attention cost), which is why B/32 runs comfortably on a CPU.

**Q7. How is CLIP's text encoder different from BERT?**

Same transformer building blocks, but CLIP's is causal (masked, GPT-2-style) and pools at the end-of-text token; BERT is bidirectional and pools `[CLS]`. CLIP's is trained to align with images, not for masked-language-modelling.

**Q8. Count the parameters.**

See 2.2: `12 d²` per layer × 12 layers ≈ 85 M + embeddings ≈ 87.5 M for vision; 63.2 M for text; 151.3 M total.

### Training objective

**Q9. What is contrastive learning? Write the loss.**

Pull matching pairs together and push non-matching apart. With N pairs, the logit matrix `s·cos(I_i,T_j)` is classified row-wise and column-wise with cross-entropy against the diagonal (InfoNCE).

**Q10. What does the temperature do?**

It scales the logits: a small temperature (large `s`) sharpens the softmax and makes hard negatives matter more. CLIP learns it; ours is 100.

**Q11. Why large batches?**

More in-batch negatives give a tighter bound on mutual information and a harder task; CLIP used 32 768.

**Q12. Why normalise embeddings?**

It makes dot product = cosine similarity, removes magnitude as a nuisance variable, and on the unit sphere Euclidean distance is a monotone function of cosine.

### Zero-shot and transfer

**Q13. How can it classify classes it was never trained on?**

Because the text encoder maps any phrase into the same space as images; the class names become the weights of a linear classifier (section 2.5).

**Q14. Why do prompt templates change accuracy?**

Training captions are sentences, so "a photo of a dog" lies closer to the image distribution than the bare token "dog"; ensembles average out wording noise. You can see it in the AI Lab template switch.

**Q15. Zero-shot vs linear probe vs fine-tuning?**

See the table in 2.6: zero-shot needs no labels, probes train 5 130 weights on frozen features, fine-tuning updates the backbone (better, but needs more data and can forget).

**Q16. Why does a CNN trained from scratch do worse in our benchmark?**

It must learn every filter from only a few thousand 32×32 images, while CLIP learned general visual features from 400 M pairs. The gap measures the value of pre-training.

**Q17. What is overfitting and what did you use against it in the CNN?**

Fitting noise in the training set. We use data augmentation (flip, shift), dropout, weight decay (AdamW), batch normalisation and a one-cycle learning-rate schedule, and we judge on held-out images.

**Q18. Why batch normalisation? Why AdamW? What is OneCycle?**

BN normalises activations per mini-batch, allowing larger learning rates and acting as a regulariser. AdamW decouples weight decay from the adaptive update. OneCycle warms the learning rate up then anneals it, which converges quickly in a fixed epoch budget.

### Explainability and limitations

**Q19. What does attention rollout show and what are its limits?**

How much each patch contributes to `[CLS]` through all layers. Attention is not the same as importance (value vectors and later MLPs matter), and ViTs sometimes put high attention on uninformative background tokens.

**Q20. Why does the tagger sometimes give odd tags?**

(1) Our 31 labels are a small, partly overlapping vocabulary (a flower photo got `presentation` and `dance`); (2) the top-5 rule always returns several tags even when none fits; (3) a bare word is a weak prompt; (4) a domain gap between web images and campus photos. Fixes: better labels, prompt ensembles, a confidence floor, a linear probe on a few hundred labelled campus photos.

**Q21. What are CLIP's known weaknesses?**

Counting, spatial relations, fine-grained categories, text-in-image reading, low-resolution inputs (our CIFAR-10 images are up-scaled from 32 px), bias inherited from web data, English-centric text, a 77-token limit.

**Q22. How would you fine-tune it with few photos?**

Freeze most layers and train a linear head or low-rank adapters (LoRA) with a small learning rate and weight decay; early-stop on validation; avoid destroying the pre-trained features (catastrophic forgetting).

**Q23. pHash vs CLIP embeddings for duplicates?**

pHash is a cheap low-level fingerprint (fails on flips/rotations, can't see semantics). CLIP cosine > ~0.95 would also catch re-photographed or re-framed copies but costs a forward pass and needs a threshold chosen on data. They are complementary.

**Q24. Is this deep learning if you didn't train CLIP?**

The project uses a deep model and we trained and evaluated deep models in the experiments (the CNN, the linear probe). In an applied system most of the value comes from reusing pretrained networks; knowing how to adapt and evaluate them (zero-shot, probe, fine-tune) is the skill.

**Q25. How would you speed it up?**

Batch requests, ONNX/quantised model, GPU, cache embeddings (already stored), approximate nearest-neighbour index (HNSW) for search.

---

## 7. Limitations (say them first)

* No model is trained inside the app; learning happens only in the experiment scripts.
* CIFAR-10 is 32×32 and is a proxy for campus photos; the benchmark subset is 2 000 test images, so differences of about 2 points are noise.
* The tag vocabulary is small and hand-written; no accuracy has been measured on real event photos yet because that needs labelled data.

## 8. Cheat sheet

```
Tokens 49+1 · layers 12 · heads 12 · d 768 · patch 32 · embedding 512 · logit scale 100
Params: vision 87.5 M · text 63.2 M · total 151.3 M · text context 77 · vocab 49 408
Attention softmax(QKᵀ/√dk)V · InfoNCE (symmetric) · zero-shot w_c = text_enc("a photo of c")
pHash 64 bit, duplicate if Hamming < 10
```
